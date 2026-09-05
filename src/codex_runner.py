"""Runs `codex exec --json` as the wrapped agent and streams structured events.

Speaks the same de-facto runner interface as ClaudeRunner (run_stream/run_collect
emitting StreamEvents plus a final raw dict), so every SSE/chat/Responses/
delegation layer works unchanged. Differences from the Claude dialect:

- codex assigns its own session id (thread_id, uuid-v7) on the FIRST event of a
  fresh run; we capture it and rebind the SessionRegistry key to it, so the
  next turn resumes via `codex exec resume <thread_id>`.
- `--json` is item-granular: there are no text deltas, so text/thinking arrive
  as whole blocks (stream_partial_messages is forced off).
- usage arrives on turn.completed; there is no cost figure (cost_usd stays 0).
- ADJUSTMENT POINT: _handle_event below encodes the 0.153.x JSONL schema
  (thread.started / turn.* / item.*). tests/test_codex_runner.py's fake binary
  pins exactly this schema; if a codex upgrade changes it, update both together.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from .agent_runner import BaseAgentRunner, SessionRegistry, StreamEvent, TurnState
from .config import CODEX_EFFORT_CHOICES

log = logging.getLogger("claude_wrapper.runner")


class CodexRunner(BaseAgentRunner):
    """Runs ``codex exec --json`` as a subprocess and streams structured events."""

    agent_label = "codex"
    wrapper_assigns_session_id = False

    def __init__(
        self,
        registry: SessionRegistry,
        workspace_root: Path,
        agent_bin: str = "codex",
        request_timeout_seconds: int = 1800,
        effort: str = "",
        clarify_system_prompt: str = "",
        clarify_disallowed_tools: tuple[str, ...] = (),
        workspace_system_prompt: str = "",
        stream_partial_messages: bool = False,
        executor=None,
    ):
        # `clarify_disallowed_tools` is accepted and ignored — AskUserQuestion
        # is a Claude tool; the kwarg stays so deps constructs both runner
        # classes identically. codex --json has no incremental deltas, so
        # partial mode is meaningless and forced off regardless of the caller.
        super().__init__(
            registry=registry,
            workspace_root=workspace_root,
            agent_bin=agent_bin,
            request_timeout_seconds=request_timeout_seconds,
            effort=effort,
            clarify_system_prompt=clarify_system_prompt,
            clarify_disallowed_tools=clarify_disallowed_tools,
            workspace_system_prompt=workspace_system_prompt,
            stream_partial_messages=False,
            executor=executor,
        )

    def _effort_choices_for(self, model: str) -> tuple[str, ...]:
        # Runner ACCEPTANCE set, independent of SETTINGS.agent (the test suite
        # constructs this runner under claude-mode frozen SETTINGS) and wider
        # than the ADVERTISED set: an explicit ":none" suffix must reach the
        # CLI, not silently fall back to codex's default (medium).
        return CODEX_EFFORT_CHOICES

    def _build_argv(
        self,
        session_uuid: str,
        model: Optional[str],
        resume: bool,
        extra_args: Optional[list[str]] = None,
        effort: Optional[str] = None,
        clarify: bool = False,
        workspace_hint: bool = False,
        capability_gated: bool = True,
    ) -> list[str]:
        argv = [self.agent_bin, "exec"]
        if resume:
            # Resume the codex-assigned thread bound to this key by a prior turn.
            argv += ["resume", session_uuid]
        # Fresh runs pass NO session flag: codex mints a uuid-v7 thread id and
        # announces it in thread.started; run_stream binds it to the key.
        argv += [
            "--json",
            "--skip-git-repo-check",  # session cwds are not git repos
            # The container IS the sandbox (network-isolated agent + squid, README
            # "Sandboxed deployment"): codex's own approvals/sandbox layer is
            # redundant here, and its write restrictions would break the very
            # tool use these turns exist for (session workspaces, generated
            # files). This flag is sanctioned HERE ONLY — the refresher, which
            # has ordinary egress and no network boundary, runs its turn under
            # codex's own read-only sandbox instead, where a sandbox that
            # fails to set up fails CLOSED (no turn, logged, retried) rather
            # than open (see entrypoint.sh cmd_codex_refresher).
            "--dangerously-bypass-approvals-and-sandbox",
            "-c", "check_for_update_on_startup=false",  # no update probe through squid
        ]
        if model:
            argv += ["--model", model]
        eff, _src = self._resolve_effort(model, effort)
        if eff:
            # No --effort flag on exec; the -c TOML override is the supported
            # path. The double quotes are part of the argument (TOML string),
            # passed as one argv element — no shell involved.
            argv += ["-c", f'model_reasoning_effort="{eff}"']
        # capability_gated: per-tool CLI gating (--disallowedTools) has no codex
        # equivalent — codex exec's command execution cannot be disabled per-run.
        # Profiles still gate the tool bridge; the container sandbox is the
        # enforcement boundary here. (ADJUSTMENT POINT: add -c overrides here if a
        # future codex grows per-tool switches.)
        if extra_args:
            argv += list(extra_args)
        argv += ["-"]  # LAST: prompt positional = read stdin to EOF. Without it a
        #              piped-stdin-plus-no-prompt run can block on "Reading
        #              additional input from stdin...".
        return argv

    def _prepare_prompt(self, prompt: str, clarify: bool, workspace_hint: bool) -> str:
        # codex exec has no --append-system-prompt; the protocols ride as a leading
        # prompt block instead. Same enable conditions as ClaudeRunner._build_argv.
        segments: list[str] = []
        if workspace_hint and self.workspace_system_prompt:
            segments.append(self.workspace_system_prompt)
        if clarify and self.clarify_system_prompt:
            segments.append(self.clarify_system_prompt)
        if not segments:
            return prompt
        return "\n\n".join(segments + [prompt])

    def _handle_event(self, evt: dict, turn: TurnState) -> list[StreamEvent]:
        etype = str(evt.get("type") or "")
        if etype == "thread.started":
            tid = str(evt.get("thread_id") or "")
            if tid:
                turn.observed_session_uuid = tid
            return []
        if etype in ("turn.started", "item.updated"):
            return []  # item.updated is high-volume noise
        if etype == "turn.completed":
            usage = evt.get("usage") or {}
            # cached_input_tokens is a SUBSET of input_tokens in the 0.153.x
            # TokenUsage (total = input + output); do NOT add it on top or the
            # usage ledger / session token cap double-counts every cache read.
            turn.input_tokens = int(usage.get("input_tokens") or 0)
            turn.output_tokens = int(usage.get("output_tokens") or 0)  # reasoning tokens assumed included
            # No per-run cost figure exists in this schema: turn.cost_usd stays
            # 0.0, which every downstream consumer already default-tolerates.
            turn.stop_reason = "stop"
            return []
        if etype == "turn.failed":
            err = evt.get("error")
            msg = err.get("message") if isinstance(err, dict) else (str(err) if err else None)
            turn.errored = msg or turn.last_error or "codex turn failed"
            return []
        if etype == "error":  # transient (retry/reconnect) notice
            # An infinite "Reconnecting… waiting for network" retry loop keeps
            # stdout silent, so the executor's per-output-line timeout is what
            # eventually kills such a run — nothing terminal to record here.
            turn.last_error = str(evt.get("message") or "codex error")
            log.warning("codex transient error: %s", turn.last_error)
            return []
        if etype in ("item.started", "item.completed"):
            item = evt.get("item") or {}
            itype = str(item.get("type") or "")
            if itype == "agent_message":
                if etype != "item.completed":
                    return []
                # Multiple agent_message items concatenate with "" in the base
                # loop — acceptable; a single final message is the overwhelmingly
                # common case.
                text = str(item.get("text") or "")
                return [StreamEvent(kind="text", text=text, raw=item)] if text else []
            if itype == "reasoning":
                if etype != "item.completed":
                    return []
                think = str(item.get("text") or item.get("summary") or "")
                return [StreamEvent(kind="thinking", text=think, raw=item)] if think else []
            if itype == "command_execution":
                if etype == "item.started":
                    return [
                        StreamEvent(
                            kind="tool_use",
                            tool_name="command_execution",
                            tool_input={"command": item.get("command")},
                            raw=item,
                        )
                    ]
                return [
                    StreamEvent(
                        kind="tool_result",
                        tool_output=str(item.get("aggregated_output") or ""),
                        raw=item,
                    )
                ]
            if itype in ("mcp_tool_call", "web_search", "file_change"):
                kind = "tool_use" if etype == "item.started" else "tool_result"
                return [
                    StreamEvent(
                        kind=kind,
                        tool_name=itype,
                        tool_input=item if kind == "tool_use" else None,
                        tool_output=None if kind == "tool_use" else "",
                        raw=item,
                    )
                ]
            if itype == "error":
                # Error-shaped items (e.g. resume-with-different-model warning) are
                # advisory; only turn.failed is terminal.
                turn.last_error = str(item.get("message") or "codex item error")
                return [StreamEvent(kind="system", raw=evt)]
            return [StreamEvent(kind="system", raw=evt)]  # todo_list, future item types
        return [StreamEvent(kind="system", raw=evt)]  # unknown event types survive

    def _stderr_indicates_dead_session(self, stderr_lc: str) -> bool:
        # Codex's exact wording is not pinned upstream; the agent-neutral
        # "errored and produced no text on resume" predicate in the base is the
        # primary self-heal path — this only adds fast-paths for obvious wordings.
        return ("thread" in stderr_lc or "session" in stderr_lc) and (
            "not found" in stderr_lc
            or "no such" in stderr_lc
            or "does not exist" in stderr_lc
        )

    def _stderr_indicates_busy_session(self, stderr_lc: str) -> bool:
        # Observed live: `thread/resume failed: thread <id> already has an
        # active writer (code -32600)`. Codex takes a writer lock on the
        # thread store for the duration of a turn, so a previous turn that is
        # still running — notably one wedged in codex's own "Reconnecting…"
        # loop after an egress failure — makes the next resume fail while the
        # thread itself is fine. Forgetting the mapping here would strand
        # that thread and start a blank one.
        return "active writer" in stderr_lc or "already has an active" in stderr_lc

    def _stderr_for_client(self, stderr: str) -> str:
        # With CODEX_RUST_LOG active, codex's Rust tracing lands on stderr —
        # at debug/trace that includes request URLs and headers, which must
        # not ride a 502 detail out to API tenants. The tail goes to the
        # server log instead; without the knob, stderr is the usual terse
        # failure hint and stays in the error string.
        if os.environ.get("RUST_LOG", "").strip():
            if stderr:
                log.error(
                    "codex stderr withheld from client error (RUST_LOG active): %s",
                    stderr[-2000:],
                )
            return "[stderr withheld: RUST_LOG active — see the agent container log]"
        return super()._stderr_for_client(stderr)
