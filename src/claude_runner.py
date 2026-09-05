from __future__ import annotations

import asyncio  # noqa: F401 — tests/test_resume_selfheal.py monkeypatches
# claude_runner.asyncio.create_subprocess_exec; the module attribute must
# survive the refactor even though nothing here calls it directly anymore.
from pathlib import Path
from typing import Optional

from .agent_runner import (  # noqa: F401 — re-exported for back-compat; agent_shim/tests import from here
    _STREAM_BUFFER_LIMIT,
    SHIM_EXIT_KEY,
    _SHIM_EXIT_PREFIX,
    AgentResult,
    BaseAgentRunner,
    ExecExit,
    LocalAgentExecutor,
    RemoteAgentExecutor,
    SessionRegistry,
    StreamEvent,
    TurnState,
    _drain_stderr,
    _read_lines,
)
from .capabilities import cli_disallowed_tools
from .config import SETTINGS, ULTRACODE_EFFORT

# Historical name, kept for every existing import site (delegate, tests).
ClaudeResult = AgentResult


class ClaudeRunner(BaseAgentRunner):
    """Runs ``claude -p`` as a subprocess and streams structured events."""

    agent_label = "claude"
    wrapper_assigns_session_id = True

    def __init__(
        self,
        registry: SessionRegistry,
        workspace_root: Path,
        claude_bin: str = "claude",
        request_timeout_seconds: int = 1800,
        effort: str = "",
        clarify_system_prompt: str = "",
        clarify_disallowed_tools: tuple[str, ...] = (),
        workspace_system_prompt: str = "",
        stream_partial_messages: bool = True,
        executor=None,
        agent_bin: Optional[str] = None,
    ):
        # `claude_bin=` stays the primary kwarg for back-compat (deps and the
        # tests construct by keyword); `agent_bin=` is the canonical alias so
        # deps can construct every runner class identically.
        super().__init__(
            registry=registry,
            workspace_root=workspace_root,
            agent_bin=agent_bin or claude_bin,
            request_timeout_seconds=request_timeout_seconds,
            effort=effort,
            clarify_system_prompt=clarify_system_prompt,
            clarify_disallowed_tools=clarify_disallowed_tools,
            workspace_system_prompt=workspace_system_prompt,
            stream_partial_messages=stream_partial_messages,
            executor=executor,
        )

    @property
    def claude_bin(self) -> str:
        return self.agent_bin

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
        # Prompt is fed via stdin (not argv) to avoid E2BIG on large prompts.
        argv = [
            self.claude_bin,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        if self.stream_partial_messages:
            # Incremental message deltas (content_block_delta) so text and
            # thinking stream token-by-token rather than one block at a time.
            argv += ["--include-partial-messages"]
        if resume:
            argv += ["--resume", session_uuid]
        else:
            argv += ["--session-id", session_uuid]
        if model:
            argv += ["--model", model]
        eff, _src = self._resolve_effort(model, effort)
        if eff == ULTRACODE_EFFORT:
            # "ultracode" is not a --effort value (the CLI ignores it and falls
            # back to default effort). It is requested via settings instead,
            # where the CLI resolves it to xhigh effort plus ultracode's
            # dynamic-workflow orchestration opt-in.
            #
            # Ultracode is GATED on dynamic workflows being enabled: with them
            # off the CLI rejects it ("Ultracode needs dynamic workflows
            # enabled") and silently runs at default effort. In a fresh headless
            # container the `enableWorkflows` setting defaults to false, so we
            # must turn it on in the same overlay — otherwise the advertised
            # "(ultracode)" model is selectable but a functional no-op. (An
            # org-policy `disableWorkflows` or account launch gate can still
            # override this; those are account-side, not settable here.)
            argv += ["--settings", '{"enableWorkflows": true, "ultracode": true}']
        elif eff:
            argv += ["--effort", eff]
        # Interactive clarification: teach Claude to pause-and-ask in plain text
        # and disable the headless-dead question-card tool. Placed so the variadic
        # --disallowedTools is terminated by the following --dangerously-skip-…
        # flag rather than greedily eating a later positional.
        #
        # The workspace protocol rides along in the same flag: the CLI takes one
        # --append-system-prompt, so both segments are joined rather than the
        # flag repeated.
        segments: list[str] = []
        if workspace_hint and self.workspace_system_prompt:
            segments.append(self.workspace_system_prompt)
        if clarify and self.clarify_system_prompt:
            segments.append(self.clarify_system_prompt)
        if segments:
            argv += ["--append-system-prompt", "\n\n".join(segments)]
        # Capability gating (chat runs only — delegation passes
        # capability_gated=False): tools the model's profile withholds, merged
        # with the clarify set into one --disallowedTools emission. Dedup
        # preserves first-seen order so the argv stays deterministic.
        disallowed: list[str] = []
        if capability_gated:
            disallowed += cli_disallowed_tools(model or SETTINGS.default_model)
        if clarify and self.clarify_disallowed_tools:
            disallowed += self.clarify_disallowed_tools
        if disallowed:
            argv += ["--disallowedTools", *dict.fromkeys(disallowed)]
        argv += ["--dangerously-skip-permissions"]
        if extra_args:
            argv += list(extra_args)
        return argv

    def _handle_event(self, evt: dict, turn: TurnState) -> list[StreamEvent]:
        out = _normalize_stream_event(evt, partial=self.stream_partial_messages)
        if evt.get("type") == "result":
            turn.stop_reason = _stop_reason_from_result(evt)
            turn.cost_usd = float(evt.get("total_cost_usd") or evt.get("cost_usd") or 0.0)
            usage = evt.get("usage") or {}
            turn.input_tokens = int(usage.get("input_tokens") or 0)
            turn.output_tokens = int(usage.get("output_tokens") or 0)
            if evt.get("subtype") and evt["subtype"] != "success":
                turn.errored = evt.get("error") or evt.get("subtype")
        return out

    def _stderr_indicates_dead_session(self, stderr_lc: str) -> bool:
        return "session" in stderr_lc and (
            "not found" in stderr_lc or "no such" in stderr_lc
        )


# ---------- helpers ----------


def _normalize_stream_event(evt: dict, partial: bool = False) -> list[StreamEvent]:
    """Convert a raw Claude Code stream-json event into StreamEvents.

    When ``partial`` is True the run was launched with
    ``--include-partial-messages``, so live text/thinking arrive as incremental
    ``stream_event`` deltas (handled below). In that mode the *consolidated*
    assistant ``text``/``thinking`` blocks are suppressed — they would otherwise
    re-emit, in one chunk, exactly what the deltas already streamed. tool_use is
    still taken from the consolidated block (its input doesn't stream usefully).
    """
    etype = evt.get("type")

    if etype == "stream_event":
        # Incremental Anthropic streaming event (only present with
        # --include-partial-messages). We care about content_block_delta for
        # text and thinking; block start/stop, tool input fragments, signatures,
        # and message-level deltas are not needed for the client stream.
        inner = evt.get("event") or {}
        if inner.get("type") != "content_block_delta":
            return []
        delta = inner.get("delta") or {}
        dtype = delta.get("type")
        if dtype == "text_delta":
            text = delta.get("text") or ""
            return [StreamEvent(kind="text", text=text, raw=inner)] if text else []
        if dtype == "thinking_delta":
            thinking = delta.get("thinking") or ""
            return [StreamEvent(kind="thinking", text=thinking, raw=inner)] if thinking else []
        return []

    if etype == "system":
        return [StreamEvent(kind="system", raw=evt)]

    if etype == "assistant":
        msg = evt.get("message") or {}
        content = msg.get("content") or []
        out: list[StreamEvent] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                if partial:
                    continue  # already streamed via text_delta
                text = block.get("text") or ""
                if text:
                    out.append(StreamEvent(kind="text", text=text, raw=block))
            elif btype == "thinking":
                if partial:
                    continue  # already streamed via thinking_delta
                out.append(StreamEvent(kind="thinking", text=block.get("thinking") or "", raw=block))
            elif btype == "tool_use":
                out.append(
                    StreamEvent(
                        kind="tool_use",
                        tool_name=block.get("name"),
                        tool_input=block.get("input") or {},
                        raw=block,
                    )
                )
        return out

    if etype == "user":
        msg = evt.get("message") or {}
        content = msg.get("content") or []
        out = []
        for block in content:
            if block.get("type") == "tool_result":
                raw_out = block.get("content")
                if isinstance(raw_out, list):
                    text = "".join(part.get("text", "") for part in raw_out if isinstance(part, dict))
                else:
                    text = str(raw_out) if raw_out is not None else ""
                out.append(StreamEvent(kind="tool_result", tool_output=text, raw=block))
        return out

    if etype == "result":
        return []  # terminal extraction happens in ClaudeRunner._handle_event

    return [StreamEvent(kind="system", raw=evt)]


def _stop_reason_from_result(evt: dict) -> str:
    subtype = evt.get("subtype") or "success"
    if subtype == "success":
        return "stop"
    if "length" in subtype or "max" in subtype:
        return "length"
    return "stop"
