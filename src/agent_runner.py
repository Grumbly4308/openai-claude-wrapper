"""Agent-neutral runner core shared by every wrapped CLI agent.

Everything here is dialect-free: the executors (local subprocess / remote
shim), the session registry, the stream plumbing (per-line timeout, salvage
buffer, exit-record-not-HTTP-error), and BaseAgentRunner's run_stream/
run_collect loop. Anything that speaks a specific CLI's language — argv
construction, stream-json normalization, dead-session stderr wordings —
lives behind the per-agent hooks a subclass overrides (see claude_runner).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

import aiofiles
import httpx

from .config import effort_choices_for

log = logging.getLogger("claude_wrapper.runner")

# Claude Code stream-json events are newline-delimited, but a single event
# (large assistant text, tool input, or tool result) can easily exceed the
# default asyncio StreamReader limit of 64 KiB and trip readline() with
# "Separator is found, but chunk is longer than limit". Give the reader
# enough headroom for realistic payloads.
_STREAM_BUFFER_LIMIT = 64 * 1024 * 1024  # 64 MiB

# Sentinel key terminating the agent shim's NDJSON stream (sandboxed topology).
# Every Claude Code stream-json event is an object keyed on "type", so a line
# opening with this key can only be the shim's exit record.
SHIM_EXIT_KEY = "__claude_wrapper_exit__"
_SHIM_EXIT_PREFIX = '{"' + SHIM_EXIT_KEY + '"'


@dataclass
class StreamEvent:
    """Normalized event emitted while the wrapped agent is running."""

    kind: str  # "text" | "tool_use" | "tool_result" | "thinking" | "final" | "error" | "system"
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None
    raw: Optional[dict] = None


@dataclass
class AgentResult:
    session_uuid: str
    final_text: str
    stop_reason: str = "stop"
    total_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    events: list[StreamEvent] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ExecExit:
    """Terminal status of one agent execution (local subprocess or remote shim)."""

    returncode: int
    stderr: str = ""
    timed_out: bool = False


@dataclass
class TurnState:
    """Mutable per-run accumulator filled by the per-agent event handler."""

    stop_reason: str = "stop"
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    errored: Optional[str] = None  # terminal error reported by the CLI stream
    last_error: Optional[str] = None  # most recent non-terminal error notice
    observed_session_uuid: Optional[str] = None  # CLI-assigned session id (codex thread_id)


class LocalAgentExecutor:
    """Runs the CLI as a subprocess in this container — the classic layout.

    The stream contract shared with RemoteAgentExecutor: yields ("line", str)
    for each raw stdout line, then exactly one ("exit", ExecExit). A per-line
    timeout kills the process and is reported via ExecExit.timed_out rather
    than raised; cancellation kills the process and propagates.
    """

    def __init__(self, label: str = "claude"):
        # Names the agent in stderr debug lines; the default keeps today's
        # "claude stderr:" wording for existing construction sites.
        self.label = label

    async def stream(
        self,
        argv: list[str],
        prompt: str,
        cwd: Path,
        env_extra: Optional[dict[str, str]],
        timeout: int,
    ):
        env = os.environ.copy()
        env.setdefault("CI", "1")
        env.setdefault("CLAUDE_CODE_DISABLE_TELEMETRY", "1")
        if env_extra:
            env.update(env_extra)

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=env,
            limit=_STREAM_BUFFER_LIMIT,
        )

        async def _feed_stdin() -> None:
            # Concurrent writer so a prompt larger than the pipe buffer
            # (typically 64 KiB on Linux) doesn't deadlock against stdout.
            try:
                proc.stdin.write(prompt.encode("utf-8"))
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with contextlib.suppress(Exception):
                    proc.stdin.close()

        stdin_task = asyncio.create_task(_feed_stdin())
        stderr_task = asyncio.create_task(_drain_stderr(proc.stderr, self.label))
        timed_out = False
        try:
            try:
                async for line in _read_lines(proc.stdout, timeout):
                    yield "line", line
            except asyncio.TimeoutError:
                timed_out = True
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
            except asyncio.CancelledError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                raise
        finally:
            returncode = await proc.wait()
            stderr_output = await stderr_task
            with contextlib.suppress(Exception):
                await stdin_task
        yield "exit", ExecExit(returncode=returncode, stderr=stderr_output, timed_out=timed_out)


class RemoteAgentExecutor:
    """Runs the CLI via the agent shim (src.agent_shim) in a sandboxed container.

    Same stream contract as LocalAgentExecutor; the shim forwards raw stream-json
    stdout lines verbatim and terminates with one SHIM_EXIT_KEY sentinel line
    carrying returncode/stderr. argv[0] (this container's claude path) is
    dropped from the request — the shim prepends its own configured binary, so
    nothing that reaches its port can exec an arbitrary program there.
    """

    def __init__(self, base_url: str, token: str = ""):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # trust_env=False: in the sandboxed topology this container carries
            # HTTP(S)_PROXY pointing at the egress proxy for *upstream* calls;
            # the shim hop is internal and must never be routed through it.
            self._client = httpx.AsyncClient(base_url=self.base_url, trust_env=False)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def stream(
        self,
        argv: list[str],
        prompt: str,
        cwd: Path,
        env_extra: Optional[dict[str, str]],
        timeout: int,
    ):
        payload = {
            "args": list(argv[1:]),
            "prompt": prompt,
            "cwd": str(cwd),
            "env": dict(env_extra or {}),
        }
        headers = {"authorization": f"Bearer {self.token}"} if self.token else {}
        exit_info: Optional[ExecExit] = None
        try:
            client = self._get_client()
            # read=timeout maps the local per-line timeout onto the HTTP read;
            # closing the stream (context exit) is what tells the shim to kill
            # the subprocess, mirroring the local kill-on-timeout/cancel.
            async with client.stream(
                "POST",
                "/run",
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(30.0, read=float(timeout)),
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    exit_info = ExecExit(
                        returncode=-1,
                        stderr=f"agent shim error {resp.status_code}: {body[:500]}",
                    )
                else:
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        if line.startswith(_SHIM_EXIT_PREFIX):
                            try:
                                meta = json.loads(line).get(SHIM_EXIT_KEY) or {}
                            except json.JSONDecodeError:
                                meta = {}
                            exit_info = ExecExit(
                                returncode=int(meta.get("returncode", -1)),
                                stderr=str(meta.get("stderr") or ""),
                                timed_out=bool(meta.get("timed_out")),
                            )
                            continue
                        yield "line", line
        except httpx.ReadTimeout:
            exit_info = ExecExit(returncode=-1, timed_out=True)
        except httpx.HTTPError as e:
            exit_info = ExecExit(returncode=-1, stderr=f"agent shim unreachable: {e}")
        yield "exit", exit_info or ExecExit(
            returncode=-1, stderr="agent shim stream ended without an exit record"
        )


class SessionRegistry:
    """Maps stable string session keys to agent session UUIDs and per-session locks.

    Claude Code's ``--session-id`` expects a UUID, so we maintain a stable
    mapping from whatever session key the caller uses (hash of the
    transcript, OpenAI ``user``, or a client-supplied id).
    """

    def __init__(self, root: Path, agent: str = "claude"):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        # Entries are tagged with the agent that minted them: claude-data
        # survives a stack switch, and resuming one agent's uuid with the
        # other's CLI buys a guaranteed failed turn before self-heal fires.
        # A mismatched (or legacy untagged ≙ claude) entry is treated as
        # absent, so a switched deployment starts fresh sessions instead.
        self.agent = agent
        self._locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def has(self, key: str) -> bool:
        """Return True if a uuid THIS agent minted is bound to this session key.

        Agent-aware on purpose, with the same predicate as get_or_create_uuid:
        prepare_messages uses this to enter replay-only mode (send only the
        trailing message, trust the CLI's own log for history), which is only
        sound when the coming run will actually resume. A mismatched or
        unreadable entry means get_or_create_uuid will mint fresh — reporting
        it as present here would pair a brand-new thread with a history-less
        prompt and silently drop the conversation on a stack switch.
        """
        try:
            with open(self._path(key), "r") as f:
                data = json.load(f)
        except Exception:
            return False
        return data.get("agent", "claude") == self.agent

    async def lock_for(self, key: str) -> asyncio.Lock:
        async with self._registry_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    async def get_or_create_uuid(self, key: str) -> tuple[str, bool]:
        """Return (uuid, created). ``created`` is True when we minted a fresh id."""
        path = self._path(key)
        if path.exists():
            try:
                async with aiofiles.open(path, "r") as f:
                    data = json.loads(await f.read())
                if data.get("agent", "claude") == self.agent:
                    return data["uuid"], False
                # Another agent's session: fall through and mint fresh.
            except Exception:
                pass
        new_uuid = str(uuid.uuid4())
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps({"key": key, "uuid": new_uuid, "agent": self.agent}))
        return new_uuid, True

    async def bind_uuid(self, key: str, new_uuid: str) -> None:
        async with aiofiles.open(self._path(key), "w") as f:
            await f.write(json.dumps({"key": key, "uuid": new_uuid, "agent": self.agent}))

    async def forget(self, key: str) -> None:
        """Drop the registry entry so the next call mints a fresh uuid."""
        path = self._path(key)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    # Synchronous twins for generator-unwind paths: a client cancel tears the
    # stream down via GeneratorExit/CancelledError, where an awaited write can
    # be swallowed by the very cancellation that triggered the cleanup. The
    # files are a few dozen bytes; the sync write is effectively instant.
    def bind_uuid_sync(self, key: str, new_uuid: str) -> None:
        with open(self._path(key), "w") as f:
            f.write(json.dumps({"key": key, "uuid": new_uuid, "agent": self.agent}))

    def forget_sync(self, key: str) -> None:
        try:
            self._path(key).unlink()
        except FileNotFoundError:
            pass


class BaseAgentRunner:
    """Agent-neutral run loop; subclasses supply the CLI dialect via hooks."""

    # Per-agent class surface — everything main.py/delegate.py depend on lives here.
    agent_label: str = "agent"  # used in every error string and log line
    wrapper_assigns_session_id: bool = True  # claude: we mint the uuid; codex: CLI assigns

    def __init__(
        self,
        registry: SessionRegistry,
        workspace_root: Path,
        agent_bin: str = "",
        request_timeout_seconds: int = 1800,
        effort: str = "",
        clarify_system_prompt: str = "",
        clarify_disallowed_tools: tuple[str, ...] = (),
        workspace_system_prompt: str = "",
        stream_partial_messages: bool = True,
        executor=None,
    ):
        self.registry = registry
        self.workspace_root = workspace_root
        self.agent_bin = agent_bin
        self.request_timeout_seconds = request_timeout_seconds
        self.effort = effort
        # How runs actually execute: a local subprocess by default, or the
        # agent shim in the sandboxed split when deps wires a RemoteAgentExecutor.
        self.executor = executor if executor is not None else LocalAgentExecutor(label=self.agent_label)
        # Emit incremental text/thinking deltas (live streaming) via
        # `--include-partial-messages`. When on, the consolidated assistant
        # text/thinking blocks are suppressed in _normalize_stream_event to avoid
        # double-emitting what the deltas already streamed.
        self.stream_partial_messages = stream_partial_messages
        # Interactive clarification protocol, applied only when a caller passes
        # clarify=True (chat/responses) AND a prompt is configured. Empty prompt
        # ⇒ globally disabled (CLAUDE_WRAPPER_CLARIFY=off), so it's a no-op.
        self.clarify_system_prompt = clarify_system_prompt
        self.clarify_disallowed_tools = clarify_disallowed_tools
        # Workspace protocol, applied only when a caller passes
        # workspace_hint=True (chat/responses) AND a prompt is configured. Empty
        # prompt ⇒ globally disabled (CLAUDE_WRAPPER_WORKSPACE_HINT=off).
        self.workspace_system_prompt = workspace_system_prompt

    # ---- per-agent hooks (overridden by subclasses) ----

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
        raise NotImplementedError

    def _prepare_prompt(self, prompt: str, clarify: bool, workspace_hint: bool) -> str:
        return prompt  # claude injects protocols via argv; codex overrides to prepend

    def _handle_event(self, evt: dict, turn: TurnState) -> list[StreamEvent]:
        raise NotImplementedError  # normalize + terminal extraction, fills TurnState

    def _effort_choices_for(self, model: str) -> tuple[str, ...]:
        # What efforts THIS runner will actually pass to its CLI. Kept separate
        # from config.effort_choices_for (which drives /v1/models advertisement)
        # so runner acceptance never depends on the process-wide SETTINGS.agent
        # (test modules construct runners under claude-mode frozen SETTINGS).
        return effort_choices_for(model)  # claude: config rules, unchanged

    def _stderr_indicates_dead_session(self, stderr_lc: str) -> bool:
        return False

    def _stderr_for_client(self, stderr: str) -> str:
        """The stderr tail embedded in client-facing error strings (the
        ``<agent> exited N: …`` detail main returns as a 502). Overridable so
        an agent whose stderr can carry sensitive tracing withholds it —
        self-heal detection is unaffected, it reads the full stderr."""
        return stderr[-500:] if stderr else ""

    # ---- shared machinery ----

    async def aclose(self) -> None:
        """Release executor resources (the remote executor's HTTP client)."""
        close = getattr(self.executor, "aclose", None)
        if close is not None:
            await close()

    def _session_cwd(self, session_key: str) -> Path:
        d = self.workspace_root / session_key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _iter_tracked(self, cwd: Path):
        (cwd / "outputs").mkdir(parents=True, exist_ok=True)
        for p in cwd.rglob("*"):
            if not p.is_file():
                continue
            try:
                rel = p.relative_to(cwd)
            except ValueError:
                continue
            # Skip uploads and hidden/system files — uploads are caller-provided,
            # dotfiles belong to the CLI's own bookkeeping.
            if rel.parts and rel.parts[0] == "uploads":
                continue
            if any(part.startswith(".") for part in rel.parts):
                continue
            yield p

    def _snapshot_outputs(self, cwd: Path) -> dict[Path, float]:
        return {p: p.stat().st_mtime for p in self._iter_tracked(cwd)}

    def _new_outputs(self, cwd: Path, before: dict[Path, float]) -> list[Path]:
        new: list[Path] = []
        for p in self._iter_tracked(cwd):
            prev = before.get(p)
            if prev is None or p.stat().st_mtime > prev:
                new.append(p)
        return sorted(new)

    def _resolve_effort(self, model: Optional[str], effort: Optional[str]) -> tuple[str, str]:
        """Resolve the effort actually applied to a run, plus where it came from.

        An explicit per-request value wins (including "", meaning "no flag");
        otherwise we fall back to the server default (CLAUDE_WRAPPER_EFFORT).
        Effort — and the ultracode settings overlay — only apply to the effort
        choices a given model accepts (see config.effort_choices_for), so it's
        dropped for models that take no effort and for an effort the model
        doesn't support (e.g. max/ultracode on Sonnet).
        Returns (effective_effort, source) where source is one of:
        "request" | "server-default" | "model-incapable" | "effort-unsupported".
        """
        if effort is not None:
            eff, source = effort, "request"
        else:
            eff, source = self.effort, "server-default"
        allowed = self._effort_choices_for(model or "")
        if not allowed:
            return "", "model-incapable"
        if eff and eff not in allowed:
            return "", "effort-unsupported"
        return eff, source

    async def run_stream(
        self,
        prompt: str,
        session_key: str,
        model: Optional[str] = None,
        env_extra: Optional[dict[str, str]] = None,
        extra_args: Optional[list[str]] = None,
        effort: Optional[str] = None,
        clarify: bool = False,
        workspace_hint: bool = False,
        capability_gated: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Yield StreamEvents as the subprocess produces them.

        The final event is always either ``final`` or ``error``.

        ``capability_gated`` applies the model's capability profile as
        --disallowedTools. On by default so every chat surface is covered;
        delegation runs pass False (they do their work through Bash).
        """
        lock = await self.registry.lock_for(session_key)
        await lock.acquire()
        turn: Optional[TurnState] = None
        session_uuid, created, completed = "", False, False
        try:
            session_uuid, created = await self.registry.get_or_create_uuid(session_key)
            cwd = self._session_cwd(session_key)
            snapshot = self._snapshot_outputs(cwd)

            argv = self._build_argv(
                session_uuid=session_uuid,
                model=model,
                resume=not created,
                extra_args=extra_args,
                effort=effort,
                clarify=clarify,
                workspace_hint=workspace_hint,
                capability_gated=capability_gated,
            )
            prompt = self._prepare_prompt(prompt, clarify, workspace_hint)

            eff_applied, eff_source = self._resolve_effort(model, effort)
            log.info(
                "launching %s session_key=%s uuid=%s resume=%s model=%s effort=%s (%s)",
                self.agent_label,
                session_key,
                session_uuid,
                not created,
                model,
                eff_applied or "cli-default",
                eff_source,
            )

            final_text_parts: list[str] = []
            turn = TurnState()
            exit_info: Optional[ExecExit] = None

            # The executor owns the process (or the shim connection): it kills
            # the subprocess on timeout/cancel and always terminates the stream
            # with one ("exit", ExecExit) event on non-cancelled paths.
            async for kind, payload in self.executor.stream(
                argv=argv,
                prompt=prompt,
                cwd=cwd,
                env_extra=env_extra,
                timeout=self.request_timeout_seconds,
            ):
                if kind == "exit":
                    exit_info = payload
                    continue
                line = payload.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue

                for normalized in self._handle_event(evt, turn):
                    if normalized.kind == "text" and normalized.text:
                        final_text_parts.append(normalized.text)
                    yield normalized
            completed = True

            # Session-id capture MUST precede the self-heal computation below:
            # bind_uuid unconditionally rewrites the key file, so if forget()
            # ran first, a failed resume in which the CLI announced a fresh id
            # would resurrect the mapping and defeat self-heal. Capture first,
            # then forget() runs last and wins.
            if turn.observed_session_uuid and turn.observed_session_uuid != session_uuid:
                await self.registry.bind_uuid(session_key, turn.observed_session_uuid)
                session_uuid = turn.observed_session_uuid

            returncode = exit_info.returncode if exit_info is not None else -1
            stderr_output = exit_info.stderr if exit_info is not None else ""
            if exit_info is not None and exit_info.timed_out and turn.errored is None:
                turn.errored = f"{self.agent_label} subprocess timed out"
            if returncode != 0 and turn.errored is None:
                turn.errored = (
                    f"{self.agent_label} exited {returncode}: "
                    f"{self._stderr_for_client(stderr_output)}"
                )
            errored = turn.errored
            # Self-heal a broken resume. If a --resume turn fails — a non-zero
            # exit, or a stream-json result with an error subtype like
            # "error_during_execution" (which exits 0) — without producing any
            # assistant text, the underlying session is unusable. The most
            # common cause is its transcript being gone (e.g. the session
            # store was wiped, or never persisted) while our key->uuid mapping
            # survived, so every retry re-resumes the same dead uuid and fails
            # identically. Drop the mapping so the NEXT request mints a fresh
            # uuid, switches to --session-id, and replays the full transcript
            # (prepare_messages leaves replay-only mode once registry.has() is
            # False). Costs one extra full-history turn but keeps the
            # conversation alive instead of permanently bricking it.
            #
            # The "no assistant text this turn" guard means a session that
            # streamed a real answer and only then hit a late error is left
            # intact — we only reset sessions that produced nothing usable.
            #
            # placeholder_unbound covers CLI-assigned-id agents (codex): a
            # first turn that dies before announcing its id leaves the
            # registry holding a wrapper-minted placeholder that can never be
            # resumed, so it must be forgotten now, not one failed turn later.
            stderr_lc = (stderr_output or "").lower()
            dead_session = self._stderr_indicates_dead_session(stderr_lc)
            resume_unusable = bool(errored) and not final_text_parts
            placeholder_unbound = (
                not self.wrapper_assigns_session_id
                and created
                and turn.observed_session_uuid is None
            )
            if (not created and (dead_session or resume_unusable)) or (
                placeholder_unbound and bool(errored)
            ):
                log.warning(
                    "resume failed for session %s uuid %s (returncode=%s error=%r); "
                    "dropping mapping so the next turn replays full history",
                    session_key,
                    session_uuid,
                    returncode,
                    errored,
                )
                await self.registry.forget(session_key)

            final_text = "".join(final_text_parts).strip()

            new_outputs = self._new_outputs(cwd, snapshot)
            yield StreamEvent(
                kind="final",
                text=final_text,
                raw={
                    "stop_reason": turn.stop_reason,
                    "cost_usd": turn.cost_usd,
                    "input_tokens": turn.input_tokens,
                    "output_tokens": turn.output_tokens,
                    "new_outputs": [str(p) for p in new_outputs],
                    "session_uuid": session_uuid,
                    "error": errored,
                },
            )
            if errored:
                yield StreamEvent(kind="error", text=errored, raw={"session_uuid": session_uuid})
        finally:
            if not completed and not self.wrapper_assigns_session_id:
                # A client cancel/disconnect unwound the generator mid-stream,
                # so the post-loop bookkeeping never ran. For CLI-assigned-id
                # agents that is poison on a FIRST turn: fresh runs pass no
                # session flag, so the registry holds a wrapper-minted
                # placeholder the CLI was never told about, and the next turn
                # would resume a thread that does not exist — a guaranteed
                # failed turn before self-heal fires. Persist what the stream
                # announced (or drop the never-announced placeholder) so the
                # next turn resumes the real thread — or replays in full.
                observed = turn.observed_session_uuid if turn is not None else None
                if observed and observed != session_uuid:
                    self.registry.bind_uuid_sync(session_key, observed)
                elif created and observed is None:
                    self.registry.forget_sync(session_key)
            lock.release()

    async def run_collect(
        self,
        prompt: str,
        session_key: str,
        model: Optional[str] = None,
        env_extra: Optional[dict[str, str]] = None,
        extra_args: Optional[list[str]] = None,
        effort: Optional[str] = None,
        clarify: bool = False,
        workspace_hint: bool = False,
        capability_gated: bool = True,
    ) -> AgentResult:
        result = AgentResult(session_uuid="", final_text="")
        text_parts: list[str] = []
        new_outputs: list[str] = []
        async for evt in self.run_stream(
            prompt=prompt,
            session_key=session_key,
            model=model,
            env_extra=env_extra,
            extra_args=extra_args,
            effort=effort,
            clarify=clarify,
            workspace_hint=workspace_hint,
            capability_gated=capability_gated,
        ):
            result.events.append(evt)
            if evt.kind == "text" and evt.text:
                text_parts.append(evt.text)
            elif evt.kind == "final":
                meta = evt.raw or {}
                result.session_uuid = meta.get("session_uuid", "")
                result.stop_reason = meta.get("stop_reason", "stop")
                result.total_cost_usd = float(meta.get("cost_usd", 0.0))
                result.input_tokens = int(meta.get("input_tokens", 0))
                result.output_tokens = int(meta.get("output_tokens", 0))
                new_outputs = list(meta.get("new_outputs", []))
                if meta.get("error"):
                    result.error = str(meta["error"])
                result.final_text = evt.text or "".join(text_parts)
            elif evt.kind == "error":
                if not result.error:
                    result.error = evt.text
        if not result.final_text:
            result.final_text = "".join(text_parts)
        result.events.append(StreamEvent(kind="system", raw={"new_outputs": new_outputs}))
        return result


# ---------- helpers ----------


async def _read_lines(stream: asyncio.StreamReader, timeout: int) -> AsyncIterator[str]:
    while True:
        try:
            line = await asyncio.wait_for(stream.readline(), timeout=timeout)
        except asyncio.TimeoutError:
            raise
        except ValueError:
            # readline() raises ValueError when a single line exceeds the
            # StreamReader buffer. Drain and concatenate partial chunks via
            # readuntil() so we still produce the full event instead of
            # killing the whole stream.
            buf = bytearray()
            while True:
                try:
                    part = await asyncio.wait_for(
                        stream.readuntil(b"\n"), timeout=timeout
                    )
                    buf.extend(part)
                    break
                except ValueError as exc:
                    partial = getattr(exc, "partial", None)
                    if not partial:
                        raise
                    buf.extend(partial)
                except asyncio.IncompleteReadError as exc:
                    if exc.partial:
                        buf.extend(exc.partial)
                    if not buf:
                        return
                    break
            line = bytes(buf)
        if not line:
            return
        yield line.decode("utf-8", errors="replace")


async def _drain_stderr(stream: Optional[asyncio.StreamReader], label: str = "claude") -> str:
    if stream is None:
        return ""
    data = await stream.read()
    text = data.decode("utf-8", errors="replace") if data else ""
    if text:
        for line in text.splitlines():
            log.debug("%s stderr: %s", label, line)
    return text
