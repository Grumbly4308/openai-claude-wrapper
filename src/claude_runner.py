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

log = logging.getLogger("claude_wrapper.runner")


@dataclass
class StreamEvent:
    """Normalized event emitted while Claude Code is running."""

    kind: str  # "text" | "tool_use" | "tool_result" | "thinking" | "final" | "error" | "system"
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None
    raw: Optional[dict] = None


@dataclass
class ClaudeResult:
    session_uuid: str
    final_text: str
    stop_reason: str = "stop"
    total_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    events: list[StreamEvent] = field(default_factory=list)
    error: Optional[str] = None


class SessionRegistry:
    """Maps stable string session keys to Claude Code UUIDs and per-session locks.

    Claude Code's ``--session-id`` expects a UUID, so we maintain a stable
    mapping from whatever session key the caller uses (hash of the
    transcript, OpenAI ``user``, or a client-supplied id).
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

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
                return data["uuid"], False
            except Exception:
                pass
        new_uuid = str(uuid.uuid4())
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps({"key": key, "uuid": new_uuid}))
        return new_uuid, True

    async def bind_uuid(self, key: str, new_uuid: str) -> None:
        async with aiofiles.open(self._path(key), "w") as f:
            await f.write(json.dumps({"key": key, "uuid": new_uuid}))


class ClaudeRunner:
    """Runs ``claude -p`` as a subprocess and streams structured events."""

    def __init__(
        self,
        registry: SessionRegistry,
        workspace_root: Path,
        claude_bin: str = "claude",
        request_timeout_seconds: int = 1800,
    ):
        self.registry = registry
        self.workspace_root = workspace_root
        self.claude_bin = claude_bin
        self.request_timeout_seconds = request_timeout_seconds

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
            # dotfiles belong to Claude Code's own bookkeeping.
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

    def _build_argv(
        self,
        prompt: str,
        session_uuid: str,
        model: Optional[str],
        resume: bool,
        extra_args: Optional[list[str]] = None,
    ) -> list[str]:
        argv = [
            self.claude_bin,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        if resume:
            argv += ["--resume", session_uuid]
        else:
            argv += ["--session-id", session_uuid]
        if model:
            argv += ["--model", model]
        argv += ["--dangerously-skip-permissions"]
        if extra_args:
            argv += list(extra_args)
        return argv

    async def run_stream(
        self,
        prompt: str,
        session_key: str,
        model: Optional[str] = None,
        env_extra: Optional[dict[str, str]] = None,
        extra_args: Optional[list[str]] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield StreamEvents as the subprocess produces them.

        The final event is always either ``final`` or ``error``.
        """
        lock = await self.registry.lock_for(session_key)
        await lock.acquire()
        try:
            session_uuid, created = await self.registry.get_or_create_uuid(session_key)
            cwd = self._session_cwd(session_key)
            snapshot = self._snapshot_outputs(cwd)

            argv = self._build_argv(
                prompt=prompt,
                session_uuid=session_uuid,
                model=model,
                resume=not created,
                extra_args=extra_args,
            )

            env = os.environ.copy()
            env.setdefault("CI", "1")
            env.setdefault("CLAUDE_CODE_DISABLE_TELEMETRY", "1")
            if env_extra:
                env.update(env_extra)

            log.info("launching claude session_key=%s uuid=%s resume=%s", session_key, session_uuid, not created)

            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env=env,
            )

            stderr_task = asyncio.create_task(_drain_stderr(proc.stderr))
            final_text_parts: list[str] = []
            stop_reason = "stop"
            usage_input = 0
            usage_output = 0
            cost = 0.0
            errored: Optional[str] = None

            try:
                async for line in _read_lines(proc.stdout, self.request_timeout_seconds):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    for normalized in _normalize_stream_event(evt):
                        if normalized.kind == "text" and normalized.text:
                            final_text_parts.append(normalized.text)
                        yield normalized

                    if evt.get("type") == "result":
                        stop_reason = _stop_reason_from_result(evt)
                        cost = float(evt.get("total_cost_usd") or evt.get("cost_usd") or 0.0)
                        usage = evt.get("usage") or {}
                        usage_input = int(usage.get("input_tokens") or 0)
                        usage_output = int(usage.get("output_tokens") or 0)
                        if evt.get("subtype") and evt["subtype"] != "success":
                            errored = evt.get("error") or evt.get("subtype")

            except asyncio.TimeoutError:
                errored = "claude subprocess timed out"
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
            except asyncio.CancelledError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                raise
            finally:
                returncode = await proc.wait()
                stderr_output = await stderr_task
                if returncode != 0 and errored is None:
                    errored = f"claude exited {returncode}: {stderr_output[-500:] if stderr_output else ''}"
                # Rotate uuid on resume-failure so the next request starts fresh.
                if (
                    returncode != 0
                    and not created
                    and stderr_output
                    and "session" in stderr_output.lower()
                    and ("not found" in stderr_output.lower() or "no such" in stderr_output.lower())
                ):
                    log.warning("session %s uuid %s missing; rotating for next call", session_key, session_uuid)
                    await self.registry.bind_uuid(session_key, str(uuid.uuid4()))

            final_text = "".join(final_text_parts).strip()

            new_outputs = self._new_outputs(cwd, snapshot)
            yield StreamEvent(
                kind="final",
                text=final_text,
                raw={
                    "stop_reason": stop_reason,
                    "cost_usd": cost,
                    "input_tokens": usage_input,
                    "output_tokens": usage_output,
                    "new_outputs": [str(p) for p in new_outputs],
                    "session_uuid": session_uuid,
                    "error": errored,
                },
            )
            if errored:
                yield StreamEvent(kind="error", text=errored, raw={"session_uuid": session_uuid})
        finally:
            lock.release()

    async def run_collect(
        self,
        prompt: str,
        session_key: str,
        model: Optional[str] = None,
        env_extra: Optional[dict[str, str]] = None,
        extra_args: Optional[list[str]] = None,
    ) -> ClaudeResult:
        result = ClaudeResult(session_uuid="", final_text="")
        text_parts: list[str] = []
        new_outputs: list[str] = []
        async for evt in self.run_stream(
            prompt=prompt,
            session_key=session_key,
            model=model,
            env_extra=env_extra,
            extra_args=extra_args,
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
        if not line:
            return
        yield line.decode("utf-8", errors="replace")


async def _drain_stderr(stream: Optional[asyncio.StreamReader]) -> str:
    if stream is None:
        return ""
    data = await stream.read()
    text = data.decode("utf-8", errors="replace") if data else ""
    if text:
        for line in text.splitlines():
            log.debug("claude stderr: %s", line)
    return text


def _normalize_stream_event(evt: dict) -> list[StreamEvent]:
    """Convert a raw Claude Code stream-json event into StreamEvents."""
    etype = evt.get("type")

    if etype == "system":
        return [StreamEvent(kind="system", raw=evt)]

    if etype == "assistant":
        msg = evt.get("message") or {}
        content = msg.get("content") or []
        out: list[StreamEvent] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text = block.get("text") or ""
                if text:
                    out.append(StreamEvent(kind="text", text=text, raw=block))
            elif btype == "thinking":
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
        return []  # handled by caller via raw event

    return [StreamEvent(kind="system", raw=evt)]


def _stop_reason_from_result(evt: dict) -> str:
    subtype = evt.get("subtype") or "success"
    if subtype == "success":
        return "stop"
    if "length" in subtype or "max" in subtype:
        return "length"
    return "stop"
