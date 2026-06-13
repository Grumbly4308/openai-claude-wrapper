"""Unit tests for resume self-healing in ClaudeRunner.run_stream.

A follow-up turn resumes the Claude session bound to a stable key via
``claude --resume <uuid>``. If that session's transcript is gone (the store
was wiped, or never persisted) while our key->uuid mapping survived, the
resume fails identically on every retry. run_stream must drop the mapping on
such a failure so the *next* turn mints a fresh uuid and replays full history,
rather than bricking the conversation forever.

These tests mock the subprocess so no real ``claude`` binary is spawned.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Point the data dir at a tempdir so importing src.* never touches /data, and
# use the static model list so import doesn't scan the CLI binary.
_TMP = tempfile.mkdtemp(prefix="claude-wrapper-selfheal-test-")
os.environ["CLAUDE_WRAPPER_DATA"] = _TMP
os.environ["CLAUDE_WRAPPER_MODEL_DISCOVERY"] = "off"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib  # noqa: E402

from src import claude_runner  # noqa: E402

# Sibling test modules (notably test_endpoints) monkeypatch
# ClaudeRunner.run_stream with a stub at import time, which would otherwise
# shadow the real implementation we need to exercise here. Reload the module so
# `claude_runner.ClaudeRunner` is a fresh class carrying the genuine run_stream.
# (This does not disturb test_endpoints: its stub lives on the pre-reload class
# object that its own app/RUNNER instance was built from.)
importlib.reload(claude_runner)
ClaudeRunner = claude_runner.ClaudeRunner
SessionRegistry = claude_runner.SessionRegistry

_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, note: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"PASS  {name}")
    else:
        _FAIL += 1
        print(f"FAIL  {name} {note}")
    assert cond, f"{name} {note}"


class _FakeStdin:
    def write(self, _data: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeProc:
    """Minimal stand-in for asyncio subprocess: scripted stdout, fixed rc.

    Must be constructed inside a running event loop — asyncio.StreamReader()
    binds to the loop at creation time.
    """

    def __init__(self, stdout_lines: list[bytes], returncode: int, stderr: bytes = b""):
        self.stdin = _FakeStdin()
        self.stdout = asyncio.StreamReader()
        for line in stdout_lines:
            self.stdout.feed_data(line)
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        if stderr:
            self.stderr.feed_data(stderr)
        self.stderr.feed_eof()
        self._returncode = returncode

    async def wait(self) -> int:
        return self._returncode

    def kill(self) -> None:
        pass


def _runner(tag: str) -> ClaudeRunner:
    return ClaudeRunner(
        registry=SessionRegistry(Path(_TMP) / f"sessions-{tag}"),
        workspace_root=Path(_TMP) / f"workspace-{tag}",
        claude_bin="claude",
    )


async def _drain_resume(
    runner: ClaudeRunner,
    key: str,
    stdout_lines: list[bytes],
    returncode: int,
    stderr: bytes = b"",
) -> None:
    """Pre-bind a uuid (so the run is a resume) then drive run_stream to completion.

    The fake proc is built here, inside the running loop, so its StreamReaders
    bind to the active loop.
    """
    await runner.registry.get_or_create_uuid(key)  # mapping now exists -> resume turn
    fake = _FakeProc(stdout_lines, returncode=returncode, stderr=stderr)

    async def _fake_exec(*_args, **_kwargs):
        return fake

    orig = claude_runner.asyncio.create_subprocess_exec
    claude_runner.asyncio.create_subprocess_exec = _fake_exec
    try:
        async for _evt in runner.run_stream(prompt="hi", session_key=key, model="claude-opus-4-8"):
            pass
    finally:
        claude_runner.asyncio.create_subprocess_exec = orig


def test_dead_resume_drops_mapping() -> None:
    """A resume that errors with no assistant text drops the mapping."""
    runner = _runner("dead")
    key = "conv-dead"
    # stream-json: an error_during_execution result, exit 0, zero assistant text.
    result = b'{"type":"result","subtype":"error_during_execution","error":"error_during_execution","usage":{}}\n'
    asyncio.run(_drain_resume(runner, key, [result], returncode=0))
    check("dead_resume.mapping_dropped", not runner.registry.has(key),
          "expected the key->uuid mapping to be forgotten so the next turn replays history")


def test_nonzero_resume_no_output_drops_mapping() -> None:
    """A resume that exits non-zero with no output also drops the mapping."""
    runner = _runner("nonzero")
    key = "conv-nonzero"
    asyncio.run(_drain_resume(runner, key, [], returncode=1, stderr=b"error_during_execution\n"))
    check("nonzero_resume.mapping_dropped", not runner.registry.has(key))


def test_healthy_resume_keeps_mapping() -> None:
    """A resume that streams a real answer keeps the mapping intact."""
    runner = _runner("ok")
    key = "conv-ok"
    assistant = (
        b'{"type":"assistant","message":{"content":[{"type":"text","text":"hello there"}]}}\n'
    )
    result = b'{"type":"result","subtype":"success","usage":{"input_tokens":5,"output_tokens":2}}\n'
    asyncio.run(_drain_resume(runner, key, [assistant, result], returncode=0))
    check("healthy_resume.mapping_kept", runner.registry.has(key),
          "a session that produced output must not be reset")


def test_late_error_after_output_keeps_mapping() -> None:
    """Streamed a real answer, THEN hit an error subtype -> session left intact."""
    runner = _runner("late")
    key = "conv-late"
    assistant = (
        b'{"type":"assistant","message":{"content":[{"type":"text","text":"partial answer"}]}}\n'
    )
    result = b'{"type":"result","subtype":"error_during_execution","error":"late","usage":{}}\n'
    asyncio.run(_drain_resume(runner, key, [assistant, result], returncode=0))
    check("late_error.mapping_kept", runner.registry.has(key),
          "a late error after real output should not reset the session")


if __name__ == "__main__":
    test_dead_resume_drops_mapping()
    test_nonzero_resume_no_output_drops_mapping()
    test_healthy_resume_keeps_mapping()
    test_late_error_after_output_keeps_mapping()
    print(f"\n{_PASS} passed, {_FAIL} failed")
