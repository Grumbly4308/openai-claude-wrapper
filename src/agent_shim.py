"""The agent container's inbound surface: run the wrapped CLI on behalf of the API.

In the sandboxed topology (docker-compose.yml) the FastAPI wrapper and
the wrapped CLI live in different containers: the wrapper is the only
externally reachable service, and the agent container — where model-driven
tool use actually executes — sits on an internal network whose only egress is
an allowlisting proxy. This shim is how the wrapper reaches the CLI across
that boundary. It accepts one request shape, spawns the configured agent
binary (`claude` by default, `codex` under CLAUDE_WRAPPER_AGENT=codex) in a
workspace directory shared between the containers, and streams the raw
stream-json stdout lines back, terminated by a single sentinel line
carrying returncode/stderr (claude_runner.SHIM_EXIT_KEY). The wrapper's
RemoteAgentExecutor consumes exactly this contract.

Deliberately minimal trust surface:
- argv[0] is never taken from the caller: which binary runs here is decided by
  THIS container's environment (CLAUDE_WRAPPER_AGENT + the per-agent *_BIN
  var, set per-service in compose), never by anything in the request —
  RunRequest deliberately carries no agent/binary field. A slash-less
  configured name is resolved against the shim's own PATH before spawning
  (_resolved_agent_bin), so the child env can't steer resolution either.
- caller-supplied env vars overlay this container's environment, so the
  HTTP(S)_PROXY pointing at the egress proxy is inherited by the CLI and every
  Bash subshell it runs — that inheritance IS the sandbox integration. The
  overlay is for run-scoped knobs only: PATH/PYTHONPATH/PYTHONHOME/
  NODE_OPTIONS/LD_* are dropped so the API container cannot steer loader,
  interpreter, or binary resolution.
- cwd must resolve inside the shared workspace root.
- an optional bearer token (CLAUDE_WRAPPER_AGENT_TOKEN) gates every request.

Run with:  entrypoint.sh agent   (uvicorn src.agent_shim:app)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import shutil
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .claude_runner import (
    SHIM_EXIT_KEY,
    _STREAM_BUFFER_LIMIT,
    _drain_stderr,
    _read_lines,
)
from .config import SETTINGS, log_agent_credential_status

log = logging.getLogger("claude_wrapper.agent_shim")

app = FastAPI(title="Claude Wrapper Agent Shim", version="0.1.0")


@app.on_event("startup")
async def _startup() -> None:
    # This container owns the writable credentials mount and is where the CLI
    # actually authenticates, so an expired login shows up here first — as
    # `claude exited 1` with empty stderr, which says nothing about the cause.
    log_agent_credential_status("Claude")


# How much stderr to ship back in the exit record. The wrapper only ever quotes
# the last 500 bytes into an error message; 2000 leaves headroom for logs.
_STDERR_TAIL = 2000


class RunRequest(BaseModel):
    args: list[str] = Field(default_factory=list)  # argv WITHOUT the binary
    prompt: str = ""
    cwd: str
    env: dict[str, str] = Field(default_factory=dict)


def _check_token(authorization: Optional[str]) -> None:
    token = SETTINGS.agent_token
    if not token:
        return
    presented = authorization or ""
    if presented.lower().startswith("bearer "):
        presented = presented[7:].strip()
    if not secrets.compare_digest(presented, token):
        raise HTTPException(status_code=401, detail="invalid agent token")


def _resolve_cwd(raw: str) -> Path:
    """Contain the run to the shared workspace volume.

    Both containers mount the workspace at the same path, so the wrapper's
    per-session cwd is valid here verbatim — anything else is rejected.
    """
    root = SETTINGS.workspace_dir.resolve()
    cwd = Path(raw).resolve()
    if cwd != root and root not in cwd.parents:
        raise HTTPException(status_code=400, detail=f"cwd must live under {root}")
    cwd.mkdir(parents=True, exist_ok=True)
    return cwd


@app.get("/healthz")
async def healthz() -> dict:
    # `agent` is the wrapper's boot-time handshake surface: main.py refuses to
    # start when the two containers disagree on which agent they are running
    # (stale .env / AGENT_URL pointing at the other stack).
    return {"status": "ok", "agent": SETTINGS.agent}


@app.post("/run")
async def run(
    req: RunRequest, authorization: Optional[str] = Header(default=None)
) -> StreamingResponse:
    _check_token(authorization)
    cwd = _resolve_cwd(req.cwd)
    return StreamingResponse(_run_stream(req, cwd), media_type="application/x-ndjson")


def _resolved_agent_bin() -> Optional[str]:
    """Absolute path of the binary this shim will spawn, or None if missing.

    POSIX execvpe resolves a slash-less argv[0] against PATH in the CHILD env
    — with the caller's env overlaid, a req.env["PATH"] pointing at the shared
    writable workspace volume would steer which binary runs. Resolving against
    the shim's OWN environment first (and denylisting PATH from the overlay,
    see _run_stream) closes that. Resolved at call time, not import time:
    tests monkeypatch SETTINGS after import.
    """
    configured = SETTINGS.agent_bin
    if os.sep in configured:
        return configured
    return shutil.which(configured)


def _exit_line(returncode: int, stderr: str, timed_out: bool = False) -> bytes:
    record = {
        SHIM_EXIT_KEY: {
            "returncode": returncode,
            "stderr": stderr[-_STDERR_TAIL:],
            "timed_out": timed_out,
        }
    }
    return (json.dumps(record) + "\n").encode("utf-8")


async def _run_stream(req: RunRequest, cwd: Path) -> AsyncIterator[bytes]:
    # Mirror LocalAgentExecutor's env semantics exactly, against THIS
    # container's environment: proxy vars and OPENWEBUI_* live here, and
    # CI/telemetry are set unless already pinned.
    env = os.environ.copy()
    env.setdefault("CI", "1")
    env.setdefault("CLAUDE_CODE_DISABLE_TELEMETRY", "1")
    # The overlay exists for run-scoped knobs (proxy hints, OPENWEBUI_*), not
    # for steering loader/interpreter/binary resolution from the API container
    # — drop the keys that would (LD_PRELOAD, NODE_OPTIONS et al. hijack an
    # absolute argv[0] just as surely as PATH hijacks a relative one).
    env.update(
        {
            k: v
            for k, v in req.env.items()
            if k not in ("PATH", "PYTHONPATH", "PYTHONHOME", "NODE_OPTIONS")
            and not k.startswith("LD_")
        }
    )

    agent_bin = _resolved_agent_bin()
    if agent_bin is None:
        # Same not-an-HTTP-error framing as the spawn failure below: the
        # response head is already committed when this generator runs.
        yield _exit_line(-1, f"agent binary {SETTINGS.agent_bin!r} not found on PATH")
        return
    argv = [agent_bin, *req.args]
    log.info("spawning %s (cwd=%s, args=%d)", agent_bin, cwd, len(req.args))
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
            env=env,
            limit=_STREAM_BUFFER_LIMIT,
        )
    except OSError as e:
        # Spawn failures must arrive as a well-formed exit record, not a
        # severed body: the response head is already committed by the time
        # this generator runs.
        yield _exit_line(-1, f"failed to spawn {agent_bin}: {e}")
        return

    async def _feed_stdin() -> None:
        try:
            proc.stdin.write(req.prompt.encode("utf-8"))
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with contextlib.suppress(Exception):
                proc.stdin.close()

    stdin_task = asyncio.create_task(_feed_stdin())
    stderr_task = asyncio.create_task(_drain_stderr(proc.stderr, SETTINGS.agent))
    timed_out = False
    try:
        try:
            async for line in _read_lines(proc.stdout, SETTINGS.request_timeout_seconds):
                if not line.endswith("\n"):
                    line += "\n"
                yield line.encode("utf-8")
        except asyncio.TimeoutError:
            timed_out = True
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        except (asyncio.CancelledError, GeneratorExit):
            # The wrapper hung up (its own timeout, or the end client
            # disconnected). Kill the CLI so the turn doesn't keep burning
            # tokens with nobody listening.
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            raise
    finally:
        returncode = await proc.wait()
        stderr_output = await stderr_task
        with contextlib.suppress(Exception):
            await stdin_task
    yield _exit_line(returncode, stderr_output, timed_out)
