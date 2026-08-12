"""The agent container's inbound surface: run `claude -p` on behalf of the API.

In the sandboxed topology (docker-compose.yml) the FastAPI wrapper and
the Claude Code CLI live in different containers: the wrapper is the only
externally reachable service, and the agent container — where model-driven
tool use actually executes — sits on an internal network whose only egress is
an allowlisting proxy. This shim is how the wrapper reaches the CLI across
that boundary. It accepts one request shape, spawns the configured `claude`
binary in a workspace directory shared between the containers, and streams the
raw stream-json stdout lines back, terminated by a single sentinel line
carrying returncode/stderr (claude_runner.SHIM_EXIT_KEY). The wrapper's
RemoteAgentExecutor consumes exactly this contract.

Deliberately minimal trust surface:
- argv[0] is never taken from the caller: the shim prepends its own configured
  binary, so nothing that reaches this port can exec an arbitrary program here.
- cwd must resolve inside the shared workspace root.
- an optional bearer token (CLAUDE_WRAPPER_AGENT_TOKEN) gates every request.
- caller-supplied env vars overlay this container's environment, so the
  HTTP(S)_PROXY pointing at the egress proxy is inherited by the CLI and every
  Bash subshell it runs — that inheritance IS the sandbox integration.

Run with:  entrypoint.sh agent   (uvicorn src.agent_shim:app)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
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
from .config import SETTINGS, log_credential_status

log = logging.getLogger("claude_wrapper.agent_shim")

app = FastAPI(title="Claude Wrapper Agent Shim", version="0.1.0")


@app.on_event("startup")
async def _startup() -> None:
    # This container owns the writable credentials mount and is where the CLI
    # actually authenticates, so an expired login shows up here first — as
    # `claude exited 1` with empty stderr, which says nothing about the cause.
    log_credential_status("Claude")


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
    return {"status": "ok"}


@app.post("/run")
async def run(
    req: RunRequest, authorization: Optional[str] = Header(default=None)
) -> StreamingResponse:
    _check_token(authorization)
    cwd = _resolve_cwd(req.cwd)
    return StreamingResponse(_run_stream(req, cwd), media_type="application/x-ndjson")


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
    env.update(req.env)

    argv = [SETTINGS.claude_bin, *req.args]
    log.info("spawning %s (cwd=%s, args=%d)", SETTINGS.claude_bin, cwd, len(req.args))
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
        yield _exit_line(-1, f"failed to spawn {SETTINGS.claude_bin}: {e}")
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
    stderr_task = asyncio.create_task(_drain_stderr(proc.stderr))
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
