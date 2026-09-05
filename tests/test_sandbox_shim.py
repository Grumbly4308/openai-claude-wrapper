"""Agent shim + remote executor: the sandboxed split's wire contract.

Covers the shim's trust boundary (token gate, cwd containment, argv[0]
substitution), the NDJSON passthrough + exit-sentinel framing, and a full
ClaudeRunner.run_collect round trip through RemoteAgentExecutor against the
in-process shim app — proving the remote path yields the same events the local
subprocess path does, with no real `claude` binary involved.

Named test_sandbox_shim (not test_agent_shim) so it collects AFTER
test_budget.py: the suite relies on test_budget being the first module to
import src.config (see the test_downloads docstring), and an earlier-sorting
module that imports src.* freezes SETTINGS before test_budget's env preamble
runs, failing the budget/stats tests.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="claude-wrapper-shim-test-")
os.environ.setdefault("CLAUDE_WRAPPER_DATA", _TMP)
os.environ.setdefault("CLAUDE_WRAPPER_MODEL_DISCOVERY", "off")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import importlib  # noqa: E402

from src import agent_shim  # noqa: E402
from src import claude_runner  # noqa: E402

# Sibling test modules (notably test_endpoints) monkeypatch
# ClaudeRunner.run_stream with a stub at import time; reload so the round-trip
# test below exercises the genuine implementation. agent_shim keeps its
# references to the pre-reload helper functions, which are the same code.
importlib.reload(claude_runner)

client = TestClient(agent_shim.app)

WS = Path(_TMP) / "workspace"
WS.mkdir(parents=True, exist_ok=True)

# Two stream-json lines a healthy `claude -p` run would emit.
_ASSISTANT = '{"type":"assistant","message":{"content":[{"type":"text","text":"hi from fake"}]}}'
_RESULT = '{"type":"result","subtype":"success","usage":{"input_tokens":1,"output_tokens":2}}'

# Three `codex exec --json` lines a healthy fresh run would emit (schema pinned
# in test_codex_runner). Every codex line opens '{"type":', which is what keeps
# the exit sentinel ('{"__claude_wrapper_exit__"') collision-free — see the
# sentinel test below.
_CODEX_THREAD = '{"type":"thread.started","thread_id":"01a06fb8-718c-7c53-b1d7-489513ad39bf"}'
_CODEX_MSG = '{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"hi from fake codex"}}'
_CODEX_DONE = '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":2}}'


@pytest.fixture
def fake_claude(tmp_path):
    """A stand-in binary: consumes stdin, emits two stream-json lines."""
    script = tmp_path / "fake-claude"
    script.write_text(
        "#!/bin/sh\n"
        "cat > /dev/null\n"
        f"printf '%s\\n' '{_ASSISTANT}'\n"
        f"printf '%s\\n' '{_RESULT}'\n"
    )
    script.chmod(0o755)
    return script


@pytest.fixture
def fake_codex(tmp_path):
    """A stand-in codex binary: consumes stdin, emits three JSONL lines."""
    script = tmp_path / "fake-codex"
    script.write_text(
        "#!/bin/sh\n"
        "cat > /dev/null\n"
        f"printf '%s\\n' '{_CODEX_THREAD}'\n"
        f"printf '%s\\n' '{_CODEX_MSG}'\n"
        f"printf '%s\\n' '{_CODEX_DONE}'\n"
    )
    script.chmod(0o755)
    return script


def _shim_settings(monkeypatch, **overrides):
    overrides.setdefault("workspace_dir", WS)
    overrides.setdefault("agent_token", "")
    monkeypatch.setattr(
        agent_shim, "SETTINGS", dataclasses.replace(agent_shim.SETTINGS, **overrides)
    )


def _run(payload: dict, headers: dict | None = None):
    return client.post("/run", json=payload, headers=headers or {})


def _payload(cwd: Path | str) -> dict:
    return {"args": ["-p"], "prompt": "hi", "cwd": str(cwd), "env": {}}


# ---------- passthrough + framing ----------


def test_stdout_lines_forwarded_verbatim_then_exit_sentinel(monkeypatch, fake_claude):
    _shim_settings(monkeypatch, claude_bin=str(fake_claude))
    r = _run(_payload(WS / "sess-frame"))
    assert r.status_code == 200
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    assert lines[0] == _ASSISTANT
    assert lines[1] == _RESULT
    exit_record = json.loads(lines[2])[claude_runner.SHIM_EXIT_KEY]
    assert exit_record["returncode"] == 0
    assert exit_record["timed_out"] is False


def test_spawn_failure_arrives_as_an_exit_record_not_a_severed_body(monkeypatch):
    """The response head is committed before the generator runs, so a missing
    binary must surface as a well-formed sentinel the executor can report."""
    _shim_settings(monkeypatch, claude_bin=str(Path(_TMP) / "no-such-binary"))
    r = _run(_payload(WS / "sess-spawnfail"))
    assert r.status_code == 200
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    assert len(lines) == 1
    exit_record = json.loads(lines[0])[claude_runner.SHIM_EXIT_KEY]
    assert exit_record["returncode"] == -1
    assert "failed to spawn" in exit_record["stderr"]


# ---------- trust boundary ----------


def test_cwd_outside_the_workspace_root_is_rejected(monkeypatch, fake_claude):
    _shim_settings(monkeypatch, claude_bin=str(fake_claude))
    assert _run(_payload("/etc")).status_code == 400
    # Traversal back out of the root is caught after resolution.
    assert _run(_payload(WS / ".." / "escape")).status_code == 400
    # The root itself and anything beneath it are fine.
    assert _run(_payload(WS)).status_code == 200


def test_token_gate(monkeypatch, fake_claude):
    _shim_settings(monkeypatch, claude_bin=str(fake_claude), agent_token="hunter2")
    assert _run(_payload(WS / "sess-tok")).status_code == 401
    assert (
        _run(_payload(WS / "sess-tok"), headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )
    assert (
        _run(_payload(WS / "sess-tok"), headers={"Authorization": "Bearer hunter2"}).status_code
        == 200
    )


# ---------- agent selection + env-overlay hardening ----------


def test_shim_spawns_codex_bin_when_agent_is_codex(monkeypatch, fake_codex):
    """Which binary runs here is decided by THIS container's env (agent +
    codex_bin), never by the request — RunRequest carries no agent/binary
    field, and the deliberately bogus claude_bin proves the selector actually
    switched rather than falling through."""
    _shim_settings(
        monkeypatch, agent="codex", codex_bin=str(fake_codex), claude_bin="/nonexistent-claude"
    )
    r = _run(_payload(WS / "sess-codex"))
    assert r.status_code == 200
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    assert lines[0] == _CODEX_THREAD
    assert lines[1] == _CODEX_MSG
    assert lines[2] == _CODEX_DONE
    exit_record = json.loads(lines[3])[claude_runner.SHIM_EXIT_KEY]
    assert exit_record["returncode"] == 0
    assert exit_record["timed_out"] is False


def test_shim_ignores_caller_path_and_loader_env(monkeypatch, tmp_path):
    """A caller-supplied PATH must not steer which binary a slash-less
    configured name resolves to (execvpe resolves a relative argv[0] against
    the CHILD env), and loader/interpreter overrides must never reach the
    child at all — the overlay is for run-scoped knobs only."""
    good = tmp_path / "good"
    evil = tmp_path / "evil"
    good.mkdir()
    evil.mkdir()
    (good / "fake-agent-envdump").write_text(
        "#!/bin/sh\n"
        "cat > /dev/null\n"
        'printf \'GOOD PATH=%s\\n\' "$PATH"\n'
        'printf \'GOOD LD_PRELOAD=%s\\n\' "${LD_PRELOAD:-unset}"\n'
        'printf \'GOOD NODE_OPTIONS=%s\\n\' "${NODE_OPTIONS:-unset}"\n'
        'printf \'GOOD PYTHONPATH=%s\\n\' "${PYTHONPATH:-unset}"\n'
    )
    (good / "fake-agent-envdump").chmod(0o755)
    # The malicious twin a steered PATH resolution would have picked instead.
    (evil / "fake-agent-envdump").write_text("#!/bin/sh\ncat > /dev/null\necho EVIL\n")
    (evil / "fake-agent-envdump").chmod(0o755)
    for var in ("LD_PRELOAD", "NODE_OPTIONS", "PYTHONPATH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PATH", f"{good}{os.pathsep}{os.environ.get('PATH', '')}")
    _shim_settings(monkeypatch, claude_bin="fake-agent-envdump")
    payload = _payload(WS / "sess-envhard")
    payload["env"] = {
        "PATH": str(evil),
        "LD_PRELOAD": "/x.so",
        "NODE_OPTIONS": "--require /x.js",
        "PYTHONPATH": "/evil-pythonpath",
    }
    r = _run(payload)
    assert r.status_code == 200
    assert "EVIL" not in r.text  # PATH steering did not pick the binary...
    assert str(evil) not in r.text  # ...and never reached the child either
    assert "GOOD LD_PRELOAD=unset" in r.text
    assert "GOOD NODE_OPTIONS=unset" in r.text
    assert "GOOD PYTHONPATH=unset" in r.text
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    exit_record = json.loads(lines[-1])[claude_runner.SHIM_EXIT_KEY]
    assert exit_record["returncode"] == 0


def test_healthz_names_the_configured_agent(monkeypatch):
    """The wrapper's boot handshake reads this field to refuse a mismatched
    wrapper/shim pair; pin both values."""
    assert client.get("/healthz").json() == {"status": "ok", "agent": "claude"}
    _shim_settings(monkeypatch, agent="codex")
    assert client.get("/healthz").json() == {"status": "ok", "agent": "codex"}


def test_codex_output_cannot_forge_the_exit_sentinel(monkeypatch, fake_codex):
    """Every codex --json line opens '{"type":', so agent output can never be
    mistaken for (or shadow) the shim's exit record — the sentinel framing
    survives the second dialect."""
    _shim_settings(
        monkeypatch, agent="codex", codex_bin=str(fake_codex), claude_bin="/nonexistent-claude"
    )
    r = _run(_payload(WS / "sess-codex-sentinel"))
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    assert all(not ln.startswith(claude_runner._SHIM_EXIT_PREFIX) for ln in lines[:-1])
    assert lines[-1].startswith(claude_runner._SHIM_EXIT_PREFIX)


# ---------- remote executor round trip ----------


def test_run_collect_through_the_remote_executor(monkeypatch, fake_claude):
    """End to end across the split: ClaudeRunner -> RemoteAgentExecutor ->
    shim app -> fake subprocess. The runner's claude_bin is deliberately bogus:
    reaching the fake binary proves the shim substituted argv[0] with its own,
    i.e. the API container cannot pick what the agent container executes."""
    _shim_settings(monkeypatch, claude_bin=str(fake_claude))
    executor = claude_runner.RemoteAgentExecutor("http://agent")
    executor._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent_shim.app), base_url="http://agent"
    )
    runner = claude_runner.ClaudeRunner(
        registry=claude_runner.SessionRegistry(Path(_TMP) / "sessions-remote"),
        workspace_root=WS,
        claude_bin="not-a-real-binary-anywhere",
        # The fake emits consolidated assistant blocks; partial mode would
        # suppress them as already-streamed duplicates (see test_resume_selfheal).
        stream_partial_messages=False,
        executor=executor,
    )

    async def scenario():
        try:
            return await runner.run_collect(
                prompt="hi", session_key="remote-e2e", model="claude-opus-4-8"
            )
        finally:
            await executor.aclose()

    result = asyncio.run(scenario())
    assert result.error is None
    assert result.final_text == "hi from fake"
    assert result.input_tokens == 1
    assert result.output_tokens == 2
    assert runner.registry.has("remote-e2e")


def test_shim_unreachable_is_an_error_not_a_crash():
    """A down agent container must surface as a runner error the API layer
    already knows how to render (502 / error chunk), never an exception."""
    executor = claude_runner.RemoteAgentExecutor("http://agent")
    executor._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(httpx.ConnectError("boom"))
        ),
        base_url="http://agent",
    )
    runner = claude_runner.ClaudeRunner(
        registry=claude_runner.SessionRegistry(Path(_TMP) / "sessions-down"),
        workspace_root=WS,
        claude_bin="claude",
        executor=executor,
    )

    async def scenario():
        try:
            return await runner.run_collect(prompt="hi", session_key="down-e2e")
        finally:
            await executor.aclose()

    result = asyncio.run(scenario())
    assert result.error is not None
    assert "agent shim unreachable" in result.error


def test_capability_gating_argv_crosses_the_shim(monkeypatch, tmp_path):
    """The profile's --disallowedTools is built in the API container; the agent
    executes it only because the shim forwards args verbatim. Pin that chain:
    with the terminal gate closed (its default), the fake binary refuses to
    answer unless the gating flags actually arrived."""
    from src.capabilities import TERMINAL_TOGGLE_ENV, reset_profile_cache

    monkeypatch.delenv(TERMINAL_TOGGLE_ENV, raising=False)
    reset_profile_cache()
    script = tmp_path / "fake-claude-gated"
    script.write_text(
        "#!/bin/sh\n"
        "cat > /dev/null\n"
        'case "$*" in *"--disallowedTools Bash"*) ;; *) echo "gating argv missing" >&2; exit 3;; esac\n'
        f"printf '%s\\n' '{_ASSISTANT}'\n"
        f"printf '%s\\n' '{_RESULT}'\n"
    )
    script.chmod(0o755)
    _shim_settings(monkeypatch, claude_bin=str(script))
    executor = claude_runner.RemoteAgentExecutor("http://agent")
    executor._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent_shim.app), base_url="http://agent"
    )
    runner = claude_runner.ClaudeRunner(
        registry=claude_runner.SessionRegistry(Path(_TMP) / "sessions-gated"),
        workspace_root=WS,
        claude_bin="unused-locally",
        stream_partial_messages=False,
        executor=executor,
    )

    async def scenario():
        try:
            return await runner.run_collect(
                prompt="hi", session_key="remote-gated", model="claude-opus-4-8"
            )
        finally:
            await executor.aclose()

    try:
        result = asyncio.run(scenario())
    finally:
        reset_profile_cache()
    assert result.error is None
    assert result.final_text == "hi from fake"
