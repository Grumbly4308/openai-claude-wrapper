"""entrypoint.sh shell logic, pinned via subprocess — no src imports.

The entrypoint is the one piece of run-time logic the pytest suite would
otherwise never exercise: it ships inside the image and only ever runs under
tini. These tests drive the script directly with HOME and the data dirs
pointed at a tmpdir, so module import order is irrelevant (nothing here
touches src.config's frozen SETTINGS).

Pinned: the codex-refresher's refusal to run with an environment credential
present (it would burn turns renewing a login codex ignores), the stale-image
diagnosis on an unknown command (existing behavior, previously unpinned),
that the refresh turn runs against a private CODEX_HOME (a fake codex binary
stands in for the CLI), and that the script parses at all. The codex-login
path needs a real codex binary and stays manual.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ENTRYPOINT = Path(__file__).resolve().parents[1] / "entrypoint.sh"


def _run(args, tmp_path, extra_env=None):
    # Minimal env: `set -u` makes the unguarded data-dir expansions fatal, so
    # every var the mkdir line reads must be present, and HOME keeps the
    # claude/codex home probes inside the tmpdir.
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "CLAUDE_WRAPPER_WORKSPACE": str(tmp_path / "workspace"),
        "CLAUDE_WRAPPER_FILES": str(tmp_path / "files"),
        "CLAUDE_WRAPPER_SESSIONS": str(tmp_path / "sessions"),
    }
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(ENTRYPOINT), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_codex_refresher_refuses_env_credential(tmp_path):
    # An env key makes codex ignore the volume login this loop exists to
    # renew; the entrypoint must refuse loudly, naming the codex vars —
    # not sleep forever measuring the wrong thing.
    proc = _run(["codex-refresher"], tmp_path, {"OPENAI_API_KEY": "x"})
    assert proc.returncode != 0
    assert "OPENAI_API_KEY / CODEX_API_KEY" in proc.stderr

    proc = _run(["codex-refresher"], tmp_path, {"CODEX_API_KEY": "x"})
    assert proc.returncode != 0
    assert "OPENAI_API_KEY / CODEX_API_KEY" in proc.stderr


def test_unknown_command_gets_stale_image_diagnosis(tmp_path):
    # A role name this image doesn't know must fail with the rebuild hint,
    # not a bare `exec: not found` crash-loop — that is what a stale image
    # looks like when compose comes from a newer checkout.
    proc = _run(["no-such-role-from-the-future"], tmp_path)
    assert proc.returncode == 127
    assert "IMAGE is stale" in proc.stderr
    assert "rebuild" in proc.stderr


def test_codex_refresh_turn_runs_in_private_home(tmp_path):
    # The volume's config.toml is agent-writable and codex executes config
    # directives (mcp_servers commands spawn as plain subprocesses) — loaded
    # in the refresher, that hands a prompt-injected agent command execution
    # with un-allowlisted egress. The refresh turn must therefore run against
    # a scratch CODEX_HOME seeded with auth.json alone, and a renewed
    # credential must still land back in the volume.
    home = tmp_path / "home"
    codex_home = home / ".codex"
    codex_home.mkdir(parents=True)
    (codex_home / "auth.json").write_text(
        json.dumps({"tokens": {"access_token": "tok"}, "last_refresh": "2020-01-01T00:00:00Z"})
    )
    (codex_home / "config.toml").write_text('[mcp_servers.evil]\ncommand = "curl"\n')

    seen = tmp_path / "seen"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    fake = fakebin / "codex"
    fake.write_text(
        "#!/bin/bash\n"
        f'echo "$CODEX_HOME" > {seen}-home\n'
        f'ls "$CODEX_HOME" > {seen}-ls\n'
        "cat > /dev/null\n"
        'printf \'{"tokens":{"access_token":"tok"},"last_refresh":"%s"}\' '
        '"$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$CODEX_HOME/auth.json"\n'
    )
    fake.chmod(0o755)

    env = {
        "PATH": f"{fakebin}:/usr/bin:/bin",
        "HOME": str(home),
        "CLAUDE_WRAPPER_WORKSPACE": str(tmp_path / "workspace"),
        "CLAUDE_WRAPPER_FILES": str(tmp_path / "files"),
        "CLAUDE_WRAPPER_SESSIONS": str(tmp_path / "sessions"),
        "CODEX_REFRESH_CHECK_SECONDS": "300",
    }
    # The loop never exits on its own; one pass finishes in well under the
    # bound, after which `timeout` reaps it (rc 124) and we assert on the
    # side effects the pass left behind.
    proc = subprocess.run(
        ["timeout", "6", "bash", str(ENTRYPOINT), "codex-refresher"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 124, proc.stderr
    turn_home = (seen.parent / "seen-home").read_text().strip()
    assert turn_home and turn_home != str(codex_home)
    assert "config.toml" not in (seen.parent / "seen-ls").read_text()
    renewed = json.loads((codex_home / "auth.json").read_text())
    assert renewed["last_refresh"] != "2020-01-01T00:00:00Z"
    assert "renewed" in proc.stderr


def test_entrypoint_parses():
    proc = subprocess.run(
        ["bash", "-n", str(ENTRYPOINT)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
