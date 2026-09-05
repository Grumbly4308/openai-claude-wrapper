"""entrypoint.sh shell logic, pinned via subprocess — no src imports.

The entrypoint is the one piece of run-time logic the pytest suite would
otherwise never exercise: it ships inside the image and only ever runs under
tini. These tests drive the script directly with HOME and the data dirs
pointed at a tmpdir, so module import order is irrelevant (nothing here
touches src.config's frozen SETTINGS).

Pinned: the codex-refresher's refusal to run with an environment credential
present (it would burn turns renewing a login codex ignores), the stale-image
diagnosis on an unknown command (existing behavior, previously unpinned), and
that the script parses at all. The codex-login and refresh-loop paths need a
real codex binary and stay manual.
"""

from __future__ import annotations

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


def test_entrypoint_parses():
    proc = subprocess.run(
        ["bash", "-n", str(ENTRYPOINT)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
