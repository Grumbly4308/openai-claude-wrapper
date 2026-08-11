"""CLI-path capability enforcement: profile → --disallowedTools (phase 2).

Pins the argv-building layer: chat runs are gated by the model's capability
profile, delegation runs never are, and with the terminal toggle set the
default profile leaves the argv byte-for-byte identical to an ungated run.

Runs under pytest or standalone: python3 tests/test_profile_argv.py
(needs the wrapper's requirements installed — imports claude_runner).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="claude-wrapper-profile-argv-test-")
os.environ["CLAUDE_WRAPPER_DATA"] = _TMP
os.environ["CLAUDE_WRAPPER_MODEL_DISCOVERY"] = "off"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capabilities import (  # noqa: E402
    PROFILE_FILE_ENV,
    PROFILE_OVERRIDES_ENV,
    TERMINAL_TOGGLE_ENV,
    reset_profile_cache,
)
from src.claude_runner import ClaudeRunner, SessionRegistry  # noqa: E402


def _reset() -> None:
    os.environ.pop(PROFILE_FILE_ENV, None)
    os.environ.pop(PROFILE_OVERRIDES_ENV, None)
    os.environ.pop(TERMINAL_TOGGLE_ENV, None)
    reset_profile_cache()


def _write_profile(doc: dict) -> None:
    path = Path(_TMP) / "profiles.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    os.environ[PROFILE_FILE_ENV] = str(path)
    reset_profile_cache()


def _runner(**kw) -> ClaudeRunner:
    return ClaudeRunner(
        registry=SessionRegistry(Path(_TMP) / "sessions"),
        workspace_root=Path(_TMP) / "workspace",
        claude_bin="claude",
        **kw,
    )


def _argv(model: str = "claude-opus-5", **kw) -> list[str]:
    return _runner()._build_argv(session_uuid="u", model=model, resume=False, **kw)


def _disallowed(argv: list[str]) -> list[str]:
    if "--disallowedTools" not in argv:
        return []
    i = argv.index("--disallowedTools") + 1
    out = []
    while i < len(argv) and not argv[i].startswith("--"):
        out.append(argv[i])
        i += 1
    return out


def test_terminal_gate_disallows_bash_by_default():
    # CLAUDE_WRAPPER_EXPOSE_TERMINAL unset → chat runs carry Bash disallowed.
    _reset()
    assert _disallowed(_argv()) == ["Bash"]


def test_toggle_set_argv_matches_ungated_byte_for_byte():
    _reset()
    os.environ[TERMINAL_TOGGLE_ENV] = "true"
    reset_profile_cache()
    gated = _argv()
    ungated = _argv(capability_gated=False)
    assert gated == ungated, f"{gated} != {ungated}"
    assert "--disallowedTools" not in gated


def test_profile_removals_map_to_tool_names():
    _reset()
    os.environ[TERMINAL_TOGGLE_ENV] = "true"
    _write_profile(
        {"models": [{"match": "claude-opus-5", "remove": ["web_search", "sub_agents"]}]}
    )
    assert _disallowed(_argv()) == ["WebSearch", "WebFetch", "Task"]


def test_delegation_is_never_gated():
    # Even a profile that strips everything must not touch delegation runs.
    _reset()
    _write_profile({"default": {"capabilities": []}})
    argv = _argv(capability_gated=False)
    assert "--disallowedTools" not in argv


def test_gating_merges_with_clarify_tools():
    _reset()  # terminal off → profile contributes Bash
    r = ClaudeRunner(
        registry=SessionRegistry(Path(_TMP) / "sessions-cl"),
        workspace_root=Path(_TMP) / "workspace-cl",
        claude_bin="claude",
        clarify_system_prompt="ask before acting",
        clarify_disallowed_tools=("AskUserQuestion", "Bash"),
    )
    argv = r._build_argv(session_uuid="u", model="claude-opus-5", resume=False, clarify=True)
    # One emission, deduped, profile tools first; the trailing flag still
    # terminates the variadic list.
    assert argv.count("--disallowedTools") == 1
    assert _disallowed(argv) == ["Bash", "AskUserQuestion"]
    assert argv[argv.index("AskUserQuestion") + 1] == "--dangerously-skip-permissions"


def test_effort_variant_uses_base_profile():
    _reset()
    os.environ[TERMINAL_TOGGLE_ENV] = "true"
    _write_profile({"models": [{"match": "claude-opus-5", "remove": ["sub_agents"]}]})
    assert _disallowed(_argv("claude-opus-5 (high)")) == ["Task"]


def main() -> int:
    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
        else:
            passed += 1
    print(f"\nRESULT pass={passed} fail={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
