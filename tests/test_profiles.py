"""Unit tests for per-model capability profiles (src/capabilities.py).

Covers the three-layer resolution (built-in default → profile file → inline
overrides), pattern matching, effort/[1m] variant inheritance, and the
validation errors the loader must raise with the offending entry named.

Runs under pytest or standalone: python3 tests/test_profiles.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

# config.SETTINGS is built at import time and mkdirs its data dir — point it at
# a tempdir so importing src.* never touches /data.
_TMP = tempfile.mkdtemp(prefix="claude-wrapper-profiles-test-")
os.environ["CLAUDE_WRAPPER_DATA"] = _TMP
# Use the static fallback list — don't scan the 250 MB CLI binary during tests.
os.environ["CLAUDE_WRAPPER_MODEL_DISCOVERY"] = "off"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capabilities import (  # noqa: E402
    DEFAULT_CAPABILITIES,
    PROFILE_FILE_ENV,
    PROFILE_OVERRIDES_ENV,
    TERMINAL_TOGGLE_ENV,
    Capability,
    ProfileConfigError,
    has_capability,
    reset_profile_cache,
    resolve_profile,
)


def _reset() -> None:
    """Fresh state: no profile env vars, empty caches."""
    os.environ.pop(PROFILE_FILE_ENV, None)
    os.environ.pop(PROFILE_OVERRIDES_ENV, None)
    os.environ.pop(TERMINAL_TOGGLE_ENV, None)
    reset_profile_cache()


def _enable_terminal() -> None:
    os.environ[TERMINAL_TOGGLE_ENV] = "true"
    reset_profile_cache()


def _write_profile(doc: dict) -> None:
    path = Path(_TMP) / "profiles.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    os.environ[PROFILE_FILE_ENV] = str(path)
    reset_profile_cache()


def _expect_error(needle: str) -> None:
    try:
        resolve_profile("claude-opus-5")
    except ProfileConfigError as e:
        assert needle in str(e), f"{needle!r} not in error: {e}"
    else:
        raise AssertionError(f"ProfileConfigError with {needle!r} not raised")


# --- defaults -----------------------------------------------------------------


def test_builtin_default_absent_config():
    # Terminal is masked until the operator opts in, everything else defaults.
    _reset()
    expected = DEFAULT_CAPABILITIES - {Capability.TERMINAL}
    assert resolve_profile("claude-opus-5") == expected


def test_builtin_default_preserves_todays_behavior():
    # With the terminal toggle set, the default matches what the CLI path does
    # today: terminal/web-search/sub-agents on, client tools allowed;
    # capabilities needing new wrapper machinery are off.
    _reset()
    _enable_terminal()
    caps = resolve_profile("claude-sonnet-5")
    assert Capability.TERMINAL in caps
    assert Capability.CLIENT_TOOLS in caps
    assert Capability.CODE_INTERPRETER not in caps
    assert Capability.MEMORY not in caps


# --- codex: advertise only what is enforced -----------------------------------


def test_codex_advertises_intrinsic_cli_capabilities(monkeypatch):
    # The codex CLI has no per-tool switches: terminal/web_search/sub_agents
    # cannot actually be removed, so a profile removal (or the terminal env
    # gate) must not falsify /v1/models — the removal would be advertisement
    # without enforcement. client_tools removal stays real (bridge-enforced).
    import dataclasses

    from src import config as src_config

    _reset()
    _write_profile(
        {"models": [{"match": "gpt-*", "remove": ["terminal", "web_search", "client_tools"]}]}
    )
    monkeypatch.setattr(
        src_config, "SETTINGS", dataclasses.replace(src_config.SETTINGS, agent="codex")
    )
    try:
        caps = resolve_profile("gpt-5.2")
        assert Capability.TERMINAL in caps
        assert Capability.SUB_AGENTS in caps
        assert Capability.CLIENT_TOOLS not in caps
        # web_search is the exception: codex's search is opt-in, so while the
        # knob is off the capability must NOT be advertised — the same
        # advertise-only-what-is-real rule, cutting the other way.
        assert Capability.WEB_SEARCH not in caps

        os.environ["CLAUDE_WRAPPER_CODEX_WEB_SEARCH"] = "true"
        reset_profile_cache()
        assert Capability.WEB_SEARCH in resolve_profile("gpt-5.2")
    finally:
        os.environ.pop("CLAUDE_WRAPPER_CODEX_WEB_SEARCH", None)
        _reset()


# --- terminal env gate --------------------------------------------------------


def test_terminal_masked_by_default():
    _reset()
    assert Capability.TERMINAL not in resolve_profile("claude-opus-5")


def test_terminal_profile_grant_masked_without_toggle():
    # A profile file alone must not be able to expose a shell to the UI.
    _reset()
    _write_profile({"models": [{"match": "claude-opus-5", "add": ["terminal"]}]})
    assert Capability.TERMINAL not in resolve_profile("claude-opus-5")


def test_terminal_toggle_enables():
    _reset()
    _enable_terminal()
    assert Capability.TERMINAL in resolve_profile("claude-opus-5")


def test_terminal_toggle_still_respects_profile_remove():
    _reset()
    _write_profile({"models": [{"match": "claude-haiku-*", "remove": ["terminal"]}]})
    _enable_terminal()
    assert Capability.TERMINAL not in resolve_profile("claude-haiku-4-5")
    assert Capability.TERMINAL in resolve_profile("claude-opus-5")


def test_file_default_replaces_builtin():
    _reset()
    _write_profile({"default": {"capabilities": ["vision"]}})
    assert resolve_profile("claude-opus-5") == frozenset({Capability.VISION})


# --- entry matching -----------------------------------------------------------


def test_replace_entry():
    _reset()
    _enable_terminal()
    _write_profile(
        {"models": [{"match": "claude-haiku-4-5", "capabilities": ["vision", "client_tools"]}]}
    )
    assert resolve_profile("claude-haiku-4-5") == frozenset(
        {Capability.VISION, Capability.CLIENT_TOOLS}
    )
    # Other models keep the default.
    assert resolve_profile("claude-opus-5") == DEFAULT_CAPABILITIES


def test_add_remove_entry():
    _reset()
    _write_profile(
        {"models": [{"match": "claude-opus-5", "add": ["memory"], "remove": ["terminal"]}]}
    )
    caps = resolve_profile("claude-opus-5")
    assert Capability.MEMORY in caps
    assert Capability.TERMINAL not in caps


def test_glob_pattern():
    _reset()
    _write_profile({"models": [{"match": "claude-haiku-*", "remove": ["sub_agents"]}]})
    assert Capability.SUB_AGENTS not in resolve_profile("claude-haiku-4-5")
    assert Capability.SUB_AGENTS in resolve_profile("claude-opus-5")


def test_entries_apply_in_order():
    _reset()
    _enable_terminal()
    _write_profile(
        {
            "models": [
                {"match": "claude-*", "remove": ["terminal"]},
                {"match": "claude-opus-5", "add": ["terminal"]},
            ]
        }
    )
    assert Capability.TERMINAL in resolve_profile("claude-opus-5")
    assert Capability.TERMINAL not in resolve_profile("claude-sonnet-5")


# --- variant inheritance ------------------------------------------------------


def test_effort_variant_inherits_base():
    _reset()
    _enable_terminal()
    _write_profile({"models": [{"match": "claude-opus-5", "remove": ["terminal"]}]})
    assert resolve_profile("claude-opus-5 (high)") == resolve_profile("claude-opus-5")
    assert Capability.TERMINAL not in resolve_profile("claude-opus-5 (xhigh)")


def test_long_context_variant_inherits_base():
    _reset()
    _enable_terminal()
    _write_profile({"models": [{"match": "claude-opus-5", "remove": ["terminal"]}]})
    assert Capability.TERMINAL not in resolve_profile("claude-opus-5[1m]")


def test_exact_1m_entry_targets_only_that_id():
    _reset()
    _enable_terminal()
    _write_profile({"models": [{"match": "claude-opus-5[1m]", "remove": ["terminal"]}]})
    assert Capability.TERMINAL not in resolve_profile("claude-opus-5[1m]")
    assert Capability.TERMINAL in resolve_profile("claude-opus-5")


# --- overrides env ------------------------------------------------------------


def test_overrides_apply_after_file():
    _reset()
    _enable_terminal()
    _write_profile({"models": [{"match": "claude-opus-5", "remove": ["terminal"]}]})
    os.environ[PROFILE_OVERRIDES_ENV] = json.dumps(
        {"models": [{"match": "claude-opus-5", "add": ["terminal"]}]}
    )
    reset_profile_cache()
    assert Capability.TERMINAL in resolve_profile("claude-opus-5")


def test_overrides_alone():
    _reset()
    os.environ[PROFILE_OVERRIDES_ENV] = json.dumps(
        {"default": {"capabilities": ["client_tools"]}}
    )
    reset_profile_cache()
    assert resolve_profile("claude-opus-5") == frozenset({Capability.CLIENT_TOOLS})


# --- validation ---------------------------------------------------------------


def test_unknown_capability_names_entry():
    _reset()
    _write_profile({"models": [{"match": "x", "add": ["warp_drive"]}]})
    _expect_error("models[0].add")
    _expect_error("warp_drive")


def test_replace_and_delta_conflict():
    _reset()
    _write_profile({"models": [{"match": "x", "capabilities": ["vision"], "add": ["memory"]}]})
    _expect_error("models[0]")


def test_entry_without_action_rejected():
    _reset()
    _write_profile({"models": [{"match": "x"}]})
    _expect_error("models[0]")


def test_unknown_keys_rejected():
    _reset()
    _write_profile({"models": [{"match": "x", "adds": ["vision"]}]})
    _expect_error("adds")


def test_missing_file_is_an_error():
    _reset()
    os.environ[PROFILE_FILE_ENV] = str(Path(_TMP) / "does-not-exist.json")
    reset_profile_cache()
    _expect_error("cannot read")


def test_invalid_json_is_an_error():
    _reset()
    path = Path(_TMP) / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    os.environ[PROFILE_FILE_ENV] = str(path)
    reset_profile_cache()
    _expect_error("invalid JSON")


def test_has_capability_helper():
    _reset()
    assert has_capability("claude-opus-5", Capability.VISION)
    assert not has_capability("claude-opus-5", Capability.IMAGE_GENERATION)


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
