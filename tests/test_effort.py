"""Unit tests for per-request effort, including the special "ultracode" choice.

`ultracode` is exposed as an effort choice but is NOT a `claude --effort`
value (the CLI ignores it and falls back to default effort). It must instead
be enabled via `--settings '{"ultracode": true}'`, which the CLI resolves to
xhigh effort plus standing dynamic-workflow orchestration. These tests pin both
the advertising/parsing layer (config) and the argv-building layer (runner).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# config.SETTINGS is built at import time and mkdirs its data dir — point it at
# a tempdir so importing src.* never touches /data.
_TMP = tempfile.mkdtemp(prefix="claude-wrapper-effort-test-")
os.environ["CLAUDE_WRAPPER_DATA"] = _TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.claude_runner import ClaudeRunner, SessionRegistry  # noqa: E402
from src.config import (  # noqa: E402
    EFFORT_LEVELS,
    advertised_models,
    split_model_effort,
)

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


def _runner() -> ClaudeRunner:
    return ClaudeRunner(
        registry=SessionRegistry(Path(_TMP) / "sessions"),
        workspace_root=Path(_TMP) / "workspace",
        claude_bin="claude",
    )


def _argv(model: str, effort) -> list[str]:
    return _runner()._build_argv(
        session_uuid="u", model=model, resume=False, effort=effort
    )


# ---------- advertising ----------


def test_ultracode_advertised_for_opus() -> None:
    models = advertised_models()
    check("advertise.ultracode.opus", "claude-opus-4-8 (ultracode)" in models)
    check("advertise.levels.opus", all(f"claude-opus-4-8 ({lvl})" in models for lvl in EFFORT_LEVELS))


def test_ultracode_not_advertised_for_nonopus() -> None:
    models = advertised_models()
    # Effort variants (incl. ultracode) only exist for effort-capable models.
    check("advertise.ultracode.sonnet.absent", "claude-sonnet-4-6 (ultracode)" not in models)
    check("advertise.sonnet.base.present", "claude-sonnet-4-6" in models)


# ---------- parsing ----------


def test_split_ultracode() -> None:
    check("split.paren", split_model_effort("claude-opus-4-8 (ultracode)") == ("claude-opus-4-8", "ultracode"))
    check("split.colon", split_model_effort("claude-opus-4-8:ultracode") == ("claude-opus-4-8", "ultracode"))
    check("split.case", split_model_effort("claude-opus-4-8 (UltraCode)") == ("claude-opus-4-8", "ultracode"))
    check("split.plain", split_model_effort("claude-opus-4-8") == ("claude-opus-4-8", None))
    check("split.level", split_model_effort("claude-opus-4-8 (max)") == ("claude-opus-4-8", "max"))


# ---------- argv mapping ----------


def test_argv_ultracode_uses_settings_not_effort() -> None:
    argv = _argv("claude-opus-4-8", "ultracode")
    check("argv.ultracode.settings", "--settings" in argv and '{"ultracode": true}' in argv)
    check("argv.ultracode.no_effort_flag", "--effort" not in argv)


def test_argv_level_uses_effort_flag() -> None:
    argv = _argv("claude-opus-4-8", "max")
    check("argv.max.effort", argv[argv.index("--effort") + 1] == "max")
    check("argv.max.no_settings", "--settings" not in argv)


def test_argv_empty_effort_passes_nothing() -> None:
    # Explicit "" means "no effort flag" — neither --effort nor --settings.
    argv = _argv("claude-opus-4-8", "")
    check("argv.empty.no_effort", "--effort" not in argv)
    check("argv.empty.no_settings", "--settings" not in argv)


def test_argv_noncapable_drops_effort() -> None:
    # A non-effort-capable model never gets a flag the CLI would ignore/reject,
    # even when an effort suffix is requested explicitly (e.g. via :max / :ultracode).
    for eff in ("max", "ultracode"):
        argv = _argv("claude-sonnet-4-6", eff)
        check(f"argv.noncapable.{eff}.no_effort", "--effort" not in argv)
        check(f"argv.noncapable.{eff}.no_settings", "--settings" not in argv)


def test_resolve_effort_source() -> None:
    # Server default applies when the request carries no effort suffix — this is
    # exactly the "bare model ran at max" case. The source label makes it visible.
    r = ClaudeRunner(
        registry=SessionRegistry(Path(_TMP) / "sessions-re"),
        workspace_root=Path(_TMP) / "workspace-re",
        claude_bin="claude",
        effort="max",
    )
    check("resolve.default", r._resolve_effort("claude-opus-4-8", None) == ("max", "server-default"))
    check("resolve.request", r._resolve_effort("claude-opus-4-8", "low") == ("low", "request"))
    check("resolve.ultracode", r._resolve_effort("claude-opus-4-8", "ultracode") == ("ultracode", "request"))
    check("resolve.incapable", r._resolve_effort("claude-sonnet-4-6", "max") == ("", "model-incapable"))
    check("resolve.explicit_empty", r._resolve_effort("claude-opus-4-8", "") == ("", "request"))


def main() -> int:
    tests = [
        test_resolve_effort_source,
        test_ultracode_advertised_for_opus,
        test_ultracode_not_advertised_for_nonopus,
        test_split_ultracode,
        test_argv_ultracode_uses_settings_not_effort,
        test_argv_level_uses_effort_flag,
        test_argv_empty_effort_passes_nothing,
        test_argv_noncapable_drops_effort,
    ]
    for t in tests:
        try:
            t()
        except AssertionError:
            pass
        except Exception as e:
            check(t.__name__, False, note=f"exception: {e!r}")
    print(f"\nRESULT pass={_PASS} fail={_FAIL}")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
