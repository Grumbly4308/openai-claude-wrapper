"""Unit tests for the wrapper-owned tools (src/wrapper_tools.py).

Covers the safe calculator, timezone handling, and the file-path memory
store: command coverage, per-conversation isolation, and path-escape
hardening.

Runs under pytest or standalone: python3 tests/test_wrapper_tools.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="claude-wrapper-wtools-test-")
os.environ["CLAUDE_WRAPPER_DATA"] = _TMP
os.environ["CLAUDE_WRAPPER_MODEL_DISCOVERY"] = "off"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capabilities import Capability  # noqa: E402
from src.wrapper_tools import execute, tool_definitions, wrapper_tool_names  # noqa: E402


def _mem(command: str, session: str = "sess-a", **kw):
    return execute("memory", {"command": command, **kw}, session)


def test_definitions_follow_capabilities():
    none = tool_definitions(frozenset())
    assert none == []
    mem = tool_definitions(frozenset({Capability.MEMORY}))
    assert [t.get("type") for t in mem] == ["memory_20250818"]
    both = tool_definitions(frozenset({Capability.MEMORY, Capability.TIME_CALC}))
    assert [t.get("name") for t in both] == ["memory", "get_current_time", "calculate"]
    assert wrapper_tool_names(frozenset({Capability.TIME_CALC})) == {
        "get_current_time",
        "calculate",
    }


def test_calculate():
    out, err = execute("calculate", {"expression": "(17.5 * 12) / 3"}, "s")
    assert not err and out.endswith("= 70")
    out, err = execute("calculate", {"expression": "2 ** 10 % 7"}, "s")
    assert not err and out.endswith("= 2")


def test_calculate_rejects_non_arithmetic():
    for expr in ("__import__('os')", "open('/etc/passwd')", "1; 2", "'a' * 3"):
        _out, err = execute("calculate", {"expression": expr}, "s")
        assert err, f"{expr!r} should be rejected"
    _out, err = execute("calculate", {"expression": "1/0"}, "s")
    assert err
    _out, err = execute("calculate", {"expression": "2 ** 99999"}, "s")
    assert err


def test_get_current_time():
    out, err = execute("get_current_time", {}, "s")
    assert not err and "+00:00" in out
    out, err = execute("get_current_time", {"timezone": "Europe/Berlin"}, "s")
    assert not err
    _out, err = execute("get_current_time", {"timezone": "Mars/Olympus"}, "s")
    assert err


def test_memory_create_view_replace_insert():
    out, err = _mem("create", path="/memories/notes.md", file_text="alpha\nbeta")
    assert not err
    out, err = _mem("view", path="/memories/notes.md")
    assert not err and out == "1\talpha\n2\tbeta"
    out, err = _mem("str_replace", path="/memories/notes.md", old_str="beta", new_str="gamma")
    assert not err
    out, err = _mem("insert", path="/memories/notes.md", insert_line=0, insert_text="top")
    assert not err
    out, err = _mem("view", path="/memories/notes.md")
    assert not err and out == "1\ttop\n2\talpha\n3\tgamma"


def test_memory_listing_rename_delete():
    _mem("create", path="/memories/a/one.md", file_text="1")
    out, err = _mem("view", path="/memories/a")
    assert not err and "- one.md" in out
    out, err = _mem("rename", old_path="/memories/a/one.md", new_path="/memories/a/two.md")
    assert not err
    out, err = _mem("delete", path="/memories/a/two.md")
    assert not err
    _out, err = _mem("view", path="/memories/a/two.md")
    assert err


def test_memory_is_per_conversation():
    _mem("create", session="conv-1", path="/memories/x.md", file_text="one")
    _out, err = _mem("view", session="conv-2", path="/memories/x.md")
    assert err, "conversation B must not see conversation A's memory"
    out, err = _mem("view", session="conv-1", path="/memories/x.md")
    assert not err and "one" in out


def test_memory_path_escapes_rejected():
    for path in ("/etc/passwd", "../x", "/memories/../../x", "/memories/a/../../x"):
        _out, err = _mem("view", path=path)
        assert err, f"{path!r} must be rejected"
    _out, err = _mem("delete", path="/memories")
    assert err, "deleting the root must be refused"


def test_memory_str_replace_requires_unique_match():
    _mem("create", path="/memories/dup.md", file_text="x\nx")
    _out, err = _mem("str_replace", path="/memories/dup.md", old_str="x", new_str="y")
    assert err


def test_unknown_tool_and_command_error():
    _out, err = execute("warp_drive", {}, "s")
    assert err
    _out, err = _mem("compress")
    assert err


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
