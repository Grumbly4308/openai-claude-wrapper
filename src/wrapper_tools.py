"""Wrapper-owned tools for the tool bridge's hybrid loop.

Client-declared tools are executed by the caller (OpenWebUI). The tools here
are executed by the WRAPPER itself, inline in the bridge loop, and the caller
never sees them — see tool_bridge's loop for the turn-taking rules. Two
capability-gated groups:

- ``memory`` (Capability.MEMORY) — the Anthropic-defined memory_20250818
  tool. The model reads/writes markdown files under a virtual ``/memories``
  directory; storage is a real per-conversation directory under the data dir
  (``<data>/memory/<session_key>/``), the same file-path model Claude Code
  uses for its own memory. Path handling is hardened: every model-supplied
  path is resolved and must stay inside the conversation's root.

- ``get_current_time`` / ``calculate`` (Capability.TIME_CALC) — cheap custom
  tools proving the loop; time via zoneinfo, arithmetic via an AST whitelist
  (never eval on raw model input).
"""

from __future__ import annotations

import ast
import logging
import re
import shutil
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .capabilities import Capability
from .config import SETTINGS

log = logging.getLogger("claude_wrapper.wrapper_tools")

_MEMORY_TOOL = {"type": "memory_20250818", "name": "memory"}

_TIME_TOOL = {
    "name": "get_current_time",
    "description": (
        "Get the current date and time. Call this whenever the answer depends "
        "on today's date or the current time."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone name, e.g. Europe/Berlin. Defaults to UTC.",
            }
        },
    },
}

_CALC_TOOL = {
    "name": "calculate",
    "description": (
        "Evaluate an arithmetic expression exactly. Supports + - * / // % ** "
        "and parentheses on numbers. Call this instead of doing arithmetic "
        "yourself when exactness matters."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "e.g. (17.5 * 12) / 3"}
        },
        "required": ["expression"],
    },
}

# Tool name -> gating capability. Client-declared tools may not shadow these
# names when the capability is active (the bridge rejects the collision).
TOOL_CAPABILITIES = {
    "memory": Capability.MEMORY,
    "get_current_time": Capability.TIME_CALC,
    "calculate": Capability.TIME_CALC,
}


def tool_definitions(caps: frozenset[Capability]) -> list[dict[str, Any]]:
    """Anthropic tool definitions for the wrapper-owned tools a profile enables."""
    out: list[dict[str, Any]] = []
    if Capability.MEMORY in caps:
        out.append(_MEMORY_TOOL)
    if Capability.TIME_CALC in caps:
        out.extend((_TIME_TOOL, _CALC_TOOL))
    return out


def wrapper_tool_names(caps: frozenset[Capability]) -> frozenset[str]:
    return frozenset(n for n, cap in TOOL_CAPABILITIES.items() if cap in caps)


def execute(name: str, tool_input: dict[str, Any], session_key: str) -> tuple[str, bool]:
    """Run a wrapper-owned tool. Returns (result text, is_error)."""
    try:
        if name == "memory":
            return _memory(tool_input, session_key), False
        if name == "get_current_time":
            return _current_time(tool_input), False
        if name == "calculate":
            return _calculate(tool_input), False
    except _ToolError as e:
        return str(e), True
    except Exception as e:  # defensive: a tool bug must not kill the turn
        log.exception("wrapper tool %s failed (session=%s)", name, session_key)
        return f"tool failed: {e}", True
    return f"unknown wrapper tool: {name}", True


class _ToolError(Exception):
    """Expected tool failure, reported to the model as is_error=True."""


# ---------- time & calculation ----------


def _current_time(tool_input: dict[str, Any]) -> str:
    from datetime import datetime, timezone

    tz_name = str(tool_input.get("timezone") or "").strip()
    if tz_name:
        try:
            tz = ZoneInfo(tz_name)
        except (KeyError, ValueError) as e:
            raise _ToolError(f"unknown timezone {tz_name!r}") from e
    else:
        tz = timezone.utc
    now = datetime.now(tz)
    return f"{now.isoformat(timespec='seconds')} ({now:%A})"


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)


def _calculate(tool_input: dict[str, Any]) -> str:
    expr = str(tool_input.get("expression") or "").strip()
    if not expr:
        raise _ToolError("expression is required")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = _eval(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise _ToolError("exponent too large")
            op = type(node.op)
            try:
                if op is ast.Add:
                    return left + right
                if op is ast.Sub:
                    return left - right
                if op is ast.Mult:
                    return left * right
                if op is ast.Div:
                    return left / right
                if op is ast.FloorDiv:
                    return left // right
                if op is ast.Mod:
                    return left % right
                return left**right
            except ZeroDivisionError as e:
                raise _ToolError("division by zero") from e
        raise _ToolError("only arithmetic on numbers is supported")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise _ToolError(f"invalid expression: {e.msg}") from e
    result = _eval(tree)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expr} = {result}"


# ---------- memory (file-path model, per conversation) ----------

_MEMORY_PREFIX = "/memories"
# Session keys are caller-derived; flatten anything path-hostile.
_SAFE_SESSION = re.compile(r"[^A-Za-z0-9._-]+")


def _memory_root(session_key: str) -> Path:
    safe = _SAFE_SESSION.sub("_", session_key or "anonymous") or "anonymous"
    root = SETTINGS.data_dir / "memory" / safe
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve(root: Path, raw_path: str) -> Path:
    p = str(raw_path or "").strip()
    if p in (_MEMORY_PREFIX, _MEMORY_PREFIX + "/"):
        return root
    if not p.startswith(_MEMORY_PREFIX + "/"):
        raise _ToolError(f"path must start with {_MEMORY_PREFIX}/ (got {p!r})")
    candidate = (root / p[len(_MEMORY_PREFIX) + 1 :]).resolve()
    if candidate != root.resolve() and root.resolve() not in candidate.parents:
        raise _ToolError(f"path escapes the memory directory: {p!r}")
    return candidate


def _memory(tool_input: dict[str, Any], session_key: str) -> str:
    root = _memory_root(session_key)
    command = str(tool_input.get("command") or "")
    if command == "view":
        return _mem_view(root, tool_input)
    if command == "create":
        target = _resolve(root, tool_input.get("path"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(tool_input.get("file_text") or ""), encoding="utf-8")
        return f"created {tool_input.get('path')}"
    if command == "str_replace":
        return _mem_str_replace(root, tool_input)
    if command == "insert":
        return _mem_insert(root, tool_input)
    if command == "delete":
        target = _resolve(root, tool_input.get("path"))
        if target == root:
            raise _ToolError("refusing to delete the memory root")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        else:
            raise _ToolError(f"not found: {tool_input.get('path')}")
        return f"deleted {tool_input.get('path')}"
    if command == "rename":
        src = _resolve(root, tool_input.get("old_path"))
        dst = _resolve(root, tool_input.get("new_path"))
        if not src.exists():
            raise _ToolError(f"not found: {tool_input.get('old_path')}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return f"renamed {tool_input.get('old_path')} -> {tool_input.get('new_path')}"
    raise _ToolError(f"unknown memory command: {command!r}")


def _mem_view(root: Path, tool_input: dict[str, Any]) -> str:
    target = _resolve(root, tool_input.get("path") or _MEMORY_PREFIX)
    if target.is_dir():
        entries = sorted(target.iterdir())
        listing = "\n".join(
            f"- {e.name}{'/' if e.is_dir() else ''}" for e in entries
        )
        return f"Directory: {tool_input.get('path') or _MEMORY_PREFIX}\n{listing or '(empty)'}"
    if not target.exists():
        raise _ToolError(f"not found: {tool_input.get('path')}")
    lines = target.read_text(encoding="utf-8").splitlines()
    view_range = tool_input.get("view_range")
    start, end = 1, len(lines)
    if isinstance(view_range, list) and len(view_range) == 2:
        start = max(1, int(view_range[0]))
        end = len(lines) if int(view_range[1]) == -1 else min(len(lines), int(view_range[1]))
    numbered = "\n".join(f"{i}\t{lines[i - 1]}" for i in range(start, end + 1))
    return numbered or "(empty file)"


def _mem_str_replace(root: Path, tool_input: dict[str, Any]) -> str:
    target = _resolve(root, tool_input.get("path"))
    if not target.is_file():
        raise _ToolError(f"not found: {tool_input.get('path')}")
    old = str(tool_input.get("old_str") or "")
    if not old:
        raise _ToolError("old_str is required")
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise _ToolError(f"old_str must occur exactly once (found {count})")
    target.write_text(text.replace(old, str(tool_input.get("new_str") or "")), encoding="utf-8")
    return f"replaced in {tool_input.get('path')}"


def _mem_insert(root: Path, tool_input: dict[str, Any]) -> str:
    target = _resolve(root, tool_input.get("path"))
    if not target.is_file():
        raise _ToolError(f"not found: {tool_input.get('path')}")
    try:
        line_no = int(tool_input.get("insert_line"))
    except (TypeError, ValueError) as e:
        raise _ToolError("insert_line must be an integer") from e
    lines = target.read_text(encoding="utf-8").splitlines()
    if not 0 <= line_no <= len(lines):
        raise _ToolError(f"insert_line out of range 0..{len(lines)}")
    lines.insert(line_no, str(tool_input.get("insert_text") or "").rstrip("\n"))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f"inserted into {tool_input.get('path')}"
