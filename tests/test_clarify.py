"""Unit tests for the interactive clarification protocol (argv + config).

The endpoint smoke tests stub the runner, so they never build real argv — these
exercise _build_argv directly to prove the --append-system-prompt /
--disallowedTools flags are emitted exactly when (and only when) intended.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Point the data dir at a tempdir so importing src.* never touches /data.
_TMP = tempfile.mkdtemp(prefix="claude-wrapper-clarify-")
os.environ["CLAUDE_WRAPPER_DATA"] = _TMP
os.environ["CLAUDE_WRAPPER_MODEL_DISCOVERY"] = "off"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.claude_runner import ClaudeRunner, SessionRegistry  # noqa: E402
from src.config import (  # noqa: E402
    DEFAULT_CLARIFY_SYSTEM_PROMPT,
    DEFAULT_WORKSPACE_SYSTEM_PROMPT,
    Settings,
)


def _runner(
    clarify_prompt: str = DEFAULT_CLARIFY_SYSTEM_PROMPT,
    tools=("AskUserQuestion",),
    workspace_prompt: str = DEFAULT_WORKSPACE_SYSTEM_PROMPT,
) -> ClaudeRunner:
    root = Path(_TMP)
    return ClaudeRunner(
        registry=SessionRegistry(root / "sessions"),
        workspace_root=root / "ws",
        claude_bin="claude",
        clarify_system_prompt=clarify_prompt,
        clarify_disallowed_tools=tuple(tools),
        workspace_system_prompt=workspace_prompt,
    )


def test_clarify_argv_adds_flags_when_requested() -> None:
    argv = _runner()._build_argv(session_uuid="u", model="claude-opus-4-8", resume=False, clarify=True)

    assert "--append-system-prompt" in argv
    i = argv.index("--append-system-prompt")
    assert argv[i + 1] == DEFAULT_CLARIFY_SYSTEM_PROMPT

    assert "--disallowedTools" in argv
    j = argv.index("--disallowedTools")
    assert argv[j + 1] == "AskUserQuestion"
    # The variadic --disallowedTools must be terminated by a flag, never a bare
    # positional that the CLI would swallow as another tool name.
    assert argv[j + 2].startswith("--")


def test_clarify_argv_absent_when_not_requested() -> None:
    argv = _runner()._build_argv(session_uuid="u", model="claude-opus-4-8", resume=False, clarify=False)
    assert "--append-system-prompt" not in argv
    assert "--disallowedTools" not in argv


def test_clarify_argv_noop_when_globally_disabled() -> None:
    # Empty configured prompt == CLAUDE_WRAPPER_CLARIFY=off: clarify=True is a no-op.
    argv = _runner(clarify_prompt="", tools=())._build_argv(
        session_uuid="u", model="claude-opus-4-8", resume=False, clarify=True
    )
    assert "--append-system-prompt" not in argv
    assert "--disallowedTools" not in argv


# ---------- workspace hint (shares the --append-system-prompt flag) ----------


def test_workspace_hint_argv_adds_the_prompt() -> None:
    argv = _runner()._build_argv(
        session_uuid="u", model="claude-opus-4-8", resume=False, workspace_hint=True
    )
    i = argv.index("--append-system-prompt")
    assert argv[i + 1] == DEFAULT_WORKSPACE_SYSTEM_PROMPT
    # No clarify => no tool ban, even though the two share the prompt flag.
    assert "--disallowedTools" not in argv


def test_workspace_hint_and_clarify_share_one_flag() -> None:
    """The CLI takes a single --append-system-prompt; repeating it would drop one."""
    argv = _runner()._build_argv(
        session_uuid="u", model="claude-opus-4-8", resume=False,
        clarify=True, workspace_hint=True,
    )
    assert argv.count("--append-system-prompt") == 1
    i = argv.index("--append-system-prompt")
    combined = argv[i + 1]
    assert DEFAULT_WORKSPACE_SYSTEM_PROMPT in combined
    assert DEFAULT_CLARIFY_SYSTEM_PROMPT in combined
    # --disallowedTools stays variadic-terminated by a following flag.
    j = argv.index("--disallowedTools")
    assert argv[j + 1] == "AskUserQuestion"
    assert argv[j + 2].startswith("--")


def test_workspace_hint_absent_when_not_requested() -> None:
    argv = _runner()._build_argv(session_uuid="u", model="claude-opus-4-8", resume=False)
    assert "--append-system-prompt" not in argv


def test_workspace_hint_noop_when_globally_disabled() -> None:
    # Empty configured prompt == CLAUDE_WRAPPER_WORKSPACE_HINT=off.
    argv = _runner(workspace_prompt="")._build_argv(
        session_uuid="u", model="claude-opus-4-8", resume=False, workspace_hint=True
    )
    assert "--append-system-prompt" not in argv


def test_workspace_prompt_carries_no_json_schema_marker() -> None:
    """The hint travels as a CLI argument and is never concatenated into the
    prompt, so nothing reads it as a structured-output declaration today. Keep
    it free of "json schema" markers anyway, so it stays inert if a future
    refactor ever does concatenate it into the prompt."""
    import re

    assert not re.search(r"json[\s_-]?schema", DEFAULT_WORKSPACE_SYSTEM_PROMPT, re.I)


def test_workspace_config_parsing() -> None:
    prev = os.environ.get("CLAUDE_WRAPPER_WORKSPACE_HINT")
    try:
        os.environ["CLAUDE_WRAPPER_WORKSPACE_HINT"] = "off"
        assert Settings.from_env().workspace_hint_enabled is False
        os.environ["CLAUDE_WRAPPER_WORKSPACE_HINT"] = "on"
        s = Settings.from_env()
        assert s.workspace_hint_enabled is True
        assert s.workspace_system_prompt == DEFAULT_WORKSPACE_SYSTEM_PROMPT
    finally:
        if prev is None:
            os.environ.pop("CLAUDE_WRAPPER_WORKSPACE_HINT", None)
        else:
            os.environ["CLAUDE_WRAPPER_WORKSPACE_HINT"] = prev


def test_clarify_config_parsing() -> None:
    prev = os.environ.get("CLAUDE_WRAPPER_CLARIFY")
    try:
        os.environ["CLAUDE_WRAPPER_CLARIFY"] = "off"
        assert Settings.from_env().clarify_enabled is False

        os.environ["CLAUDE_WRAPPER_CLARIFY"] = "true"
        s = Settings.from_env()
        assert s.clarify_enabled is True
        assert s.clarify_disallowed_tools == ("AskUserQuestion",)
        assert "Clarification protocol" in s.clarify_system_prompt
    finally:
        if prev is None:
            os.environ.pop("CLAUDE_WRAPPER_CLARIFY", None)
        else:
            os.environ["CLAUDE_WRAPPER_CLARIFY"] = prev
