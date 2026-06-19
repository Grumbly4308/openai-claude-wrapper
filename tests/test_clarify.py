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
from src.config import DEFAULT_CLARIFY_SYSTEM_PROMPT, Settings  # noqa: E402


def _runner(clarify_prompt: str = DEFAULT_CLARIFY_SYSTEM_PROMPT, tools=("AskUserQuestion",)) -> ClaudeRunner:
    root = Path(_TMP)
    return ClaudeRunner(
        registry=SessionRegistry(root / "sessions"),
        workspace_root=root / "ws",
        claude_bin="claude",
        clarify_system_prompt=clarify_prompt,
        clarify_disallowed_tools=tuple(tools),
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
