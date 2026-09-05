"""CodexRunner: argv shape, prompt prep, JSONL normalization, resume + self-heal.

Named test_codex_runner so it collects AFTER test_budget.py (see the
test_sandbox_shim docstring for the module-ordering constraint).

The suite's process-wide SETTINGS is frozen in claude mode; every test here
passes anyway because effort acceptance goes through
CodexRunner._effort_choices_for, which never consults SETTINGS.agent — do NOT
add a config reload to this module.

The fake_codex scripts pin the 0.153.x `codex exec --json` JSONL schema the
normalizer encodes (the ADJUSTMENT POINT in src/codex_runner.py) — if a codex
upgrade changes the schema, update both together.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Point the data dir at a tempdir so importing src.* never touches /data, and
# use the static model list so import doesn't scan the CLI binary.
_TMP = tempfile.mkdtemp(prefix="claude-wrapper-codex-runner-test-")
os.environ.setdefault("CLAUDE_WRAPPER_DATA", _TMP)
os.environ.setdefault("CLAUDE_WRAPPER_MODEL_DISCOVERY", "off")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib  # noqa: E402

from src import agent_runner, codex_runner  # noqa: E402

# Sibling test modules (notably test_endpoints) monkeypatch runner methods with
# stubs at import time; reload so the round trips below exercise the genuine
# run_stream. codex_runner reloads second so it re-imports the fresh
# agent_runner classes.
importlib.reload(agent_runner)
importlib.reload(codex_runner)
CodexRunner = codex_runner.CodexRunner
SessionRegistry = agent_runner.SessionRegistry

THREAD_ID = "01a06fb8-718c-7c53-b1d7-489513ad39bf"

# The JSONL a healthy `codex exec --json` run emits (0.153.x schema).
_HAPPY_JSONL = (
    '{"type":"thread.started","thread_id":"%s"}' % THREAD_ID,
    '{"type":"turn.started"}',
    '{"type":"item.started","item":{"id":"item_0","type":"command_execution",'
    '"command":"echo hi","aggregated_output":"","exit_code":null,"status":"in_progress"}}',
    '{"type":"item.completed","item":{"id":"item_0","type":"command_execution",'
    '"command":"echo hi","aggregated_output":"hi\\n","exit_code":0,"status":"completed"}}',
    '{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"done"}}',
    '{"type":"turn.completed","usage":{"input_tokens":42,"cached_input_tokens":10,'
    '"cache_write_input_tokens":0,"output_tokens":7,"reasoning_output_tokens":2}}',
)


def _fake_codex(tmp_path: Path, lines=(), returncode: int = 0) -> tuple[Path, Path]:
    """A stand-in binary: logs its argv, consumes stdin, emits JSONL lines.

    printf leaves %s arguments untouched, so the escaped "\\n" inside the JSON
    reaches stdout as two characters — exactly what json.loads expects.
    """
    script = tmp_path / "fake-codex"
    argv_log = tmp_path / "argv.log"
    body = ["#!/bin/sh", f'echo "$@" >> "{argv_log}"', "cat > /dev/null"]
    body += [f"printf '%s\\n' '{line}'" for line in lines]
    body.append(f"exit {returncode}")
    script.write_text("\n".join(body) + "\n")
    script.chmod(0o755)
    return script, argv_log


def _runner(tag: str, bin_path: str = "codex", **kwargs) -> CodexRunner:
    return CodexRunner(
        registry=SessionRegistry(Path(_TMP) / f"sessions-{tag}", agent="codex"),
        workspace_root=Path(_TMP) / f"workspace-{tag}",
        agent_bin=bin_path,
        **kwargs,
    )


# ---------- argv construction (no subprocess) ----------


def test_fresh_argv_exact_order():
    argv = _runner("argv")._build_argv(
        session_uuid="u-1", model="gpt-5.2-codex", resume=False, effort="high"
    )
    assert argv == [
        "codex",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "-c",
        "check_for_update_on_startup=false",
        "--model",
        "gpt-5.2-codex",
        "-c",
        'model_reasoning_effort="high"',
        "-",
    ]


def test_resume_argv_inserts_resume_right_after_exec():
    argv = _runner("argv-resume")._build_argv(
        session_uuid=THREAD_ID, model="gpt-5.2-codex", resume=True, effort="high"
    )
    assert argv[:4] == ["codex", "exec", "resume", THREAD_ID]
    assert argv[4] == "--json"
    assert argv[-1] == "-"


def test_effort_none_reaches_the_cli():
    """An explicit "none" must not silently fall back to codex's default medium."""
    argv = _runner("argv-none")._build_argv(
        session_uuid="u", model="gpt-5.2", resume=False, effort="none"
    )
    assert 'model_reasoning_effort="none"' in argv
    assert argv[-1] == "-"


def test_claude_only_efforts_are_dropped():
    for eff in ("max", "ultracode"):
        argv = _runner("argv-drop")._build_argv(
            session_uuid="u", model="gpt-5.2", resume=False, effort=eff
        )
        assert not any("model_reasoning_effort" in a for a in argv), eff


def test_capability_gating_adds_nothing():
    runner = _runner("argv-gate")
    gated = runner._build_argv(
        session_uuid="u", model="gpt-5.2", resume=False, capability_gated=True
    )
    ungated = runner._build_argv(
        session_uuid="u", model="gpt-5.2", resume=False, capability_gated=False
    )
    assert gated == ungated


# ---------- prompt preparation ----------


def test_prepare_prompt_prepends_protocol_segments():
    runner = _runner(
        "prompt", clarify_system_prompt="CLARIFY", workspace_system_prompt="WORKSPACE"
    )
    assert (
        runner._prepare_prompt("hi", clarify=True, workspace_hint=True)
        == "WORKSPACE\n\nCLARIFY\n\nhi"
    )
    assert runner._prepare_prompt("hi", clarify=True, workspace_hint=False) == "CLARIFY\n\nhi"
    assert runner._prepare_prompt("hi", clarify=False, workspace_hint=True) == "WORKSPACE\n\nhi"
    assert runner._prepare_prompt("hi", clarify=False, workspace_hint=False) == "hi"


def test_prepare_prompt_passthrough_when_disabled():
    # Empty prompts ⇒ globally disabled, even when the caller flags are set.
    runner = _runner("prompt-off")
    assert runner._prepare_prompt("hi", clarify=True, workspace_hint=True) == "hi"


# ---------- round trip + resume wiring (real subprocess via fake binary) ----------


def test_run_collect_round_trip_and_resume(tmp_path):
    script, argv_log = _fake_codex(tmp_path, _HAPPY_JSONL)
    runner = _runner("happy", bin_path=str(script))
    key = "conv-happy"

    async def _go():
        first = await runner.run_collect(prompt="hi", session_key=key, model="gpt-5.2-codex")
        second = await runner.run_collect(prompt="again", session_key=key, model="gpt-5.2-codex")
        return first, second

    first, second = asyncio.run(_go())

    assert first.error is None
    assert first.final_text == "done"
    kinds = [e.kind for e in first.events]
    use = first.events[kinds.index("tool_use")]
    assert use.tool_name == "command_execution"
    assert use.tool_input == {"command": "echo hi"}
    assert kinds.index("tool_use") < kinds.index("tool_result")
    assert first.events[kinds.index("tool_result")].tool_output == "hi\n"
    # cached_input_tokens is a subset of input_tokens — never added on top.
    assert first.input_tokens == 42
    assert first.output_tokens == 7
    assert first.total_cost_usd == 0.0
    assert first.session_uuid == THREAD_ID

    # bind_uuid fired: the registry now holds codex's thread id, agent-tagged.
    entry = json.loads((runner.registry.root / f"{key}.json").read_text())
    assert entry["uuid"] == THREAD_ID
    assert entry["agent"] == "codex"

    # Second turn resumed the codex-assigned thread, not the wrapper placeholder.
    lines = argv_log.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("exec --json")
    assert lines[1].startswith(f"exec resume {THREAD_ID} --json")
    assert second.session_uuid == THREAD_ID


# ---------- self-heal ----------


def test_failed_resume_drops_mapping_even_after_rebind(tmp_path):
    """Pins the capture-before-self-heal ordering: the fresh thread id announced
    mid-run is bound first, and forget() still wins — a bind_uuid landing after
    forget would resurrect the mapping and brick the conversation."""
    other_id = "01a06fb8-0000-7c53-b1d7-000000000000"
    script, _ = _fake_codex(
        tmp_path,
        (
            '{"type":"thread.started","thread_id":"%s"}' % other_id,
            '{"type":"turn.failed","error":{"message":"thread not found"}}',
        ),
    )
    runner = _runner("heal-resume", bin_path=str(script))
    key = "conv-heal"

    async def _go():
        await runner.registry.get_or_create_uuid(key)  # mapping exists -> resume turn
        return await runner.run_collect(prompt="hi", session_key=key, model="gpt-5.2")

    result = asyncio.run(_go())
    assert result.error == "thread not found"
    assert any(e.kind == "error" for e in result.events)
    assert not runner.registry.has(key)


def test_dead_first_turn_forgets_the_placeholder(tmp_path):
    """A first turn that dies before thread.started leaves a wrapper-minted
    placeholder that can never be resumed — the placeholder_unbound arm drops
    it now, not one failed turn later."""
    script, _ = _fake_codex(tmp_path, (), returncode=1)
    runner = _runner("heal-fresh", bin_path=str(script))
    key = "conv-fresh"
    result = asyncio.run(runner.run_collect(prompt="hi", session_key=key, model="gpt-5.2"))
    assert result.error is not None
    assert result.error.startswith("codex exited 1")
    assert not runner.registry.has(key)


# ---------- tolerance ----------


def test_unknown_events_survive_as_system(tmp_path):
    script, _ = _fake_codex(
        tmp_path,
        (
            '{"type":"thread.started","thread_id":"%s"}' % THREAD_ID,
            '{"type":"totally.new.event","x":1}',
            '{"type":"item.completed","item":{"id":"item_9","type":"todo_list","items":[]}}',
            '{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"ok"}}',
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}',
        ),
    )
    runner = _runner("tolerant", bin_path=str(script))
    result = asyncio.run(
        runner.run_collect(prompt="hi", session_key="conv-tolerant", model="gpt-5.2")
    )
    assert result.error is None
    assert result.final_text == "ok"
    system_types = [e.raw.get("type") for e in result.events if e.kind == "system" and e.raw]
    assert "totally.new.event" in system_types
    assert "item.completed" in system_types  # the unknown todo_list item, whole-event raw


def test_transient_error_does_not_fail_the_turn(tmp_path):
    script, _ = _fake_codex(
        tmp_path,
        (
            '{"type":"thread.started","thread_id":"%s"}' % THREAD_ID,
            '{"type":"error","message":"stream disconnected; retrying"}',
            '{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"recovered"}}',
            '{"type":"turn.completed","usage":{"input_tokens":3,"output_tokens":2}}',
        ),
    )
    runner = _runner("transient", bin_path=str(script))
    result = asyncio.run(
        runner.run_collect(prompt="hi", session_key="conv-transient", model="gpt-5.2")
    )
    assert result.error is None
    assert not any(e.kind == "error" for e in result.events)
    assert result.final_text == "recovered"


# ---------- client cancel mid-first-turn ----------


def test_cancel_mid_first_turn_persists_the_announced_thread_id(tmp_path):
    # A client disconnect unwinds run_stream at a yield, skipping the
    # post-loop bind. Codex was already told nothing about the wrapper's
    # placeholder (fresh runs pass no session flag), so losing the announced
    # thread id here used to guarantee a dead resume — and a 502 — on the
    # next message. The unwind path must persist what the stream announced.
    script, _ = _fake_codex(tmp_path, _HAPPY_JSONL)
    runner = _runner("cancel-bind", bin_path=str(script))
    key = "conv-cancel"

    async def _go():
        gen = runner.run_stream(prompt="hi", session_key=key, model="gpt-5.2")
        async for _ in gen:
            break  # client gone after the first frame
        await gen.aclose()

    asyncio.run(_go())
    entry = json.loads((Path(_TMP) / "sessions-cancel-bind" / f"{key}.json").read_text())
    assert entry == {"key": key, "uuid": THREAD_ID, "agent": "codex"}


def test_cancel_before_thread_started_drops_the_placeholder(tmp_path):
    # Unwound before codex announced any id: the placeholder can never be
    # resumed, so it must be forgotten now — the next turn then mints fresh
    # and replays in full instead of burning a turn on a dead resume.
    script, _ = _fake_codex(
        tmp_path,
        lines=(
            '{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"hi"}}',
        ),
    )
    runner = _runner("cancel-drop", bin_path=str(script))
    key = "conv-cancel-drop"

    async def _go():
        gen = runner.run_stream(prompt="hi", session_key=key, model="gpt-5.2")
        async for _ in gen:
            break
        await gen.aclose()

    asyncio.run(_go())
    assert not (Path(_TMP) / "sessions-cancel-drop" / f"{key}.json").exists()


# ---------- registry agent isolation ----------


def test_registry_entries_are_agent_scoped():
    root = Path(_TMP) / "sessions-isolation"

    async def _go():
        claude_reg = SessionRegistry(root, agent="claude")
        codex_reg = SessionRegistry(root, agent="codex")

        # A claude-minted entry is invisible under codex: fresh mint, not resume.
        await claude_reg.get_or_create_uuid("k1")
        _, created = await codex_reg.get_or_create_uuid("k1")
        assert created
        # ...and the codex-tagged rewrite is now invisible under claude.
        _, created = await claude_reg.get_or_create_uuid("k1")
        assert created

        # Legacy untagged entries read as claude: resumed there, fresh under codex.
        (root / "k2.json").write_text(json.dumps({"key": "k2", "uuid": "legacy-uuid"}))
        u, created = await claude_reg.get_or_create_uuid("k2")
        assert u == "legacy-uuid" and not created
        (root / "k3.json").write_text(json.dumps({"key": "k3", "uuid": "legacy-uuid"}))
        _, created = await codex_reg.get_or_create_uuid("k3")
        assert created

        # has() must apply the same agent predicate as get_or_create_uuid:
        # prepare_messages keys replay-only mode off it, so a mismatch that
        # reads as "present" would pair a fresh thread with a trailing-message
        # prompt and silently drop the conversation history on a stack switch.
        (root / "k4.json").write_text(
            json.dumps({"key": "k4", "uuid": "u4", "agent": "claude"})
        )
        assert claude_reg.has("k4")
        assert not codex_reg.has("k4")
        # Legacy untagged reads as claude; corrupt reads as absent for both.
        (root / "k5.json").write_text(json.dumps({"key": "k5", "uuid": "u5"}))
        assert claude_reg.has("k5")
        assert not codex_reg.has("k5")
        (root / "k6.json").write_text("not json")
        assert not claude_reg.has("k6")
        assert not codex_reg.has("k6")

    asyncio.run(_go())
