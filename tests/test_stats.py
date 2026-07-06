"""Tests for the instant `stats` / `context` chat commands and /v1/usage.

Covers three layers:
  - command detection (whole-message match only, optional / prefix),
  - the interception in run_chat_completion, which must answer from the ledger
    *without* invoking Claude — including while the conversation is paused at a
    budget checkpoint,
  - the /v1/usage/{session_id} endpoint's JSON shape.

The runner is stubbed, so no real Claude Code subprocess is launched. Env is set
before importing src.* (same values as test_budget) so the module-level
SETTINGS / USAGE_LEDGER come up with a small block (allowance 2000 x 5% = 100
tokens) that the stub's 120-token replies immediately exceed.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Build SETTINGS/USAGE_LEDGER against a tempdir with the cap enabled.
_TMP = tempfile.mkdtemp(prefix="claude-wrapper-stats-test-")
os.environ["CLAUDE_WRAPPER_DATA"] = _TMP
os.environ["CLAUDE_WRAPPER_DEFAULT_MODEL"] = "claude-opus-4-8"
os.environ["CLAUDE_WRAPPER_MODEL_DISCOVERY"] = "off"
os.environ["CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE"] = "2000"
os.environ["CLAUDE_WRAPPER_SESSION_BLOCK_PERCENT"] = "5"
os.environ.pop("CLAUDE_WRAPPER_API_KEYS", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.claude_runner import ClaudeResult  # noqa: E402

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


# ---------- command detection ----------


def test_usage_command_detection() -> None:
    from src.main import _usage_command
    from src.models import ChatMessage

    def msgs(text: str):
        return [ChatMessage(role="user", content=text)]

    check("cmd.stats", _usage_command(msgs("stats")) == "stats")
    check("cmd.context", _usage_command(msgs("context")) == "context")
    check("cmd.slash", _usage_command(msgs("/stats")) == "stats")
    check("cmd.case_punct", _usage_command(msgs(" Context! ")) == "context")
    check("cmd.mid_sentence", _usage_command(msgs("show me stats please")) is None)
    check("cmd.substring", _usage_command(msgs("statistics")) is None)
    check("cmd.empty", _usage_command([]) is None)
    # Uses the last user message, not an earlier one.
    check(
        "cmd.last_only",
        _usage_command(
            [
                ChatMessage(role="user", content="stats"),
                ChatMessage(role="assistant", content="report"),
                ChatMessage(role="user", content="now do something else"),
            ]
        )
        is None,
    )


# ---------- report rendering ----------


def test_usage_report_disabled_ledger() -> None:
    import src.main as main_module
    from src.usage import UsageLedger, UsageState

    prev = main_module.USAGE_LEDGER
    main_module.USAGE_LEDGER = UsageLedger(Path(_TMP) / "led-off", block_tokens=0)
    try:
        text = main_module._usage_report("sid", UsageState())
        check("report.disabled", "disabled" in text, note=text)
    finally:
        main_module.USAGE_LEDGER = prev


# ---------- interception gate ----------

_CALLS = {"n": 0}


async def _stub_run_collect(prompt, session_key, model=None, env_extra=None, extra_args=None, effort=None, **_kwargs):
    _CALLS["n"] += 1
    return ClaudeResult(
        session_uuid="stub-uuid",
        final_text="ok",
        stop_reason="stop",
        input_tokens=60,
        output_tokens=60,
        events=[],
    )


def _content(resp) -> str:
    data = json.loads(resp.body.decode() if isinstance(resp.body, (bytes, bytearray)) else resp.body)
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def test_stats_answers_without_claude() -> None:
    from src.deps import RUNNER, USAGE_LEDGER
    from src.main import run_chat_completion
    from src.models import ChatCompletionRequest, ChatMessage

    check("stats.ledger_enabled", USAGE_LEDGER.enabled)
    had_own = "run_collect" in RUNNER.__dict__
    prev = RUNNER.__dict__.get("run_collect")
    RUNNER.run_collect = _stub_run_collect  # instance attr → called without self
    _CALLS["n"] = 0
    sid = "stats-int-1"

    def req(messages):
        return ChatCompletionRequest(model="claude-opus-4-8", session_id=sid, messages=messages)

    async def run():
        # 1) fresh conversation: `stats` reports zeros without running Claude.
        r1 = await run_chat_completion(req([ChatMessage(role="user", content="stats")]))
        n1 = _CALLS["n"]
        # 2) a real turn runs and records 120 tokens (block is 100 → now over).
        r2 = await run_chat_completion(req([ChatMessage(role="user", content="hello")]))
        n2 = _CALLS["n"]
        # 3) `stats` while over budget → still the report, not the checkpoint,
        #    and still no Claude call.
        r3 = await run_chat_completion(
            req(
                [
                    ChatMessage(role="user", content="hello"),
                    ChatMessage(role="assistant", content="ok"),
                    ChatMessage(role="user", content="stats"),
                ]
            )
        )
        n3 = _CALLS["n"]
        # 4) an ordinary prompt still hits the checkpoint afterwards.
        r4 = await run_chat_completion(
            req(
                [
                    ChatMessage(role="user", content="hello"),
                    ChatMessage(role="assistant", content="ok"),
                    ChatMessage(role="user", content="tell me more"),
                ]
            )
        )
        n4 = _CALLS["n"]
        # 5) `/context` is an alias for the same report.
        r5 = await run_chat_completion(req([ChatMessage(role="user", content="/context")]))
        n5 = _CALLS["n"]
        return (r1, n1), (r2, n2), (r3, n3), (r4, n4), (r5, n5)

    try:
        (r1, n1), (r2, n2), (r3, n3), (r4, n4), (r5, n5) = asyncio.run(run())
    finally:
        if had_own:
            RUNNER.run_collect = prev
        else:
            del RUNNER.run_collect

    check("stats.fresh_no_claude", n1 == 0, note=f"n1={n1}")
    check("stats.fresh_content", "Usage stats" in _content(r1) and "0 tokens" in _content(r1), note=_content(r1))
    check("stats.turn_ran", n2 == 1 and _content(r2) == "ok", note=f"n2={n2}")
    check("stats.over_budget_no_claude", n3 == 1, note=f"n3={n3}")
    check(
        "stats.over_budget_content",
        "Usage stats" in _content(r3) and "120 tokens across 1 request" in _content(r3),
        note=_content(r3),
    )
    check("stats.checkpoint_still_gates", n4 == 1 and "Usage checkpoint" in _content(r4), note=_content(r4))
    check("stats.context_alias", n5 == 1 and "Usage stats" in _content(r5), note=_content(r5))


# ---------- REST endpoint ----------


def test_usage_endpoint_shape() -> None:
    from src.main import session_usage

    async def run():
        return await session_usage("stats-int-1")

    data = asyncio.run(run())
    check("endpoint.object", data["object"] == "usage.session")
    check("endpoint.tracking", data["tracking_enabled"] is True)
    check("endpoint.spent", data["spent_tokens"] == 120, note=str(data))
    check("endpoint.remaining", data["remaining_tokens"] == 0, note=str(data))
    check("endpoint.over", data["over_budget"] is True)
    check("endpoint.block", data["block_tokens"] == 100 and data["allowance_tokens"] == 100)
    # Unknown session → clean zeros, not an error.
    unknown = asyncio.run(session_usage("never-seen"))
    check("endpoint.unknown_zeros", unknown["spent_tokens"] == 0 and not unknown["over_budget"])


def main() -> int:
    tests = [
        test_usage_command_detection,
        test_usage_report_disabled_ledger,
        test_stats_answers_without_claude,
        test_usage_endpoint_shape,
    ]
    for t in tests:
        try:
            t()
        except AssertionError:
            pass
        except Exception as e:  # pragma: no cover - surfaces unexpected errors
            check(t.__name__, False, note=f"exception: {e!r}")
    print(f"\nRESULT pass={_PASS} fail={_FAIL}")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
