"""Unit + integration tests for the per-conversation token budget.

Covers three layers:
  - UsageLedger accounting (accumulate / over-budget / grant / persistence / disabled),
  - the block-size arithmetic in config (allowance x percent),
  - the "continue" keyword detection and the enforcement gate in run_chat_completion,
    which must short-circuit with a checkpoint message *without* invoking Claude.

The runner is stubbed, so no real Claude Code subprocess is launched. Env is set
before importing src.* so the module-level SETTINGS / USAGE_LEDGER come up with a
small block (allowance 2000 x 5% = 100 tokens) that the stub's 120-token replies
immediately exceed.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Build SETTINGS/USAGE_LEDGER against a tempdir with the cap enabled.
_TMP = tempfile.mkdtemp(prefix="claude-wrapper-budget-test-")
os.environ["CLAUDE_WRAPPER_DATA"] = _TMP
os.environ["CLAUDE_WRAPPER_DEFAULT_MODEL"] = "claude-opus-4-8"
os.environ["CLAUDE_WRAPPER_MODEL_DISCOVERY"] = "off"
os.environ["CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE"] = "2000"
os.environ["CLAUDE_WRAPPER_SESSION_BLOCK_PERCENT"] = "5"
os.environ.pop("CLAUDE_WRAPPER_API_KEYS", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.claude_runner import ClaudeResult  # noqa: E402
from src.usage import UsageLedger  # noqa: E402

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


# ---------- UsageLedger accounting ----------


def test_ledger_accumulate_grant_and_persist() -> None:
    async def run() -> None:
        root = Path(tempfile.mkdtemp(prefix="led-")) / "usage"
        led = UsageLedger(root, block_tokens=100)
        check("ledger.enabled", led.enabled)

        s = await led.record("k", 60)
        check("ledger.record.under", s.spent_tokens == 60 and not s.over_budget)
        s = await led.record("k", 60)
        check("ledger.record.over", s.spent_tokens == 120 and s.over_budget, note=str(s))
        check("ledger.requests", s.requests == 2)

        s = await led.grant("k")
        check("ledger.grant.raises_allowance", s.allowance_tokens == 200)
        check("ledger.grant.clears_over", not s.over_budget)
        check("ledger.grant.bumps_grants", s.grants == 2)

        # Fresh ledger over the same dir must see the persisted counters.
        led2 = UsageLedger(root, block_tokens=100)
        s2 = await led2.snapshot("k")
        check(
            "ledger.persist",
            s2.spent_tokens == 120 and s2.grants == 2 and s2.requests == 2,
            note=str(s2),
        )

    asyncio.run(run())


def test_ledger_disabled_is_noop() -> None:
    async def run() -> None:
        led = UsageLedger(Path(tempfile.mkdtemp(prefix="led0-")) / "usage", block_tokens=0)
        check("ledger.disabled.enabled_false", not led.enabled)
        s = await led.record("k", 9999)
        check("ledger.disabled.record_noop", s.spent_tokens == 0 and not s.over_budget)
        s = await led.snapshot("k")
        check("ledger.disabled.snapshot_clean", not s.over_budget)

    asyncio.run(run())


# ---------- block-size arithmetic ----------


def test_plan_resolution() -> None:
    from src.config import _normalize_plan

    # Normalizer maps free-form plan strings to canonical keys.
    cases = {
        "pro": "pro",
        "PRO $20": "pro",
        "max 5x": "max_5x",
        "MAX $100": "max_5x",
        "max": "max_5x",  # bare "max" defaults to 5x
        "max 20x": "max_20x",
        "MAX $200": "max_20x",
        "": "",
        "garbage": "",
    }
    for raw, expected in cases.items():
        check(f"plan.norm.{raw or 'empty'}", _normalize_plan(raw) == expected, note=_normalize_plan(raw))

    # End-to-end resolution via Settings.from_env (anchor default 1,000,000).
    saved = {k: os.environ.get(k) for k in (
        "CLAUDE_WRAPPER_SESSION_PLAN",
        "CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE",
        "CLAUDE_WRAPPER_PRO_SESSION_TOKENS",
    )}
    try:
        os.environ.pop("CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE", None)
        os.environ.pop("CLAUDE_WRAPPER_PRO_SESSION_TOKENS", None)

        # Default Pro anchor is 1,500,000 (calibrated); plans scale from it.
        os.environ["CLAUDE_WRAPPER_SESSION_PLAN"] = "pro"
        s = config.Settings.from_env()
        check("plan.pro.allowance", s.session_token_allowance == 1_500_000 and s.session_plan == "pro")

        os.environ["CLAUDE_WRAPPER_SESSION_PLAN"] = "max $100"
        s = config.Settings.from_env()
        check("plan.max5x.allowance", s.session_token_allowance == 7_500_000 and s.session_plan == "max_5x")

        os.environ["CLAUDE_WRAPPER_SESSION_PLAN"] = "max $200"
        s = config.Settings.from_env()
        check("plan.max20x.allowance", s.session_token_allowance == 30_000_000 and s.session_plan == "max_20x")

        # Custom Pro anchor scales every plan.
        os.environ["CLAUDE_WRAPPER_PRO_SESSION_TOKENS"] = "2000000"
        os.environ["CLAUDE_WRAPPER_SESSION_PLAN"] = "max 20x"
        s = config.Settings.from_env()
        check("plan.anchor.scales", s.session_token_allowance == 40_000_000)
        os.environ.pop("CLAUDE_WRAPPER_PRO_SESSION_TOKENS", None)

        # Explicit allowance wins over the plan.
        os.environ["CLAUDE_WRAPPER_SESSION_PLAN"] = "pro"
        os.environ["CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE"] = "777000"
        s = config.Settings.from_env()
        check("plan.explicit.wins", s.session_token_allowance == 777_000 and s.session_plan == "custom")
        os.environ.pop("CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE", None)

        # Unset plan → defaults to Max 5x (cap is ON out of the box).
        os.environ.pop("CLAUDE_WRAPPER_SESSION_PLAN", None)
        s = config.Settings.from_env()
        check("plan.default.max5x", s.session_plan == "max_5x" and s.session_token_allowance == 7_500_000, note=str(s.session_token_allowance))

        # Explicit "off" disables the cap.
        os.environ["CLAUDE_WRAPPER_SESSION_PLAN"] = "off"
        s = config.Settings.from_env()
        check("plan.off.disabled", s.session_token_allowance == 0 and s.session_block_tokens == 0)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_block_math() -> None:
    prev_alloc = os.environ.get("CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE")
    prev_plan = os.environ.get("CLAUDE_WRAPPER_SESSION_PLAN")
    try:
        os.environ["CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE"] = "1000000"
        os.environ["CLAUDE_WRAPPER_SESSION_BLOCK_PERCENT"] = "5"
        s = config.Settings.from_env()
        check("block.5pct", s.session_block_tokens == 50000, note=str(s.session_block_tokens))

        # Explicit allowance 0 + plan off → cap fully disabled.
        os.environ["CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE"] = "0"
        os.environ["CLAUDE_WRAPPER_SESSION_PLAN"] = "off"
        check("block.disabled", config.Settings.from_env().session_block_tokens == 0)
    finally:
        os.environ["CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE"] = prev_alloc or "2000"
        os.environ["CLAUDE_WRAPPER_SESSION_BLOCK_PERCENT"] = "5"
        if prev_plan is None:
            os.environ.pop("CLAUDE_WRAPPER_SESSION_PLAN", None)
        else:
            os.environ["CLAUDE_WRAPPER_SESSION_PLAN"] = prev_plan


# ---------- keyword detection ----------


def test_is_continue() -> None:
    from src.main import _is_continue
    from src.models import ChatMessage

    def msgs(text: str):
        return [ChatMessage(role="user", content=text)]

    check("cont.plain", _is_continue(msgs("continue")))
    check("cont.punct", _is_continue(msgs(" Continue. ")))
    check("cont.yes", _is_continue(msgs("yes")))
    check("cont.trailing", _is_continue(msgs("yes, continue")))
    check("cont.leading", _is_continue(msgs("continue please")))
    check("cont.multiword", _is_continue(msgs("keep going")))
    check("cont.no", not _is_continue(msgs("what is the capital of France?")))
    check("cont.empty", not _is_continue([]))
    # Uses the last user message, not an earlier one.
    check(
        "cont.last_only",
        not _is_continue(
            [
                ChatMessage(role="user", content="continue"),
                ChatMessage(role="assistant", content="ok"),
                ChatMessage(role="user", content="now do something else"),
            ]
        ),
    )


# ---------- enforcement gate ----------

_CALLS = {"n": 0}


async def _stub_run_collect(prompt, session_key, model=None, env_extra=None, extra_args=None, effort=None):
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


def test_enforcement_pauses_then_continues() -> None:
    from src.deps import RUNNER, USAGE_LEDGER
    from src.main import run_chat_completion
    from src.models import ChatCompletionRequest, ChatMessage

    check("gate.ledger_enabled", USAGE_LEDGER.enabled)
    # Stub on the shared singleton, but restore afterwards: an instance attribute
    # here would otherwise shadow the class-level stub other test modules install
    # (e.g. test_endpoints) when they run later in the same pytest process.
    had_own = "run_collect" in RUNNER.__dict__
    prev = RUNNER.__dict__.get("run_collect")
    RUNNER.run_collect = _stub_run_collect  # instance attr → called without self
    _CALLS["n"] = 0
    sid = "budget-int-1"

    def req(messages):
        return ChatCompletionRequest(model="claude-opus-4-8", session_id=sid, messages=messages)

    async def run():
        # Capture the runner call-count after each turn — checking it only at the
        # end would always see the final total.
        # 1) first turn runs and records 120 tokens (block is 100 → now over).
        r1 = await run_chat_completion(req([ChatMessage(role="user", content="hello")]))
        n1 = _CALLS["n"]
        # 2) over budget + ordinary prompt → checkpoint, no Claude call.
        r2 = await run_chat_completion(
            req(
                [
                    ChatMessage(role="user", content="hello"),
                    ChatMessage(role="assistant", content="ok"),
                    ChatMessage(role="user", content="tell me more"),
                ]
            )
        )
        n2 = _CALLS["n"]
        # 3) over budget + "continue" → grant a block and run again.
        r3 = await run_chat_completion(
            req(
                [
                    ChatMessage(role="user", content="hello"),
                    ChatMessage(role="assistant", content="ok"),
                    ChatMessage(role="user", content="continue"),
                ]
            )
        )
        n3 = _CALLS["n"]
        return (r1, n1), (r2, n2), (r3, n3)

    try:
        (r1, n1), (r2, n2), (r3, n3) = asyncio.run(run())
    finally:
        if had_own:
            RUNNER.run_collect = prev
        else:
            del RUNNER.run_collect

    check("gate.first_ran", n1 == 1 and _content(r1) == "ok", note=f"n1={n1}")
    check("gate.pause_content", "Usage checkpoint" in _content(r2), note=_content(r2))
    check("gate.pause_no_claude", n2 == 1, note=f"n2={n2}")
    check("gate.continue_ran", n3 == 2 and _content(r3) == "ok", note=f"n3={n3}")


def main() -> int:
    tests = [
        test_ledger_accumulate_grant_and_persist,
        test_ledger_disabled_is_noop,
        test_plan_resolution,
        test_block_math,
        test_is_continue,
        test_enforcement_pauses_then_continues,
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
