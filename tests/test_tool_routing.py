"""Routing of tools-carrying chat requests: bridge vs. agentic CLI.

Pure predicate tests. src.tool_routing imports nothing from the package, so
these need no app, no settings singleton and no usage ledger — the mode is
passed in as an argument rather than read from SETTINGS.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ChatCompletionRequest
from src.tool_routing import tool_call_is_owed, use_tool_bridge

TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}

TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "web_search", "arguments": '{"q":"x"}'},
}


def _req(**kw) -> ChatCompletionRequest:
    body = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "make me a csv"}],
    }
    body.update(kw)
    return ChatCompletionRequest(**body)


# ---- no tools: never the bridge, in any mode ----


def test_no_tools_never_bridges():
    for mode in ("bridge", "agentic"):
        assert use_tool_bridge(_req(), mode) is False
        assert use_tool_bridge(_req(tools=[]), mode) is False
        assert use_tool_bridge(_req(tools=None), mode) is False


# ---- default mode is byte-identical to the historical behavior ----


def test_bridge_mode_routes_every_tools_request_to_the_bridge():
    assert use_tool_bridge(_req(tools=[TOOL]), "bridge") is True
    assert use_tool_bridge(_req(tools=[TOOL], tool_choice="auto"), "bridge") is True
    assert use_tool_bridge(_req(tools=[TOOL], tool_choice="none"), "bridge") is True


def test_unknown_mode_behaves_as_bridge():
    # A typo in CLAUDE_WRAPPER_TOOLS_MODE must not silently change routing.
    assert use_tool_bridge(_req(tools=[TOOL]), "agentik") is True
    assert use_tool_bridge(_req(tools=[TOOL]), "") is True


# ---- agentic mode: ambiguous turns run the CLI ----


def test_agentic_mode_runs_cli_when_no_call_is_owed():
    assert use_tool_bridge(_req(tools=[TOOL]), "agentic") is False
    assert use_tool_bridge(_req(tools=[TOOL], tool_choice="auto"), "agentic") is False
    assert use_tool_bridge(_req(tools=[TOOL], tool_choice="none"), "agentic") is False


def test_agentic_mode_still_bridges_a_forced_call():
    assert use_tool_bridge(_req(tools=[TOOL], tool_choice="required"), "agentic") is True
    assert use_tool_bridge(_req(tools=[TOOL], tool_choice="REQUIRED"), "agentic") is True
    named = {"type": "function", "function": {"name": "web_search"}}
    assert use_tool_bridge(_req(tools=[TOOL], tool_choice=named), "agentic") is True


def test_agentic_mode_still_bridges_a_mid_loop_transcript():
    tail_tool_result = _req(
        tools=[TOOL],
        messages=[
            {"role": "user", "content": "search"},
            {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
            {"role": "tool", "tool_call_id": "call_1", "content": "results"},
        ],
    )
    assert use_tool_bridge(tail_tool_result, "agentic") is True

    tail_tool_calls = _req(
        tools=[TOOL],
        messages=[
            {"role": "user", "content": "search"},
            {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
        ],
    )
    assert use_tool_bridge(tail_tool_calls, "agentic") is True


def test_settled_tool_history_does_not_strand_the_conversation():
    """The regression guard for the whole feature.

    A transcript whose tool exchange is finished business and whose tail is a
    fresh user message is NOT mid-loop. Testing `any(m.role == "tool" ...)`
    over the whole transcript instead of the tail would pin such a conversation
    to the bridge forever after its first tool call, so it could never again
    produce a file.
    """
    req = _req(
        tools=[TOOL],
        messages=[
            {"role": "user", "content": "search"},
            {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
            {"role": "tool", "tool_call_id": "call_1", "content": "results"},
            {"role": "assistant", "content": "here is what I found"},
            {"role": "user", "content": "now put that in a csv"},
        ],
    )
    assert tool_call_is_owed(req) is False
    assert use_tool_bridge(req, "agentic") is False
    assert use_tool_bridge(req, "bridge") is True


def test_empty_messages_is_not_owed():
    assert tool_call_is_owed(_req(tools=[TOOL], messages=[])) is False
