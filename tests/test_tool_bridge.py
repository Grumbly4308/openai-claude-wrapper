"""Function-calling (tools) tests for the tool bridge.

The Anthropic Messages API is mocked with httpx.MockTransport, so every test
also captures the exact outbound payload and asserts the OpenAI→Anthropic
translation, not just the response shape. The agentic path is stubbed the same
way test_endpoints.py does, to prove tools-absent requests never touch the
bridge.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import re
import sys
import tempfile
from pathlib import Path

# ---- environment setup before importing anything from src ----
_TMP = tempfile.mkdtemp(prefix="claude-wrapper-test-tools-")
os.environ.setdefault("CLAUDE_WRAPPER_DATA", _TMP)
os.environ.setdefault("CLAUDE_WRAPPER_DEFAULT_MODEL", "claude-sonnet-4-6")
os.environ.setdefault("CLAUDE_WRAPPER_MODEL_DISCOVERY", "off")
os.environ.pop("CLAUDE_WRAPPER_API_KEYS", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import pytest
from fastapi.testclient import TestClient

from src import main as src_main
from src import tool_bridge
from src.claude_runner import ClaudeResult, StreamEvent
from src.deps import RUNNER
from src.main import app

client = TestClient(app)

_seq = 0


def _post(body: dict):
    """POST a chat completion with a unique session_id per request.

    The usage ledger is a process-wide singleton that test_budget.py enables
    (with a tiny block) for the whole pytest run; without a fresh session each
    request, the budget checkpoint fires instead of the bridge.
    """
    global _seq
    _seq += 1
    return client.post("/v1/chat/completions", json={"session_id": f"toolbridge-{_seq}", **body})

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def _fn_tool(name: str) -> dict:
    """A minimal client function-tool declaration."""
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object", "properties": {}}},
    }


def _anthropic_tool_use_response() -> dict:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_abc123",
                "name": "web_search",
                "input": {"query": "weather in Paris"},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


class _Capture:
    """MockTransport handler that records outbound requests."""

    def __init__(self, responses):
        self.requests: list[dict] = []
        self._responses = list(responses)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.last_headers = dict(request.headers)
        body = self._responses.pop(0)
        if isinstance(body, bytes):
            return httpx.Response(
                200, content=body, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=body)


@pytest.fixture
def bridge(monkeypatch):
    """Install a capture transport and an env API key; restore afterwards."""

    def _install(*responses):
        capture = _Capture(responses)
        monkeypatch.setattr(
            tool_bridge, "_client", httpx.AsyncClient(transport=httpx.MockTransport(capture))
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        return capture

    yield _install
    tool_bridge._client = None


# ---------- request translation ----------


def test_tool_call_non_streaming(bridge):
    capture = bridge(_anthropic_tool_use_response())
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "temperature": 0,
            "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}],
            "tools": [WEB_SEARCH_TOOL],
        },
    )
    assert r.status_code == 200
    choice = r.json()["choices"][0]

    # tool_calls populated, finish_reason tool_calls, content explicit null
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    (tc,) = choice["message"]["tool_calls"]
    assert tc["id"] == "toolu_abc123"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "web_search"
    # arguments MUST be a serialized JSON string, not an object
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"query": "weather in Paris"}

    # outbound: parameters -> input_schema verbatim, default tool_choice auto
    (sent,) = capture.requests
    assert sent["tools"] == [
        {
            "name": "web_search",
            "description": "Search the web",
            "input_schema": WEB_SEARCH_TOOL["function"]["parameters"],
        }
    ]
    assert sent["tool_choice"] == {"type": "auto"}
    assert sent["model"] == "claude-haiku-4-5"
    assert sent["temperature"] == 0
    # api-key auth: x-api-key header, no oauth beta
    assert capture.last_headers.get("x-api-key") == "test-key"
    assert "anthropic-beta" not in capture.last_headers
    # usage + wrapper parity fields survive
    body = r.json()
    assert body["usage"]["total_tokens"] == 120
    assert body["session_id"]
    assert body["effort"]["source"] == "tool-bridge"


def test_two_turn_round_trip(bridge):
    final = {
        "id": "msg_02",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "text", "text": "It is sunny, 19C."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 150, "output_tokens": 12},
    }
    capture = bridge(final)
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [
                {"role": "user", "content": "What is the weather in Paris right now?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "toolu_abc123",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": "{\"query\": \"weather in Paris\"}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "toolu_abc123",
                    "content": "Paris: sunny, 19C",
                },
            ],
            "tools": [WEB_SEARCH_TOOL],
        },
    )
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "It is sunny, 19C."
    assert "tool_calls" not in choice["message"]

    (sent,) = capture.requests
    user, assistant, tool_result = sent["messages"]
    assert assistant["role"] == "assistant"
    (tu,) = assistant["content"]
    # id round-trips losslessly and arguments were parsed to an object
    assert tu == {
        "type": "tool_use",
        "id": "toolu_abc123",
        "name": "web_search",
        "input": {"query": "weather in Paris"},
    }
    assert tool_result["role"] == "user"
    (tr,) = tool_result["content"]
    assert tr["type"] == "tool_result"
    assert tr["tool_use_id"] == "toolu_abc123"
    assert tr["content"] == "Paris: sunny, 19C"


def test_parallel_tool_calls(bridge):
    resp = _anthropic_tool_use_response()
    resp["content"].append(
        {
            "type": "tool_use",
            "id": "toolu_def456",
            "name": "web_search",
            "input": {"query": "Paris humidity"},
        }
    )
    capture = bridge(resp, resp)
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "Weather and humidity in Paris?"}],
            "tools": [WEB_SEARCH_TOOL],
        },
    )
    tcs = r.json()["choices"][0]["message"]["tool_calls"]
    assert [t["id"] for t in tcs] == ["toolu_abc123", "toolu_def456"]
    assert all(isinstance(t["function"]["arguments"], str) for t in tcs)

    # Round-trip: both results must merge into ONE user message.
    r2 = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [
                {"role": "user", "content": "Weather and humidity in Paris?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "toolu_abc123", "type": "function",
                         "function": {"name": "web_search", "arguments": "{\"query\": \"a\"}"}},
                        {"id": "toolu_def456", "type": "function",
                         "function": {"name": "web_search", "arguments": "{\"query\": \"b\"}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "toolu_abc123", "content": "sunny"},
                {"role": "tool", "tool_call_id": "toolu_def456", "content": "60%"},
            ],
            "tools": [WEB_SEARCH_TOOL],
        },
    )
    assert r2.status_code == 200
    sent = capture.requests[1]
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "user"]
    results = sent["messages"][2]["content"]
    assert [b["tool_use_id"] for b in results] == ["toolu_abc123", "toolu_def456"]


def test_forced_and_none_tool_choice(bridge):
    capture = bridge(_anthropic_tool_use_response(), _anthropic_tool_use_response(),
                     _anthropic_tool_use_response())
    base = {
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": "Population of Tokyo?"}],
        "tools": [WEB_SEARCH_TOOL],
    }
    _post({**base, "tool_choice": {"type": "function", "function": {"name": "web_search"}}})
    assert capture.requests[0]["tool_choice"] == {"type": "tool", "name": "web_search"}

    _post({**base, "tool_choice": "required"})
    assert capture.requests[1]["tool_choice"] == {"type": "any"}

    _post({**base, "tool_choice": "none"})
    assert "tools" not in capture.requests[2]
    assert "tool_choice" not in capture.requests[2]


def test_parallel_tool_calls_disabled(bridge):
    capture = bridge(_anthropic_tool_use_response())
    _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
            "parallel_tool_calls": False,
        },
    )
    assert capture.requests[0]["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }


def test_system_message_and_model_mapping(bridge):
    capture = bridge(_anthropic_tool_use_response())
    _post(
        {
            "model": "claude-opus-4-8[1m] (xhigh)",
            "messages": [
                {"role": "system", "content": "You are a researcher."},
                {"role": "user", "content": "hi"},
            ],
            "tools": [WEB_SEARCH_TOOL],
        },
    )
    sent = capture.requests[0]
    assert sent["model"] == "claude-opus-4-8"
    assert {"type": "text", "text": "You are a researcher."} in sent["system"]
    assert "context-1m-2025-08-07" in capture.last_headers.get("anthropic-beta", "")


# ---------- streaming ----------


def _sse(events: list[dict]) -> bytes:
    return b"".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events
    )


def test_streaming_tool_call_deltas(bridge):
    events = [
        {"type": "message_start",
         "message": {"id": "msg_01", "usage": {"input_tokens": 50, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "Searching."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "toolu_xyz", "name": "web_search", "input": {}}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": "{\"query\": "}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": "\"Tokyo population\"}"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
         "usage": {"output_tokens": 30}},
        {"type": "message_stop"},
    ]
    bridge(_sse(events))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "claude-haiku-4-5",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "Population of Tokyo? Use web_search."}],
            "tools": [WEB_SEARCH_TOOL],
        },
    ) as r:
        assert r.status_code == 200
        lines = [l for l in r.iter_lines() if l.startswith("data:")]

    payloads = [l[5:].strip() for l in lines]
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1] if p != "[DONE]"]

    tc_frames = [
        c["choices"][0]["delta"]["tool_calls"]
        for c in chunks
        if c.get("choices") and c["choices"][0]["delta"].get("tool_calls")
    ]
    # First frame: index, id, type, name, empty arguments.
    first = tc_frames[0][0]
    assert first == {
        "index": 0,
        "id": "toolu_xyz",
        "type": "function",
        "function": {"name": "web_search", "arguments": ""},
    }
    # Later frames: only index + argument fragment; assemble into valid JSON.
    assembled = "".join(f[0]["function"]["arguments"] for f in tc_frames)
    assert all(f[0]["index"] == 0 for f in tc_frames)
    assert json.loads(assembled) == {"query": "Tokyo population"}

    # Text alongside the call still streams as content.
    contents = [
        c["choices"][0]["delta"].get("content", "")
        for c in chunks
        if c.get("choices")
    ]
    assert "Searching." in contents

    finish = [
        c["choices"][0]["finish_reason"]
        for c in chunks
        if c.get("choices") and c["choices"][0].get("finish_reason")
    ]
    assert finish == ["tool_calls"]

    usage_chunks = [c for c in chunks if not c.get("choices")]
    assert usage_chunks and usage_chunks[0]["usage"]["total_tokens"] == 80


# ---------- tools absent: the agentic path is untouched ----------


def test_no_tools_uses_agentic_path(bridge):
    capture = bridge()  # would raise IndexError if the bridge were hit

    async def _stub_run_collect(prompt, session_key, **_kwargs):
        return ClaudeResult(session_uuid="u", final_text="agentic ok", input_tokens=1, output_tokens=1)

    # Instance attribute on the RUNNER singleton (restored below): other test
    # modules replace ClaudeRunner methods at class level on import, so a
    # class-level monkeypatch here could be shadowed or shadow them.
    had_own = "run_collect" in RUNNER.__dict__
    prev = RUNNER.__dict__.get("run_collect")
    RUNNER.run_collect = _stub_run_collect
    try:
        for body in (
            {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]},
            {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}], "tools": []},
        ):
            r = _post(body)
            assert r.status_code == 200
            msg = r.json()["choices"][0]["message"]
            assert msg["content"] == "agentic ok"
            assert "tool_calls" not in msg
    finally:
        if had_own:
            RUNNER.run_collect = prev
        else:
            del RUNNER.run_collect
    assert capture.requests == []


# ---------- tools present: always the bridge; no tools: always the CLI ----------
#
# There is no mode switch. A turn cannot be served both ways -- the CLI runs its
# own tool loop and cannot surface a caller-declared tool -- so `tools` on the
# wire means the bridge, and the bridge has no workspace and produces no files.
# A chat UI that wants downloads sends no tools (Open WebUI: Function Calling
# "Native" -> "Default"), which is the second test below.


@contextlib.contextmanager
def _stub_runner(final_text="cli ok", new_outputs=None):
    """Replace RUNNER.run_collect for the duration, as instance attribute.

    Same reasoning as test_no_tools_uses_agentic_path (the CLI path): other modules patch
    ClaudeRunner at class level on import, so class-level patching here could be
    shadowed or shadow them.
    """

    async def _stub_run_collect(prompt, session_key, **_kwargs):
        events = []
        if new_outputs is not None:
            events.append(
                StreamEvent(kind="system", raw={"new_outputs": [str(p) for p in new_outputs]})
            )
        return ClaudeResult(
            session_uuid="u",
            final_text=final_text,
            input_tokens=1,
            output_tokens=1,
            events=events,
        )

    had_own = "run_collect" in RUNNER.__dict__
    prev = RUNNER.__dict__.get("run_collect")
    RUNNER.run_collect = _stub_run_collect
    try:
        yield
    finally:
        if had_own:
            RUNNER.run_collect = prev
        else:
            del RUNNER.run_collect


def _set_settings(monkeypatch, **extra):
    monkeypatch.setattr(
        src_main, "SETTINGS", dataclasses.replace(src_main.SETTINGS, **extra)
    )


def test_every_tools_request_stays_on_the_bridge(bridge, monkeypatch):
    """Explicit assertion of the promise the other 14 tests rely on.

    Unconditional: no setting relaxes it. Clients that own their agent loop
    (the Vercel AI SDK, LangChain) depend on getting a tool_call back rather
    than prose, and dropping their tools to run the CLI would break the loop on
    its opening turn.
    """
    capture = bridge(_anthropic_tool_use_response())
    with _stub_runner():
        r = _post(
            {
                "model": "claude-haiku-4-5",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [WEB_SEARCH_TOOL],
            }
        )
    assert r.status_code == 200
    assert len(capture.requests) == 1
    assert r.json()["choices"][0]["message"]["tool_calls"]


def test_a_toolless_request_gets_a_clickable_download_link(bridge, monkeypatch, tmp_path):
    """The end-to-end goal, in the shape Open WebUI sends once Function Calling
    is set to "Default": no tools on the wire, a file produced, a link in the
    reply. This is the supported route to downloads from a chat UI."""
    capture = bridge()
    _set_settings(
        monkeypatch,
        public_base_url="https://wrapper.example",
        download_signing_key="k" * 32,
        download_url_ttl_seconds=3600,
    )
    out = tmp_path / "report.csv"
    out.write_text("a,b\n1,2\n")
    with _stub_runner(final_text="done", new_outputs=[out]):
        r = _post(
            {
                "model": "claude-haiku-4-5",
                "messages": [{"role": "user", "content": "make me a csv"}],
            }
        )
    assert r.status_code == 200
    assert capture.requests == []
    content = r.json()["choices"][0]["message"]["content"]
    assert re.search(
        r"\[report\.csv\]\(https://wrapper\.example/v1/files/file-[0-9a-f]{32}/content"
        r"\?exp=\d+&sig=[A-Za-z0-9_-]+\)",
        content,
    ), content


def test_oauth_auth_headers(bridge, monkeypatch, tmp_path):
    capture = bridge(_anthropic_tool_use_response())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-token-123")
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
        },
    )
    assert r.status_code == 200
    assert capture.last_headers["authorization"] == "Bearer oauth-token-123"
    assert "oauth-2025-04-20" in capture.last_headers["anthropic-beta"]
    # Claude Code identity line injected as the FIRST system block.
    assert capture.requests[0]["system"][0]["text"].startswith("You are Claude Code")


# ---------- tools + response_format together ----------
#
# A request may declare tools AND response_format: AI SDK clients do this when a
# structured-output call is allowed to call tools first. The bridge path is
# chosen before the CLI path's JSON-mode handling ever runs, so it has to
# apply the output-format instruction and the raw-JSON reduction itself —
# otherwise the client JSON.parses a fenced or prose-wrapped body and dies.


def _anthropic_text_response(text: str) -> dict:
    return {
        "id": "msg_json",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def test_json_mode_instruction_and_strip(bridge):
    capture = bridge(_anthropic_text_response('```json\n{"answer": 42}\n```'))
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "answer as json"}],
            "tools": [WEB_SEARCH_TOOL],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "result",
                    "schema": {"type": "object", "properties": {"answer": {"type": "number"}}},
                },
            },
        },
    )
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    # Fence stripped: the client can JSON.parse this verbatim.
    assert json.loads(content) == {"answer": 42}

    # The instruction reached the model, as the LAST system block, with schema.
    (sent,) = capture.requests
    assert "Output format" in sent["system"][-1]["text"]
    assert "JSON Schema" in sent["system"][-1]["text"]
    assert '"answer"' in sent["system"][-1]["text"]


def test_json_mode_no_instruction_without_response_format(bridge):
    capture = bridge(_anthropic_text_response('```json\n{"answer": 42}\n```'))
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "answer"}],
            "tools": [WEB_SEARCH_TOOL],
        },
    )
    assert r.status_code == 200
    # Untouched without response_format — fence and all.
    assert r.json()["choices"][0]["message"]["content"] == '```json\n{"answer": 42}\n```'
    (sent,) = capture.requests
    assert not any("Output format" in b["text"] for b in sent.get("system", []))


def test_json_mode_leaves_tool_calls_alone(bridge):
    """Content next to a tool call is commentary, not the structured answer."""
    bridge(_anthropic_tool_use_response())
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "search then answer as json"}],
            "tools": [WEB_SEARCH_TOOL],
            "response_format": {"type": "json_object"},
        },
    )
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    (tc,) = choice["message"]["tool_calls"]
    assert json.loads(tc["function"]["arguments"]) == {"query": "weather in Paris"}


def test_json_mode_streaming_buffers_and_strips(bridge):
    # Fence split across chunk boundaries: only whole-stream stripping works.
    events = [
        {"type": "message_start",
         "message": {"id": "msg_01", "usage": {"input_tokens": 5, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": '```json\n{"answer": '}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": '42}\n```'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 9}},
        {"type": "message_stop"},
    ]
    bridge(_sse(events))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "session_id": "toolbridge-json-stream",
            "model": "claude-haiku-4-5",
            "stream": True,
            "messages": [{"role": "user", "content": "answer as json"}],
            "tools": [WEB_SEARCH_TOOL],
            "response_format": {"type": "json_object"},
        },
    ) as r:
        assert r.status_code == 200
        payloads = [l[5:].strip() for l in r.iter_lines() if l.startswith("data:")]

    chunks = [json.loads(p) for p in payloads if p != "[DONE]"]
    content = "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunks if c.get("choices")
    )
    assert json.loads(content) == {"answer": 42}


def test_json_mode_unparseable_errors(bridge):
    """Prose in JSON mode => 502 quoting the model, not a 200 that breaks parse."""
    bridge(_anthropic_text_response("I need more details before I can build that."))
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "answer as json"}],
            "tools": [WEB_SEARCH_TOOL],
            "response_format": {"type": "json_object"},
        },
    )
    assert r.status_code == 502
    err = r.json()["error"]
    assert "I need more details" in err["message"]
    assert "json_object" in err["message"]
    assert err["type"] == "api_error"


def test_json_mode_streaming_unparseable_errors(bridge):
    events = [
        {"type": "message_start",
         "message": {"id": "msg_01", "usage": {"input_tokens": 5, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "I need more "}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "details first."}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 9}},
        {"type": "message_stop"},
    ]
    bridge(_sse(events))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "session_id": "toolbridge-json-stream-err",
            "model": "claude-haiku-4-5",
            "stream": True,
            "messages": [{"role": "user", "content": "answer as json"}],
            "tools": [WEB_SEARCH_TOOL],
            "response_format": {"type": "json_object"},
        },
    ) as r:
        assert r.status_code == 200
        payloads = [l[5:].strip() for l in r.iter_lines() if l.startswith("data:")]

    chunks = [json.loads(p) for p in payloads if p != "[DONE]"]
    # No prose leaked onto the content channel...
    content = "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunks if c.get("choices")
    )
    assert content == ""
    # ...it came back on the error channel instead, quoting the reply.
    errors = [c["error"]["message"] for c in chunks if "error" in c]
    assert len(errors) == 1
    assert "I need more details first." in errors[0]


# ---------- capability profiles on the bridge (phase 3) ----------


@pytest.fixture
def profiles(monkeypatch):
    """Point the profile loader at a per-test file; reset caches around it."""
    from src.capabilities import PROFILE_FILE_ENV, reset_profile_cache

    def _set(doc: dict):
        path = Path(_TMP) / f"profiles-{len(str(doc))}.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        monkeypatch.setenv(PROFILE_FILE_ENV, str(path))
        reset_profile_cache()

    yield _set
    reset_profile_cache()


def test_client_tools_denied_by_profile(bridge, profiles):
    bridge(_anthropic_tool_use_response())
    profiles({"models": [{"match": "claude-haiku-4-5", "remove": ["client_tools"]}]})
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
        }
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error" and err["param"] == "tools"
    assert "client_tools" in err["message"] and "web_search" in err["message"]


def test_code_interpreter_injected_after_client_tools(bridge, profiles):
    capture = bridge(_anthropic_tool_use_response())
    profiles({"models": [{"match": "claude-haiku-4-5", "add": ["code_interpreter"]}]})
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
        }
    )
    assert r.status_code == 200
    sent = capture.requests[0]["tools"]
    assert [t.get("name") for t in sent] == ["web_search", "code_execution"]
    assert sent[1]["type"] == "code_execution_20260521"
    # tool_choice still governs the client tools.
    assert capture.requests[0]["tool_choice"] == {"type": "auto"}


def test_bridge_web_search_needs_env_opt_in(bridge, profiles, monkeypatch):
    # Capability on (default) but no env opt-in → no injection.
    capture = bridge(_anthropic_tool_use_response())
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
        }
    )
    assert r.status_code == 200
    assert [t.get("name") for t in capture.requests[0]["tools"]] == ["web_search"]

    # Env opt-in → injected, basic variant for Haiku. The client tool keeps a
    # distinct name here; a client tool NAMED web_search suppresses injection
    # instead (see test_client_tool_shadows_server_web_search).
    monkeypatch.setenv("CLAUDE_WRAPPER_BRIDGE_WEB_SEARCH", "true")
    capture = bridge(_anthropic_tool_use_response())
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_fn_tool("lookup")],
        }
    )
    assert r.status_code == 200
    sent = capture.requests[0]["tools"]
    assert len(sent) == 2
    assert sent[0]["name"] == "lookup"
    assert sent[1] == {"type": "web_search_20250305", "name": "web_search"}


def test_client_tool_shadows_server_web_search(bridge, profiles, monkeypatch):
    """A client tool reusing a server tool's name wins; the server tool is not
    injected — the Messages API rejects duplicate names, so sending both used
    to 502 opaquely."""
    monkeypatch.setenv("CLAUDE_WRAPPER_BRIDGE_WEB_SEARCH", "true")
    capture = bridge(_anthropic_tool_use_response())
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
        }
    )
    assert r.status_code == 200
    sent = capture.requests[0]["tools"]
    assert len(sent) == 1
    assert sent[0]["name"] == "web_search" and "type" not in sent[0]


def test_web_search_version_tracks_model_family(bridge, profiles, monkeypatch):
    monkeypatch.setenv("CLAUDE_WRAPPER_BRIDGE_WEB_SEARCH", "true")
    capture = bridge(_anthropic_tool_use_response())
    r = _post(
        {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_fn_tool("lookup")],
        }
    )
    assert r.status_code == 200
    assert capture.requests[0]["tools"][1]["type"] == "web_search_20260209"


# ---------- wrapper-owned tools & the hybrid loop (phase 4) ----------


def _anthropic_text_response(text: str = "The answer is 4.") -> dict:
    return {
        "id": "msg_02",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 40, "output_tokens": 10},
    }


def _wrapper_call_response(name: str, tool_input: dict, tool_id: str = "toolu_w1") -> dict:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [
            {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 30, "output_tokens": 5},
    }


def test_hybrid_loop_executes_wrapper_tool(bridge, profiles):
    capture = bridge(
        _wrapper_call_response("calculate", {"expression": "2+2"}),
        _anthropic_text_response(),
    )
    profiles({"models": [{"match": "claude-haiku-4-5", "add": ["time_calc"]}]})
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "what is 2+2?"}],
            "tools": [WEB_SEARCH_TOOL],
        }
    )
    assert r.status_code == 200
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == "The answer is 4."
    assert not msg.get("tool_calls")
    # Two upstream rounds; the second carries the echoed assistant turn and
    # the wrapper's tool_result.
    assert len(capture.requests) == 2
    second = capture.requests[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["content"][0]["name"] == "calculate"
    assert second[-1]["role"] == "user"
    assert second[-1]["content"][0]["tool_use_id"] == "toolu_w1"
    assert "= 4" in second[-1]["content"][0]["content"]
    # Wrapper tool definitions were injected after the client's.
    names = [t.get("name") for t in capture.requests[0]["tools"]]
    assert names == ["web_search", "get_current_time", "calculate"]
    # Usage accumulates across rounds.
    assert r.json()["usage"]["prompt_tokens"] == 70


def test_mixed_turn_returns_client_calls_and_drops_wrapper_calls(bridge, profiles):
    mixed = _wrapper_call_response("calculate", {"expression": "1+1"})
    mixed["content"].append(
        {"type": "tool_use", "id": "toolu_c1", "name": "web_search", "input": {"query": "x"}}
    )
    capture = bridge(mixed)
    profiles({"models": [{"match": "claude-haiku-4-5", "add": ["time_calc"]}]})
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
        }
    )
    assert r.status_code == 200
    msg = r.json()["choices"][0]["message"]
    calls = msg["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["web_search"]
    # One round only: the client loop takes over; the wrapper call was
    # neither executed nor surfaced.
    assert len(capture.requests) == 1


def test_wrapper_tool_name_collision_is_rejected(bridge, profiles):
    bridge()
    profiles({"models": [{"match": "claude-haiku-4-5", "add": ["time_calc"]}]})
    calc_tool = {
        "type": "function",
        "function": {"name": "calculate", "parameters": {"type": "object", "properties": {}}},
    }
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [calc_tool],
        }
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert "calculate" in err["message"]


def test_hybrid_loop_round_cap(bridge, profiles):
    responses = [
        _wrapper_call_response("calculate", {"expression": "1+1"}, tool_id=f"toolu_{i}")
        for i in range(8)
    ]
    capture = bridge(*responses)
    profiles({"models": [{"match": "claude-haiku-4-5", "add": ["time_calc"]}]})
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "loop forever"}],
            "tools": [WEB_SEARCH_TOOL],
        }
    )
    assert r.status_code == 502
    err = r.json()["error"]
    assert "rounds" in err["message"] and err["code"] == "tool_loop_limit"
    assert len(capture.requests) == 8


def test_memory_persists_under_the_data_dir(bridge, profiles):
    capture = bridge(
        _wrapper_call_response(
            "memory",
            {"command": "create", "path": "/memories/prefs.md", "file_text": "likes tabs"},
        ),
        _anthropic_text_response("Noted."),
    )
    profiles({"models": [{"match": "claude-haiku-4-5", "add": ["memory"]}]})
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "remember I like tabs"}],
            "tools": [WEB_SEARCH_TOOL],
        }
    )
    assert r.status_code == 200
    assert len(capture.requests) == 2
    # Glob the live settings' data dir — under pytest, whichever test module
    # imported first owns CLAUDE_WRAPPER_DATA, so _TMP may not be it.
    from src.config import SETTINGS

    stored = list((SETTINGS.data_dir / "memory").rglob("prefs.md"))
    assert stored and stored[0].read_text() == "likes tabs"


def test_streaming_hybrid_loop_suppresses_wrapper_calls(bridge, profiles):
    round1 = [
        {"type": "message_start",
         "message": {"id": "msg_01", "usage": {"input_tokens": 30, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": "toolu_w1", "name": "calculate", "input": {}}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": "{\"expression\": \"2+2\"}"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]
    round2 = [
        {"type": "message_start",
         "message": {"id": "msg_02", "usage": {"input_tokens": 40, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "It is 4."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 6}},
        {"type": "message_stop"},
    ]
    capture = bridge(_sse(round1), _sse(round2))
    profiles({"models": [{"match": "claude-haiku-4-5", "add": ["time_calc"]}]})
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "claude-haiku-4-5",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "2+2?"}],
            "tools": [WEB_SEARCH_TOOL],
        },
    ) as r:
        assert r.status_code == 200
        lines = [l for l in r.iter_lines() if l.startswith("data:")]
    chunks = [json.loads(p) for p in (l[5:].strip() for l in lines) if p != "[DONE]"]

    # The wrapper's calculate call never surfaced as an OpenAI tool_call.
    assert not any(
        c["choices"][0]["delta"].get("tool_calls") for c in chunks if c.get("choices")
    )
    content = "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunks if c.get("choices")
    )
    assert content == "It is 4."
    finish = [
        c["choices"][0]["finish_reason"]
        for c in chunks
        if c.get("choices") and c["choices"][0].get("finish_reason")
    ]
    assert finish == ["stop"]
    # Round 2 carried the echoed assistant turn and the tool_result.
    assert len(capture.requests) == 2
    echoed = capture.requests[1]["messages"]
    assert echoed[-2]["content"][0] == {
        "type": "tool_use", "id": "toolu_w1", "name": "calculate",
        "input": {"expression": "2+2"},
    }
    assert "= 4" in echoed[-1]["content"][0]["content"]
    # Usage sums both rounds.
    usage = [c for c in chunks if not c.get("choices")][0]["usage"]
    assert usage["prompt_tokens"] == 70 and usage["completion_tokens"] == 11


# ---------- OpenAI compat matrix: one test per contract quirk ----------


def test_duplicate_tool_names_deduped_last_wins(bridge):
    """OpenAI tolerates duplicate names; the Messages API rejects them. The
    bridge dedupes (last definition wins) instead of 502ing opaquely."""
    capture = bridge(_anthropic_tool_use_response())
    first = _fn_tool("web_search")
    second = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "the newer definition",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [first, second],
        }
    )
    assert r.status_code == 200
    sent = capture.requests[0]["tools"]
    assert len(sent) == 1
    assert sent[0]["description"] == "the newer definition"
    assert sent[0]["input_schema"]["properties"] == {"q": {"type": "string"}}


def test_tool_names_sanitized_and_reverse_mapped(bridge):
    """Dotted/namespaced names go upstream API-safe and come back verbatim —
    in tool defs, forced tool_choice, echoed history, and response calls."""
    resp = _anthropic_tool_use_response()
    resp["content"][0]["name"] = "repo_search_v2"  # what the API would echo
    capture = bridge(resp)
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [
                {"role": "user", "content": "search the repo"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "toolu_prev",
                            "type": "function",
                            "function": {"name": "repo.search/v2", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "toolu_prev", "content": "nothing"},
            ],
            "tools": [_fn_tool("repo.search/v2")],
            "tool_choice": {"type": "function", "function": {"name": "repo.search/v2"}},
        }
    )
    assert r.status_code == 200
    sent = capture.requests[0]
    # Outbound: definition, forced choice, and history echo all sanitized.
    assert sent["tools"][0]["name"] == "repo_search_v2"
    assert sent["tool_choice"] == {"type": "tool", "name": "repo_search_v2"}
    assistant = [m for m in sent["messages"] if m["role"] == "assistant"][0]
    assert assistant["content"][0]["name"] == "repo_search_v2"
    # Inbound: the client gets its own spelling back.
    (tc,) = r.json()["choices"][0]["message"]["tool_calls"]
    assert tc["function"]["name"] == "repo.search/v2"


def test_sanitized_name_collision_disambiguated(bridge):
    """"a.b" and "a_b" must not both sanitize to the same upstream name."""
    capture = bridge(_anthropic_tool_use_response())
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_fn_tool("a_b"), _fn_tool("a.b")],
        }
    )
    assert r.status_code == 200
    names = [t["name"] for t in capture.requests[0]["tools"]]
    assert names == ["a_b", "a_b_2"]


def test_sanitized_name_wrapper_collision_rejected(bridge, profiles):
    """A name that only collides with a wrapper tool AFTER sanitization is
    still caught — the guard runs on the names the API will actually see."""
    bridge()
    profiles({"models": [{"match": "claude-haiku-4-5", "add": ["time_calc"]}]})
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_fn_tool("get.current.time")],
        }
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert "get.current.time" in err["message"]


def test_max_completion_tokens_wins_over_max_tokens(bridge):
    capture = bridge(_anthropic_tool_use_response())
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
            "max_tokens": 100,
            "max_completion_tokens": 555,
        }
    )
    assert r.status_code == 200
    assert capture.requests[0]["max_tokens"] == 555


def test_legacy_functions_translated_and_answered_in_kind(bridge):
    """openai<1.0 shape: `functions`/`function_call` in, `message.function_call`
    + finish_reason "function_call" out."""
    capture = bridge(_anthropic_tool_use_response())
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "functions": [
                {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                }
            ],
            "function_call": {"name": "web_search"},
        }
    )
    assert r.status_code == 200
    # Outbound: translated to tools + forced tool_choice.
    sent = capture.requests[0]
    assert sent["tools"][0]["name"] == "web_search"
    assert sent["tool_choice"] == {"type": "tool", "name": "web_search"}
    # Inbound: the legacy response shape, not tool_calls.
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "function_call"
    assert "tool_calls" not in choice["message"]
    fc = choice["message"]["function_call"]
    assert fc["name"] == "web_search"
    assert json.loads(fc["arguments"]) == {"query": "weather in Paris"}


def test_legacy_functions_streaming_uses_function_call_deltas(bridge):
    events = [
        {"type": "message_start",
         "message": {"id": "msg_01", "usage": {"input_tokens": 10, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": "toolu_1", "name": "web_search", "input": {}}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": "{\"query\": \"paris\"}"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]
    bridge(_sse(events))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "session_id": "toolbridge-legacy-stream",
            "model": "claude-haiku-4-5",
            "stream": True,
            "messages": [{"role": "user", "content": "weather?"}],
            "functions": [{"name": "web_search", "parameters": {"type": "object"}}],
        },
    ) as r:
        assert r.status_code == 200
        payloads = [l[5:].strip() for l in r.iter_lines() if l.startswith("data:")]
    chunks = [json.loads(p) for p in payloads if p != "[DONE]"]
    deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
    assert not any(d.get("tool_calls") for d in deltas)
    fc_frames = [d["function_call"] for d in deltas if d.get("function_call")]
    assert fc_frames[0]["name"] == "web_search"
    assert json.loads("".join(f.get("arguments", "") for f in fc_frames)) == {"query": "paris"}
    finish = [
        c["choices"][0]["finish_reason"]
        for c in chunks
        if c.get("choices") and c["choices"][0].get("finish_reason")
    ]
    assert finish == ["function_call"]


def test_preflight_rejects_non_function_tool_type(bridge):
    bridge()
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "web_search", "function": {"name": "x"}}],
        }
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["param"] == "tools[0].type"


def test_preflight_rejects_empty_tool_name(bridge):
    bridge()
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_fn_tool("  ")],
        }
    )
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "tools[0].function.name"


def test_preflight_rejects_bad_tool_choice(bridge):
    bridge()
    base = {
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [WEB_SEARCH_TOOL],
    }
    r = _post({**base, "tool_choice": "always"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "tool_choice"

    # A forced function must be declared in tools.
    r = _post({**base, "tool_choice": {"type": "function", "function": {"name": "nope"}}})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["param"] == "tool_choice.function.name"
    assert "nope" in err["message"]


def test_preflight_rejects_tool_message_without_id(bridge):
    bridge()
    r = _post(
        {
            "model": "claude-haiku-4-5",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "tool", "content": "result"},
            ],
            "tools": [WEB_SEARCH_TOOL],
        }
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["param"] == "messages[1].tool_call_id"


def test_preflight_error_streams_on_the_error_channel(bridge):
    """Streaming requests can't 400 after the head is sent mid-turn, but a
    pre-flight failure surfaces as an OpenAI-shaped SSE error, not a repr."""
    bridge()
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "session_id": "toolbridge-preflight-stream",
            "model": "claude-haiku-4-5",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
            "tool_choice": "always",
        },
    ) as r:
        assert r.status_code == 200
        payloads = [l[5:].strip() for l in r.iter_lines() if l.startswith("data:")]
    chunks = [json.loads(p) for p in payloads if p != "[DONE]"]
    (err,) = [c["error"] for c in chunks if "error" in c]
    assert err["type"] == "invalid_request_error"
    assert "always" in err["message"] and "{" not in err["message"]
