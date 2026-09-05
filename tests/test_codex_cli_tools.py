"""Client-declared function calling served by the codex CLI (src/codex_cli_tools.py).

The path a ChatGPT-plan deployment uses: no OpenAI Platform key anywhere, the
tool contract carried in the prompt and parsed back out of codex's reply.

Named test_codex_cli_tools so it collects after test_budget.py (see the
test_sandbox_shim docstring for the module-ordering constraint).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="claude-wrapper-test-codex-cli-tools-")
os.environ.setdefault("CLAUDE_WRAPPER_DATA", _TMP)
os.environ.setdefault("CLAUDE_WRAPPER_MODEL_DISCOVERY", "off")
os.environ.pop("CLAUDE_WRAPPER_API_KEYS", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src import codex_cli_tools, config  # noqa: E402
from src import main as src_main  # noqa: E402
from src.agent_runner import AgentResult  # noqa: E402
from src.deps import RUNNER  # noqa: E402
from src.main import app  # noqa: E402
from src.models import ChatCompletionRequest  # noqa: E402

client = TestClient(app)
_seq = 0

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _post(body: dict):
    global _seq
    _seq += 1
    return client.post("/v1/chat/completions", json={"session_id": f"clitools-{_seq}", **body})


@pytest.fixture
def codex_mode(monkeypatch):
    """agent=codex with no Platform credential — the plan-only deployment."""
    saved = config._supported_models_cache
    for mod in (src_main, config):
        monkeypatch.setattr(mod, "SETTINGS", dataclasses.replace(mod.SETTINGS, agent="codex"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_WRAPPER_CODEX_TOOL_MODE", raising=False)
    monkeypatch.setattr(
        src_main.openai_bridge, "has_platform_credential", lambda: False
    )
    yield
    config._supported_models_cache = saved


@contextlib.contextmanager
def _cli_reply(text: str):
    """Stub the runner turn, capturing the prompt the loop built."""
    seen: dict = {}

    async def _stub(prompt, session_key, **kwargs):
        seen["prompt"] = prompt
        seen["kwargs"] = kwargs
        return AgentResult(
            session_uuid="u", final_text=text, input_tokens=11, output_tokens=7
        )

    had_own = "run_collect" in RUNNER.__dict__
    prev = RUNNER.__dict__.get("run_collect")
    RUNNER.run_collect = _stub
    try:
        yield seen
    finally:
        if had_own:
            RUNNER.run_collect = prev
        else:
            del RUNNER.run_collect


# ---------- protocol construction ----------


def test_protocol_lists_declared_tools_and_forced_choice():
    req = ChatCompletionRequest(
        model="gpt-5.2",
        messages=[{"role": "user", "content": "hi"}],
        tools=[WEB_SEARCH_TOOL],
        tool_choice={"type": "function", "function": {"name": "web_search"}},
    )
    block = codex_cli_tools.build_tool_protocol(req)
    assert "web_search" in block and "Search the web" in block
    assert '"query"' in block  # the schema itself, so arguments can be shaped
    assert "MUST call the tool `web_search`" in block


def test_protocol_encodes_none_and_required_choices():
    base = dict(model="gpt-5.2", messages=[{"role": "user", "content": "hi"}],
                tools=[WEB_SEARCH_TOOL])
    assert "Do NOT call any tool" in codex_cli_tools.build_tool_protocol(
        ChatCompletionRequest(**base, tool_choice="none")
    )
    assert "MUST call at least one tool" in codex_cli_tools.build_tool_protocol(
        ChatCompletionRequest(**base, tool_choice="required")
    )


# ---------- envelope parsing ----------


def test_parse_envelope_shapes():
    d = frozenset({"web_search"})
    p = codex_cli_tools.parse_envelope

    content, calls = p('{"tool_calls":[{"name":"web_search","arguments":{"query":"paris"}}]}', d)
    assert content is None and len(calls) == 1
    assert calls[0]["type"] == "function" and calls[0]["id"].startswith("call_")
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "paris"}

    # Fenced output, the most common deviation from "no code fence".
    assert p('```json\n{"content":"sunny"}\n```', d) == ("sunny", None)

    # OpenAI-native nesting, and arguments already a JSON string.
    _, calls = p('{"tool_calls":[{"function":{"name":"web_search","arguments":"{\\"q\\":1}"}}]}', d)
    assert calls[0]["function"]["arguments"] == '{"q":1}'

    # A hallucinated tool the client never declared must not reach it.
    content, calls = p('{"tool_calls":[{"name":"rm_rf","arguments":{}}]}', d)
    assert calls is None and "rm_rf" in content

    # Unparseable output degrades to a plain answer, never an error.
    assert p("just prose", d) == ("just prose", None)

    # Non-object arguments still become valid JSON so clients can parse them.
    _, calls = p('{"tool_calls":[{"name":"web_search","arguments":"paris"}]}', d)
    assert json.loads(calls[0]["function"]["arguments"]) == {"input": "paris"}


# ---------- end to end through the API ----------


def test_tools_request_returns_tool_calls_without_any_api_key(codex_mode):
    with _cli_reply('{"tool_calls":[{"name":"web_search","arguments":{"query":"paris"}}]}') as seen:
        r = _post({
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "weather in paris?"}],
            "tools": [WEB_SEARCH_TOOL],
        })
    assert r.status_code == 200, r.text
    body = r.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "web_search"
    assert json.loads(call["function"]["arguments"]) == {"query": "paris"}
    # The protocol rode in the prompt, and the user's question came with it.
    assert "Tool-calling protocol" in seen["prompt"]
    assert "weather in paris?" in seen["prompt"]
    # Clarify must be off: it competes with the protocol for the reply shape.
    assert seen["kwargs"]["clarify"] is False
    assert body["usage"]["prompt_tokens"] == 11


def test_direct_answer_returns_plain_content(codex_mode):
    with _cli_reply('{"content":"It is sunny in Paris."}'):
        r = _post({
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [WEB_SEARCH_TOOL],
        })
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "It is sunny in Paris."
    assert choice["message"].get("tool_calls") is None


def test_tool_results_close_the_loop(codex_mode):
    # Second leg: the client executed the tool and sent the result back.
    with _cli_reply('{"content":"Paris is 18C."}') as seen:
        r = _post({
            "model": "gpt-5.2",
            "messages": [
                {"role": "user", "content": "weather in paris?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "web_search", "arguments": '{"query":"paris"}'}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "Paris: 18C, clear"},
                {"role": "user", "content": "so?"},
            ],
            "tools": [WEB_SEARCH_TOOL],
        })
    assert r.json()["choices"][0]["message"]["content"] == "Paris is 18C."
    # The tool result reached the CLI — otherwise the model answers blind.
    assert "18C, clear" in seen["prompt"]


def test_streaming_emits_tool_calls_then_done(codex_mode):
    with _cli_reply('{"tool_calls":[{"name":"web_search","arguments":{"query":"x"}}]}'):
        r = client.post("/v1/chat/completions", json={
            "session_id": "clitools-stream",
            "model": "gpt-5.2",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
        })
    assert r.status_code == 200
    payloads = [
        line[len("data: "):]
        for line in r.text.splitlines()
        if line.startswith("data: ")
    ]
    assert payloads[-1] == "[DONE]"
    frames = [json.loads(p) for p in payloads[:-1]]
    assert frames[0]["choices"][0]["delta"] == {"role": "assistant"}
    tc = frames[1]["choices"][0]["delta"]["tool_calls"][0]
    assert tc["index"] == 0 and tc["function"]["name"] == "web_search"
    assert frames[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_mode_bridge_forces_the_passthrough(codex_mode, monkeypatch):
    # An operator who wants native function calling must still get the
    # passthrough's credential error, not a silent downgrade to the CLI loop.
    monkeypatch.setenv("CLAUDE_WRAPPER_CODEX_TOOL_MODE", "bridge")
    r = _post({
        "model": "gpt-5.2",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [WEB_SEARCH_TOOL],
    })
    assert r.status_code == 502
    assert "OPENAI_API_KEY" in r.json()["error"]["message"]


def test_mode_cli_used_even_when_a_key_exists(codex_mode, monkeypatch):
    monkeypatch.setenv("CLAUDE_WRAPPER_CODEX_TOOL_MODE", "cli")
    monkeypatch.setattr(src_main.openai_bridge, "has_platform_credential", lambda: True)
    with _cli_reply('{"content":"cli served this"}'):
        r = _post({
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
        })
    assert r.json()["choices"][0]["message"]["content"] == "cli served this"


def test_invalid_mode_fails_closed(monkeypatch):
    monkeypatch.setenv("CLAUDE_WRAPPER_CODEX_TOOL_MODE", "sideways")
    with pytest.raises(ValueError) as e:
        config.codex_tool_mode()
    assert "CLAUDE_WRAPPER_CODEX_TOOL_MODE" in str(e.value)
