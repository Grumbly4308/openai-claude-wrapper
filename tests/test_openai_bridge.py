"""Function-calling (tools) tests for the OpenAI passthrough bridge.

The OpenAI chat.completions API is mocked with httpx.MockTransport, mirroring
test_tool_bridge.py's harness, so every test also captures the exact outbound
payload. What is pinned here is the *pure-proxy* contract: the wrapper's own
extension fields are stripped and nothing else is touched — no history
translation, no name sanitization, no synthetic stream chunks. The main.py
module switch (SETTINGS.agent=codex → openai_bridge, tools-less → RUNNER) and
the owned_by wiring are pinned end-to-end at the bottom.

Named test_openai_bridge so it collects after test_budget.py — the suite
relies on test_budget being the first module to import src.config (see the
test_sandbox_shim docstring).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# ---- environment setup before importing anything from src ----
_TMP = tempfile.mkdtemp(prefix="claude-wrapper-test-openai-bridge-")
os.environ.setdefault("CLAUDE_WRAPPER_DATA", _TMP)
os.environ.setdefault("CLAUDE_WRAPPER_DEFAULT_MODEL", "claude-sonnet-4-6")
os.environ.setdefault("CLAUDE_WRAPPER_MODEL_DISCOVERY", "off")
os.environ.pop("CLAUDE_WRAPPER_API_KEYS", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src import config
from src import main as src_main
from src import openai_bridge, tool_bridge
from src.claude_runner import ClaudeResult
from src.deps import RUNNER
from src.main import app
from src.models import ChatCompletionRequest

client = TestClient(app)

_seq = 0


def _post(body: dict):
    """POST a chat completion with a unique session_id per request.

    Same reasoning as test_tool_bridge: the usage ledger is a process-wide
    singleton test_budget.py enables with a tiny block, so a reused session
    would hit the budget checkpoint instead of the bridge.
    """
    global _seq
    _seq += 1
    return client.post("/v1/chat/completions", json={"session_id": f"oaibridge-{_seq}", **body})


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


def _openai_tool_call_response() -> dict:
    return {
        "id": "chatcmpl-up1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.2",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": "{\"query\": \"weather in Paris\"}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
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
        if isinstance(body, Exception):
            raise body
        if isinstance(body, httpx.Response):
            return body
        if isinstance(body, bytes):
            return httpx.Response(
                200, content=body, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=body)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """The env key outranks everything, so start every test without one."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def bridge(monkeypatch):
    """Install a capture transport and an env API key; restore afterwards."""

    def _install(*responses):
        capture = _Capture(responses)
        monkeypatch.setattr(
            openai_bridge, "_client", httpx.AsyncClient(transport=httpx.MockTransport(capture))
        )
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        return capture

    yield _install
    openai_bridge._client = None


@pytest.fixture
def codex_mode(monkeypatch):
    """SETTINGS.agent=codex on the modules that hold refs to it.

    The model-list cache is snapshotted and restored: a /v1/models call in
    here must not memoize the codex list into later-sorting claude-mode tests
    (or vice versa when this module runs standalone).
    """
    saved_cache = config._supported_models_cache
    for mod in (src_main, config):
        monkeypatch.setattr(mod, "SETTINGS", dataclasses.replace(mod.SETTINGS, agent="codex"))
    yield
    config._supported_models_cache = saved_cache


# ---------- auth resolution ----------


def _auth_json(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(data))
    return path


def test_env_key_wins_over_apikey_file(monkeypatch, tmp_path):
    path = _auth_json(tmp_path, {"auth_mode": "apikey", "OPENAI_API_KEY": "sk-file"})
    monkeypatch.setattr(openai_bridge, "_CODEX_CREDENTIALS_FILE", path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert openai_bridge.resolve_auth() == {"authorization": "Bearer sk-env"}


def test_blank_env_key_falls_through_to_apikey_file(monkeypatch, tmp_path):
    """Compose always delivers the var, usually as "" — that is not a key."""
    path = _auth_json(tmp_path, {"auth_mode": "apikey", "OPENAI_API_KEY": "sk-file"})
    monkeypatch.setattr(openai_bridge, "_CODEX_CREDENTIALS_FILE", path)
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    assert openai_bridge.resolve_auth() == {"authorization": "Bearer sk-file"}


def test_chatgpt_only_login_is_502_naming_both_remedies(monkeypatch, tmp_path):
    """Plan tokens cannot call the Platform API, so there is no fallback —
    the error names both working options AND the unusable login it found."""
    path = _auth_json(
        tmp_path, {"auth_mode": "chatgpt", "tokens": {"access_token": "tok"}}
    )
    monkeypatch.setattr(openai_bridge, "_CODEX_CREDENTIALS_FILE", path)
    with pytest.raises(HTTPException) as exc:
        openai_bridge.resolve_auth()
    assert exc.value.status_code == 502
    err = exc.value.detail["error"]
    assert err["code"] == "no_upstream_credential" and err["type"] == "api_error"
    assert "OPENAI_API_KEY" in err["message"]
    assert "codex login --with-api-key" in err["message"]
    assert str(path) in err["message"] and "not usable here" in err["message"]


def test_no_credential_at_all_is_502(monkeypatch, tmp_path):
    monkeypatch.setattr(openai_bridge, "_CODEX_CREDENTIALS_FILE", tmp_path / "absent.json")
    with pytest.raises(HTTPException) as exc:
        openai_bridge.resolve_auth()
    assert exc.value.status_code == 502
    err = exc.value.detail["error"]
    assert err["code"] == "no_upstream_credential"
    # No phantom login is reported when the file holds none.
    assert "was found" not in err["message"]


# ---------- payload build (through the live endpoint, agent=codex) ----------


def test_outbound_key_set_pinned_and_wrapper_fields_stripped(codex_mode, bridge):
    """The exact top-level key set is pinned: the wrapper extensions are
    stripped (inline_generated_files has a non-None default that would 400
    upstream) and NOTHING else leaks or is dropped."""
    capture = bridge(_openai_tool_call_response())
    r = _post(
        {
            "model": "gpt-5.2",
            "temperature": 0,
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": [WEB_SEARCH_TOOL],
            "inline_generated_files": True,
            "clarify": False,
        }
    )
    assert r.status_code == 200
    (sent,) = capture.requests
    assert set(sent) == {"model", "messages", "stream", "tools", "temperature"}
    assert sent["model"] == "gpt-5.2"
    assert sent["stream"] is False
    assert sent["tools"] == [WEB_SEARCH_TOOL]
    assert capture.last_headers["authorization"] == "Bearer sk-test"

    # The 200 → BridgeResult mapping: tool_calls verbatim, ids/arguments
    # untouched, arguments still a string; usage carried over; the wrapper's
    # envelope extras (session_id, effort) re-added by main as with claude.
    body = r.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert (
        choice["message"]["tool_calls"]
        == _openai_tool_call_response()["choices"][0]["message"]["tool_calls"]
    )
    assert body["usage"]["total_tokens"] == 120
    assert body["session_id"]
    assert body["effort"]["source"] == "tool-bridge"


def test_reasoning_effort_present_iff_suffix_given(codex_mode, bridge):
    """Only an explicit suffix maps to reasoning_effort — the server-default
    effort must never poison the payload — and `:none` is honored too."""
    capture = bridge(
        _openai_tool_call_response(),
        _openai_tool_call_response(),
        _openai_tool_call_response(),
    )
    base = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [WEB_SEARCH_TOOL],
    }
    r = _post({**base, "model": "gpt-5.2:high"})
    assert capture.requests[0]["model"] == "gpt-5.2"
    assert capture.requests[0]["reasoning_effort"] == "high"
    # The non-streaming path re-derives effort inside the bridge, so the
    # envelope's claim and the payload agree.
    assert r.json()["effort"] == {
        "applied": "high",
        "source": "tool-bridge",
        "requested": "high",
    }

    _post({**base, "model": "gpt-5.2:none"})
    assert capture.requests[1]["reasoning_effort"] == "none"

    r = _post({**base, "model": "gpt-5.2"})
    assert "reasoning_effort" not in capture.requests[2]
    assert r.json()["effort"]["applied"] == "api-default"


def test_response_format_and_tool_choice_pass_through_verbatim(codex_mode, bridge):
    capture = bridge(_openai_tool_call_response())
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "result",
            "schema": {"type": "object", "properties": {"answer": {"type": "number"}}},
        },
    }
    tool_choice = {"type": "function", "function": {"name": "web_search"}}
    r = _post(
        {
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
            "tool_choice": tool_choice,
            "response_format": response_format,
        }
    )
    assert r.status_code == 200
    (sent,) = capture.requests
    assert sent["tool_choice"] == tool_choice
    assert sent["response_format"] == response_format


def test_legacy_functions_forwarded_and_answered_in_kind(codex_mode, bridge):
    """The legacy parameter family goes upstream as received — NOT alongside
    the tools/tool_choice the request validator synthesized from it — and the
    legacy response shape comes back through main's existing envelope path."""
    legacy_response = _openai_tool_call_response()
    legacy_response["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "function_call": {"name": "web_search", "arguments": "{\"query\": \"paris\"}"},
    }
    legacy_response["choices"][0]["finish_reason"] = "function_call"
    capture = bridge(legacy_response)
    r = _post(
        {
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "weather?"}],
            "functions": [
                {"name": "web_search", "parameters": {"type": "object", "properties": {}}}
            ],
            "function_call": {"name": "web_search"},
        }
    )
    assert r.status_code == 200
    (sent,) = capture.requests
    assert "functions" in sent and "function_call" in sent
    assert "tools" not in sent and "tool_choice" not in sent
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "function_call"
    assert "tool_calls" not in choice["message"]
    assert choice["message"]["function_call"] == {
        "name": "web_search",
        "arguments": "{\"query\": \"paris\"}",
    }


# ---------- complete(): status-mapping ladder ----------


def test_upstream_400_envelope_passes_through(codex_mode, bridge):
    """Request-shape errors are upstream's to explain — status and envelope
    reach the client verbatim (the honor-contracts rule)."""
    bridge(
        httpx.Response(
            400,
            json={
                "error": {
                    "message": "bad temperature",
                    "type": "invalid_request_error",
                    "param": "temperature",
                    "code": None,
                }
            },
        )
    )
    r = _post(
        {
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
        }
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["message"] == "bad temperature" and err["param"] == "temperature"


def test_upstream_401_becomes_fixed_message_502(codex_mode, bridge, caplog):
    """OpenAI 401 bodies echo a redacted rendering of the presented key; that
    belongs to the operator's log, never to a wrapper tenant."""
    bridge(
        httpx.Response(
            401, json={"error": {"message": "Incorrect API key provided: sk-oops-123"}}
        )
    )
    with caplog.at_level(logging.ERROR, logger="claude_wrapper.openai_bridge"):
        r = _post(
            {
                "model": "gpt-5.2",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [WEB_SEARCH_TOOL],
            }
        )
    assert r.status_code == 502
    err = r.json()["error"]
    assert err["message"] == openai_bridge._CREDENTIAL_REJECTED_MSG
    assert "sk-oops-123" not in err["message"]
    assert err["type"] == "api_error" and err["code"] == "upstream_error"
    # ...but the body did reach the server log.
    assert "sk-oops-123" in caplog.text


def test_upstream_5xx_wrapped_as_502(codex_mode, bridge):
    bridge(httpx.Response(500, text="upstream exploded"))
    r = _post(
        {
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
        }
    )
    assert r.status_code == 502
    err = r.json()["error"]
    assert err["message"] == "openai api error 500: upstream exploded"
    assert err["code"] == "upstream_error"


def test_connect_error_is_502_unreachable(codex_mode, bridge):
    bridge(httpx.ConnectError("connection refused"))
    r = _post(
        {
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
        }
    )
    assert r.status_code == 502
    assert "openai api unreachable" in r.json()["error"]["message"]


# ---------- stream(): pure proxy ----------


def _chunk_base() -> dict:
    return {
        "id": "chatcmpl-up1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-5.2",
    }


def _upstream_chunks(include_usage_chunk: bool = True) -> list[dict]:
    base = _chunk_base()
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                              "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {"content": "Searching"},
                              "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function",
             "function": {"name": "web_search", "arguments": ""}}]},
            "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "{\"query\": \"paris\"}"}}]},
            "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    if include_usage_chunk:
        chunks.append({**base, "choices": [],
                       "usage": {"prompt_tokens": 50, "completion_tokens": 30,
                                 "total_tokens": 80}})
    return chunks


def _sse_bytes(chunks: list[dict]) -> bytes:
    out = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (out + "data: [DONE]\n\n").encode()


def _collect_stream(body: dict, on_usage=None) -> tuple[str, list[str]]:
    """Drive openai_bridge.stream directly; returns (raw text, data payloads)."""
    req = ChatCompletionRequest(**body)
    run_model, effort = config.split_model_effort(req.model)
    effort_info = {"applied": "api-default", "source": "tool-bridge", "requested": effort}

    async def _go():
        out = []
        async for chunk in openai_bridge.stream(
            req, run_model, req.model, "sess-1", effort_info, on_usage=on_usage
        ):
            out.append(chunk)
        return out

    raw = b"".join(asyncio.run(_go())).decode("utf-8")
    payloads = [l[5:].strip() for l in raw.split("\n\n") if l.startswith("data:")]
    return raw, payloads


_STREAM_BODY = {
    "model": "gpt-5.2",
    "stream": True,
    "messages": [{"role": "user", "content": "weather?"}],
    "tools": [WEB_SEARCH_TOOL],
}


def test_stream_forwards_upstream_chunks_verbatim(bridge):
    """Preamble + upstream chunks byte-for-byte + exactly one [DONE]. No
    synthetic first chunk: the first data frame is upstream's own (its id),
    and no frame carries the wrapper's session_id/effort extras."""
    chunks = _upstream_chunks()
    capture = bridge(_sse_bytes(chunks))
    recorded: list[tuple[int, int]] = []

    async def _rec(i: int, o: int) -> None:
        recorded.append((i, o))

    raw, payloads = _collect_stream(_STREAM_BODY, on_usage=_rec)
    assert raw.startswith(": ")  # proxy-buffer preamble first
    assert payloads[-1] == "[DONE]" and payloads.count("[DONE]") == 1
    # Verbatim: the forwarded frames are the upstream lines, not
    # re-serializations — minus the injected usage chunk, which is consumed.
    assert payloads[:-1] == [json.dumps(c) for c in chunks[:-1]]
    assert "session_id" not in raw and "effort" not in raw
    # The wrapper asked for usage on the ledger's behalf and recorded it.
    assert capture.requests[0]["stream_options"] == {"include_usage": True}
    assert recorded == [(50, 30)]


def test_stream_client_requested_usage_chunk_is_forwarded(bridge):
    chunks = _upstream_chunks()
    capture = bridge(_sse_bytes(chunks))
    recorded: list[tuple[int, int]] = []

    async def _rec(i: int, o: int) -> None:
        recorded.append((i, o))

    body = {**_STREAM_BODY, "stream_options": {"include_usage": True}}
    _, payloads = _collect_stream(body, on_usage=_rec)
    # The client asked for the chunk itself, so it goes through untouched...
    assert payloads[:-1] == [json.dumps(c) for c in chunks]
    # ...and the ledger still records from it.
    assert recorded == [(50, 30)]
    assert capture.requests[0]["stream_options"] == {"include_usage": True}


def test_stream_options_merged_not_clobbered(bridge):
    # The wrapper's piggybacked include_usage must not discard the client's
    # other stream_options keys.
    capture = bridge(_sse_bytes(_upstream_chunks()))

    async def _rec(i: int, o: int) -> None:
        pass

    body = {**_STREAM_BODY, "stream_options": {"include_obfuscation": False}}
    _collect_stream(body, on_usage=_rec)
    assert capture.requests[0]["stream_options"] == {
        "include_obfuscation": False,
        "include_usage": True,
    }


def test_stream_upstream_500_surfaces_on_the_error_channel(bridge):
    bridge(httpx.Response(500, text="upstream exploded"))
    _, payloads = _collect_stream(_STREAM_BODY)
    assert payloads[-1] == "[DONE]"
    (err,) = [json.loads(p)["error"] for p in payloads[:-1] if "error" in p]
    assert err["message"] == "openai api error 500: upstream exploded"
    assert err["type"] == "upstream_error"


def test_stream_upstream_401_emits_the_fixed_message(bridge, caplog):
    bridge(
        httpx.Response(
            401, json={"error": {"message": "Incorrect API key provided: sk-oops-456"}}
        )
    )
    with caplog.at_level(logging.ERROR, logger="claude_wrapper.openai_bridge"):
        raw, payloads = _collect_stream(_STREAM_BODY)
    (err,) = [json.loads(p)["error"] for p in payloads if p != "[DONE]"]
    assert err["message"] == openai_bridge._CREDENTIAL_REJECTED_MSG
    assert "sk-oops-456" not in raw
    assert "sk-oops-456" in caplog.text


def test_stream_end_to_end_has_no_synthetic_first_chunk(codex_mode, bridge):
    """Through the live endpoint: main routes the stream to this bridge and
    the first data frame the client sees is upstream's own chunk."""
    chunks = _upstream_chunks()
    bridge(_sse_bytes(chunks))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"session_id": "oaibridge-stream-e2e", **_STREAM_BODY},
    ) as r:
        assert r.status_code == 200
        payloads = [l[5:].strip() for l in r.iter_lines() if l.startswith("data:")]
    assert payloads[-1] == "[DONE]"
    first = json.loads(payloads[0])
    assert first["id"] == "chatcmpl-up1"
    assert "session_id" not in first


# ---------- capability profiles gate the bridge ----------


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


def test_client_tools_denied_by_profile(codex_mode, bridge, profiles):
    """The one non-proxy gate: same 400 as the claude bridge, same param."""
    bridge(_openai_tool_call_response())
    profiles({"models": [{"match": "gpt-5.2", "remove": ["client_tools"]}]})
    r = _post(
        {
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
        }
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error" and err["param"] == "tools"
    assert "client_tools" in err["message"] and "web_search" in err["message"]


def test_codex_tuned_ids_rejected_for_tools_requests(codex_mode, bridge):
    """*-codex ids are Responses-API-only upstream: a tools request on one must
    die as a clear 400 here, not as an opaque upstream-404 passthrough."""
    capture = bridge(_openai_tool_call_response())
    for model in ("gpt-5.2-codex", "gpt-5.3-codex-spark:high"):
        r = _post(
            {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [WEB_SEARCH_TOOL],
            }
        )
        assert r.status_code == 400
        err = r.json()["error"]
        assert err["param"] == "model" and err["code"] == "model_not_bridge_capable"
        assert "Responses API" in err["message"]
    assert capture.requests == []  # nothing reached upstream


def test_codex_tuned_gate_skipped_on_custom_base_url(codex_mode, bridge, monkeypatch):
    """A custom OPENAI_BASE_URL backend may well serve codex ids on
    chat.completions — the gate is scoped to the real api.openai.com."""
    capture = bridge(_openai_tool_call_response())
    monkeypatch.setattr(openai_bridge, "OPENAI_BASE_URL", "http://vllm.internal:8000")
    r = _post(
        {
            "model": "gpt-5.2-codex",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
        }
    )
    assert r.status_code == 200
    assert capture.requests[0]["model"] == "gpt-5.2-codex"


# ---------- main.py routing + owned_by under agent=codex ----------


@contextlib.contextmanager
def _stub_runner(final_text="agentic ok"):
    """Replace RUNNER.run_collect for the duration, as instance attribute —
    same reasoning as test_tool_bridge (class-level patches from other modules
    could shadow, or be shadowed by, a class-level patch here)."""

    async def _stub_run_collect(prompt, session_key, **_kwargs):
        return ClaudeResult(
            session_uuid="u", final_text=final_text, input_tokens=1, output_tokens=1
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


def test_codex_agent_routes_tools_requests_to_the_openai_bridge(codex_mode, monkeypatch):
    calls: list[str] = []

    async def _stub_complete(req, run_model, session_key=""):
        calls.append(run_model)
        return openai_bridge.BridgeResult(
            content="bridged", tool_calls=None, finish_reason="stop",
            input_tokens=1, output_tokens=1,
        )

    async def _wrong_bridge(*_a, **_k):  # pragma: no cover - failure path
        raise AssertionError("the anthropic bridge must not serve agent=codex")

    monkeypatch.setattr(openai_bridge, "complete", _stub_complete)
    monkeypatch.setattr(tool_bridge, "complete", _wrong_bridge)
    r = _post(
        {
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [WEB_SEARCH_TOOL],
        }
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "bridged"
    assert calls == ["gpt-5.2"]


def test_codex_agent_toolless_requests_stay_on_the_runner(codex_mode, monkeypatch):
    async def _no_bridge(*_a, **_k):  # pragma: no cover - failure path
        raise AssertionError("a tool-less request must not touch the bridge")

    monkeypatch.setattr(openai_bridge, "complete", _no_bridge)
    with _stub_runner():
        r = _post({"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == "agentic ok"
    assert "tool_calls" not in msg


def test_models_owned_by_openai_under_codex(codex_mode):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data and all(m["owned_by"] == "openai" for m in data)
