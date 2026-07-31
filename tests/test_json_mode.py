"""JSON mode (response_format) tests.

Structured-output clients — Vane and anything else built on the Vercel AI
SDK's generateObject — send ``response_format`` and JSON.parse the returned
content verbatim; a ```json fence or surrounding prose breaks them (see
https://github.com/ItzCrazyKns/Vane/issues/959). These tests prove the
contract: the output-format instruction reaches the prompt, fenced or
prose-wrapped replies are reduced to raw JSON on both the sync and streaming
paths, reasoning/progress frames never enter a JSON content stream, and
requests *without* response_format behave exactly as before.

The runner is stubbed, so no real Claude Code subprocess is launched.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# ---- environment setup before importing anything from src ----
_TMP = tempfile.mkdtemp(prefix="claude-wrapper-jsonmode-")
os.environ["CLAUDE_WRAPPER_DATA"] = _TMP
os.environ["CLAUDE_WRAPPER_DEFAULT_MODEL"] = "claude-opus-4-8"
os.environ["CLAUDE_WRAPPER_MODEL_DISCOVERY"] = "off"
os.environ.pop("CLAUDE_WRAPPER_API_KEYS", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- stub ClaudeRunner before importing the FastAPI app ----
from src.claude_runner import ClaudeResult, ClaudeRunner, StreamEvent  # noqa: E402

# Per-test knobs: what the stub model replies, and what prompt it received.
_STATE = {
    "final_text": "",
    "stream_texts": [],
    "last_prompt": "",
}


async def _stub_run_collect(self, prompt, session_key, model=None, effort=None, **_kwargs):
    _STATE["last_prompt"] = prompt
    return ClaudeResult(
        session_uuid="stub-uuid",
        final_text=_STATE["final_text"],
        stop_reason="stop",
        input_tokens=1,
        output_tokens=1,
        events=[],
    )


async def _stub_run_stream(self, prompt, session_key, model=None, effort=None, **_kwargs):
    _STATE["last_prompt"] = prompt
    yield StreamEvent(kind="thinking", text="pondering the JSON...")
    for piece in _STATE["stream_texts"]:
        yield StreamEvent(kind="text", text=piece)
    yield StreamEvent(
        kind="final",
        raw={"stop_reason": "stop", "new_outputs": [], "input_tokens": 1, "output_tokens": 1},
    )


ClaudeRunner.run_collect = _stub_run_collect
ClaudeRunner.run_stream = _stub_run_stream

# ---- now import the app ----
from fastapi.testclient import TestClient  # noqa: E402

from src.main import app, extract_raw_json  # noqa: E402

client = TestClient(app)

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


def _chat(messages, **extra):
    body = {"model": "claude-opus-4-8", "messages": messages, **extra}
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200, r.text
    return r


def _parse_sse(body: str):
    contents, reasonings = [], []
    finish = None
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload.strip() == "[DONE]":
            continue
        obj = json.loads(payload)
        for choice in obj.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                contents.append(delta["content"])
            if delta.get("reasoning_content"):
                reasonings.append(delta["reasoning_content"])
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return "".join(contents), "".join(reasonings), finish


# ---------- extract_raw_json unit cases ----------


def test_extract_raw_json() -> None:
    raw = '{"suggestions": ["a", "b"]}'
    check("extract.asis", extract_raw_json(raw) == raw)
    check("extract.fenced", extract_raw_json(f"```json\n{raw}\n```") == raw)
    check("extract.fenced_nolang", extract_raw_json(f"```\n{raw}\n```") == raw)
    check(
        "extract.preamble_fence",
        extract_raw_json(f"Here is the JSON you asked for:\n\n```json\n{raw}\n```\nHope that helps!") == raw,
    )
    check("extract.bare_preamble", extract_raw_json(f"Sure! {raw}") == raw)
    check("extract.array", extract_raw_json('```json\n[1, 2, 3]\n```') == "[1, 2, 3]")
    check("extract.nested_braces", extract_raw_json('note {not json} then {"a": {"b": 1}}') == '{"a": {"b": 1}}')
    check("extract.no_json", extract_raw_json("I could not produce JSON, sorry.") is None)
    check("extract.empty", extract_raw_json("") is None)


# ---------- sync path ----------


def test_sync_json_object_strips_fences() -> None:
    _STATE["final_text"] = '```json\n{"suggestions": ["x", "y"]}\n```'
    r = _chat(
        [{"role": "user", "content": "suggest things (sync json test)"}],
        response_format={"type": "json_object"},
    )
    content = r.json()["choices"][0]["message"]["content"]
    check("sync.raw_json", json.loads(content) == {"suggestions": ["x", "y"]}, note=content)
    check("sync.instruction_in_prompt", "Output format" in _STATE["last_prompt"])


def test_sync_json_schema_in_prompt() -> None:
    _STATE["final_text"] = '{"suggestions": []}'
    schema = {
        "type": "object",
        "properties": {"suggestions": {"type": "array", "items": {"type": "string"}}},
        "required": ["suggestions"],
    }
    _chat(
        [{"role": "user", "content": "suggest things (schema test)"}],
        response_format={"type": "json_schema", "json_schema": {"name": "suggestions", "schema": schema, "strict": True}},
    )
    check("schema.instruction", "JSON Schema" in _STATE["last_prompt"])
    check("schema.body_in_prompt", '"suggestions"' in _STATE["last_prompt"])


def test_sync_json_unparseable_passthrough() -> None:
    _STATE["final_text"] = "I could not produce JSON, sorry."
    r = _chat(
        [{"role": "user", "content": "suggest things (unparseable test)"}],
        response_format={"type": "json_object"},
    )
    content = r.json()["choices"][0]["message"]["content"]
    check("sync.passthrough", content == "I could not produce JSON, sorry.", note=content)


def test_sync_without_response_format_untouched() -> None:
    fenced = 'Here you go:\n```json\n{"a": 1}\n```'
    _STATE["final_text"] = fenced
    r = _chat([{"role": "user", "content": "normal chat (no json mode)"}])
    content = r.json()["choices"][0]["message"]["content"]
    check("plain.content_verbatim", content == fenced, note=content)
    check("plain.no_instruction", "Output format" not in _STATE["last_prompt"])


def test_response_format_text_is_not_json_mode() -> None:
    fenced = '```json\n{"a": 1}\n```'
    _STATE["final_text"] = fenced
    r = _chat(
        [{"role": "user", "content": "text response_format test"}],
        response_format={"type": "text"},
    )
    content = r.json()["choices"][0]["message"]["content"]
    check("text_mode.content_verbatim", content == fenced, note=content)


# ---------- streaming path ----------


def test_stream_json_buffers_and_strips() -> None:
    # Fence split across chunk boundaries: only whole-stream stripping works.
    _STATE["stream_texts"] = ['```json\n{"suggestions": [', '"x", "y"]}\n```']
    r = _chat(
        [{"role": "user", "content": "suggest things (stream json test)"}],
        response_format={"type": "json_object"},
        stream=True,
    )
    content, reasoning, finish = _parse_sse(r.text)
    check("stream.raw_json", json.loads(content) == {"suggestions": ["x", "y"]}, note=content)
    check("stream.no_reasoning_field", reasoning == "")
    check("stream.no_details_block", "<details" not in content and "<think>" not in content)
    check("stream.finish", finish == "stop")


def test_stream_without_response_format_untouched() -> None:
    _STATE["stream_texts"] = ["hello ", "stream"]
    r = _chat(
        [{"role": "user", "content": "normal chat (stream, no json mode)"}],
        stream=True,
    )
    content, _reasoning, finish = _parse_sse(r.text)
    # Default reasoning channel ("details") wraps thinking inside content —
    # exactly the pre-existing behavior JSON mode must not disturb.
    check("stream.plain_answer", content.endswith("hello stream"), note=content)
    check("stream.plain_reasoning_kept", "<details" in content, note=content)
    check("stream.plain_finish", finish == "stop")


# ---------- runner ----------


def main() -> int:
    tests = [
        test_extract_raw_json,
        test_sync_json_object_strips_fences,
        test_sync_json_schema_in_prompt,
        test_sync_json_unparseable_passthrough,
        test_sync_without_response_format_untouched,
        test_response_format_text_is_not_json_mode,
        test_stream_json_buffers_and_strips,
        test_stream_without_response_format_untouched,
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
