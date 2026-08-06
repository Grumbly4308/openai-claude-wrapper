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

# Per-test knobs: what the stub model replies, what prompt it received, and
# which files the run "generated" (paths that must exist on disk — the wrapper
# skips anything it cannot stat).
_STATE = {
    "final_text": "",
    "stream_texts": [],
    "last_prompt": "",
    "new_outputs": [],
}


async def _stub_run_collect(self, prompt, session_key, model=None, effort=None, **_kwargs):
    _STATE["last_prompt"] = prompt
    events = []
    if _STATE["new_outputs"]:
        events.append(StreamEvent(kind="system", raw={"new_outputs": list(_STATE["new_outputs"])}))
    return ClaudeResult(
        session_uuid="stub-uuid",
        final_text=_STATE["final_text"],
        stop_reason="stop",
        input_tokens=1,
        output_tokens=1,
        events=events,
    )


async def _stub_run_stream(self, prompt, session_key, model=None, effort=None, **_kwargs):
    _STATE["last_prompt"] = prompt
    yield StreamEvent(kind="thinking", text="pondering the JSON...")
    for piece in _STATE["stream_texts"]:
        yield StreamEvent(kind="text", text=piece)
    yield StreamEvent(
        kind="final",
        raw={
            "stop_reason": "stop",
            "new_outputs": list(_STATE["new_outputs"]),
            "input_tokens": 1,
            "output_tokens": 1,
        },
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


# ---------- prompt-declared JSON: RETIRED ----------
#
# There used to be a heuristic here ("the sniff") that guessed from prompt text
# — a schema marker plus an imperative — that a client wanted JSON, then
# appended a no-fences hint and unfenced the reply. It was built for Vane on
# the belief that Vane declared nothing the wrapper could see. That turned out
# to be false: Vane sends response_format json_schema, so the sniff sat in the
# `else` branch and could never fire on the traffic it was written for. It was
# removed rather than left flag-gated, because its only remaining effect was to
# rewrite ORDINARY chat replies on a regex false positive.
#
# These tests are the inverse of the ones they replace: a prompt that used to
# trip the sniff must now be treated as completely ordinary. They exist so the
# path cannot creep back in unnoticed.

_SNIFFED = "Use this json schema: {\"a\": 1}. Respond with JSON only."


def test_sniff_prompt_gets_no_hint() -> None:
    # The prompt reaches Claude verbatim — no appended formatting instruction.
    _STATE["final_text"] = '{"a": 1}'
    _chat([{"role": "user", "content": _SNIFFED}])
    check("retired.no_hint_in_prompt", "no markdown code fences" not in _STATE["last_prompt"])
    check("retired.no_json_mode_instruction", "JSON Schema" not in _STATE["last_prompt"])


def test_sniff_prompt_reply_is_not_unfenced() -> None:
    # Without response_format the reply is prose as far as the wrapper knows,
    # fences and all. Only a real declaration earns rewriting.
    fenced = '```json\n{"a": 1}\n```'
    _STATE["final_text"] = fenced
    r = _chat([{"role": "user", "content": _SNIFFED}])
    content = r.json()["choices"][0]["message"]["content"]
    check("retired.content_verbatim", content == fenced, note=content)


def test_sniff_prompt_with_trailing_prose_is_untouched() -> None:
    reply = 'Here you go:\n```json\n{"a": 1}\n```\nHope that helps!'
    _STATE["final_text"] = reply
    r = _chat([{"role": "user", "content": _SNIFFED}])
    content = r.json()["choices"][0]["message"]["content"]
    check("retired.sole_block_untouched", content == reply, note=content)


def test_ordinary_json_chat_still_passes_through() -> None:
    # Unchanged from before the retirement: chat that merely mentions JSON has
    # to pass through byte for byte.
    fenced = 'Sure:\n```json\n{"a": 1}\n```'
    _STATE["final_text"] = fenced
    r = _chat([{"role": "user", "content": "Return a JSON object with a key"}])
    content = r.json()["choices"][0]["message"]["content"]
    check("nosniff.content_verbatim", content == fenced, note=content)
    check("nosniff.no_hint", "no markdown code fences" not in _STATE["last_prompt"])


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


# ---------- the generated-file trailer vs. JSON mode ----------
#
# Every other stub in the suite reports new_outputs=[], so until now the
# suppression branches had never been exercised with attachments actually
# present — which is exactly how they got clobbered twice (#20/#21, restored by
# c95c2f1 and 8854a17). These pin the contract with a real file in play:
#
#   real JSON mode  => never a trailer, at any of the four emission sites
#   anything else   => trailer present
#
# There used to be a third row here for the prompt-declared JSON sniff, which
# suppressed the trailer on the two non-streaming sites as a side effect of
# unfencing the reply. The sniff is retired, so real JSON mode is now the only
# thing that suppresses.
#
# Registration into the file store stays unconditional either way; only the
# markdown trailer is gated.

_TRAILER = "Generated files:"


def _generated(tmp_name="artifact.txt"):
    p = Path(_TMP) / tmp_name
    p.write_text("payload")
    return [str(p)]


def _responses(**extra):
    body = {"model": "claude-opus-4-8", "input": "make me a file", **extra}
    r = client.post("/v1/responses", json=body)
    assert r.status_code == 200, r.text
    return r


def test_trailer_present_without_json_mode() -> None:
    """Control case: the trailer only means something if it appears here."""
    _STATE["new_outputs"] = _generated("plain-sync.txt")
    _STATE["final_text"] = "here you go"
    r = _chat([{"role": "user", "content": "make me a file"}])
    content = r.json()["choices"][0]["message"]["content"]
    check("trailer.sync_present", _TRAILER in content, note=content)

    _STATE["stream_texts"] = ["here ", "you go"]
    _STATE["new_outputs"] = _generated("plain-stream.txt")
    r = _chat([{"role": "user", "content": "make me a file (stream)"}], stream=True)
    content, _reasoning, _finish = _parse_sse(r.text)
    check("trailer.stream_present", _TRAILER in content, note=content)
    _STATE["stream_texts"] = []
    _STATE["new_outputs"] = []


def test_trailer_suppressed_in_json_mode_all_sites() -> None:
    raw = '{"a": 1}'

    # chat, non-streaming
    _STATE["new_outputs"] = _generated("json-sync.txt")
    _STATE["final_text"] = raw
    r = _chat(
        [{"role": "user", "content": "make me a file"}],
        response_format={"type": "json_object"},
    )
    content = r.json()["choices"][0]["message"]["content"]
    check("trailer.json_sync_suppressed", _TRAILER not in content, note=content)
    check("trailer.json_sync_parses", json.loads(content) == {"a": 1}, note=content)
    # ...but the file is still registered and retrievable.
    attachments = r.json()["choices"][0]["message"].get("attachments") or []
    check("trailer.json_sync_still_registered", len(attachments) == 1, note=str(attachments))

    # chat, streaming
    _STATE["stream_texts"] = [raw]
    _STATE["new_outputs"] = _generated("json-stream.txt")
    r = _chat(
        [{"role": "user", "content": "make me a file (stream)"}],
        response_format={"type": "json_object"},
        stream=True,
    )
    content, _reasoning, _finish = _parse_sse(r.text)
    check("trailer.json_stream_suppressed", _TRAILER not in content, note=content)
    check("trailer.json_stream_parses", json.loads(content) == {"a": 1}, note=content)
    _STATE["stream_texts"] = []

    # /v1/responses, non-streaming
    _STATE["new_outputs"] = _generated("json-resp-sync.txt")
    _STATE["final_text"] = raw
    r = _responses(text={"format": {"type": "json_object"}})
    text = json.dumps(r.json())
    check("trailer.json_resp_sync_suppressed", _TRAILER not in text, note=text[:400])

    # /v1/responses, streaming
    _STATE["stream_texts"] = [raw]
    _STATE["new_outputs"] = _generated("json-resp-stream.txt")
    r = _responses(text={"format": {"type": "json_object"}}, stream=True)
    check("trailer.json_resp_stream_suppressed", _TRAILER not in r.text, note=r.text[:400])
    _STATE["stream_texts"] = []
    _STATE["new_outputs"] = []


def test_trailer_kept_on_a_would_be_sniffed_prompt() -> None:
    # The retirement's trailer consequence, on both non-streaming sites. These
    # two turns used to have their trailer suppressed as a side effect of the
    # sniff unfencing the reply; with no response_format on the wire they are
    # ordinary turns now, so a generated file must be surfaced as usual.
    _STATE["new_outputs"] = _generated("sniff-sync.txt")
    _STATE["final_text"] = '```json\n{"a": 1}\n```'
    r = _chat([{"role": "user", "content": _SNIFFED}])
    content = r.json()["choices"][0]["message"]["content"]
    check("trailer.retired_sync_kept", _TRAILER in content, note=content)

    _STATE["new_outputs"] = _generated("sniff-resp.txt")
    r = _responses(input=_SNIFFED)
    text = json.dumps(r.json())
    check("trailer.retired_resp_sync_kept", _TRAILER in text, note=text[:400])
    _STATE["new_outputs"] = []


def test_trailer_kept_on_a_prose_reply() -> None:
    # Unchanged: a plain prose reply with a generated file keeps its trailer.
    _STATE["new_outputs"] = _generated("sniff-noop.txt")
    _STATE["final_text"] = "I could not produce that."
    r = _chat([{"role": "user", "content": _SNIFFED}])
    content = r.json()["choices"][0]["message"]["content"]
    check("trailer.prose_keeps_trailer", _TRAILER in content, note=content)
    _STATE["new_outputs"] = []


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
        test_trailer_present_without_json_mode,
        test_trailer_suppressed_in_json_mode_all_sites,
        test_sniff_prompt_gets_no_hint,
        test_sniff_prompt_reply_is_not_unfenced,
        test_sniff_prompt_with_trailing_prose_is_untouched,
        test_ordinary_json_chat_still_passes_through,
        test_trailer_kept_on_a_would_be_sniffed_prompt,
        test_trailer_kept_on_a_prose_reply,
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
