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
    "last_clarify": None,
}


async def _stub_run_collect(self, prompt, session_key, model=None, effort=None, **_kwargs):
    _STATE["last_prompt"] = prompt
    _STATE["last_clarify"] = _kwargs.get("clarify")
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
    _STATE["last_clarify"] = _kwargs.get("clarify")
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

from src.json_mode import (  # noqa: E402
    prompt_requests_json as _prompt_requests_json,
    responses_text_format,
    unfence_json as _unfence_json,
    unfence_sole_json_block as _unfence_sole_json_block,
)
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


def test_sync_json_unparseable_errors() -> None:
    """No JSON at all => 502 quoting the model, never a 200 the client chokes on.

    Handing prose back with a 200 makes the client die in JSON.parse on the
    first character, which tells nobody anything. The error carries the reply.
    """
    _STATE["final_text"] = "I could not produce JSON, sorry."
    body = {
        "model": "claude-opus-4-8",
        "messages": [{"role": "user", "content": "suggest things (unparseable test)"}],
        "response_format": {"type": "json_object"},
    }
    r = client.post("/v1/chat/completions", json=body)
    check("sync.unparseable_status", r.status_code == 502, note=str(r.status_code))
    detail = r.json().get("detail", "")
    check("sync.unparseable_quotes_reply", "I could not produce JSON, sorry." in detail, note=detail)
    check("sync.unparseable_names_mode", "json_object" in detail, note=detail)


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


# ---------- clarification protocol ----------
#
# The clarify protocol tells Claude to make its ENTIRE reply a list of questions
# when it hits an ambiguity. That is pure prose with no JSON in it, so it
# reaches a generateObject client as an unparseable body and dies in
# JSON.parse — and nobody is on the far end to answer it anyway. JSON mode must
# therefore force clarify OFF, even when the client explicitly asked for it.


def test_json_mode_disables_clarify() -> None:
    _STATE["final_text"] = '{"a": 1}'
    _STATE["last_clarify"] = None
    _chat(
        [{"role": "user", "content": "extract fields (clarify off test)"}],
        response_format={"type": "json_object"},
    )
    check("clarify.off_in_json_mode", _STATE["last_clarify"] is False, note=str(_STATE["last_clarify"]))


def test_json_mode_overrides_explicit_clarify() -> None:
    _STATE["final_text"] = '{"a": 1}'
    _STATE["last_clarify"] = None
    _chat(
        [{"role": "user", "content": "extract fields (explicit clarify test)"}],
        response_format={"type": "json_schema", "json_schema": {"name": "r", "schema": {"type": "object"}}},
        clarify=True,
    )
    check("clarify.json_wins_over_explicit", _STATE["last_clarify"] is False, note=str(_STATE["last_clarify"]))


def test_plain_chat_keeps_clarify() -> None:
    _STATE["final_text"] = "hi"
    _STATE["last_clarify"] = None
    _chat([{"role": "user", "content": "normal chat (clarify stays on)"}])
    check("clarify.on_without_json", _STATE["last_clarify"] is True, note=str(_STATE["last_clarify"]))


def test_json_mode_disables_clarify_streaming() -> None:
    _STATE["stream_texts"] = ['{"a": 1}']
    _STATE["last_clarify"] = None
    _chat(
        [{"role": "user", "content": "extract fields (stream clarify test)"}],
        response_format={"type": "json_object"},
        stream=True,
    )
    check("clarify.off_in_json_stream", _STATE["last_clarify"] is False, note=str(_STATE["last_clarify"]))


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


def test_stream_json_unparseable_errors() -> None:
    """Streaming can't 502 (head already sent) => error frame, and no content."""
    _STATE["stream_texts"] = ["I could not ", "produce JSON, sorry."]
    r = _chat(
        [{"role": "user", "content": "suggest things (stream unparseable test)"}],
        response_format={"type": "json_object"},
        stream=True,
    )
    content, _reasoning, _finish = _parse_sse(r.text)
    check("stream.unparseable_no_content", content == "", note=content)
    errors = [
        json.loads(line[len("data: "):])["error"]["message"]
        for line in r.text.splitlines()
        if line.startswith("data: ")
        and line[len("data: "):].strip() != "[DONE]"
        and "error" in json.loads(line[len("data: "):])
    ]
    check("stream.unparseable_error_frame", len(errors) == 1, note=str(errors))
    check("stream.unparseable_quotes_reply", "produce JSON, sorry." in errors[0], note=errors[0])


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


# ---------- /v1/responses ----------
#
# The Responses API declares structured output as `text: {"format": {…}}`, with
# the schema inlined rather than nested under `json_schema`. This is the shape
# the Vercel AI SDK sends by default — `openai(model)` resolves to its Responses
# model — so a generateObject call lands here, not on /v1/chat/completions. The
# same contract must hold: instruction in the prompt, raw JSON out, a real error
# instead of prose.


def _responses(input_text: str, **extra):
    body = {"model": "claude-opus-4-8", "input": input_text, **extra}
    return client.post("/v1/responses", json=body)


def _parse_responses_sse(body: str):
    """-> (concatenated deltas, terminal event type, terminal response object)."""
    deltas, terminal_type, terminal = [], None, None
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        obj = json.loads(line[len("data: "):])
        if obj.get("type") == "response.output_text.delta":
            deltas.append(obj.get("delta") or "")
        elif obj.get("type") in ("response.completed", "response.failed"):
            terminal_type = obj["type"]
            terminal = obj.get("response") or {}
    return "".join(deltas), terminal_type, terminal


_TEXT_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "suggestions",
        "schema": {
            "type": "object",
            "properties": {"suggestions": {"type": "array", "items": {"type": "string"}}},
            "required": ["suggestions"],
        },
        "strict": True,
    }
}


def test_responses_text_format_normalization() -> None:
    """The Responses `text.format` shape maps onto chat's `response_format`."""
    norm = responses_text_format(_TEXT_FORMAT)
    check("respfmt.type", norm["type"] == "json_schema", note=str(norm))
    check("respfmt.schema_kept", norm["json_schema"]["schema"]["required"] == ["suggestions"], note=str(norm))
    check("respfmt.json_object", responses_text_format({"format": {"type": "json_object"}}) == {"type": "json_object"})
    # Chat-style nesting from a lenient client is taken as-is.
    nested = responses_text_format({"format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}})
    check("respfmt.nested_envelope", nested["json_schema"] == {"schema": {"type": "object"}}, note=str(nested))
    check("respfmt.plain_text", responses_text_format({"format": {"type": "text"}}) is None)
    check("respfmt.absent", responses_text_format(None) is None)


def test_responses_sync_json_strips_fences() -> None:
    _STATE["final_text"] = '```json\n{"suggestions": ["x", "y"]}\n```'
    r = _responses("suggest things (responses json test)", text=_TEXT_FORMAT)
    check("responses.status", r.status_code == 200, note=r.text)
    out = r.json()["output_text"]
    check("responses.raw_json", json.loads(out) == {"suggestions": ["x", "y"]}, note=out)
    check("responses.instruction_in_prompt", "Output format" in _STATE["last_prompt"])
    check("responses.schema_in_prompt", "JSON Schema" in _STATE["last_prompt"])
    check("responses.clarify_off", _STATE["last_clarify"] is False, note=str(_STATE["last_clarify"]))


def test_responses_sync_unparseable_errors() -> None:
    _STATE["final_text"] = "Before diving into the data, I need to know which fields matter."
    r = _responses("extract fields (responses unparseable test)", text=_TEXT_FORMAT)
    check("responses.unparseable_status", r.status_code == 502, note=str(r.status_code))
    detail = r.json().get("detail", "")
    check("responses.unparseable_quotes_reply", "Before diving into the data" in detail, note=detail)


def test_responses_plain_untouched() -> None:
    fenced = 'Here you go:\n```json\n{"a": 1}\n```'
    _STATE["final_text"] = fenced
    r = _responses("normal responses turn (no structured output)")
    check("responses.plain_verbatim", r.json()["output_text"] == fenced, note=r.text)
    check("responses.plain_no_instruction", "Output format" not in _STATE["last_prompt"])


def test_responses_stream_json_buffers_and_strips() -> None:
    _STATE["stream_texts"] = ['```json\n{"suggestions": [', '"x", "y"]}\n```']
    r = _responses("suggest things (responses stream json test)", text=_TEXT_FORMAT, stream=True)
    deltas, terminal_type, terminal = _parse_responses_sse(r.text)
    check("responses.stream_raw_json", json.loads(deltas) == {"suggestions": ["x", "y"]}, note=deltas)
    check("responses.stream_completed", terminal_type == "response.completed", note=str(terminal_type))
    check(
        "responses.stream_output_text",
        json.loads(terminal["output_text"]) == {"suggestions": ["x", "y"]},
        note=str(terminal.get("output_text")),
    )


def test_responses_stream_json_unparseable_fails() -> None:
    """Head already sent => fail on the stream's own channel, emit no text."""
    _STATE["stream_texts"] = ["Before diving in, ", "which fields matter?"]
    r = _responses("extract fields (responses stream unparseable)", text=_TEXT_FORMAT, stream=True)
    deltas, terminal_type, terminal = _parse_responses_sse(r.text)
    check("responses.stream_bad_no_text", deltas == "", note=deltas)
    check("responses.stream_bad_failed", terminal_type == "response.failed", note=str(terminal_type))
    check(
        "responses.stream_bad_quotes_reply",
        "Before diving in" in (terminal.get("error") or {}).get("message", ""),
        note=str(terminal.get("error")),
    )


# ---------- prompt-declared JSON (no response_format on the wire) ----------
#
# Vane sends neither response_format nor tools — its request logs as
# `json_mode=off` — and puts the schema in the prompt, then JSON.parses the
# reply. Claude fences the JSON, and the client dies on the backtick. The
# wrapper cannot switch on real JSON mode for these turns (a false positive
# would 502 an ordinary chat answer), so it does the two things that are
# harmless when the guess is wrong: ask for it unfenced, and unwrap a reply
# that is nothing but a fence.

_SNIFFED = (
    "Extract the fields from this article.\n\n"
    "JSON schema:\n"
    '{"type":"object","properties":{"title":{"type":"string"}}}\n'
    "You MUST respond with a JSON object matching the schema above."
)


def test_prompt_sniff_detection() -> None:
    check("sniff.detects", _prompt_requests_json(_SNIFFED))
    # Either half alone is ordinary chat about JSON, not a machine request.
    check("sniff.marker_only", not _prompt_requests_json("What is a JSON schema, in plain English?"))
    check("sniff.directive_only", not _prompt_requests_json("Answer in JSON if you feel like it."))
    check("sniff.plain_chat", not _prompt_requests_json("Tell me about the Battle of Hastings."))


def test_unfence_json_unit() -> None:
    raw = '{"title": "x"}'
    check("unfence.fenced", _unfence_json(f"```json\n{raw}\n```") == raw)
    check("unfence.fenced_nolang", _unfence_json(f"```\n{raw}\n```") == raw)
    # Prose around the fence => a chat answer that merely contains JSON. Untouched.
    prose = f"Here you go:\n```json\n{raw}\n```\nHope that helps!"
    check("unfence.prose_untouched", _unfence_json(prose) == prose)
    # A fence that isn't JSON is a code block. Untouched.
    code = "```python\nprint('hi')\n```"
    check("unfence.code_untouched", _unfence_json(code) == code)
    check("unfence.unfenced_passthrough", _unfence_json(raw) == raw)


def test_unfence_sole_json_block_unit() -> None:
    """The escalation: a lone JSON fence is unwrapped even with prose around it.

    `unfence_json` alone matches a fence and nothing else, so the single
    trailing sentence Claude habitually adds was enough to ship backticks to
    a client whose next move is JSON.parse.
    """
    raw = '{"title": "x"}'
    check("sole.trailer", _unfence_sole_json_block(f"```json\n{raw}\n```\n\nLet me know!") == raw)
    check("sole.preamble", _unfence_sole_json_block(f"Here you go:\n```json\n{raw}\n```") == raw)
    check("sole.both_sides", _unfence_sole_json_block(f"Sure:\n```json\n{raw}\n```\nHope that helps!") == raw)
    check("sole.bare_fence", _unfence_sole_json_block(f"```json\n{raw}\n```") == raw)
    check("sole.no_lang", _unfence_sole_json_block(f"```\n{raw}\n```") == raw)

    # Narrower than extract_raw_json on purpose. Two fenced blocks is a reply
    # weighing options, not a structured-output answer — picking one would be
    # a guess, so it is left alone.
    two = f"Option A:\n```json\n{raw}\n```\nOption B:\n```json\n{{\"title\": \"y\"}}\n```"
    check("sole.two_blocks_untouched", _unfence_sole_json_block(two) == two)
    # A fence that isn't JSON is a code block, whatever the prompt asked for.
    code = "Try this:\n```python\nprint('hi')\n```"
    check("sole.non_json_untouched", _unfence_sole_json_block(code) == code)
    # No fence at all: nothing to unwrap.
    prose = "A JSON schema describes the shape of a document."
    check("sole.no_fence_untouched", _unfence_sole_json_block(prose) == prose)


def test_sniffed_reply_with_trailer_is_unfenced() -> None:
    """End-to-end regression for the reported bug.

    Vane's generateObject died on `Unexpected token '`'` because the reply was
    a fence plus a sign-off, which strict unfencing left completely alone.
    """
    _STATE["final_text"] = '```json\n{"title": "x"}\n```\n\nLet me know if you need changes.'
    r = _chat([{"role": "user", "content": _SNIFFED}])
    content = r.json()["choices"][0]["message"]["content"]
    check("sniff.trailer_status", r.status_code == 200, note=str(r.status_code))
    check("sniff.trailer_parses", json.loads(content) == {"title": "x"}, note=content)


def test_responses_sniffed_reply_with_trailer_is_unfenced() -> None:
    """Same escalation on /v1/responses, the other sniffed path."""
    _STATE["final_text"] = 'Here you go:\n```json\n{"title": "x"}\n```'
    r = _responses(_SNIFFED)
    check(
        "sniff.responses_trailer",
        json.loads(r.json()["output_text"]) == {"title": "x"},
        note=r.text,
    )


def test_sniffed_reply_is_unfenced() -> None:
    _STATE["final_text"] = '```json\n{"title": "x"}\n```'
    r = _chat([{"role": "user", "content": _SNIFFED}])
    content = r.json()["choices"][0]["message"]["content"]
    check("sniff.unfenced", json.loads(content) == {"title": "x"}, note=content)
    check("sniff.hint_in_prompt", "no markdown code fences" in _STATE["last_prompt"])
    # Prompt-declared JSON is a guess, so it must NOT take the hard-mode paths:
    # clarify stays on and a prose reply is still a 200, not a 502.
    check("sniff.clarify_untouched", _STATE["last_clarify"] is True, note=str(_STATE["last_clarify"]))


def test_sniffed_prose_reply_still_passes_through() -> None:
    """A mis-sniffed chat turn must be delivered verbatim, never turned into an error."""
    _STATE["final_text"] = "A JSON schema describes the shape of a JSON document."
    r = _chat([{"role": "user", "content": "What is a JSON schema? Respond with JSON examples."}])
    content = r.json()["choices"][0]["message"]["content"]
    check("sniff.prose_200", r.status_code == 200)
    check("sniff.prose_verbatim", content == _STATE["final_text"], note=content)


def test_unsniffed_chat_keeps_its_fences() -> None:
    """An ordinary chat turn keeps markdown fences — OWUI renders them."""
    fenced = '```json\n{"a": 1}\n```'
    _STATE["final_text"] = fenced
    r = _chat([{"role": "user", "content": "show me an example config block"}])
    content = r.json()["choices"][0]["message"]["content"]
    check("sniff.chat_fence_kept", content == fenced, note=content)
    check("sniff.no_hint_in_prompt", "no markdown code fences" not in _STATE["last_prompt"])


def test_responses_sniffed_reply_is_unfenced() -> None:
    _STATE["final_text"] = '```json\n{"title": "x"}\n```'
    r = _responses(_SNIFFED)
    check("sniff.responses_unfenced", json.loads(r.json()["output_text"]) == {"title": "x"}, note=r.text)


# ---------- wrapper-authored replies ----------
#
# Some turns never reach Claude: the `stats`/`context` commands and the
# token-budget checkpoint are answered by the wrapper in prose. In JSON mode
# that prose is just as unparseable to the client as a prose model reply, so it
# must not go out with a 200.


def test_instant_command_errors_in_json_mode() -> None:
    body = {
        "model": "claude-opus-4-8",
        "messages": [{"role": "user", "content": "stats"}],
        "response_format": {"type": "json_object"},
    }
    r = client.post("/v1/chat/completions", json=body)
    check("instant.chat_status", r.status_code == 502, note=str(r.status_code))
    check("instant.chat_names_cause", "wrapper answered this turn" in r.json().get("detail", ""), note=r.text)


def test_instant_command_ok_without_json_mode() -> None:
    body = {"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "stats"}]}
    r = client.post("/v1/chat/completions", json=body)
    check("instant.chat_plain_200", r.status_code == 200, note=str(r.status_code))
    check("instant.chat_plain_text", "Usage stats" in r.json()["choices"][0]["message"]["content"], note=r.text)


def test_responses_instant_command_errors_in_json_mode() -> None:
    r = _responses("stats", text=_TEXT_FORMAT)
    check("instant.responses_status", r.status_code == 502, note=str(r.status_code))
    check("instant.responses_names_cause", "wrapper answered this turn" in r.json().get("detail", ""), note=r.text)


# ---------- runner ----------


def main() -> int:
    tests = [
        test_extract_raw_json,
        test_sync_json_object_strips_fences,
        test_sync_json_schema_in_prompt,
        test_sync_json_unparseable_errors,
        test_stream_json_unparseable_errors,
        test_sync_without_response_format_untouched,
        test_response_format_text_is_not_json_mode,
        test_json_mode_disables_clarify,
        test_json_mode_overrides_explicit_clarify,
        test_plain_chat_keeps_clarify,
        test_json_mode_disables_clarify_streaming,
        test_stream_json_buffers_and_strips,
        test_stream_without_response_format_untouched,
        test_responses_text_format_normalization,
        test_responses_sync_json_strips_fences,
        test_responses_sync_unparseable_errors,
        test_responses_plain_untouched,
        test_responses_stream_json_buffers_and_strips,
        test_responses_stream_json_unparseable_fails,
        test_prompt_sniff_detection,
        test_unfence_json_unit,
        test_sniffed_reply_is_unfenced,
        test_sniffed_prose_reply_still_passes_through,
        test_unsniffed_chat_keeps_its_fences,
        test_responses_sniffed_reply_is_unfenced,
        test_instant_command_errors_in_json_mode,
        test_instant_command_ok_without_json_mode,
        test_responses_instant_command_errors_in_json_mode,
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
