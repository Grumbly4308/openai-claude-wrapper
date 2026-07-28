"""JSON mode (the OpenAI ``response_format`` parameter).

Structured-output clients (Vercel AI SDK generateObject → Vane, etc.) send
response_format {"type": "json_object"|"json_schema"} and JSON.parse the
returned content verbatim — a ```json fence or any surrounding prose breaks
them. Claude habitually fences JSON when it's only asked for via prompt, so
JSON mode both instructs the model (raw JSON only) and strips whatever wrapping
slips through before the content leaves the wrapper.

Lives in its own module because BOTH generation paths need it: the agentic CLI
path in main.py and the direct Messages API call in tool_bridge.py (a request
may carry tools *and* response_format). tool_bridge is imported by main, so a
shared home is what keeps that from being a circular import.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


_JSON_TYPES = ("json_object", "json_schema")


def wants_json(req: Any) -> bool:
    rf = getattr(req, "response_format", None)
    return rf is not None and rf.type in _JSON_TYPES


def responses_text_format(text: Any) -> Optional[dict[str, Any]]:
    """Normalize a Responses-API ``text.format`` block into ResponseFormat kwargs.

    /v1/responses carries structured output in a different shape than
    /v1/chat/completions: `text: {"format": {"type": "json_schema", "name": …,
    "schema": {…}, "strict": true}}` — the schema inlined, not nested under a
    `json_schema` key. This is not an exotic corner: the Vercel AI SDK's
    `openai(model)` resolves to the Responses model by default, so a
    generateObject call lands on /v1/responses and declares its schema *only*
    this way. Reading `response_format` alone would see nothing and treat the
    turn as free-text prose.

    Returns kwargs for models.ResponseFormat, or None when the request isn't
    asking for structured output.
    """
    if not isinstance(text, dict):
        return None
    fmt = text.get("format")
    if not isinstance(fmt, dict) or fmt.get("type") not in _JSON_TYPES:
        return None
    if fmt["type"] == "json_object":
        return {"type": "json_object"}
    # Lenient clients sometimes nest the chat-style envelope instead of
    # inlining; take it as-is when present, otherwise treat the format block
    # itself (minus `type`) as the envelope.
    envelope = fmt.get("json_schema")
    if not isinstance(envelope, dict):
        envelope = {k: v for k, v in fmt.items() if k != "type"}
    return {"type": "json_schema", "json_schema": envelope}


def json_instruction(req: Any) -> str:
    lines = [
        "## Output format",
        "Respond with a single raw JSON value. Do not wrap it in markdown "
        "code fences and do not add any text before or after the JSON.",
    ]
    rf = getattr(req, "response_format", None)
    if rf is not None and rf.type == "json_schema" and rf.json_schema:
        schema = rf.json_schema.get("schema") or rf.json_schema
        lines.append("The JSON MUST validate against this JSON Schema:")
        lines.append(json.dumps(schema, indent=2))
    return "\n".join(lines)


# ---------- prompt-declared JSON (no response_format on the wire) ----------
#
# Some structured-output clients never declare anything the wrapper can see:
# they put the schema in the PROMPT and JSON.parse the reply anyway. That is
# what Vane does — its requests arrive with no response_format and no tools
# (`json_mode=off` in the request log) — and Claude, asked for JSON in prose
# with nothing forbidding markdown, fences it: ```json\n{…}\n``` → the client
# dies on the backtick.
#
# The wrapper cannot turn on real JSON mode for these turns: a false positive
# there would 502 an ordinary chat answer. What it can do is two things that
# are harmless when the guess is wrong:
#
#   1. Ask for the JSON unfenced (a formatting-only hint, conditional on the
#      answer being JSON at all — it cannot turn a prose answer into JSON).
#   2. Unwrap a reply that is *nothing but* one fenced JSON block. A reply with
#      prose around the fence is left exactly as it is, so a chat answer that
#      merely contains a JSON snippet is never touched.

_SCHEMA_MARKER_RE = re.compile(r"json[\s_-]?schema", re.IGNORECASE)
_JSON_DIRECTIVE_RE = re.compile(
    r"\b(respond|reply|answer|output|return)\b[^.\n]{0,60}\bjson\b", re.IGNORECASE
)

FENCE_HINT = (
    "## Output format\n"
    "If your reply is a JSON value, output it raw — no markdown code fences "
    "around it and no text before or after it."
)


def prompt_requests_json(prompt: str) -> bool:
    """Does this prompt look like a machine asking for a JSON-only reply?

    Deliberately requires BOTH a schema marker and an imperative ("respond
    with JSON"), because either alone matches ordinary chat about JSON. Even
    so this is a guess, so both things it gates are no-ops when it guesses
    wrong — see the module note above.
    """
    text = prompt or ""
    return bool(_SCHEMA_MARKER_RE.search(text) and _JSON_DIRECTIVE_RE.search(text))


_FENCE_ONLY_RE = re.compile(r"\A```[a-zA-Z0-9]*[ \t]*\n?(.*?)\n?[ \t]*```\Z", re.DOTALL)


def unfence_json(text: str) -> str:
    """Unwrap a reply that is one fenced JSON block and nothing else.

    Returns `text` unchanged for anything else — prose around the fence, a
    fence whose contents are not JSON, no fence at all.
    """
    s = (text or "").strip()
    m = _FENCE_ONLY_RE.match(s)
    if m is None:
        return text
    inner = m.group(1).strip()
    try:
        json.loads(inner)
    except json.JSONDecodeError:
        return text
    return inner


# How much of the model's reply to quote back in the error. Enough to see what
# it actually said (a clarifying question, a refusal) without pasting an essay
# into an error field.
_ERROR_SNIPPET_CHARS = 500


def _mode_and_snippet(req: Any, text: str) -> tuple[str, str]:
    rf = getattr(req, "response_format", None)
    mode = getattr(rf, "type", None) or "json"
    snippet = (text or "").strip()
    if len(snippet) > _ERROR_SNIPPET_CHARS:
        snippet = snippet[:_ERROR_SNIPPET_CHARS] + "…"
    return mode, snippet or "(empty reply)"


def json_mode_error(req: Any, text: str) -> str:
    """Error message for a JSON-mode reply that contains no JSON at all.

    Passing the prose through with a 200 is worse than useless to a
    structured-output client: it JSON.parses the body verbatim and dies with a
    ``SyntaxError: Unexpected token`` pointing at the first character, which
    says nothing about what went wrong. Quoting the reply inside a proper
    upstream error keeps the model's actual words (a clarifying question, a
    refusal to fabricate) while letting the client surface a real API error.
    """
    mode, snippet = _mode_and_snippet(req, text)
    return (
        f"model returned no JSON in {mode} mode; it replied with prose instead: {snippet!r}"
    )


def instant_reply_error(req: Any, text: str) -> str:
    """Error message for a wrapper-authored reply that lands in JSON mode.

    Some turns never reach Claude: the per-conversation token-budget checkpoint
    and the instant `stats`/`context` chat commands are answered by the wrapper
    itself, in prose. That is right for a chat UI and unusable for a
    structured-output client, which JSON.parses the body — and the budget
    checkpoint in particular is *sticky*, so every subsequent generateObject
    call would fail the same way with nothing in the logs explaining why. Name
    the cause in the error instead.
    """
    mode, snippet = _mode_and_snippet(req, text)
    return (
        f"wrapper answered this turn itself (token-budget checkpoint or chat command) "
        f"and its reply is prose, which cannot be returned in {mode} mode: {snippet!r}"
    )


_JSON_FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\s*(.*?)\s*```", re.DOTALL)


def extract_raw_json(text: str) -> Optional[str]:
    """Best-effort recovery of a raw JSON value from a model reply.

    Tries, in order: the reply as-is, each fenced code block, then the first
    parseable JSON object/array found anywhere in the text (handles preamble
    like "Here is the JSON:"). Returns None when nothing parses — callers
    should then pass the original text through rather than mask the reply.
    """
    s = (text or "").strip()
    if not s:
        return None
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass
    for m in _JSON_FENCE_RE.finditer(s):
        candidate = m.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch in "{[":
            try:
                _, end = decoder.raw_decode(s, i)
                return s[i:end]
            except json.JSONDecodeError:
                continue
    return None
