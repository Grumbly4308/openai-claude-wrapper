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


def wants_json(req: Any) -> bool:
    rf = getattr(req, "response_format", None)
    return rf is not None and rf.type in ("json_object", "json_schema")


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


# How much of the model's reply to quote back in the error. Enough to see what
# it actually said (a clarifying question, a refusal) without pasting an essay
# into an error field.
_ERROR_SNIPPET_CHARS = 500


def json_mode_error(req: Any, text: str) -> str:
    """Error message for a JSON-mode reply that contains no JSON at all.

    Passing the prose through with a 200 is worse than useless to a
    structured-output client: it JSON.parses the body verbatim and dies with a
    ``SyntaxError: Unexpected token`` pointing at the first character, which
    says nothing about what went wrong. Quoting the reply inside a proper
    upstream error keeps the model's actual words (a clarifying question, a
    refusal to fabricate) while letting the client surface a real API error.
    """
    rf = getattr(req, "response_format", None)
    mode = getattr(rf, "type", None) or "json"
    snippet = (text or "").strip()
    if len(snippet) > _ERROR_SNIPPET_CHARS:
        snippet = snippet[:_ERROR_SNIPPET_CHARS] + "…"
    if not snippet:
        snippet = "(empty reply)"
    return (
        f"model returned no JSON in {mode} mode; it replied with prose instead: {snippet!r}"
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
