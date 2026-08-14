"""OpenAI function calling served by calling the Anthropic Messages API directly.

When a chat request declares ``tools``, the CLIENT owns the agent loop for
its own tools: the wrapper returns the model's tool call(s) and stops; the
client executes the tool and sends the result back as a ``role: "tool"``
message on the next request. Two capability-gated exceptions ride the same
request (see the hybrid-loop notes on complete()/stream()): Anthropic
server-side tools (web search, code execution) execute upstream, and
wrapper-owned tools (memory, time/calc — wrapper_tools.py) execute here,
invisible to the caller. Wrapper tools are therefore only active on
tools-carrying requests; tool-less chats run on the CLI path, which has its
own built-ins.

That is the opposite of the wrapper's normal path (the Claude Code CLI running
its own agentic loop with built-in tools), so requests with tools bypass the
CLI entirely. This module holds everything that path needs: auth resolution,
OpenAI→Anthropic request/history translation, and Anthropic→OpenAI response
translation for both sync and streaming responses.

Auth resolution order (per request):
  1. ANTHROPIC_API_KEY            → x-api-key header
  2. CLAUDE_CODE_OAUTH_TOKEN      → Authorization: Bearer + oauth beta header
  3. ~/.claude/.credentials.json  → same, using the CLI's own OAuth access
     token. The file is re-read on every request because the CLI refreshes it.
     With an OAuth token the API requires the Claude Code identity line as the
     first system block, so it is injected ahead of any caller system prompt.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import httpx
from fastapi import HTTPException

from . import wrapper_tools
from .capabilities import Capability, resolve_profile
from .config import CREDENTIALS_FILE, SETTINGS, _bool_env
from .wrapper_tools import wrapper_tool_names
from .json_mode import extract_raw_json, json_instruction, json_mode_error, wants_json
from .models import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatMessage,
    DeltaMessage,
    ImageContent,
    TextContent,
    Usage,
)

log = logging.getLogger("claude_wrapper.tool_bridge")

ANTHROPIC_BASE_URL = os.environ.get(
    "CLAUDE_WRAPPER_ANTHROPIC_BASE_URL", "https://api.anthropic.com"
).rstrip("/")
_ANTHROPIC_VERSION = "2023-06-01"
_OAUTH_BETA = "oauth-2025-04-20"
_CONTEXT_1M_BETA = "context-1m-2025-08-07"
# Required as the first system block when authenticating with a Claude Code
# OAuth token — the API rejects OAuth requests that don't carry it.
_CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

# The Messages API requires max_tokens; used when the client doesn't send one.
_DEFAULT_MAX_TOKENS = int(os.environ.get("CLAUDE_WRAPPER_TOOLS_MAX_TOKENS", "8192"))

_CREDENTIALS_FILE = CREDENTIALS_FILE

# Same proxy-buffer-flushing preamble the main chat stream uses (see main.py).
_STREAM_PREAMBLE_BYTES = int(os.environ.get("CLAUDE_WRAPPER_SSE_PREAMBLE_BYTES", "2048"))

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


def openai_error(
    status: int,
    message: str,
    err_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> HTTPException:
    """An HTTPException whose body is the OpenAI error envelope.

    OpenAI clients surface errors from ``{"error": {message, type, param,
    code}}``, not FastAPI's ``{"detail": ...}`` — main.py's exception handler
    returns this detail verbatim. Returned (not raised) so call sites read
    ``raise openai_error(...)``.
    """
    return HTTPException(
        status_code=status,
        detail={"error": {"message": message, "type": err_type, "param": param, "code": code}},
    )


@dataclass
class BridgeResult:
    """Non-streaming outcome, ready to be wrapped in a chat.completion."""

    content: Optional[str]
    tool_calls: Optional[list[dict[str, Any]]]
    finish_reason: str
    input_tokens: int
    output_tokens: int


# Shared client; tests replace it with one built on httpx.MockTransport.
_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, read=float(SETTINGS.request_timeout_seconds))
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ---------- auth ----------


def _oauth_access_token() -> str:
    try:
        data = json.loads(_CREDENTIALS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    oauth = data.get("claudeAiOauth") or {}
    token = str(oauth.get("accessToken") or "")
    expires_ms = oauth.get("expiresAt")
    if token and isinstance(expires_ms, (int, float)) and expires_ms / 1000 < time.time():
        # Use it anyway: clock skew is common, and the CLI rewrites this file
        # when it runs. But that refresh only happens if the CLI actually runs
        # AND can reach the network — when egress is broken the token stays
        # expired and every turn 401s, so don't imply the fix is automatic.
        log.warning(
            "OAuth access token in %s is past its expiry. It refreshes when the CLI "
            "next runs with working egress; if this repeats, re-mint with `setup-token`.",
            _CREDENTIALS_FILE,
        )
    return token


def resolve_auth() -> tuple[dict[str, str], bool]:
    """Return (auth headers, is_oauth). Raises 502 when no credential exists."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        return {"x-api-key": api_key}, False
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip() or _oauth_access_token()
    if token:
        return {"authorization": f"Bearer {token}"}, True
    raise openai_error(
        502,
        "function calling requires direct Anthropic API access, but no "
        "credential was found: set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN, "
        f"or log the Claude Code CLI in so {_CREDENTIALS_FILE} exists",
        err_type="api_error",
        code="no_upstream_credential",
    )


# ---------- capability gating & server-side tool injection ----------

# Server web search in the bridge is genuinely new behavior (direct API tool
# billing), while the web_search *capability* defaults on for the CLI path's
# sake — so bridge injection takes its own operator opt-in, mirroring the
# terminal gate. Code execution needs no extra gate: its capability defaults
# off, so a profile grant is already an explicit operator action.
_BRIDGE_WEB_SEARCH_ENV = "CLAUDE_WRAPPER_BRIDGE_WEB_SEARCH"

# Ceiling on hybrid-loop rounds per turn (wrapper tool executions and
# pause_turn continuations) — a runaway-loop backstop, not a tuning knob.
_MAX_TOOL_ROUNDS = int(os.environ.get("CLAUDE_WRAPPER_BRIDGE_MAX_TOOL_ROUNDS", "8"))

_WS_MODERN = "web_search_20260209"  # Opus 4.6+/Sonnet 4.6+/Claude 5+ families
_WS_BASIC = "web_search_20250305"  # everything older, incl. Haiku 4.5
_CODE_EXECUTION_TYPE = "code_execution_20260521"

_WS_FAMILY_RE = re.compile(r"^claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d{1,2}))?")


def _web_search_type(api_model: str) -> str:
    m = _WS_FAMILY_RE.match(api_model)
    if not m:
        return _WS_BASIC
    fam, major, minor = m.group(1), int(m.group(2)), m.group(3)
    if fam in ("fable", "mythos"):
        return _WS_MODERN
    if fam == "haiku":
        return _WS_BASIC
    if major >= 5 or (major == 4 and minor is not None and int(minor) >= 6):
        return _WS_MODERN
    return _WS_BASIC


def _server_tools(run_model: str, caps: frozenset[Capability]) -> list[dict[str, Any]]:
    """Anthropic server-side tools the model's profile enables for this run."""
    out: list[dict[str, Any]] = []
    if Capability.WEB_SEARCH in caps and _bool_env(_BRIDGE_WEB_SEARCH_ENV, False):
        api_model, _ = _api_model(run_model)
        out.append({"type": _web_search_type(api_model), "name": "web_search"})
    if Capability.CODE_INTERPRETER in caps:
        out.append({"type": _CODE_EXECUTION_TYPE, "name": "code_execution"})
    return out


# ---------- OpenAI -> Anthropic translation ----------


def _api_model(run_model: str) -> tuple[str, bool]:
    """Map a wrapper model id to an API model id. Returns (model, wants_1m)."""
    m = (run_model or "").strip()
    wants_1m = m.endswith("[1m]")
    if wants_1m:
        m = m[: -len("[1m]")].strip()
    # Minor-less discovery ids ("opus-5", "sonnet-5") lack the claude- prefix.
    if m and not m.startswith("claude-"):
        m = f"claude-{m}"
    return m, wants_1m


# The Messages API requires tool names to match this; OpenAI clients send
# dotted/namespaced names ("mcp.server.tool") that the OpenAI API tolerates.
_TOOL_NAME_BAD = re.compile(r"[^a-zA-Z0-9_-]")
_TOOL_NAME_MAX = 64


def _preflight(req: ChatCompletionRequest) -> None:
    """Reject malformed tools requests here, with an OpenAI-shaped 400 naming
    the offending param, instead of letting them die upstream as an opaque 502."""
    declared: set[str] = set()
    for i, t in enumerate(req.tools or []):
        if t.type != "function":
            raise openai_error(
                400,
                f"tools[{i}].type must be 'function', got {t.type!r}",
                param=f"tools[{i}].type",
            )
        if not t.function.name.strip():
            raise openai_error(
                400,
                f"tools[{i}].function.name must be a non-empty string",
                param=f"tools[{i}].function.name",
            )
        declared.add(t.function.name)
    tc = req.tool_choice
    if isinstance(tc, str) and tc not in ("auto", "none", "required"):
        raise openai_error(
            400,
            f"tool_choice must be 'auto', 'none', 'required' or a named function, got {tc!r}",
            param="tool_choice",
        )
    if isinstance(tc, dict):
        name = ((tc.get("function") or {}).get("name")) or ""
        if not name:
            raise openai_error(
                400, "tool_choice.function.name is required", param="tool_choice.function.name"
            )
        if name not in declared:
            raise openai_error(
                400,
                f"tool_choice names function {name!r}, which is not declared in tools",
                param="tool_choice.function.name",
            )
    for i, msg in enumerate(req.messages):
        if msg.role == "tool" and not (msg.tool_call_id or "").strip():
            raise openai_error(
                400,
                f"messages[{i}] has role 'tool' but no tool_call_id",
                param=f"messages[{i}].tool_call_id",
            )


def _sanitize_name(name: str) -> str:
    return _TOOL_NAME_BAD.sub("_", name)[:_TOOL_NAME_MAX] or "tool"


def _sanitize_names(names: list[str]) -> dict[str, str]:
    """Original -> API-safe name, deterministic. Post-sanitize collisions
    ("a.b" and "a_b") get a numeric suffix in declaration order."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for name in names:
        safe = _sanitize_name(name)
        if safe in used:
            i = 2
            while True:
                candidate = f"{safe[: _TOOL_NAME_MAX - len(str(i)) - 1]}_{i}"
                if candidate not in used:
                    safe = candidate
                    break
                i += 1
        mapping[name] = safe
        used.add(safe)
    return mapping


def _client_tools(
    req: ChatCompletionRequest,
) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]], dict[str, str]]:
    """Map tools + tool_choice; returns (tools, tool_choice, orig→safe names).

    ``tool_choice: "none"`` sends no tools at all. Duplicate names are deduped
    last-definition-wins (the OpenAI API tolerates duplicates, the Messages API
    rejects them), then names are sanitized to the API's charset/length.
    """
    if req.tool_choice == "none":
        return [], None, {}
    by_name: dict[str, Any] = {}
    total = 0
    for t in req.tools or []:
        if t.type == "function":
            by_name[t.function.name] = t.function
            total += 1
    if not by_name:
        return [], None, {}
    if total > len(by_name):
        log.warning(
            "deduplicated %d duplicate tool name(s); last definition wins", total - len(by_name)
        )
    name_map = _sanitize_names(list(by_name))
    tools = [
        {
            "name": name_map[name],
            "description": fn.description or "",
            # `parameters` maps to `input_schema` verbatim — never rewritten.
            "input_schema": fn.parameters or {"type": "object", "properties": {}},
        }
        for name, fn in by_name.items()
    ]
    choice: dict[str, Any]
    if isinstance(req.tool_choice, dict):
        name = ((req.tool_choice.get("function") or {}).get("name")) or ""
        choice = {"type": "tool", "name": name_map.get(name, name)}
    elif req.tool_choice == "required":
        choice = {"type": "any"}
    else:  # "auto" or absent
        choice = {"type": "auto"}
    if req.parallel_tool_calls is False:
        choice["disable_parallel_tool_use"] = True
    return tools, choice, name_map


def _message_text(msg: ChatMessage) -> str:
    if msg.content is None:
        return ""
    if isinstance(msg.content, str):
        return msg.content
    return "\n".join(p.text for p in msg.content if isinstance(p, TextContent) and p.text)


def _user_blocks(msg: ChatMessage) -> list[dict[str, Any]]:
    if isinstance(msg.content, str):
        return [{"type": "text", "text": msg.content}] if msg.content.strip() else []
    blocks: list[dict[str, Any]] = []
    for part in msg.content or []:
        if isinstance(part, TextContent):
            if part.text.strip():
                blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImageContent):
            url = part.image_url.url
            if url.startswith("data:"):
                try:
                    meta, data = url.split(",", 1)
                    media_type = meta.split(":", 1)[1].split(";", 1)[0] or "image/png"
                    blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": data},
                        }
                    )
                except (IndexError, ValueError):
                    continue
            elif url.startswith(("http://", "https://")):
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        # Other part kinds (audio/file) have no Messages-API equivalent here and
        # only occur on the agentic path, which materializes them to disk.
    return blocks


def messages_to_anthropic(
    messages: list[ChatMessage], name_map: Optional[dict[str, str]] = None
) -> tuple[list[str], list[dict[str, Any]]]:
    """Translate an OpenAI transcript to (system texts, Anthropic messages).

    - assistant.tool_calls become tool_use blocks (arguments parsed to objects,
      ids reused verbatim so the round-trip is lossless; names go through the
      same orig→safe map as the tool definitions, so echoed history matches);
    - role:"tool" messages become tool_result blocks in a USER message, with
      consecutive tool messages merged into one (the API wants every parallel
      result in the single next user turn);
    - consecutive same-role messages are merged, which the API requires.
    """
    system_texts: list[str] = []
    out: list[dict[str, Any]] = []

    def _append(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for msg in messages:
        if msg.role in ("system", "developer"):
            text = _message_text(msg)
            if text.strip():
                system_texts.append(text)
        elif msg.role == "tool":
            _append(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "",
                        "content": _message_text(msg),
                    }
                ],
            )
        elif msg.role == "assistant":
            blocks: list[dict[str, Any]] = []
            text = _message_text(msg)
            if text.strip():
                blocks.append({"type": "text", "text": text})
            for tc in msg.tool_calls or []:
                raw_args = tc.function.arguments if tc.function else None
                try:
                    parsed = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    parsed = {}
                if not isinstance(parsed, dict):
                    parsed = {}
                raw_name = (tc.function.name if tc.function else "") or ""
                if name_map and raw_name in name_map:
                    name = name_map[raw_name]
                else:
                    # History may cite a tool no longer declared; still keep
                    # the name inside the API's charset.
                    name = _sanitize_name(raw_name) if raw_name else ""
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": name,
                        "input": parsed,
                    }
                )
            _append("assistant", blocks)
        else:  # user
            _append("user", _user_blocks(msg))
    return system_texts, out


def build_request(
    req: ChatCompletionRequest, run_model: str, stream: bool
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """Build the Messages API payload + headers for a tools request.

    Returns (payload, headers, safe→original tool-name map) — responses run
    tool names back through the map so the client only ever sees its own.
    """
    _preflight(req)
    auth_headers, is_oauth = resolve_auth()
    model, wants_1m = _api_model(run_model)
    tools, tool_choice, name_map = _client_tools(req)
    system_texts, messages = messages_to_anthropic(req.messages, name_map)
    if not messages:
        raise openai_error(400, "no prompt content derived from messages", param="messages")

    system_blocks: list[dict[str, Any]] = []
    if is_oauth:
        system_blocks.append({"type": "text", "text": _CLAUDE_CODE_IDENTITY})
    system_blocks.extend({"type": "text", "text": t} for t in system_texts)
    # A request may declare tools AND response_format (AI SDK clients do this
    # when a structured-output call is allowed to call tools first). The
    # instruction goes last so it wins over any caller system prompt, and only
    # governs the final text answer — tool_use blocks are unaffected.
    if wants_json(req):
        system_blocks.append({"type": "text", "text": json_instruction(req)})

    payload: dict[str, Any] = {
        "model": model,
        # max_completion_tokens is OpenAI's successor to the deprecated
        # max_tokens; honor it first when a client sends both.
        "max_tokens": req.max_completion_tokens or req.max_tokens or _DEFAULT_MAX_TOKENS,
        "messages": messages,
        "stream": stream,
    }
    if system_blocks:
        payload["system"] = system_blocks
    if req.temperature is not None:
        payload["temperature"] = min(max(req.temperature, 0.0), 1.0)
    if req.top_p is not None:
        payload["top_p"] = req.top_p
    caps = resolve_profile(run_model)
    rev_map = {safe: orig for orig, safe in name_map.items()}
    if tools and Capability.CLIENT_TOOLS not in caps:
        declared = ", ".join(sorted(name_map))
        raise openai_error(
            400,
            f"model '{run_model}' does not accept client-declared tools "
            f"(capability 'client_tools' is not in its profile); declared: {declared}",
            param="tools",
        )
    # Injected tools go after the client's, in fixed order, so the rendered
    # tool list stays deterministic for prompt caching. Wrapper-owned tools
    # (memory, time/calc) are executed by the bridge's hybrid loop, never
    # surfaced to the caller.
    client_names = {t["name"] for t in tools}
    wrapper_defs = wrapper_tools.tool_definitions(caps)
    shadowed = client_names & {d.get("name") for d in wrapper_defs}
    if shadowed:
        # Report the caller's spelling, not the sanitized one.
        names = sorted(rev_map.get(n, n) for n in shadowed)
        raise openai_error(
            400,
            f"client tool name(s) {names} collide with wrapper-owned "
            f"tools enabled for model '{run_model}'; rename the client tool or "
            "remove the capability from the model's profile",
            param="tools",
        )
    # A client tool may reuse a server-side tool's name (clients commonly
    # declare their own "web_search"); the API rejects the duplicate, so the
    # client's definition wins and the server tool sits this request out.
    server_defs = []
    for s in _server_tools(run_model, caps):
        if s["name"] in client_names:
            log.warning(
                "client tool %r shadows the %s server tool; not injecting it",
                s["name"],
                s["type"],
            )
        else:
            server_defs.append(s)
    all_tools = tools + server_defs + wrapper_defs
    if all_tools:
        payload["tools"] = all_tools
        if tools:
            payload["tool_choice"] = tool_choice

    betas = []
    if is_oauth:
        betas.append(_OAUTH_BETA)
    if wants_1m:
        betas.append(_CONTEXT_1M_BETA)
    headers = {
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
        **auth_headers,
    }
    if betas:
        headers["anthropic-beta"] = ",".join(betas)
    return payload, headers, rev_map


# ---------- Anthropic -> OpenAI translation ----------


def _finish_reason(stop_reason: Optional[str], has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    return _STOP_REASON_MAP.get(stop_reason or "", "stop")


async def complete(
    req: ChatCompletionRequest, run_model: str, session_key: str = ""
) -> BridgeResult:
    """Non-streaming tools request, with the wrapper's hybrid tool loop.

    Client-declared tools stay client-looped: their calls are returned and
    the turn ends. Wrapper-owned tools (memory, time/calc — see
    wrapper_tools.py) are executed here and fed back, looping until the model
    answers or wants a client tool. Turn-taking rules:

    - only wrapper tool calls  -> execute all, append results, loop;
    - any client tool call     -> return it; wrapper calls in the same turn
      are dropped unexecuted (the model re-issues them next turn — returning
      them to a client that can't run them, or faking their results, would
      corrupt the protocol);
    - ``pause_turn``           -> re-send to let a server-side tool continue;
    - anything else            -> final answer.

    Assistant content is echoed back verbatim between rounds — thinking and
    server-tool blocks included, which the API requires for a tool loop.
    """
    payload, headers, rev_map = build_request(req, run_model, stream=False)
    wrapper_names = wrapper_tool_names(resolve_profile(run_model))
    client = _get_client()
    input_tokens = output_tokens = 0
    data: dict[str, Any] = {}

    for _round in range(_MAX_TOOL_ROUNDS):
        try:
            resp = await client.post(
                f"{ANTHROPIC_BASE_URL}/v1/messages", json=payload, headers=headers
            )
        except httpx.HTTPError as e:
            raise openai_error(
                502, f"anthropic api unreachable: {e}", err_type="api_error", code="upstream_error"
            )
        if resp.status_code != 200:
            raise openai_error(
                502,
                f"anthropic api error {resp.status_code}: {resp.text[:500]}",
                err_type="api_error",
                code="upstream_error",
            )
        data = resp.json()
        usage = data.get("usage") or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)

        blocks = data.get("content") or []
        wrapper_calls = [
            b for b in blocks if b.get("type") == "tool_use" and b.get("name") in wrapper_names
        ]
        client_calls = [
            b for b in blocks if b.get("type") == "tool_use" and b.get("name") not in wrapper_names
        ]
        paused = data.get("stop_reason") == "pause_turn"
        if client_calls or (not wrapper_calls and not paused):
            break
        payload["messages"].append({"role": "assistant", "content": blocks})
        if wrapper_calls:
            results = []
            for call in wrapper_calls:
                content_text, is_error = wrapper_tools.execute(
                    call.get("name") or "", call.get("input") or {}, session_key
                )
                result: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": call.get("id") or "",
                    "content": content_text,
                }
                if is_error:
                    result["is_error"] = True
                results.append(result)
            payload["messages"].append({"role": "user", "content": results})
    else:
        raise openai_error(
            502,
            f"wrapper tool loop exceeded {_MAX_TOOL_ROUNDS} rounds without an answer",
            err_type="api_error",
            code="tool_loop_limit",
        )

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in data.get("content") or []:
        btype = block.get("type")
        if btype == "text" and block.get("text"):
            text_parts.append(block["text"])
        elif btype == "tool_use" and block.get("name") not in wrapper_names:
            name = block.get("name") or ""
            tool_calls.append(
                {
                    "id": block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        # Give the client back the name it declared, not the
                        # sanitized one the API saw.
                        "name": rev_map.get(name, name),
                        # OpenAI clients JSON.parse this — it MUST be a string.
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )

    content = "".join(text_parts)
    # JSON mode: the client JSON.parses the content verbatim. Reduce it to the
    # raw JSON value. Skipped when the turn produced tool calls — there the
    # content is incidental commentary, not the structured answer, and the
    # client reads tool_calls instead. If nothing parses, fail loudly with the
    # model's own words rather than return a body that dies in JSON.parse.
    if not tool_calls and wants_json(req):
        cleaned = extract_raw_json(content)
        if cleaned is None:
            raise openai_error(
                502, json_mode_error(req, content), err_type="api_error", code="json_mode_failed"
            )
        content = cleaned
    return BridgeResult(
        content=content if content else None,
        tool_calls=tool_calls or None,
        finish_reason=_finish_reason(data.get("stop_reason"), bool(tool_calls)),
        # Accumulated across all wrapper-loop rounds, not just the last call.
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _materialize_block(acc: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a complete content block from its start event + deltas.

    Needed to echo the assistant turn back verbatim when the streaming hybrid
    loop continues a turn. Result blocks that arrive whole (server tool
    results, redacted_thinking) pass through as their start event.
    """
    block = dict(acc.get("start") or {})
    btype = block.get("type")
    if btype == "text":
        block["text"] = acc.get("text", "")
    elif btype in ("tool_use", "server_tool_use", "mcp_tool_use"):
        partial = acc.get("partial", "")
        if partial:
            try:
                block["input"] = json.loads(partial)
            except json.JSONDecodeError:
                block["input"] = {}
        else:
            block["input"] = block.get("input") or {}
    elif btype == "thinking":
        block["thinking"] = acc.get("thinking", "")
        if acc.get("signature"):
            block["signature"] = acc["signature"]
    return block


async def stream(
    req: ChatCompletionRequest,
    run_model: str,
    model_label: str,
    session_key: str,
    effort_info: dict[str, Any],
    on_usage: Optional[Callable[[int, int], Awaitable[None]]] = None,
) -> AsyncIterator[bytes]:
    """Streaming tools request, translated event-by-event into OpenAI chunks.

    Tool-call argument fragments are forwarded straight from Anthropic's
    input_json_delta events — never buffered into one chunk.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(delta: DeltaMessage, finish: Optional[str] = None, **extra: Any) -> bytes:
        c = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_label,
            choices=[ChatCompletionChunkChoice(index=0, delta=delta, finish_reason=finish)],
            **extra,
        )
        return f"data: {c.model_dump_json(exclude_none=True)}\n\n".encode("utf-8")

    if _STREAM_PREAMBLE_BYTES > 0:
        yield b": " + b" " * _STREAM_PREAMBLE_BYTES + b"\n\n"
    yield chunk(
        DeltaMessage(role="assistant", content=""),
        session_id=session_key,
        effort=effort_info,
    )

    # Anthropic content-block index -> position in the OpenAI tool_calls array
    # (client tools only; reset per loop round). `next_tc_index` and
    # `any_client_calls` are cumulative across rounds.
    tc_index_of: dict[int, int] = {}
    next_tc_index = 0
    any_client_calls = False
    # JSON mode: the client parses the concatenated content as JSON, so answer
    # deltas are buffered (a ```json fence can span chunk boundaries) and
    # emitted as one cleaned chunk right before the terminator. Tool-call
    # argument fragments still stream through untouched.
    json_mode = wants_json(req)
    json_parts: list[str] = []
    stop_reason: Optional[str] = None
    input_tokens = output_tokens = 0
    errored: Optional[str] = None
    errored_type = "upstream_error"
    # Legacy `functions` clients read delta.function_call, not delta.tool_calls,
    # and the shape carries a single call — extra parallel calls are dropped.
    legacy = req.functions is not None
    rev_map: dict[str, str] = {}

    try:
        payload, headers, rev_map = build_request(req, run_model, stream=True)
        wrapper_names = wrapper_tool_names(resolve_profile(run_model))
        client = _get_client()
        # Hybrid loop, same turn-taking rules as complete(): wrapper-owned
        # tool calls are executed here and the upstream call repeated on the
        # same client stream; client tool calls end the loop (the caller owns
        # that loop). Every round reassembles the assistant blocks verbatim —
        # thinking, signatures, and server-tool blocks included — because a
        # continued turn must echo them back exactly.
        for _round in range(_MAX_TOOL_ROUNDS):
            tc_index_of = {}
            round_blocks: dict[int, dict[str, Any]] = {}
            stop_reason = None
            async with client.stream(
                "POST", f"{ANTHROPIC_BASE_URL}/v1/messages", json=payload, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    errored = f"anthropic api error {resp.status_code}: {body[:500]}"
                else:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        try:
                            evt = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            continue
                        etype = evt.get("type")

                        if etype == "message_start":
                            usage = (evt.get("message") or {}).get("usage") or {}
                            input_tokens += int(usage.get("input_tokens") or 0)
                        elif etype == "content_block_start":
                            block = evt.get("content_block") or {}
                            a_idx = int(evt.get("index") or 0)
                            round_blocks[a_idx] = {"start": block, "text": "", "partial": ""}
                            if (
                                block.get("type") == "tool_use"
                                and block.get("name") not in wrapper_names
                            ):
                                any_client_calls = True
                                name = block.get("name") or ""
                                name = rev_map.get(name, name)
                                if legacy and next_tc_index > 0:
                                    # The legacy shape has no second slot.
                                    log.warning(
                                        "dropping parallel call to %r: legacy "
                                        "function_call responses carry one call",
                                        name,
                                    )
                                elif legacy:
                                    tc_index_of[a_idx] = next_tc_index
                                    next_tc_index += 1
                                    yield chunk(
                                        DeltaMessage(
                                            function_call={"name": name, "arguments": ""}
                                        )
                                    )
                                else:
                                    tc_index_of[a_idx] = next_tc_index
                                    yield chunk(
                                        DeltaMessage(
                                            tool_calls=[
                                                {
                                                    "index": next_tc_index,
                                                    "id": block.get("id")
                                                    or f"toolu_{uuid.uuid4().hex[:24]}",
                                                    "type": "function",
                                                    "function": {
                                                        "name": name,
                                                        "arguments": "",
                                                    },
                                                }
                                            ]
                                        )
                                    )
                                    next_tc_index += 1
                        elif etype == "content_block_delta":
                            delta = evt.get("delta") or {}
                            dtype = delta.get("type")
                            acc = round_blocks.setdefault(
                                int(evt.get("index") or 0), {"start": {}, "text": "", "partial": ""}
                            )
                            if dtype == "text_delta" and delta.get("text"):
                                acc["text"] += delta["text"]
                                if json_mode:
                                    json_parts.append(delta["text"])
                                else:
                                    yield chunk(DeltaMessage(content=delta["text"]))
                            elif dtype == "input_json_delta":
                                fragment = delta.get("partial_json") or ""
                                acc["partial"] += fragment
                                idx = tc_index_of.get(int(evt.get("index") or 0))
                                if fragment and idx is not None:
                                    if legacy:
                                        yield chunk(
                                            DeltaMessage(function_call={"arguments": fragment})
                                        )
                                    else:
                                        yield chunk(
                                            DeltaMessage(
                                                tool_calls=[
                                                    {
                                                        "index": idx,
                                                        "function": {"arguments": fragment},
                                                    }
                                                ]
                                            )
                                        )
                            elif dtype == "thinking_delta" and delta.get("thinking"):
                                acc["thinking"] = acc.get("thinking", "") + delta["thinking"]
                            elif dtype == "signature_delta" and delta.get("signature"):
                                acc["signature"] = delta["signature"]
                        elif etype == "message_delta":
                            stop_reason = (evt.get("delta") or {}).get("stop_reason") or stop_reason
                            usage = evt.get("usage") or {}
                            output_tokens += int(usage.get("output_tokens") or 0)
                        elif etype == "error":
                            errored = str(
                                (evt.get("error") or {}).get("message") or "upstream error"
                            )
                        elif etype == "message_stop":
                            break
            if errored:
                break
            assistant_blocks = [
                _materialize_block(round_blocks[i]) for i in sorted(round_blocks)
            ]
            wrapper_calls = [
                b
                for b in assistant_blocks
                if b.get("type") == "tool_use" and b.get("name") in wrapper_names
            ]
            paused = stop_reason == "pause_turn"
            if any_client_calls or (not wrapper_calls and not paused):
                break
            payload["messages"].append({"role": "assistant", "content": assistant_blocks})
            if wrapper_calls:
                results: list[dict[str, Any]] = []
                for call in wrapper_calls:
                    content_text, is_error = wrapper_tools.execute(
                        call.get("name") or "", call.get("input") or {}, session_key
                    )
                    result: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": call.get("id") or "",
                        "content": content_text,
                    }
                    if is_error:
                        result["is_error"] = True
                    results.append(result)
                payload["messages"].append({"role": "user", "content": results})
        else:
            errored = f"wrapper tool loop exceeded {_MAX_TOOL_ROUNDS} rounds without an answer"
    except HTTPException as e:
        # openai_error details carry the envelope; surface its message/type on
        # the SSE error channel instead of a stringified dict.
        if isinstance(e.detail, dict) and isinstance(e.detail.get("error"), dict):
            errored = str(e.detail["error"].get("message") or "request failed")
            errored_type = str(e.detail["error"].get("type") or errored_type)
        else:
            errored = str(e.detail)
    except httpx.HTTPError as e:
        errored = f"anthropic api stream failed: {e}"
    except Exception as e:  # pragma: no cover - defensive
        log.exception("tool-bridge stream failed (session=%s)", session_key)
        errored = f"internal wrapper error: {e}"

    if on_usage is not None and (input_tokens or output_tokens):
        try:
            await on_usage(input_tokens, output_tokens)
        except Exception:  # pragma: no cover
            log.exception("usage recording failed (session=%s)", session_key)

    # Flush the buffered JSON-mode answer. Emitted here (not inside the loop) so
    # it also reaches the client on the mid-stream error path. Cleaned only when
    # the turn produced no tool calls — see the same guard in complete(). With
    # the response head already sent a 502 is impossible, so an unparseable
    # reply goes out on the stream's error channel instead of as content.
    if json_parts:
        body = "".join(json_parts)
        if any_client_calls:
            yield chunk(DeltaMessage(content=body))
        else:
            cleaned = extract_raw_json(body)
            if cleaned is None:
                errored = errored or json_mode_error(req, body)
            else:
                yield chunk(DeltaMessage(content=cleaned))

    finish = _finish_reason(stop_reason, any_client_calls)
    if legacy and finish == "tool_calls":
        finish = "function_call"
    yield chunk(DeltaMessage(), finish=finish)
    if (req.stream_options or {}).get("include_usage"):
        usage_chunk = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_label,
            choices=[],
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
        yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n".encode("utf-8")
    if errored:
        err_payload = {"error": {"message": errored, "type": errored_type}}
        yield f"data: {json.dumps(err_payload)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"
