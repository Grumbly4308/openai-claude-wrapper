"""OpenAI function calling served by calling the Anthropic Messages API directly.

When a chat request declares ``tools``, the CLIENT owns the agent loop: the
wrapper must not execute anything, must not use its own built-in tools, and
must not try to answer the question. It returns the model's tool call(s) and
stops; the client executes the tool and sends the result back as a
``role: "tool"`` message on the next request.

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
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import httpx
from fastapi import HTTPException

from .config import SETTINGS
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

_CREDENTIALS_FILE = Path(
    os.environ.get(
        "CLAUDE_WRAPPER_CREDENTIALS_FILE",
        str(Path.home() / ".claude" / ".credentials.json"),
    )
)

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
        # The CLI refreshes this file whenever it runs; we don't hold the
        # refresh-token flow ourselves. Use the token anyway (clock skew is
        # common) but leave a trail if the API then 401s.
        log.warning(
            "OAuth access token in %s looks expired; any `claude` invocation refreshes it",
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
    raise HTTPException(
        status_code=502,
        detail=(
            "function calling requires direct Anthropic API access, but no "
            "credential was found: set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN, "
            f"or log the Claude Code CLI in so {_CREDENTIALS_FILE} exists"
        ),
    )


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


def _tools_to_anthropic(req: ChatCompletionRequest) -> tuple[Optional[list[dict]], Optional[dict]]:
    """Map tools + tool_choice. ``tool_choice: "none"`` sends no tools at all."""
    if req.tool_choice == "none":
        return None, None
    tools = [
        {
            "name": t.function.name,
            "description": t.function.description or "",
            # `parameters` maps to `input_schema` verbatim — never rewritten.
            "input_schema": t.function.parameters or {"type": "object", "properties": {}},
        }
        for t in (req.tools or [])
        if t.type == "function"
    ]
    if not tools:
        return None, None
    choice: dict[str, Any]
    if isinstance(req.tool_choice, dict):
        name = ((req.tool_choice.get("function") or {}).get("name")) or ""
        choice = {"type": "tool", "name": name}
    elif req.tool_choice == "required":
        choice = {"type": "any"}
    else:  # "auto" or absent
        choice = {"type": "auto"}
    if req.parallel_tool_calls is False:
        choice["disable_parallel_tool_use"] = True
    return tools, choice


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


def messages_to_anthropic(messages: list[ChatMessage]) -> tuple[list[str], list[dict[str, Any]]]:
    """Translate an OpenAI transcript to (system texts, Anthropic messages).

    - assistant.tool_calls become tool_use blocks (arguments parsed to objects,
      ids reused verbatim so the round-trip is lossless);
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
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": (tc.function.name if tc.function else "") or "",
                        "input": parsed,
                    }
                )
            _append("assistant", blocks)
        else:  # user
            _append("user", _user_blocks(msg))
    return system_texts, out


def build_request(
    req: ChatCompletionRequest, run_model: str, stream: bool
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build the Messages API payload + headers for a tools request."""
    auth_headers, is_oauth = resolve_auth()
    model, wants_1m = _api_model(run_model)
    system_texts, messages = messages_to_anthropic(req.messages)
    if not messages:
        raise HTTPException(status_code=400, detail="no prompt content derived from messages")

    system_blocks: list[dict[str, Any]] = []
    if is_oauth:
        system_blocks.append({"type": "text", "text": _CLAUDE_CODE_IDENTITY})
    system_blocks.extend({"type": "text", "text": t} for t in system_texts)

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": req.max_tokens or _DEFAULT_MAX_TOKENS,
        "messages": messages,
        "stream": stream,
    }
    if system_blocks:
        payload["system"] = system_blocks
    if req.temperature is not None:
        payload["temperature"] = min(max(req.temperature, 0.0), 1.0)
    if req.top_p is not None:
        payload["top_p"] = req.top_p
    tools, tool_choice = _tools_to_anthropic(req)
    if tools:
        payload["tools"] = tools
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
    return payload, headers


# ---------- Anthropic -> OpenAI translation ----------


def _finish_reason(stop_reason: Optional[str], has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    return _STOP_REASON_MAP.get(stop_reason or "", "stop")


async def complete(req: ChatCompletionRequest, run_model: str) -> BridgeResult:
    """Non-streaming tools request: one Messages API call, no loop."""
    payload, headers = build_request(req, run_model, stream=False)
    client = _get_client()
    try:
        resp = await client.post(
            f"{ANTHROPIC_BASE_URL}/v1/messages", json=payload, headers=headers
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"anthropic api unreachable: {e}")
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"anthropic api error {resp.status_code}: {resp.text[:500]}",
        )
    data = resp.json()

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in data.get("content") or []:
        btype = block.get("type")
        if btype == "text" and block.get("text"):
            text_parts.append(block["text"])
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        # OpenAI clients JSON.parse this — it MUST be a string.
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )

    usage = data.get("usage") or {}
    content = "".join(text_parts)
    return BridgeResult(
        content=content if content else None,
        tool_calls=tool_calls or None,
        finish_reason=_finish_reason(data.get("stop_reason"), bool(tool_calls)),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
    )


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

    # Anthropic content-block index -> position in the OpenAI tool_calls array.
    tc_index_of: dict[int, int] = {}
    stop_reason: Optional[str] = None
    input_tokens = output_tokens = 0
    errored: Optional[str] = None

    try:
        payload, headers = build_request(req, run_model, stream=True)
        client = _get_client()
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
                        input_tokens = int(usage.get("input_tokens") or 0)
                    elif etype == "content_block_start":
                        block = evt.get("content_block") or {}
                        if block.get("type") == "tool_use":
                            idx = len(tc_index_of)
                            tc_index_of[int(evt.get("index") or 0)] = idx
                            yield chunk(
                                DeltaMessage(
                                    tool_calls=[
                                        {
                                            "index": idx,
                                            "id": block.get("id")
                                            or f"toolu_{uuid.uuid4().hex[:24]}",
                                            "type": "function",
                                            "function": {
                                                "name": block.get("name") or "",
                                                "arguments": "",
                                            },
                                        }
                                    ]
                                )
                            )
                    elif etype == "content_block_delta":
                        delta = evt.get("delta") or {}
                        dtype = delta.get("type")
                        if dtype == "text_delta" and delta.get("text"):
                            yield chunk(DeltaMessage(content=delta["text"]))
                        elif dtype == "input_json_delta":
                            fragment = delta.get("partial_json") or ""
                            idx = tc_index_of.get(int(evt.get("index") or 0))
                            if fragment and idx is not None:
                                yield chunk(
                                    DeltaMessage(
                                        tool_calls=[
                                            {"index": idx, "function": {"arguments": fragment}}
                                        ]
                                    )
                                )
                    elif etype == "message_delta":
                        stop_reason = (evt.get("delta") or {}).get("stop_reason") or stop_reason
                        usage = evt.get("usage") or {}
                        output_tokens = int(usage.get("output_tokens") or output_tokens)
                    elif etype == "error":
                        errored = str((evt.get("error") or {}).get("message") or "upstream error")
                    elif etype == "message_stop":
                        break
    except HTTPException as e:
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

    yield chunk(DeltaMessage(), finish=_finish_reason(stop_reason, bool(tc_index_of)))
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
        err_payload = {"error": {"message": errored, "type": "upstream_error"}}
        yield f"data: {json.dumps(err_payload)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"
