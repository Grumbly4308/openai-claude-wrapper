from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse

from . import download_tokens, tool_bridge
from .config import (
    SETTINGS,
    advertised_models,
    log_credential_status,
    split_model_effort,
    supported_models,
)
from .converters import derive_session_id
from .deps import (
    FILE_STORE,
    PREPARER,
    RUNNER,
    USAGE_LEDGER,
    auth_dependency,
    download_auth_dependency,
)
from .request_origin import RequestOriginMiddleware, current_origin
# Re-exported under the historical private names (and `extract_raw_json`, which
# tests import from here) — see json_mode.py for why it is a separate module.
from .json_mode import (
    extract_raw_json,
    instant_reply_error as _instant_reply_error,
    json_instruction as _json_instruction,
    json_mode_error as _json_mode_error,
    wants_json as _wants_json,
)
from .models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChoiceMessage,
    DeltaMessage,
    ModelInfo,
    ModelList,
    ResponsesRequest,
    Usage,
)
from .usage import UsageState
from .routes_assistants import router as assistants_router
from .routes_audio import router as audio_router
from .routes_batches import router as batches_router
from .routes_embeddings import router as embeddings_router
from .routes_fine_tuning import router as fine_tuning_router
from .routes_images import router as images_router
from .routes_moderations import router as moderations_router
from .routes_realtime import router as realtime_router
from .routes_text import router as text_router
from .routes_vector_stores import router as vector_stores_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("claude_wrapper.api")

# How often to emit an SSE keep-alive comment while the runner is producing no
# visible output (extended thinking, tool/subagent work). Long reasoning phases
# — common with xhigh/ultracode effort and the 1M-context models — can run for
# many minutes before the first assistant text. Without bytes on the wire, an
# idle-timeout proxy or the client's own read timeout severs the chunked
# response mid-stream, which surfaces to the client as a truncated/incomplete
# payload. Comment lines (": ...\n\n") are ignored by OpenAI-compatible SSE
# parsers but keep every idle timer in the path from firing.
_STREAM_HEARTBEAT_SECONDS = float(os.environ.get("CLAUDE_WRAPPER_SSE_HEARTBEAT", "15"))

# A bare SSE comment ("\n\n") keeps the socket warm but is invisible to the user
# and — crucially — a response-buffering reverse proxy may not flush headers/body
# until it has seen enough *real* bytes, leaving the client blocked reading the
# status line until an idle timer severs the connection (aiohttp:
# "ServerDisconnectedError" while still in resp.start). Two mitigations:
#
#  - A one-time preamble: a chunky comment emitted before anything else so the
#    proxy flushes the response head immediately. ~2 KiB beats the common 1–4 KiB
#    proxy buffer; set CLAUDE_WRAPPER_SSE_PREAMBLE_BYTES=0 to disable.
#  - Periodic *visible* progress: during a long silent stretch (Claude thinking
#    or running tools/subagents on a hard problem) emit a real reasoning_content
#    frame on a slower cadence than the heartbeat, so the feed shows life and the
#    stream carries genuine data. Set CLAUDE_WRAPPER_SSE_PROGRESS_SECONDS=0 off.
_STREAM_PREAMBLE_BYTES = int(os.environ.get("CLAUDE_WRAPPER_SSE_PREAMBLE_BYTES", "2048"))
_STREAM_PROGRESS_SECONDS = float(os.environ.get("CLAUDE_WRAPPER_SSE_PROGRESS_SECONDS", "25"))
# Whether to surface tool/subagent activity in the feed (as reasoning_content).
_STREAM_SHOW_ACTIVITY = os.environ.get("CLAUDE_WRAPPER_SSE_SHOW_ACTIVITY", "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

# Which channel carries reasoning/progress frames (thinking, tool activity, the
# periodic "still working" tick) to the client:
#   - "details" (default): wrap reasoning in a <details type="reasoning"> block on
#     the *content* stream. This is OWUI's own internal representation of a
#     reasoning block, so its frontend renders it as a collapsible panel
#     REGARDLESS of provider. Necessary because OWUI classifies this wrapper as an
#     "openai" provider, and its get_reasoning_format() returns None for openai —
#     i.e. it ignores BOTH the reasoning_content field AND inline <think> tags, so
#     no native reasoning channel will ever render here.
#   - "reasoning_content": the structured DeepSeek-R1 delta field. Renders natively
#     ONLY on providers OWUI recognizes for it (e.g. llama.cpp), not openai.
#   - "think_tags": inline <think>…</think> in content. Ollama-provider only; OWUI
#     does not parse it for openai connections (shows literal tags).
#   - "none": suppress reasoning/progress frames entirely (answer text only).
# For the content-wrapped modes (details, think_tags) the block is opened lazily
# on the first reasoning frame and closed before the first answer token (and again
# at stream end as a safety net), so content never carries an unbalanced tag.
_REASONING_CHANNEL = os.environ.get(
    "CLAUDE_WRAPPER_REASONING_CHANNEL", "details"
).strip().lower()

# (open, close) token pairs for the channels that wrap reasoning inside the
# content stream. Other channels (reasoning_content, none) are not in this map.
_CONTENT_WRAP = {
    "details": (
        '<details type="reasoning">\n<summary>💭 Thinking…</summary>\n\n',
        "\n\n</details>\n\n",
    ),
    "think_tags": ("<think>\n", "\n</think>\n\n"),
}


# Shared SSE response headers. Disabling proxy buffering (X-Accel-Buffering) and
# caching is what lets keep-alive comments and incremental chunks actually reach
# the client instead of being held until the response completes.
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


app = FastAPI(title="Claude Code OpenAI Wrapper", version="0.1.0")


@app.on_event("startup")
async def _startup() -> None:
    # Build the model list once on load by scanning the installed Claude Code
    # binary (memoized thereafter). Logged so the resolved set is visible at boot.
    models = supported_models()
    log.info("model list ready: %d models — %s", len(models), ", ".join(models))

    # Credential state. Every other line here reports configuration; this one
    # reports a clock, and it is the failure this deployment actually hits — an
    # expired login surfaces as a 401 the CLI attributes to itself, with nothing
    # pointing at the file on disk. Under the sandboxed topology the CLI runs in
    # the agent container, so this container's view is of a read-only mount; the
    # agent reports the same status for the copy it owns.
    log_credential_status("Claude")

    # Where turns execute: locally, or in the sandboxed agent container.
    if SETTINGS.agent_url:
        log.info(
            "agent execution: REMOTE via %s (sandboxed agent container; workspace "
            "must be a volume shared with it at the same path)",
            SETTINGS.agent_url,
        )
    else:
        log.info("agent execution: local subprocess (%s)", SETTINGS.claude_bin)

    # Surface KB passthrough state at boot. The OpenWebUI addendum is silently
    # skipped when OPENWEBUI_BASE_URL is unset, which makes "Claude can't see the
    # KB" hard to diagnose — so make it explicit in the logs either way.
    if SETTINGS.openwebui_base_url:
        log.info(
            "knowledge-base passthrough ENABLED — base_url=%s api_key=%s default_collection=%s",
            SETTINGS.openwebui_base_url,
            "set" if SETTINGS.openwebui_api_key else "MISSING",
            SETTINGS.openwebui_default_collection or "(none)",
        )
        if not SETTINGS.openwebui_api_key:
            log.warning(
                "OPENWEBUI_BASE_URL is set but OPENWEBUI_API_KEY is empty; "
                "OpenWebUI retrieval endpoints normally require auth and will 401."
            )
    else:
        log.info(
            "knowledge-base passthrough DISABLED — set OPENWEBUI_BASE_URL to teach "
            "Claude to query the OpenWebUI retrieval API."
        )

    # Interactive clarification protocol state at boot.
    if SETTINGS.clarify_enabled:
        log.info(
            "interactive clarification ENABLED — chat/responses pause to ask in text; "
            "disallowed tools: %s",
            ", ".join(SETTINGS.clarify_disallowed_tools) or "(none)",
        )
    else:
        log.info("interactive clarification DISABLED (CLAUDE_WRAPPER_CLARIFY=off)")

    # Generated-file downloads. Same reasoning as the knowledge-base block above:
    # every way this is misconfigured fails *quietly* — a link that renders as
    # plain text, or never appears at all — and none of them say why. Three
    # independent settings have to line up, so state all three at boot.
    if SETTINGS.public_base_url:
        link_base, base_src = SETTINGS.public_base_url, "CLAUDE_WRAPPER_PUBLIC_BASE_URL"
    elif SETTINGS.derive_base_url:
        link_base, base_src = "(derived per-request from the inbound Host)", "CLAUDE_WRAPPER_DERIVE_BASE_URL=on"
    else:
        link_base, base_src = "", ""
    if link_base:
        # Signing/TTL are reported from download_links_signed — whether verify()
        # actually runs on the route — not from the raw signing key, which can
        # be set while nothing checks it (explicit key, no API keys). And a TTL
        # is only a revocation window when something enforces it.
        if SETTINGS.download_links_signed:
            signing = "on"
            ttl = (
                f"{SETTINGS.download_url_ttl_seconds}s"
                if SETTINGS.download_url_ttl_seconds
                else "never expires (still signed)"
            )
        else:
            signing = "off (no API keys; route is unauthenticated)"
            ttl = "n/a (nothing is verified, so links never expire)"
        log.info(
            "generated-file downloads ENABLED — link base=%s (%s), signing=%s, ttl=%s, workspace hint=%s",
            link_base,
            base_src,
            signing,
            ttl,
            # The hint is also gated off per-request in JSON mode (see
            # _resolve_workspace_hint), so "on" is the server default, not a
            # promise about every request.
            "on (server default; JSON-mode requests excluded)"
            if SETTINGS.workspace_hint_enabled
            else "OFF",
        )
        if SETTINGS.download_signing_key and not SETTINGS.download_links_signed:
            log.warning(
                "CLAUDE_WRAPPER_DOWNLOAD_SIGNING_KEY is set but CLAUDE_WRAPPER_API_KEYS is "
                "empty, so the download route never verifies signatures — links carry exp/sig "
                "but ANY request can read any stored file. Set API keys to enforce signing."
            )
        if SETTINGS.public_base_url and not SETTINGS.public_base_url.startswith(("http://", "https://")):
            log.warning(
                "CLAUDE_WRAPPER_PUBLIC_BASE_URL=%r has no http(s):// scheme, so generated "
                "markdown links will not be clickable in a browser. Set the full URL the "
                "BROWSER uses, scheme included.",
                SETTINGS.public_base_url,
            )
        if not SETTINGS.workspace_hint_enabled:
            log.warning(
                "download links are configured but CLAUDE_WRAPPER_WORKSPACE_HINT is off, so "
                "Claude is never told its files are delivered and will usually paste content "
                "inline instead of writing a file. Set it on for a chat-UI deployment."
            )
        if not SETTINGS.public_base_url:
            log.warning(
                "CLAUDE_WRAPPER_PUBLIC_BASE_URL is unset, so link hosts are derived from each "
                "request. Behind a chat UI that reaches this wrapper on an internal hostname "
                "(e.g. http://claude-wrapper:8000) the links will not resolve in a browser. "
                "Set it to the URL the BROWSER uses."
            )
    else:
        log.info(
            "generated-file downloads DISABLED — no CLAUDE_WRAPPER_PUBLIC_BASE_URL and "
            "CLAUDE_WRAPPER_DERIVE_BASE_URL=off, so the file trailer stays plain text."
        )
        if SETTINGS.workspace_hint_enabled:
            log.warning(
                "downloads are disabled but CLAUDE_WRAPPER_WORKSPACE_HINT is on, so Claude is "
                "told its files come back as download links that can never render — replies "
                "will reference files the user cannot fetch. Turn the hint off, or set "
                "CLAUDE_WRAPPER_PUBLIC_BASE_URL / CLAUDE_WRAPPER_DERIVE_BASE_URL=on."
            )


@app.on_event("shutdown")
async def _shutdown() -> None:
    await PREPARER.aclose()
    await tool_bridge.aclose()
    await RUNNER.aclose()


app.include_router(text_router)
app.include_router(embeddings_router)
app.include_router(moderations_router)
app.include_router(audio_router)
app.include_router(images_router)
app.include_router(batches_router)
app.include_router(assistants_router)
app.include_router(vector_stores_router)
app.include_router(fine_tuning_router)
app.include_router(realtime_router)

# Records the origin each request arrived on, so generated-file links can be
# absolute (and therefore clickable) without the operator configuring
# CLAUDE_WRAPPER_PUBLIC_BASE_URL. Pure-ASGI on purpose — see request_origin.py.
app.add_middleware(RequestOriginMiddleware)


# ---------- health & models ----------


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/v1/models", response_model=ModelList, dependencies=[Depends(auth_dependency)])
async def list_models() -> ModelList:
    now = int(time.time())
    return ModelList(data=[ModelInfo(id=m, created=now) for m in advertised_models()])


@app.get("/v1/models/{model_id}", dependencies=[Depends(auth_dependency)])
async def retrieve_model(model_id: str) -> ModelInfo:
    base, _effort = split_model_effort(model_id)
    if base not in supported_models():
        raise HTTPException(status_code=404, detail=f"unknown model: {model_id}")
    return ModelInfo(id=model_id, created=int(time.time()))


@app.get("/v1/usage/{session_id}", dependencies=[Depends(auth_dependency)])
async def session_usage(session_id: str) -> dict:
    """Programmatic twin of the `stats`/`context` chat commands: one
    conversation's token spend and remaining allowance, straight from the
    ledger. `session_id` is the session key the chat endpoints return."""
    state = await USAGE_LEDGER.snapshot(session_id)
    return {
        "object": "usage.session",
        "session_id": session_id,
        "tracking_enabled": USAGE_LEDGER.enabled,
        "spent_tokens": state.spent_tokens,
        "requests": state.requests,
        "blocks_granted": state.grants,
        "block_tokens": state.block_tokens,
        "allowance_tokens": state.allowance_tokens,
        "remaining_tokens": max(0, state.allowance_tokens - state.spent_tokens),
        "over_budget": state.over_budget,
        "session_allowance_tokens": SETTINGS.session_token_allowance,
        "session_plan": SETTINGS.session_plan,
        "block_percent": SETTINGS.session_block_percent,
    }


# ---------- files API ----------


@app.post("/v1/files", dependencies=[Depends(auth_dependency)])
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    purpose: str = Form(default="user_data"),
) -> dict:
    async def _chunks() -> AsyncIterator[bytes]:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            yield chunk

    try:
        record = await FILE_STORE.save_stream(
            chunks=_chunks(),
            filename=file.filename or "upload.bin",
            purpose=purpose,
            mime_type=file.content_type,
            max_bytes=SETTINGS.max_upload_bytes,
        )
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    return record.to_openai()


@app.get("/v1/files", dependencies=[Depends(auth_dependency)])
async def list_files(purpose: Optional[str] = None) -> dict:
    records = await FILE_STORE.list(purpose=purpose)
    return {"object": "list", "data": [r.to_openai() for r in records]}


@app.get("/v1/files/{file_id}", dependencies=[Depends(auth_dependency)])
async def retrieve_file(file_id: str) -> dict:
    record = await FILE_STORE.get(file_id)
    if record is None:
        raise HTTPException(status_code=404, detail="file not found")
    return record.to_openai()


@app.delete("/v1/files/{file_id}", dependencies=[Depends(auth_dependency)])
async def delete_file(file_id: str) -> dict:
    deleted = await FILE_STORE.delete(file_id)
    return {"id": file_id, "object": "file", "deleted": deleted}


@app.get("/v1/files/{file_id}/content", dependencies=[Depends(download_auth_dependency)])
async def download_file(file_id: str) -> StreamingResponse:
    record = await FILE_STORE.get(file_id)
    if record is None:
        raise HTTPException(status_code=404, detail="file not found")

    path = FILE_STORE.blob_path(record)

    async def _iter() -> AsyncIterator[bytes]:
        import aiofiles

        async with aiofiles.open(path, "rb") as f:
            while True:
                chunk = await f.read(1 << 16)
                if not chunk:
                    break
                yield chunk

    headers = {
        "Content-Disposition": f'attachment; filename="{record.filename}"',
        "Content-Length": str(record.bytes),
    }
    return StreamingResponse(_iter(), media_type=record.mime_type, headers=headers)


# ---------- chat completions ----------


class _InstantReply:
    """Sentinel returned by _prepare_run when the wrapper answers a turn itself.

    Used for the budget checkpoint and for the instant `stats` / `context` chat
    commands. Carries the already-rendered message so each endpoint shape
    (chat-shaped vs Responses-shaped) can wrap it without running Claude.
    """

    __slots__ = ("session_key", "text")

    def __init__(self, session_key: str, text: str) -> None:
        self.session_key = session_key
        self.text = text


async def _prepare_run(req: ChatCompletionRequest):
    """Shared prelude for every text-generation endpoint.

    Resolves the session key, answers instant chat commands, enforces the
    per-conversation token budget, builds the prompt, and resolves the model.
    Returns either ``(prompt, session_key, model)`` ready to run, or an
    ``_InstantReply`` the caller renders in its own response shape. Used by
    /v1/chat/completions, /v1/completions, /v1/responses, and the batches worker.
    """
    session_key = derive_session_id(req.messages, req.session_id, req.user)

    # Instant chat commands: a message that is exactly `stats` or `context`
    # (optionally /-prefixed) is answered by the wrapper itself — current token
    # spend and remaining allowance, straight from the ledger. Checked before
    # the budget gate so the report stays reachable (and free) even while a
    # conversation is paused at a checkpoint.
    if _usage_command(req.messages):
        state = await USAGE_LEDGER.snapshot(session_key)
        return _InstantReply(session_key, _usage_report(session_key, state))

    # Per-conversation token budget. If this conversation has already spent its
    # current allowance, pause *before* spawning Claude and ask the user to
    # confirm — unless their latest message is a "continue", which buys one more
    # block. Disabled (no-op) unless a session token allowance is configured.
    if USAGE_LEDGER.enabled:
        state = await USAGE_LEDGER.snapshot(session_key)
        if state.over_budget:
            if _is_continue(req.messages):
                await USAGE_LEDGER.grant(session_key)
            else:
                return _InstantReply(session_key, _budget_message(state))

    prompt, _attachments = await PREPARER.prepare_messages(req.messages, session_key)
    if _wants_json(req):
        prompt = f"{prompt}\n\n{_json_instruction(req)}"
    model = req.model if req.model and req.model != "auto" else SETTINGS.default_model

    if not prompt.strip():
        raise HTTPException(status_code=400, detail="no prompt content derived from messages")

    return prompt, session_key, model


def _log_request(kind: str, req: Any) -> None:
    """One compact line per generation request.

    A JSON-mode failure is reported from the *client* side ("Unexpected token
    'B'…"), which says nothing about which wrapper surface served the turn or
    whether the structured-output declaration was even seen — the two questions
    that actually locate the bug. `json_mode=off` on a turn the client thought
    was structured is the whole diagnosis in one field.
    """
    rf = getattr(req, "response_format", None)
    log.info(
        "%s: model=%s stream=%s json_mode=%s tools=%d",
        kind,
        req.model,
        bool(getattr(req, "stream", False)),
        getattr(rf, "type", None) or "off",
        len(getattr(req, "tools", None) or []),
    )


async def run_chat_completion(req: ChatCompletionRequest):
    """Shared implementation reused by /v1/chat/completions, /v1/completions,
    and the batches worker."""
    _log_request("chat/completions" + (" [tool-bridge]" if req.tools else ""), req)
    # Function calling: the CLIENT owns the agent loop, so the request is served
    # by the tool bridge (a direct Messages API call) and stops at the tool_call.
    # The bridge has no Claude Code CLI, hence no session workspace and no
    # generated files — a tools-carrying turn cannot produce a download link.
    #
    # There is no way to serve one turn both ways: the CLI runs its own tool
    # loop and cannot surface a caller-declared tool. So this is not a knob. A
    # chat UI that wants file downloads should stop sending `tools` instead —
    # in Open WebUI, set the model's Function Calling from "Native" to
    # "Default", which makes it run its own tool-selection call and send a
    # plain completion here. See README, "Generated files".
    if req.tools:
        return await _tool_bridge_completion(req)
    prep = await _prepare_run(req)
    if isinstance(prep, _InstantReply):
        return _json_safe_instant_reply(req, prep)
    prompt, session_key, model = prep

    if req.stream:
        return StreamingResponse(
            _stream_response(req, prompt, session_key, model),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    return await _sync_response(req, prompt, session_key, model)


@app.post("/v1/chat/completions", dependencies=[Depends(auth_dependency)])
async def chat_completions(req: ChatCompletionRequest):
    return await run_chat_completion(req)


async def _tool_bridge_completion(req: ChatCompletionRequest):
    """Serve a function-calling request via the tool bridge.

    Keeps the wrapper-level conveniences (session_id echo, per-conversation
    budget gate, usage ledger) but none of the agentic machinery — no prompt
    flattening, no CLI session, no KB addendum, no built-in tools.
    """
    session_key = derive_session_id(req.messages, req.session_id, req.user)

    if USAGE_LEDGER.enabled:
        state = await USAGE_LEDGER.snapshot(session_key)
        if state.over_budget:
            if _is_continue(req.messages):
                await USAGE_LEDGER.grant(session_key)
            else:
                return _json_safe_instant_reply(
                    req, _InstantReply(session_key, _budget_message(state))
                )

    model = req.model if req.model and req.model != "auto" else SETTINGS.default_model
    run_model, effort = split_model_effort(model)
    # Effort is a Claude Code CLI concept; the direct Messages API call has no
    # equivalent, so it is reported as unapplied rather than silently claimed.
    effort_info = {"applied": "api-default", "source": "tool-bridge", "requested": effort}

    if req.stream:
        async def _record(in_tok: int, out_tok: int) -> None:
            if USAGE_LEDGER.enabled:
                await USAGE_LEDGER.record(session_key, in_tok + out_tok)

        return StreamingResponse(
            tool_bridge.stream(
                req, run_model, model, session_key, effort_info, on_usage=_record
            ),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    result = await tool_bridge.complete(req, run_model)
    if USAGE_LEDGER.enabled:
        await USAGE_LEDGER.record(session_key, result.input_tokens + result.output_tokens)

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content=result.content,
                    tool_calls=result.tool_calls,
                ),
                finish_reason=result.finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=result.input_tokens,
            completion_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
        ),
        session_id=session_key,
        effort=effort_info,
    )
    data = response.model_dump(exclude_none=True)
    # Per the OpenAI spec a tool-call message carries an explicit null content,
    # not an absent key (and never "") — exclude_none above would drop it.
    data["choices"][0]["message"].setdefault("content", None)
    return JSONResponse(content=data)


async def _sync_response(
    req: ChatCompletionRequest,
    prompt: str,
    session_key: str,
    model: str,
) -> JSONResponse:
    run_model, effort = split_model_effort(model)
    result = await RUNNER.run_collect(
        prompt=prompt, session_key=session_key, model=run_model, effort=effort,
        clarify=_resolve_clarify(req), workspace_hint=_resolve_workspace_hint(req),
    )

    if result.error and not result.final_text:
        raise HTTPException(status_code=502, detail=f"claude failed: {result.error}")

    if USAGE_LEDGER.enabled:
        await USAGE_LEDGER.record(session_key, result.input_tokens + result.output_tokens)

    new_outputs: list[str] = []
    for evt in result.events:
        if evt.kind == "system" and evt.raw and isinstance(evt.raw.get("new_outputs"), list):
            new_outputs = list(evt.raw["new_outputs"])

    attachments = await _register_generated_files(
        paths=[Path(p) for p in new_outputs],
        session_key=session_key,
        inline=req.inline_generated_files,
    )

    final_text = result.final_text
    if _wants_json(req):
        # JSON mode: the client will JSON.parse the content verbatim. Reduce
        # the reply to the raw JSON value and never append the markdown file
        # trailer. If no JSON parses at all, pass the reply through unchanged
        # rather than mask what the model actually said.
        cleaned = extract_raw_json(final_text)
        if cleaned is not None:
            final_text = cleaned
    elif attachments and not req.inline_generated_files:
        final_text = _append_file_references(final_text, attachments)

    choice_msg = ChoiceMessage(
        role="assistant",
        content=final_text,
        attachments=attachments if attachments else None,
    )
    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=choice_msg,
                finish_reason=result.stop_reason or "stop",
            )
        ],
        usage=Usage(
            prompt_tokens=result.input_tokens,
            completion_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
        ),
        session_id=session_key,
        effort=_effort_info(run_model, effort),
    )
    return JSONResponse(content=response.model_dump(exclude_none=True))


async def _stream_response(
    req: ChatCompletionRequest,
    prompt: str,
    session_key: str,
    model: str,
) -> AsyncIterator[bytes]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    run_model, effort = split_model_effort(model)
    clarify = _resolve_clarify(req)
    workspace_hint = _resolve_workspace_hint(req)

    # JSON mode: the client parses the concatenated content as JSON, so nothing
    # non-JSON may enter the content stream — no reasoning/progress frames, no
    # file trailer. Answer deltas are buffered (a ```json fence can span chunk
    # boundaries) and emitted as one cleaned chunk right before the terminator.
    json_mode = _wants_json(req)
    json_parts: list[str] = []

    first_chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=""),
                finish_reason=None,
            )
        ],
        session_id=session_key,
        effort=_effort_info(run_model, effort),
    )
    # Preamble first: a chunky comment flushes the response head past any
    # buffering proxy immediately, so the client gets headers + first bytes now
    # rather than blocking on resp.start until an idle timer kills a connection
    # that never saw a status line.
    if _STREAM_PREAMBLE_BYTES > 0:
        yield b": " + b" " * _STREAM_PREAMBLE_BYTES + b"\n\n"
    yield _sse_chunk(first_chunk)

    finish_reason: Optional[str] = None
    new_outputs: list[str] = []
    errored: Optional[str] = None
    usage_tokens = 0

    # Timing for the visible "still working" progress tick. `stream_start` anchors
    # the elapsed display; `last_activity` is reset by any visible output, so
    # ticks fire only during genuine silence, at _STREAM_PROGRESS_SECONDS cadence.
    stream_start = time.monotonic()
    last_activity = stream_start

    # Reasoning/progress routing. In a content-wrapped mode (details, think_tags)
    # every reasoning frame goes into a single block on the content channel; the
    # block is opened lazily and closed before the first answer token. `nonlocal`
    # lets the nested emitters flip this flag as they open/close the block.
    reasoning_open = False

    def _reasoning_frame(text: str) -> Optional[bytes]:
        nonlocal reasoning_open
        if json_mode or _REASONING_CHANNEL == "none":
            return None
        wrap = _CONTENT_WRAP.get(_REASONING_CHANNEL)
        if wrap is not None:
            body = text if reasoning_open else wrap[0] + text
            reasoning_open = True
            return _content_sse(chunk_id, created, model, body)
        return _reasoning_sse(chunk_id, created, model, text)

    def _close_reasoning() -> Optional[bytes]:
        nonlocal reasoning_open
        if reasoning_open:
            reasoning_open = False
            wrap = _CONTENT_WRAP.get(_REASONING_CHANNEL)
            if wrap is not None:
                return _content_sse(chunk_id, created, model, wrap[1])
        return None

    # Pump runner events through a queue so we can interleave keep-alive
    # heartbeats. The producer task owns the run_stream generator — and thus the
    # Claude subprocess and the session lock — while the consumer below only
    # reads the queue. This decoupling is deliberate: wrapping the generator's
    # __anext__ in asyncio.wait_for() would, on every heartbeat timeout, cancel
    # the await *inside* run_stream, killing the subprocess and releasing the
    # lock mid-run. With a queue, a quiet stretch only times out queue.get().
    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    async def _pump() -> None:
        try:
            async for evt in RUNNER.run_stream(
                prompt=prompt, session_key=session_key, model=run_model, effort=effort,
                clarify=clarify, workspace_hint=workspace_hint,
            ):
                await queue.put(evt)
        except Exception as e:  # pragma: no cover - defensive
            await queue.put(e)
        finally:
            await queue.put(_DONE)

    producer = asyncio.create_task(_pump())

    # Everything past the first chunk is wrapped so the SSE stream is ALWAYS
    # terminated cleanly. Starlette only writes the chunked-encoding terminator
    # if this generator runs to completion; if it raises after the first byte,
    # the client receives a truncated body and aiohttp-based clients (Open WebUI)
    # surface it as "TransferEncodingError: Not enough data to satisfy transfer
    # length header". So any unexpected error here becomes a visible error chunk
    # + [DONE] rather than a severed connection.
    try:
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=_STREAM_HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    # No event for a while — Claude is thinking or running tools.
                    # Always keep the socket warm with a lightweight comment so
                    # idle-timeout proxies (and the client's own read timeout)
                    # don't sever a stream that simply hasn't produced text yet.
                    yield b": keep-alive\n\n"
                    now = time.monotonic()
                    if _STREAM_PROGRESS_SECONDS > 0 and now - last_activity >= _STREAM_PROGRESS_SECONDS:
                        # On a slower cadence than the heartbeat, emit a *visible*
                        # progress frame: it shows the run is alive in the feed and
                        # is real data that flushes any proxy a comment wouldn't.
                        frame = _reasoning_frame(
                            f"⏳ Still working… ({_format_elapsed(now - stream_start)} elapsed)\n"
                        )
                        if frame is not None:
                            yield frame
                        last_activity = now
                    continue

                if item is _DONE:
                    break
                if isinstance(item, Exception):
                    errored = str(item)
                    finish_reason = "stop"
                    continue

                evt = item
                if evt.kind == "text" and evt.text:
                    if json_mode:
                        json_parts.append(evt.text)
                        last_activity = time.monotonic()
                        continue
                    # Close any open <think> block (think_tags mode) so reasoning
                    # never bleeds into the answer content.
                    close = _close_reasoning()
                    if close is not None:
                        yield close
                    chunk = ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=DeltaMessage(content=evt.text),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield _sse_chunk(chunk)
                    last_activity = time.monotonic()
                elif evt.kind == "thinking" and evt.text:
                    # Stream reasoning on its own channel: gives live progress
                    # during long think phases and doubles as real byte flow,
                    # while keeping the answer content clean.
                    frame = _reasoning_frame(evt.text)
                    if frame is not None:
                        yield frame
                    last_activity = time.monotonic()
                elif evt.kind == "tool_use" and _STREAM_SHOW_ACTIVITY:
                    # Surface what Claude is doing during the no-answer-text phase
                    # (tool calls, subagent work) so the feed shows real progress
                    # instead of an apparently-stalled spinner.
                    frame = _reasoning_frame(_format_tool_use(evt) + "\n")
                    if frame is not None:
                        yield frame
                    last_activity = time.monotonic()
                elif evt.kind == "final":
                    meta = evt.raw or {}
                    finish_reason = meta.get("stop_reason") or "stop"
                    new_outputs = list(meta.get("new_outputs") or [])
                    usage_tokens = int(meta.get("input_tokens") or 0) + int(meta.get("output_tokens") or 0)
                    if meta.get("error"):
                        errored = str(meta["error"])
                elif evt.kind == "error":
                    errored = evt.text or errored
        finally:
            # On early exit — a client disconnect propagates CancelledError into
            # this generator — cancel the pump so run_stream tears down the
            # subprocess and releases the session lock. After a normal drain the
            # task is already finished and this is a no-op.
            if not producer.done():
                producer.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await producer

        # If the run ended while a <think> block was still open (reasoning but no
        # trailing answer text), close it before any trailer/terminator so the
        # content never ships an unbalanced tag.
        close = _close_reasoning()
        if close is not None:
            yield close

        # Post-stream bookkeeping. These touch disk (the token ledger) and the
        # file store, so they can fail — but a failure here must not truncate an
        # otherwise-complete response. Any error is folded into `errored` below.
        if USAGE_LEDGER.enabled:
            await USAGE_LEDGER.record(session_key, usage_tokens)

        attachments = await _register_generated_files(
            paths=[Path(p) for p in new_outputs],
            session_key=session_key,
            inline=req.inline_generated_files,
        )
        if attachments and not json_mode:
            trailer = "\n\n" + _append_file_references("", attachments).strip()
            trailer_chunk = ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=DeltaMessage(content=trailer),
                        finish_reason=None,
                    )
                ],
            )
            yield _sse_chunk(trailer_chunk)
    except asyncio.CancelledError:
        # Client disconnected: the socket is gone, so emitting a terminator would
        # only raise again. Propagate so Starlette/uvicorn finish tearing down.
        raise
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("streaming response failed mid-stream (session=%s)", session_key)
        if not errored:
            errored = f"internal wrapper error: {exc}"
        finish_reason = finish_reason or "stop"

    # Always-emitted clean terminator. Guarded so that even a serialization
    # failure when building the final chunk still closes the stream with [DONE]
    # rather than leaving a dangling chunked body.
    try:
        # Safety net for the error path, which skips the post-loop close above:
        # never leave a <think> block unterminated in the content stream.
        close = _close_reasoning()
        if close is not None:
            yield close
        # JSON mode: flush the buffered answer as one content chunk, reduced to
        # the raw JSON value. Emitted here (not in the post-loop section) so the
        # buffer also reaches the client on the mid-stream error path.
        if json_parts:
            body = "".join(json_parts)
            json_chunk = ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=DeltaMessage(content=extract_raw_json(body) or body),
                        finish_reason=None,
                    )
                ],
            )
            yield _sse_chunk(json_chunk)
        final_chunk = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason=finish_reason or "stop",
                )
            ],
        )
        yield _sse_chunk(final_chunk)
        if errored:
            err_payload = {"error": {"message": errored, "type": "upstream_error"}}
            yield f"data: {json.dumps(err_payload)}\n\n".encode("utf-8")
    except Exception:  # pragma: no cover - last-resort
        log.exception("failed to emit stream terminator (session=%s)", session_key)

    yield b"data: [DONE]\n\n"


def _sse_chunk(chunk: ChatCompletionChunk) -> bytes:
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode("utf-8")


def _content_sse(chunk_id: str, created: int, model: str, text: str) -> bytes:
    """A chat chunk carrying text on the answer (content) channel.

    Used for answer tokens, the file-reference trailer, and — when
    CLAUDE_WRAPPER_REASONING_CHANNEL=think_tags — reasoning wrapped in
    <think>…</think> for Open WebUI builds that don't render reasoning_content.
    """
    chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=DeltaMessage(content=text),
                finish_reason=None,
            )
        ],
    )
    return _sse_chunk(chunk)


def _reasoning_sse(chunk_id: str, created: int, model: str, text: str) -> bytes:
    """A chat chunk carrying live progress on the reasoning channel.

    Used for thinking, tool/subagent activity, and the periodic working tick —
    all of which are progress, not answer content, so they ride reasoning_content
    (rendered by Open WebUI as a collapsible "Thinking" section) and double as
    real byte flow that keeps proxies and read timers happy.
    """
    chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=DeltaMessage(reasoning_content=text),
                finish_reason=None,
            )
        ],
    )
    return _sse_chunk(chunk)


def _format_elapsed(seconds: float) -> str:
    secs = int(seconds)
    if secs < 60:
        return f"{secs}s"
    return f"{secs // 60}m{secs % 60:02d}s"


def _format_tool_use(evt) -> str:
    """One-line human summary of a tool_use event for the progress feed."""
    name = getattr(evt, "tool_name", None) or "tool"
    inp = getattr(evt, "tool_input", None) or {}
    summary = ""
    if isinstance(inp, dict):
        # Show the most informative argument for common tools.
        for key in ("command", "file_path", "path", "pattern", "query", "url", "prompt", "description"):
            val = inp.get(key)
            if isinstance(val, str) and val.strip():
                summary = " ".join(val.strip().split())
                break
    if len(summary) > 120:
        summary = summary[:117] + "…"
    return f"🔧 {name}: {summary}" if summary else f"🔧 {name}"


def _effort_info(run_model: str, requested_effort: Optional[str]) -> dict:
    """Resolved effort for the response: what was applied, and its origin.

    Mirrors the per-request launch log so clients can confirm an effort choice
    took effect rather than silently falling back to the server default.
    """
    applied, source = RUNNER._resolve_effort(run_model, requested_effort)
    return {"applied": applied or "cli-default", "source": source, "requested": requested_effort}


def _resolve_workspace_hint(req) -> bool:
    """Whether to tell Claude its cwd is a workspace that delivers new files.

    JSON mode forces it OFF: a structured-output client wants the value in the
    reply body, and a hint nudging Claude to put the deliverable in a file
    instead would starve it. Same reasoning as _resolve_clarify. The server-level
    switch is enforced in the runner (an empty configured prompt makes
    workspace_hint=True a no-op).
    """
    return not _wants_json(req)


def _resolve_clarify(req) -> bool:
    """Per-request intent for the interactive clarification protocol.

    Absent/None => on (the interactive default); explicit false opts a request
    out. The server-level switch is enforced in the runner (an empty configured
    prompt makes clarify=True a no-op), so this only governs per-request intent.

    JSON mode forces it OFF regardless of what the client asked for. The
    clarification protocol tells Claude to make its ENTIRE reply a list of
    questions when it hits an ambiguity, which is pure prose with no JSON in it
    — so the reply reaches a structured-output client (Vercel AI SDK
    generateObject → Vane) as an unparseable body and dies in JSON.parse. There
    is also nobody on the far end who *can* answer: a generateObject call is a
    one-shot machine request, not a chat turn. Asking is never the right move
    there, so the wire contract wins over the interactive default.
    """
    if _wants_json(req):
        return False
    val = getattr(req, "clarify", None)
    return True if val is None else bool(val)


# ---------- responses API (/v1/responses) ----------
#
# OpenAI's "ask and response" primitive. Two things distinguish it from
# /v1/chat/completions and force a dedicated implementation rather than a thin
# reshape of the chat response:
#
#  1. Conversation chaining is by id: the client passes the previous response's
#     `id` back as `previous_response_id` to continue the thread. We make that
#     work by deriving the response id FROM the session key (resp_<session_key>),
#     so handing it back deterministically reattaches to the same Claude session
#     via `derive_session_id`'s explicit-id path. A throwaway random id would
#     silently start a fresh session every turn.
#  2. The streaming wire format is a typed event sequence (response.created,
#     response.output_text.delta, response.completed …), NOT chat.completion
#     chunks. SDK clients parse on the event `type`, so chat chunks would be
#     unintelligible to them.

_RESPONSE_ID_PREFIX = "resp_"


def _response_id(session_key: str) -> str:
    return f"{_RESPONSE_ID_PREFIX}{session_key}"


def _session_from_response_id(response_id: Optional[str]) -> Optional[str]:
    """Recover the session key a `previous_response_id` points at.

    Inverse of `_response_id`. Tolerates ids without our prefix (a client may
    pass back a session key directly) by treating them as the key verbatim.
    """
    if not response_id:
        return None
    if response_id.startswith(_RESPONSE_ID_PREFIX):
        return response_id[len(_RESPONSE_ID_PREFIX) :]
    return response_id


def _responses_envelope(
    rreq: ResponsesRequest,
    session_key: str,
    model: str,
    run_model: str,
    effort: Optional[str],
    *,
    text: str,
    status: str,
    input_tokens: int,
    output_tokens: int,
    created: int,
    item_id: str,
    error: Optional[str] = None,
) -> dict:
    """Build a Responses-API `response` object (shared by sync + the terminal
    streaming event)."""
    output: list[dict] = []
    if text:
        output.append(
            {
                "type": "message",
                "id": item_id,
                "status": "completed" if status != "in_progress" else "in_progress",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    envelope = {
        "id": _response_id(session_key),
        "object": "response",
        "created_at": created,
        "status": status,
        "model": model,
        "output": output,
        "output_text": text,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "instructions": rreq.instructions,
        "temperature": rreq.temperature,
        "top_p": rreq.top_p,
        "max_output_tokens": rreq.max_output_tokens,
        "previous_response_id": rreq.previous_response_id,
        "metadata": rreq.metadata or {},
        # Non-standard parity field: which effort actually ran (mirrors chat).
        "effort": _effort_info(run_model, effort),
    }
    if error is not None:
        envelope["error"] = {"message": error, "type": "upstream_error"}
    return envelope


async def run_responses(rreq: ResponsesRequest, messages: list[ChatMessage]):
    """Entry point for /v1/responses, shared by the route and the batches worker.

    `messages` is the chat-shaped conversation already flattened from the
    Responses `input` by the route layer.
    """
    _log_request("responses", rreq)
    # `previous_response_id` wins over the message anchor so an explicit chain
    # always reattaches to the right session.
    session_key = derive_session_id(
        messages, _session_from_response_id(rreq.previous_response_id), rreq.user
    )
    chat_req = ChatCompletionRequest(
        model=rreq.model,
        messages=messages,
        stream=rreq.stream,
        temperature=rreq.temperature,
        top_p=rreq.top_p,
        max_tokens=rreq.max_output_tokens,
        user=rreq.user,
        session_id=session_key,
        # Carried across so _prepare_run appends the raw-JSON output instruction
        # and turns the clarification protocol off, exactly as on the chat path.
        response_format=rreq.response_format,
    )

    prep = await _prepare_run(chat_req)
    if isinstance(prep, _InstantReply):
        if _wants_json(rreq):
            raise HTTPException(
                status_code=502, detail=_instant_reply_error(rreq, prep.text)
            )
        return _responses_instant_reply(rreq, prep.session_key, prep.text)
    prompt, session_key, model = prep

    if rreq.stream:
        return StreamingResponse(
            _responses_stream(rreq, prompt, session_key, model),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    return await _responses_sync(rreq, prompt, session_key, model)


async def _responses_sync(
    rreq: ResponsesRequest, prompt: str, session_key: str, model: str
) -> JSONResponse:
    run_model, effort = split_model_effort(model)
    result = await RUNNER.run_collect(
        prompt=prompt, session_key=session_key, model=run_model, effort=effort,
        clarify=_resolve_clarify(rreq), workspace_hint=_resolve_workspace_hint(rreq),
    )
    if result.error and not result.final_text:
        raise HTTPException(status_code=502, detail=f"claude failed: {result.error}")

    if USAGE_LEDGER.enabled:
        await USAGE_LEDGER.record(session_key, result.input_tokens + result.output_tokens)

    new_outputs: list[str] = []
    for evt in result.events:
        if evt.kind == "system" and evt.raw and isinstance(evt.raw.get("new_outputs"), list):
            new_outputs = list(evt.raw["new_outputs"])

    attachments = await _register_generated_files(
        paths=[Path(p) for p in new_outputs], session_key=session_key, inline=False
    )
    final_text = result.final_text
    if _wants_json(rreq):
        # Same contract as the chat path: reduce the reply to the raw JSON value
        # (never the file trailer), and fail loudly with the model's own words
        # when nothing parses rather than hand the client prose to JSON.parse.
        cleaned = extract_raw_json(final_text)
        if cleaned is None:
            raise HTTPException(status_code=502, detail=_json_mode_error(rreq, final_text))
        final_text = cleaned
    elif attachments:
        final_text = _append_file_references(final_text, attachments)

    envelope = _responses_envelope(
        rreq,
        session_key,
        model,
        run_model,
        effort,
        text=final_text,
        status="completed",
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        created=int(time.time()),
        item_id=f"msg_{uuid.uuid4().hex[:24]}",
    )
    return JSONResponse(content=envelope)


async def _responses_stream(
    rreq: ResponsesRequest, prompt: str, session_key: str, model: str
) -> AsyncIterator[bytes]:
    run_model, effort = split_model_effort(model)
    clarify = _resolve_clarify(rreq)
    workspace_hint = _resolve_workspace_hint(rreq)
    created = int(time.time())
    item_id = f"msg_{uuid.uuid4().hex[:24]}"
    seq = 0
    # JSON mode: the client concatenates the text deltas and parses the result,
    # so answer text is buffered and emitted as one cleaned delta at the end (a
    # ```json fence can straddle chunk boundaries), and the file trailer is
    # suppressed. Mirrors the chat stream.
    json_mode = _wants_json(rreq)

    def ev(event_type: str, payload: dict) -> bytes:
        nonlocal seq
        body = {"type": event_type, "sequence_number": seq, **payload}
        seq += 1
        # Both an `event:` line and `type` in the data — SDKs key on one or the
        # other depending on transport.
        return f"event: {event_type}\ndata: {json.dumps(body)}\n\n".encode("utf-8")

    def envelope(status: str, text: str, in_tok: int, out_tok: int, error: Optional[str] = None) -> dict:
        return _responses_envelope(
            rreq, session_key, model, run_model, effort,
            text=text, status=status, input_tokens=in_tok, output_tokens=out_tok,
            created=created, item_id=item_id, error=error,
        )

    # Opening events — the skeleton an SDK needs before deltas start flowing.
    yield ev("response.created", {"response": envelope("in_progress", "", 0, 0)})
    yield ev("response.in_progress", {"response": envelope("in_progress", "", 0, 0)})
    yield ev(
        "response.output_item.added",
        {"output_index": 0, "item": {"id": item_id, "type": "message",
                                     "status": "in_progress", "role": "assistant", "content": []}},
    )
    yield ev(
        "response.content_part.added",
        {"item_id": item_id, "output_index": 0, "content_index": 0,
         "part": {"type": "output_text", "text": "", "annotations": []}},
    )

    # Same decoupled producer/consumer pattern as the chat stream: the pump owns
    # the runner (and thus the subprocess + session lock); the consumer only
    # reads the queue, so a quiet think phase times out queue.get() — emitting a
    # keep-alive — without cancelling the run mid-flight.
    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()
    text_parts: list[str] = []
    in_tok = out_tok = 0
    new_outputs: list[str] = []
    errored: Optional[str] = None

    async def _pump() -> None:
        try:
            async for evt in RUNNER.run_stream(
                prompt=prompt, session_key=session_key, model=run_model, effort=effort,
                clarify=clarify, workspace_hint=workspace_hint,
            ):
                await queue.put(evt)
        except Exception as e:  # pragma: no cover - defensive
            await queue.put(e)
        finally:
            await queue.put(_DONE)

    producer = asyncio.create_task(_pump())

    try:
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=_STREAM_HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield b": keep-alive\n\n"
                    continue

                if item is _DONE:
                    break
                if isinstance(item, Exception):
                    errored = str(item)
                    continue

                evt = item
                if evt.kind == "text" and evt.text:
                    text_parts.append(evt.text)
                    if json_mode:
                        continue
                    yield ev(
                        "response.output_text.delta",
                        {"item_id": item_id, "output_index": 0,
                         "content_index": 0, "delta": evt.text},
                    )
                elif evt.kind == "final":
                    meta = evt.raw or {}
                    new_outputs = list(meta.get("new_outputs") or [])
                    in_tok = int(meta.get("input_tokens") or 0)
                    out_tok = int(meta.get("output_tokens") or 0)
                    if meta.get("error"):
                        errored = str(meta["error"])
                elif evt.kind == "error":
                    errored = evt.text or errored
        finally:
            if not producer.done():
                producer.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await producer

        if USAGE_LEDGER.enabled:
            await USAGE_LEDGER.record(session_key, in_tok + out_tok)

        attachments = await _register_generated_files(
            paths=[Path(p) for p in new_outputs], session_key=session_key, inline=False
        )
        if attachments and not json_mode:
            trailer = "\n\n" + _append_file_references("", attachments).strip()
            text_parts.append(trailer)
            yield ev(
                "response.output_text.delta",
                {"item_id": item_id, "output_index": 0, "content_index": 0, "delta": trailer},
            )

        full_text = "".join(text_parts)
        if json_mode:
            # Flush the buffered answer as a single delta, reduced to the raw
            # JSON value. If nothing parses, the response head is long gone, so
            # the turn fails on the stream's own channel (response.failed) with
            # no text emitted at all — never prose the client will choke on.
            cleaned = extract_raw_json(full_text)
            if cleaned is None:
                errored = errored or _json_mode_error(rreq, full_text)
                full_text = ""
            else:
                full_text = cleaned
                yield ev(
                    "response.output_text.delta",
                    {"item_id": item_id, "output_index": 0,
                     "content_index": 0, "delta": full_text},
                )
        yield ev(
            "response.output_text.done",
            {"item_id": item_id, "output_index": 0, "content_index": 0, "text": full_text},
        )
        yield ev(
            "response.content_part.done",
            {"item_id": item_id, "output_index": 0, "content_index": 0,
             "part": {"type": "output_text", "text": full_text, "annotations": []}},
        )
        yield ev(
            "response.output_item.done",
            {"output_index": 0, "item": {"id": item_id, "type": "message", "status": "completed",
                                         "role": "assistant",
                                         "content": [{"type": "output_text", "text": full_text,
                                                      "annotations": []}]}},
        )
        if errored:
            yield ev("response.failed", {"response": envelope("failed", full_text, in_tok, out_tok, error=errored)})
        else:
            yield ev("response.completed", {"response": envelope("completed", full_text, in_tok, out_tok)})
    except asyncio.CancelledError:
        # Client disconnected — the socket is gone; just unwind.
        raise
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("responses streaming failed mid-stream (session=%s)", session_key)
        with contextlib.suppress(Exception):
            yield ev(
                "response.failed",
                {"response": envelope("failed", "".join(text_parts), in_tok, out_tok,
                                      error=f"internal wrapper error: {exc}")},
            )
    # NOTE: the Responses streaming protocol terminates on the terminal event
    # (response.completed/failed) — there is no chat-style `data: [DONE]`
    # sentinel, and emitting one makes strict SDK parsers choke.


def _responses_instant_reply(rreq: ResponsesRequest, session_key: str, text: str):
    """Render a wrapper-authored message (budget checkpoint, `stats`/`context`
    report) in Responses shape, without a Claude run."""
    base_model = rreq.model if rreq.model and rreq.model != "auto" else SETTINGS.default_model
    run_model, effort = split_model_effort(base_model)
    if rreq.stream:
        return StreamingResponse(
            _responses_static_stream(rreq, session_key, base_model, run_model, effort, text),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    envelope = _responses_envelope(
        rreq, session_key, base_model, run_model, effort,
        text=text, status="completed", input_tokens=0, output_tokens=0,
        created=int(time.time()), item_id=f"msg_{uuid.uuid4().hex[:24]}",
    )
    return JSONResponse(content=envelope)


async def _responses_static_stream(
    rreq: ResponsesRequest,
    session_key: str,
    model: str,
    run_model: str,
    effort: Optional[str],
    text: str,
) -> AsyncIterator[bytes]:
    """Stream a fixed, already-known message as a complete Responses event
    sequence. Used for wrapper-authored replies (budget checkpoint,
    `stats`/`context` report), where there is no Claude run."""
    created = int(time.time())
    item_id = f"msg_{uuid.uuid4().hex[:24]}"
    seq = 0

    def ev(event_type: str, payload: dict) -> bytes:
        nonlocal seq
        body = {"type": event_type, "sequence_number": seq, **payload}
        seq += 1
        return f"event: {event_type}\ndata: {json.dumps(body)}\n\n".encode("utf-8")

    def envelope(status: str) -> dict:
        return _responses_envelope(
            rreq, session_key, model, run_model, effort,
            text=text, status=status, input_tokens=0, output_tokens=0,
            created=created, item_id=item_id,
        )

    yield ev("response.created", {"response": envelope("in_progress")})
    yield ev(
        "response.output_item.added",
        {"output_index": 0, "item": {"id": item_id, "type": "message",
                                     "status": "in_progress", "role": "assistant", "content": []}},
    )
    yield ev(
        "response.content_part.added",
        {"item_id": item_id, "output_index": 0, "content_index": 0,
         "part": {"type": "output_text", "text": "", "annotations": []}},
    )
    yield ev(
        "response.output_text.delta",
        {"item_id": item_id, "output_index": 0, "content_index": 0, "delta": text},
    )
    yield ev(
        "response.output_text.done",
        {"item_id": item_id, "output_index": 0, "content_index": 0, "text": text},
    )
    yield ev("response.completed", {"response": envelope("completed")})


# ---------- per-conversation budget gating & instant chat commands ----------


def _last_user_text(messages: list[ChatMessage]) -> str:
    """Flatten the most recent user message to plain text.

    Content may be a bare string or a list of content parts (multimodal); we pull
    text from whichever shape it is and ignore non-text parts.
    """
    for msg in reversed(messages):
        if msg.role != "user":
            continue
        content = msg.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        parts = [getattr(p, "text", "") for p in content if getattr(p, "text", "")]
        return "\n".join(parts)
    return ""


def _normalize_keyword(s: str) -> str:
    return s.strip().lower().strip(".!?,;:'\"() \t\r\n")


def _is_continue(messages: list[ChatMessage]) -> bool:
    """Whether the latest user message is a 'continue' confirmation.

    Matches a configured keyword as the whole message or as a leading/trailing
    word, so "continue", "yes, continue", and "continue please" all resume.
    """
    text = _normalize_keyword(_last_user_text(messages))
    if not text:
        return False
    for kw in SETTINGS.budget_continue_keywords:
        if text == kw or text.startswith(kw + " ") or text.endswith(" " + kw):
            return True
    return False


_USAGE_COMMANDS = frozenset({"stats", "context"})


def _usage_command(messages: list[ChatMessage]) -> Optional[str]:
    """The usage command named by the latest user message, or None.

    Only a message that IS the command ("stats", "/context", "Stats!") matches —
    never one that merely contains the word — so ordinary prompts can't be
    short-circuited by accident.
    """
    text = _normalize_keyword(_last_user_text(messages)).lstrip("/")
    return text if text in _USAGE_COMMANDS else None


def _usage_report(session_key: str, state: UsageState) -> str:
    """Instant usage summary for the `stats` / `context` chat commands."""
    if not USAGE_LEDGER.enabled:
        return (
            "📊 **Usage stats** — token accounting is disabled on this server "
            "(no session allowance configured), so there is nothing to report. "
            "Set `CLAUDE_WRAPPER_SESSION_PLAN` (or "
            "`CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE`) to enable it."
        )
    remaining = max(0, state.allowance_tokens - state.spent_tokens)
    session_pct = 100.0 * state.spent_tokens / SETTINGS.session_token_allowance
    plan = SETTINGS.session_plan or "custom"
    req_s = "s" if state.requests != 1 else ""
    block_s = "s" if state.grants != 1 else ""
    return (
        f"📊 **Usage stats**\n"
        f"- **Spent (this conversation):** {state.spent_tokens:,} tokens across "
        f"{state.requests} request{req_s} ({session_pct:.1f}% of the session allowance)\n"
        f"- **Remaining before the next checkpoint:** {remaining:,} of "
        f"{state.allowance_tokens:,} tokens "
        f"({state.grants} × {state.block_tokens:,}-token block{block_s})\n"
        f"- **Session allowance:** {SETTINGS.session_token_allowance:,} tokens "
        f"({plan} plan), {SETTINGS.session_block_percent:g}% per block\n"
        f"- **Session key:** `{session_key}`"
    )


def _budget_message(state: UsageState) -> str:
    pct = f"{SETTINGS.session_block_percent:g}"
    return (
        f"⏸️ **Usage checkpoint.** This conversation has used "
        f"**{state.spent_tokens:,} tokens**, reaching its **{state.block_tokens:,}-token** "
        f"budget block ({pct}% of the configured session allowance). "
        f"Reply **continue** to allow another block, or start a new chat to reset."
    )


def _json_safe_instant_reply(req: ChatCompletionRequest, prep: "_InstantReply"):
    """Render an instant reply, or refuse to in JSON mode.

    A wrapper-authored message is prose; a structured-output client parses the
    body and dies on it. Fail with the reason instead of shipping a 200 the
    client cannot read. See json_mode.instant_reply_error.
    """
    if _wants_json(req):
        raise HTTPException(status_code=502, detail=_instant_reply_error(req, prep.text))
    return _instant_reply(req, prep.session_key, prep.text)


def _instant_reply(req: ChatCompletionRequest, session_key: str, text: str):
    """Return a wrapper-authored message in the request's shape without running
    Claude (budget checkpoint, `stats`/`context` report)."""
    if req.stream:
        return StreamingResponse(
            _instant_reply_stream(req, session_key, text),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChoiceMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=Usage(),
        session_id=session_key,
    )
    return JSONResponse(content=response.model_dump(exclude_none=True))


async def _instant_reply_stream(
    req: ChatCompletionRequest, session_key: str, text: str
) -> AsyncIterator[bytes]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    yield _sse_chunk(
        ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=req.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=text),
                    finish_reason=None,
                )
            ],
            session_id=session_key,
        )
    )
    yield _sse_chunk(
        ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=req.model,
            choices=[
                ChatCompletionChunkChoice(index=0, delta=DeltaMessage(), finish_reason="stop")
            ],
        )
    )
    yield b"data: [DONE]\n\n"


# ---------- generated-file handling ----------


async def _register_generated_files(
    paths: list[Path],
    session_key: str,
    inline: bool,
) -> list[dict]:
    records: list[dict] = []
    for p in paths:
        try:
            if not p.exists() or not p.is_file():
                continue
            record = await FILE_STORE.ingest_path(
                src=p,
                filename=p.name,
                purpose="assistant_output",
                session_id=session_key,
                source="generated",
            )
            entry = record.to_openai()
            # Non-OpenAI key on an already non-OpenAI dict, so SDK clients get
            # the link without having to parse it back out of the markdown.
            url = _file_download_url(record.id)
            if url:
                entry["url"] = url
            if inline:
                data = p.read_bytes()
                entry["content_base64"] = base64.b64encode(data).decode("ascii")
            records.append(entry)
        except Exception:  # pragma: no cover
            log.exception("failed to register generated file %s", p)
    return records


def _file_download_url(file_id: str) -> Optional[str]:
    """Download URL for a generated file, or None when the wrapper has no way to
    name itself (the caller then degrades to plain text).

    The query carries a per-file capability, because a browser clicking a link
    in chat sends no Authorization header. Signature deliberately unchanged:
    _append_file_references picks between its two line formats purely on whether
    this returns a URL.
    """
    base = SETTINGS.public_base_url or current_origin()
    if not base:
        return None
    query = download_tokens.mint(
        file_id, SETTINGS.download_signing_key, SETTINGS.download_url_ttl_seconds
    )
    url = f"{base}/v1/files/{file_id}/content"
    return f"{url}?{query}" if query else url


def _append_file_references(text: str, attachments: list[dict]) -> str:
    lines = []
    for a in attachments:
        filename = a.get("filename") or a["id"]
        meta = f"{a.get('mime_type')}, {a.get('bytes')} bytes"
        url = _file_download_url(a["id"])
        if url:
            lines.append(f"- [{filename}]({url}) ({meta}, file_id=`{a['id']}`)")
        else:
            lines.append(f"- {filename} ({meta}) → file_id={a['id']}")
    ref_block = "Generated files:\n" + "\n".join(lines)
    if not text:
        return ref_block
    return f"{text}\n\n{ref_block}"


