from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Header,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, StreamingResponse

from .config import SETTINGS, advertised_models, split_model_effort, supported_models
from .converters import derive_session_id
from .deps import FILE_STORE, PREPARER, RUNNER, USAGE_LEDGER, auth_dependency
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


@app.on_event("shutdown")
async def _shutdown() -> None:
    await PREPARER.aclose()


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


@app.get("/v1/files/{file_id}/content", dependencies=[Depends(auth_dependency)])
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


class _BudgetPause:
    """Sentinel returned by _prepare_run when a conversation hit its token cap.

    Carries the bookkeeping each endpoint shape needs to render its own
    checkpoint message (chat-shaped vs Responses-shaped) without running Claude.
    """

    __slots__ = ("session_key", "state")

    def __init__(self, session_key: str, state: UsageState) -> None:
        self.session_key = session_key
        self.state = state


async def _prepare_run(req: ChatCompletionRequest):
    """Shared prelude for every text-generation endpoint.

    Resolves the session key, enforces the per-conversation token budget, builds
    the prompt, and resolves the model. Returns either ``(prompt, session_key,
    model)`` ready to run, or a ``_BudgetPause`` the caller renders in its own
    response shape. Used by /v1/chat/completions, /v1/completions, /v1/responses,
    and the batches worker.
    """
    session_key = derive_session_id(req.messages, req.session_id, req.user)

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
                return _BudgetPause(session_key, state)

    prompt, _attachments = await PREPARER.prepare_messages(req.messages, session_key)
    model = req.model if req.model and req.model != "auto" else SETTINGS.default_model

    if not prompt.strip():
        raise HTTPException(status_code=400, detail="no prompt content derived from messages")

    return prompt, session_key, model


async def run_chat_completion(req: ChatCompletionRequest):
    """Shared implementation reused by /v1/chat/completions, /v1/completions,
    and the batches worker."""
    prep = await _prepare_run(req)
    if isinstance(prep, _BudgetPause):
        return _budget_pause(req, prep.session_key, prep.state)
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


async def _sync_response(
    req: ChatCompletionRequest,
    prompt: str,
    session_key: str,
    model: str,
) -> JSONResponse:
    run_model, effort = split_model_effort(model)
    result = await RUNNER.run_collect(
        prompt=prompt, session_key=session_key, model=run_model, effort=effort,
        clarify=_resolve_clarify(req),
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
    if attachments and not req.inline_generated_files:
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
                prompt=prompt, session_key=session_key, model=run_model, effort=effort, clarify=clarify
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
                        yield _reasoning_sse(
                            chunk_id,
                            created,
                            model,
                            f"⏳ Still working… ({_format_elapsed(now - stream_start)} elapsed)\n",
                        )
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
                    yield _reasoning_sse(chunk_id, created, model, evt.text)
                    last_activity = time.monotonic()
                elif evt.kind == "tool_use" and _STREAM_SHOW_ACTIVITY:
                    # Surface what Claude is doing during the no-answer-text phase
                    # (tool calls, subagent work) so the feed shows real progress
                    # instead of an apparently-stalled spinner.
                    yield _reasoning_sse(chunk_id, created, model, _format_tool_use(evt) + "\n")
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
        if attachments:
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


def _resolve_clarify(req) -> bool:
    """Per-request intent for the interactive clarification protocol.

    Absent/None => on (the interactive default); explicit false opts a request
    out. The server-level switch is enforced in the runner (an empty configured
    prompt makes clarify=True a no-op), so this only governs per-request intent.
    """
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
    )

    prep = await _prepare_run(chat_req)
    if isinstance(prep, _BudgetPause):
        return _responses_budget_pause(rreq, prep.session_key, prep.state)
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
        clarify=_resolve_clarify(rreq),
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
    if attachments:
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
    created = int(time.time())
    item_id = f"msg_{uuid.uuid4().hex[:24]}"
    seq = 0

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
                prompt=prompt, session_key=session_key, model=run_model, effort=effort, clarify=clarify
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
        if attachments:
            trailer = "\n\n" + _append_file_references("", attachments).strip()
            text_parts.append(trailer)
            yield ev(
                "response.output_text.delta",
                {"item_id": item_id, "output_index": 0, "content_index": 0, "delta": trailer},
            )

        full_text = "".join(text_parts)
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


def _responses_budget_pause(rreq: ResponsesRequest, session_key: str, state: UsageState):
    """Render the per-conversation checkpoint in Responses shape (no Claude run)."""
    text = _budget_message(state)
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
    sequence. Used for the budget checkpoint, where there is no Claude run."""
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


# ---------- per-conversation budget gating ----------


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


def _budget_message(state: UsageState) -> str:
    pct = f"{SETTINGS.session_block_percent:g}"
    return (
        f"⏸️ **Usage checkpoint.** This conversation has used "
        f"**{state.spent_tokens:,} tokens**, reaching its **{state.block_tokens:,}-token** "
        f"budget block ({pct}% of the configured session allowance). "
        f"Reply **continue** to allow another block, or start a new chat to reset."
    )


def _budget_pause(req: ChatCompletionRequest, session_key: str, state: UsageState):
    """Return a checkpoint message in the request's shape without running Claude."""
    text = _budget_message(state)
    if req.stream:
        return StreamingResponse(
            _budget_pause_stream(req, session_key, text),
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


async def _budget_pause_stream(
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
            if inline:
                data = p.read_bytes()
                entry["content_base64"] = base64.b64encode(data).decode("ascii")
            records.append(entry)
        except Exception:  # pragma: no cover
            log.exception("failed to register generated file %s", p)
    return records


def _file_download_url(file_id: str) -> Optional[str]:
    base = SETTINGS.public_base_url
    if not base:
        return None
    return f"{base}/v1/files/{file_id}/content"


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


