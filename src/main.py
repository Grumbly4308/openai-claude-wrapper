from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import mimetypes
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

from .config import SETTINGS, SUPPORTED_MODELS
from .converters import derive_session_id
from .deps import FILE_STORE, PREPARER, RUNNER, auth_dependency
from .models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChoiceMessage,
    DeltaMessage,
    ModelInfo,
    ModelList,
    Usage,
)
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


app = FastAPI(title="Claude Code OpenAI Wrapper", version="0.1.0")


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
    return ModelList(data=[ModelInfo(id=m, created=now) for m in SUPPORTED_MODELS])


@app.get("/v1/models/{model_id}", dependencies=[Depends(auth_dependency)])
async def retrieve_model(model_id: str) -> ModelInfo:
    if model_id not in SUPPORTED_MODELS:
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


async def run_chat_completion(req: ChatCompletionRequest):
    """Shared implementation reused by /v1/chat/completions, /v1/completions,
    /v1/responses, and the batches worker."""
    session_key = derive_session_id(req.messages, req.session_id, req.user)
    prompt, _attachments = await PREPARER.prepare_messages(req.messages, session_key)
    model = req.model if req.model and req.model != "auto" else SETTINGS.default_model

    if not prompt.strip():
        raise HTTPException(status_code=400, detail="no prompt content derived from messages")

    if req.stream:
        return StreamingResponse(
            _stream_response(req, prompt, session_key, model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
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
    result = await RUNNER.run_collect(prompt=prompt, session_key=session_key, model=model)

    if result.error and not result.final_text:
        raise HTTPException(status_code=502, detail=f"claude failed: {result.error}")

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
    )
    yield _sse_chunk(first_chunk)

    finish_reason: Optional[str] = None
    new_outputs: list[str] = []
    errored: Optional[str] = None

    try:
        async for evt in RUNNER.run_stream(prompt=prompt, session_key=session_key, model=model):
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
            elif evt.kind == "final":
                meta = evt.raw or {}
                finish_reason = meta.get("stop_reason") or "stop"
                new_outputs = list(meta.get("new_outputs") or [])
                if meta.get("error"):
                    errored = str(meta["error"])
            elif evt.kind == "error":
                errored = evt.text or errored
    except Exception as e:  # pragma: no cover - defensive
        errored = str(e)
        finish_reason = "stop"

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

    final_chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason=finish_reason or ("stop" if not errored else "stop"),
            )
        ],
    )
    yield _sse_chunk(final_chunk)
    if errored:
        err_payload = {"error": {"message": errored, "type": "upstream_error"}}
        yield f"data: {json.dumps(err_payload)}\n\n".encode("utf-8")

    yield b"data: [DONE]\n\n"


def _sse_chunk(chunk: ChatCompletionChunk) -> bytes:
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode("utf-8")


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


