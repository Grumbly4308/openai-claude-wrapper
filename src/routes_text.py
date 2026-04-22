"""Legacy /v1/completions and new /v1/responses — both wrap chat_completions."""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .deps import auth_dependency
from .models import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ResponsesRequest,
    TextContent,
    Usage,
)


router = APIRouter()


def _chat_request_from_prompt(prompt: str, base: CompletionRequest) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=base.model,
        messages=[ChatMessage(role="user", content=prompt)],
        stream=base.stream,
        temperature=base.temperature,
        top_p=base.top_p,
        max_tokens=base.max_tokens,
        user=base.user,
    )


@router.post("/v1/completions", dependencies=[Depends(auth_dependency)])
async def legacy_completions(req: CompletionRequest):
    from .main import run_chat_completion  # lazy to avoid circular import

    prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
    if req.stream and len(prompts) > 1:
        raise HTTPException(status_code=400, detail="stream with multiple prompts not supported")

    if req.stream:
        inner = await run_chat_completion(_chat_request_from_prompt(prompts[0], req))
        return inner  # StreamingResponse already shaped for chat; clients tolerate this

    choices: list[CompletionChoice] = []
    total_in = total_out = 0
    for idx, p in enumerate(prompts):
        inner: JSONResponse = await run_chat_completion(_chat_request_from_prompt(p, req))
        body: dict[str, Any] = inner.body if hasattr(inner, "body") else {}
        import json as _json

        try:
            data = _json.loads(body.decode() if isinstance(body, (bytes, bytearray)) else body)
        except Exception:
            data = {}
        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        text = msg.get("content") or ""
        if req.echo:
            text = p + text
        choices.append(
            CompletionChoice(
                text=text,
                index=idx,
                finish_reason=(data.get("choices") or [{}])[0].get("finish_reason") or "stop",
            )
        )
        usage = data.get("usage") or {}
        total_in += int(usage.get("prompt_tokens") or 0)
        total_out += int(usage.get("completion_tokens") or 0)

    resp = CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=req.model,
        choices=choices,
        usage=Usage(prompt_tokens=total_in, completion_tokens=total_out, total_tokens=total_in + total_out),
    )
    return JSONResponse(content=resp.model_dump(exclude_none=True))


def _responses_to_chat_messages(req: ResponsesRequest) -> list[ChatMessage]:
    msgs: list[ChatMessage] = []
    if req.instructions:
        msgs.append(ChatMessage(role="system", content=req.instructions))
    if isinstance(req.input, str):
        msgs.append(ChatMessage(role="user", content=req.input))
        return msgs
    for item in req.input:
        if not isinstance(item, dict):
            msgs.append(ChatMessage(role="user", content=str(item)))
            continue
        role = item.get("role") or "user"
        content = item.get("content")
        if isinstance(content, str):
            msgs.append(ChatMessage(role=role, content=content))
        elif isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict):
                    ctype = c.get("type")
                    if ctype in ("input_text", "output_text", "text"):
                        parts.append(TextContent(type="text", text=c.get("text") or ""))
                    elif ctype == "input_image" and c.get("image_url"):
                        from .models import ImageContent, ImageURL

                        parts.append(ImageContent(type="image_url", image_url=ImageURL(url=c["image_url"])))
                    elif ctype == "input_file" and c.get("file_id"):
                        from .models import FileContent, FilePayload

                        parts.append(FileContent(type="file", file=FilePayload(file_id=c["file_id"])))
            msgs.append(ChatMessage(role=role, content=parts or [TextContent(type="text", text="")]))
        elif content is None:
            continue
        else:
            msgs.append(ChatMessage(role=role, content=str(content)))
    return msgs


@router.post("/v1/responses", dependencies=[Depends(auth_dependency)])
async def responses_api(req: ResponsesRequest):
    from .main import run_chat_completion

    messages = _responses_to_chat_messages(req)
    if not messages:
        raise HTTPException(status_code=400, detail="no input provided")

    chat_req = ChatCompletionRequest(
        model=req.model,
        messages=messages,
        stream=req.stream,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_output_tokens,
        user=req.user,
        session_id=req.previous_response_id,
    )
    inner = await run_chat_completion(chat_req)
    if req.stream:
        return inner

    import json as _json

    body = inner.body if hasattr(inner, "body") else b""
    try:
        data = _json.loads(body.decode() if isinstance(body, (bytes, bytearray)) else body)
    except Exception:
        data = {}
    msg = ((data.get("choices") or [{}])[0].get("message") or {})
    text = msg.get("content") or ""
    usage = data.get("usage") or {}

    resp = {
        "id": f"resp-{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(time.time()),
        "model": req.model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": f"msg-{uuid.uuid4().hex[:16]}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "output_text": text,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        },
        "metadata": req.metadata or {},
        "previous_response_id": req.previous_response_id,
    }
    return JSONResponse(content=resp)
