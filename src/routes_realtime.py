"""Realtime endpoints — WebSocket bridge to chat completions.

OpenAI's realtime API is a bidirectional voice+text protocol. We ship a
minimal text-only subset over a WebSocket at ``/v1/realtime``:

    client → {"type": "session.update", "session": {"instructions": "..."}}
    client → {"type": "response.create", "input": "hello"}
    server → {"type": "response.output_text.delta", "delta": "..."}
    server → {"type": "response.completed"}

This gives OpenAI-realtime SDKs a functioning handshake for text-only
chat, without claiming to support the audio channel (which the wrapper
exposes separately through /v1/audio/*).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .claude_runner import StreamEvent
from .config import SETTINGS
from .converters import derive_session_id
from .deps import PREPARER, RUNNER
from .models import ChatMessage


log = logging.getLogger("claude_wrapper.realtime")
router = APIRouter()


@router.get("/v1/realtime/sessions")
async def realtime_session_info():
    return JSONResponse(
        content={
            "id": f"sess_{uuid.uuid4().hex[:24]}",
            "object": "realtime.session",
            "model": SETTINGS.default_model,
            "modalities": ["text"],
            "url": "/v1/realtime",
        }
    )


@router.websocket("/v1/realtime")
async def realtime_ws(ws: WebSocket) -> None:
    # Auth: OpenAI clients send the key in the ``Sec-WebSocket-Protocol``
    # header or as an ``Authorization`` header on the upgrade request.
    if SETTINGS.require_auth:
        auth = ws.headers.get("authorization") or ""
        token = auth[7:] if auth.lower().startswith("bearer ") else auth
        if token not in SETTINGS.api_keys:
            await ws.close(code=4401)
            return

    await ws.accept()
    session_id = f"rt-{uuid.uuid4().hex[:16]}"
    instructions: str = ""
    await ws.send_text(
        json.dumps(
            {
                "type": "session.created",
                "session": {"id": session_id, "model": SETTINGS.default_model, "modalities": ["text"]},
            }
        )
    )

    try:
        while True:
            raw = await ws.receive_text()
            try:
                evt = json.loads(raw)
            except Exception:
                await ws.send_text(json.dumps({"type": "error", "error": {"message": "invalid JSON"}}))
                continue
            etype = evt.get("type")

            if etype == "session.update":
                sess = evt.get("session") or {}
                instructions = sess.get("instructions") or instructions
                await ws.send_text(json.dumps({"type": "session.updated", "session": {"id": session_id}}))
                continue

            if etype == "response.create":
                await _handle_response_create(ws, evt, session_id, instructions)
                continue

            if etype == "input_text.append":
                # gather until response.create
                continue

            await ws.send_text(
                json.dumps({"type": "error", "error": {"message": f"unsupported event type: {etype}"}})
            )
    except WebSocketDisconnect:
        return


async def _handle_response_create(
    ws: WebSocket, evt: dict[str, Any], session_id: str, instructions: str
) -> None:
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    text = evt.get("input") or (evt.get("response") or {}).get("instructions") or ""
    if not isinstance(text, str):
        text = str(text)

    messages = []
    if instructions:
        messages.append(ChatMessage(role="system", content=instructions))
    messages.append(ChatMessage(role="user", content=text))

    prompt, _ = await PREPARER.prepare_messages(messages, session_id)
    await ws.send_text(json.dumps({"type": "response.created", "response": {"id": response_id}}))

    output_item_id = f"item_{uuid.uuid4().hex[:16]}"
    await ws.send_text(
        json.dumps(
            {
                "type": "response.output_item.added",
                "item": {"id": output_item_id, "type": "message", "role": "assistant"},
            }
        )
    )

    finish_reason = "stop"
    async for event in RUNNER.run_stream(prompt=prompt, session_key=session_id, model=SETTINGS.default_model):
        if event.kind == "text" and event.text:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "response.output_text.delta",
                        "response_id": response_id,
                        "item_id": output_item_id,
                        "delta": event.text,
                    }
                )
            )
        elif event.kind == "final":
            meta = event.raw or {}
            finish_reason = meta.get("stop_reason") or "stop"
        elif event.kind == "error":
            await ws.send_text(
                json.dumps({"type": "error", "error": {"message": event.text or "upstream error"}})
            )

    await ws.send_text(
        json.dumps(
            {
                "type": "response.output_text.done",
                "response_id": response_id,
                "item_id": output_item_id,
            }
        )
    )
    await ws.send_text(
        json.dumps(
            {
                "type": "response.completed",
                "response": {"id": response_id, "status": "completed", "finish_reason": finish_reason},
            }
        )
    )
