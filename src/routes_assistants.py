"""Assistants API mapped onto the existing session registry.

    POST/GET/DELETE /v1/assistants            — CRUD a model+instructions config
    POST/GET/DELETE /v1/threads               — thread ≡ session_id
    POST            /v1/threads/{id}/messages — append a user message
    GET             /v1/threads/{id}/messages — list thread messages
    POST            /v1/threads/{id}/runs     — invoke assistant → chat completion
    GET             /v1/threads/{id}/runs/{run_id} — run status

The mapping: an *assistant* is a saved (model, system-instructions, name)
triple. A *thread* is the same concept as a wrapper session — messages
accumulate on disk and the assistant's run replays them through the chat
completions pipeline, preserving state via session_id.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from .config import SETTINGS
from .deps import auth_dependency
from .models import (
    AssistantCreateRequest,
    ChatCompletionRequest,
    ChatMessage,
    RunCreateRequest,
    ThreadCreateRequest,
    ThreadMessageCreateRequest,
)


log = logging.getLogger("claude_wrapper.assistants")
router = APIRouter()


ASSISTANTS_DIR: Path = SETTINGS.data_dir / "assistants"
THREADS_DIR: Path = SETTINGS.data_dir / "threads"
RUNS_DIR: Path = SETTINGS.data_dir / "runs"
for d in (ASSISTANTS_DIR, THREADS_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class Assistant:
    id: str
    object: str = "assistant"
    created_at: int = 0
    name: Optional[str] = None
    description: Optional[str] = None
    model: str = ""
    instructions: Optional[str] = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self) -> None:
        (ASSISTANTS_DIR / f"{self.id}.json").write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, aid: str) -> Optional["Assistant"]:
        p = ASSISTANTS_DIR / f"{aid}.json"
        if not p.exists():
            return None
        return cls(**json.loads(p.read_text()))


@dataclass
class Thread:
    id: str
    object: str = "thread"
    created_at: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self) -> None:
        (THREADS_DIR / f"{self.id}.json").write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, tid: str) -> Optional["Thread"]:
        p = THREADS_DIR / f"{tid}.json"
        if not p.exists():
            return None
        return cls(**json.loads(p.read_text()))

    def messages_path(self) -> Path:
        return THREADS_DIR / f"{self.id}.messages.jsonl"

    def append_message(self, role: str, content: Any, metadata: Optional[dict] = None) -> dict:
        mid = f"msg_{uuid.uuid4().hex[:24]}"
        entry = {
            "id": mid,
            "object": "thread.message",
            "created_at": int(time.time()),
            "thread_id": self.id,
            "role": role,
            "content": content if isinstance(content, list) else [{"type": "text", "text": {"value": str(content)}}],
            "metadata": metadata or {},
        }
        with self.messages_path().open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def list_messages(self) -> list[dict]:
        p = self.messages_path()
        if not p.exists():
            return []
        out = []
        for ln in p.read_text().splitlines():
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except Exception:
                    continue
        return out


@dataclass
class Run:
    id: str
    object: str = "thread.run"
    created_at: int = 0
    thread_id: str = ""
    assistant_id: str = ""
    status: str = "queued"
    model: str = ""
    instructions: Optional[str] = None
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    last_error: Optional[dict] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self) -> None:
        (RUNS_DIR / f"{self.id}.json").write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, rid: str) -> Optional["Run"]:
        p = RUNS_DIR / f"{rid}.json"
        if not p.exists():
            return None
        return cls(**json.loads(p.read_text()))


# ---------- assistants CRUD ----------


@router.post("/v1/assistants", dependencies=[Depends(auth_dependency)])
async def create_assistant(req: AssistantCreateRequest):
    a = Assistant(
        id=f"asst_{uuid.uuid4().hex[:24]}",
        created_at=int(time.time()),
        name=req.name,
        description=req.description,
        model=req.model,
        instructions=req.instructions,
        tools=req.tools or [],
        metadata=req.metadata or {},
    )
    a.save()
    return JSONResponse(content=asdict(a))


@router.get("/v1/assistants", dependencies=[Depends(auth_dependency)])
async def list_assistants(limit: int = 20):
    items = []
    for p in sorted(ASSISTANTS_DIR.glob("asst_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            items.append(json.loads(p.read_text()))
        except Exception:
            continue
    return JSONResponse(content={"object": "list", "data": items, "has_more": False})


@router.get("/v1/assistants/{assistant_id}", dependencies=[Depends(auth_dependency)])
async def get_assistant(assistant_id: str):
    a = Assistant.load(assistant_id)
    if a is None:
        raise HTTPException(status_code=404, detail="assistant not found")
    return JSONResponse(content=asdict(a))


@router.post("/v1/assistants/{assistant_id}", dependencies=[Depends(auth_dependency)])
async def modify_assistant(assistant_id: str, req: AssistantCreateRequest):
    a = Assistant.load(assistant_id)
    if a is None:
        raise HTTPException(status_code=404, detail="assistant not found")
    if req.model:
        a.model = req.model
    if req.name is not None:
        a.name = req.name
    if req.description is not None:
        a.description = req.description
    if req.instructions is not None:
        a.instructions = req.instructions
    if req.tools is not None:
        a.tools = req.tools
    if req.metadata is not None:
        a.metadata = req.metadata
    a.save()
    return JSONResponse(content=asdict(a))


@router.delete("/v1/assistants/{assistant_id}", dependencies=[Depends(auth_dependency)])
async def delete_assistant(assistant_id: str):
    p = ASSISTANTS_DIR / f"{assistant_id}.json"
    deleted = p.exists()
    if deleted:
        p.unlink()
    return JSONResponse(content={"id": assistant_id, "object": "assistant.deleted", "deleted": deleted})


# ---------- threads CRUD ----------


@router.post("/v1/threads", dependencies=[Depends(auth_dependency)])
async def create_thread(req: ThreadCreateRequest):
    t = Thread(id=f"thread_{uuid.uuid4().hex[:24]}", created_at=int(time.time()), metadata=req.metadata or {})
    t.save()
    if req.messages:
        for m in req.messages:
            t.append_message(role=m.get("role") or "user", content=m.get("content") or "", metadata=m.get("metadata"))
    return JSONResponse(content=asdict(t))


@router.get("/v1/threads/{thread_id}", dependencies=[Depends(auth_dependency)])
async def get_thread(thread_id: str):
    t = Thread.load(thread_id)
    if t is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return JSONResponse(content=asdict(t))


@router.delete("/v1/threads/{thread_id}", dependencies=[Depends(auth_dependency)])
async def delete_thread(thread_id: str):
    p = THREADS_DIR / f"{thread_id}.json"
    mp = THREADS_DIR / f"{thread_id}.messages.jsonl"
    deleted = p.exists()
    if deleted:
        p.unlink()
    if mp.exists():
        mp.unlink()
    return JSONResponse(content={"id": thread_id, "object": "thread.deleted", "deleted": deleted})


# ---------- messages ----------


@router.post("/v1/threads/{thread_id}/messages", dependencies=[Depends(auth_dependency)])
async def create_message(thread_id: str, req: ThreadMessageCreateRequest):
    t = Thread.load(thread_id)
    if t is None:
        raise HTTPException(status_code=404, detail="thread not found")
    entry = t.append_message(role=req.role, content=req.content, metadata=req.metadata)
    return JSONResponse(content=entry)


@router.get("/v1/threads/{thread_id}/messages", dependencies=[Depends(auth_dependency)])
async def list_messages(thread_id: str, limit: int = 100):
    t = Thread.load(thread_id)
    if t is None:
        raise HTTPException(status_code=404, detail="thread not found")
    msgs = t.list_messages()
    return JSONResponse(content={"object": "list", "data": msgs[-limit:], "has_more": len(msgs) > limit})


# ---------- runs ----------


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                # thread-message shape {"type": "text", "text": {"value": "..."}}
                if c.get("type") == "text":
                    t = c.get("text")
                    if isinstance(t, dict):
                        parts.append(t.get("value") or "")
                    else:
                        parts.append(str(t or ""))
        return "\n".join(parts)
    return str(content or "")


async def _execute_run(run: Run) -> None:
    from .main import run_chat_completion

    run.status = "in_progress"
    run.started_at = int(time.time())
    run.save()

    thread = Thread.load(run.thread_id)
    assistant = Assistant.load(run.assistant_id)
    if thread is None or assistant is None:
        run.status = "failed"
        run.failed_at = int(time.time())
        run.last_error = {"code": "not_found", "message": "thread or assistant missing"}
        run.save()
        return

    messages: list[ChatMessage] = []
    sys_instr = run.instructions or assistant.instructions
    if sys_instr:
        messages.append(ChatMessage(role="system", content=sys_instr))
    for m in thread.list_messages():
        messages.append(ChatMessage(role=m.get("role") or "user", content=_extract_text(m.get("content"))))

    chat_req = ChatCompletionRequest(
        model=run.model or assistant.model,
        messages=messages,
        session_id=run.thread_id,
    )
    try:
        resp = await run_chat_completion(chat_req)
        body = resp.body if hasattr(resp, "body") else b""
        data = json.loads(body.decode() if isinstance(body, (bytes, bytearray)) else body)
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        thread.append_message(role="assistant", content=content)
        run.status = "completed"
        run.completed_at = int(time.time())
    except Exception as e:
        run.status = "failed"
        run.failed_at = int(time.time())
        run.last_error = {"code": "server_error", "message": str(e)}
    run.save()


@router.post("/v1/threads/{thread_id}/runs", dependencies=[Depends(auth_dependency)])
async def create_run(thread_id: str, req: RunCreateRequest):
    t = Thread.load(thread_id)
    if t is None:
        raise HTTPException(status_code=404, detail="thread not found")
    a = Assistant.load(req.assistant_id)
    if a is None:
        raise HTTPException(status_code=404, detail="assistant not found")

    run = Run(
        id=f"run_{uuid.uuid4().hex[:24]}",
        created_at=int(time.time()),
        thread_id=thread_id,
        assistant_id=req.assistant_id,
        status="queued",
        model=req.model or a.model,
        instructions=req.instructions or a.instructions,
        metadata=req.metadata or {},
    )
    run.save()
    # run synchronously by default — simpler semantics for a single-instance wrapper
    await _execute_run(run)
    return JSONResponse(content=asdict(run))


@router.get("/v1/threads/{thread_id}/runs/{run_id}", dependencies=[Depends(auth_dependency)])
async def get_run(thread_id: str, run_id: str):
    run = Run.load(run_id)
    if run is None or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail="run not found")
    return JSONResponse(content=asdict(run))


@router.get("/v1/threads/{thread_id}/runs", dependencies=[Depends(auth_dependency)])
async def list_runs(thread_id: str, limit: int = 20):
    items = []
    for p in sorted(RUNS_DIR.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text())
            if data.get("thread_id") == thread_id:
                items.append(data)
            if len(items) >= limit:
                break
        except Exception:
            continue
    return JSONResponse(content={"object": "list", "data": items, "has_more": False})
