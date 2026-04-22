"""POST /v1/batches — OpenAI-compatible batch submission.

A "batch" is a JSONL file previously uploaded to /v1/files with
``purpose=batch``. Each line is an object:

    {"custom_id": "...", "method": "POST", "url": "/v1/chat/completions",
     "body": {...}}

We replay each request serially through the in-process ASGI app, write
the results to another JSONL file, and register it back with /v1/files
as ``purpose=batch_output``. The ``endpoint`` field is validated against
what OpenAI supports.

Supported endpoints: /v1/chat/completions, /v1/completions,
/v1/embeddings, /v1/responses, /v1/moderations.
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
from .deps import FILE_STORE, auth_dependency
from .models import BatchCreateRequest


log = logging.getLogger("claude_wrapper.batches")
router = APIRouter()


BATCHES_DIR: Path = SETTINGS.data_dir / "batches"
BATCHES_DIR.mkdir(parents=True, exist_ok=True)


ALLOWED_ENDPOINTS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/responses",
    "/v1/moderations",
}


@dataclass
class Batch:
    id: str
    object: str = "batch"
    endpoint: str = ""
    input_file_id: str = ""
    completion_window: str = "24h"
    status: str = "validating"
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    created_at: int = 0
    in_progress_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    request_counts: dict[str, int] = field(default_factory=lambda: {"total": 0, "completed": 0, "failed": 0})
    metadata: dict[str, Any] = field(default_factory=dict)

    def path(self) -> Path:
        return BATCHES_DIR / f"{self.id}.json"

    def save(self) -> None:
        self.path().write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, batch_id: str) -> Optional["Batch"]:
        p = BATCHES_DIR / f"{batch_id}.json"
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return cls(**data)


async def _dispatch(endpoint: str, body: dict) -> tuple[int, Any]:
    """Call an endpoint via the in-process ASGI app using httpx.AsyncClient."""
    import httpx

    from .main import app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://batch") as client:
        headers = {}
        if SETTINGS.require_auth and SETTINGS.api_keys:
            headers["Authorization"] = f"Bearer {next(iter(SETTINGS.api_keys))}"
        resp = await client.post(endpoint, json=body, headers=headers, timeout=SETTINGS.request_timeout_seconds)
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, {"raw": resp.text}


async def _run_batch(batch: Batch) -> None:
    batch.status = "in_progress"
    batch.in_progress_at = int(time.time())
    batch.save()

    rec = await FILE_STORE.get(batch.input_file_id)
    if rec is None:
        batch.status = "failed"
        batch.failed_at = int(time.time())
        batch.save()
        return

    input_bytes = FILE_STORE.blob_path(rec).read_bytes()
    lines = [ln for ln in input_bytes.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    batch.request_counts["total"] = len(lines)
    batch.save()

    out_lines: list[str] = []
    err_lines: list[str] = []
    for ln in lines:
        try:
            req = json.loads(ln)
        except Exception as e:
            err_lines.append(json.dumps({"error": f"invalid JSON line: {e}", "line": ln[:200]}))
            batch.request_counts["failed"] += 1
            continue
        custom_id = req.get("custom_id") or f"req-{uuid.uuid4().hex[:12]}"
        url = req.get("url") or batch.endpoint
        body = req.get("body") or {}
        status, result = await _dispatch(url, body)
        entry = {
            "id": f"batch_req_{uuid.uuid4().hex[:12]}",
            "custom_id": custom_id,
            "response": {"status_code": status, "request_id": custom_id, "body": result} if 200 <= status < 400 else None,
            "error": None if 200 <= status < 400 else {"code": str(status), "message": result},
        }
        out_lines.append(json.dumps(entry))
        if 200 <= status < 400:
            batch.request_counts["completed"] += 1
        else:
            batch.request_counts["failed"] += 1
        batch.save()

    out_record = await FILE_STORE.save_bytes(
        data=("\n".join(out_lines) + "\n").encode("utf-8"),
        filename=f"{batch.id}_output.jsonl",
        purpose="batch_output",
        mime_type="application/jsonl",
    )
    batch.output_file_id = out_record.id
    if err_lines:
        err_record = await FILE_STORE.save_bytes(
            data=("\n".join(err_lines) + "\n").encode("utf-8"),
            filename=f"{batch.id}_errors.jsonl",
            purpose="batch_errors",
            mime_type="application/jsonl",
        )
        batch.error_file_id = err_record.id
    batch.status = "completed"
    batch.completed_at = int(time.time())
    batch.save()


@router.post("/v1/batches", dependencies=[Depends(auth_dependency)])
async def create_batch(req: BatchCreateRequest):
    if req.endpoint not in ALLOWED_ENDPOINTS:
        raise HTTPException(status_code=400, detail=f"endpoint not supported in batches: {req.endpoint}")
    rec = await FILE_STORE.get(req.input_file_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="input_file_id not found")

    batch = Batch(
        id=f"batch_{uuid.uuid4().hex[:24]}",
        endpoint=req.endpoint,
        input_file_id=req.input_file_id,
        completion_window=req.completion_window,
        status="validating",
        created_at=int(time.time()),
        metadata=req.metadata or {},
    )
    batch.save()
    asyncio.create_task(_run_batch(batch))
    return JSONResponse(content=asdict(batch))


@router.get("/v1/batches/{batch_id}", dependencies=[Depends(auth_dependency)])
async def retrieve_batch(batch_id: str):
    batch = Batch.load(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="batch not found")
    return JSONResponse(content=asdict(batch))


@router.post("/v1/batches/{batch_id}/cancel", dependencies=[Depends(auth_dependency)])
async def cancel_batch(batch_id: str):
    batch = Batch.load(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="batch not found")
    if batch.status in ("completed", "failed", "cancelled"):
        return JSONResponse(content=asdict(batch))
    batch.status = "cancelling"
    batch.save()
    return JSONResponse(content=asdict(batch))


@router.get("/v1/batches", dependencies=[Depends(auth_dependency)])
async def list_batches(limit: int = 20, after: Optional[str] = None):
    entries = sorted(BATCHES_DIR.glob("batch_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    data = []
    for p in entries[:limit]:
        try:
            data.append(json.loads(p.read_text()))
        except Exception:
            continue
    return JSONResponse(content={"object": "list", "data": data, "has_more": len(entries) > limit})
