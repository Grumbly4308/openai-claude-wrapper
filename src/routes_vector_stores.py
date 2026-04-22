"""Vector stores endpoint.

    POST/GET/DELETE /v1/vector_stores
    POST/GET/DELETE /v1/vector_stores/{id}/files
    POST            /v1/vector_stores/{id}/search

Each store is a JSON manifest + an npy matrix of row-major embeddings.
File ingest reads raw bytes from /v1/files, decodes text where possible,
chunks it, computes embeddings via the /v1/embeddings endpoint, and
appends rows to the matrix.

Search uses cosine similarity via numpy (or a pure-python fallback).
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from .config import SETTINGS
from .deps import FILE_STORE, auth_dependency
from .models import (
    EmbeddingsRequest,
    VectorStoreCreateRequest,
    VectorStoreFileAdd,
    VectorStoreSearchRequest,
)


log = logging.getLogger("claude_wrapper.vector_stores")
router = APIRouter()


VS_DIR: Path = SETTINGS.data_dir / "vector_stores"
VS_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_CHARS = 1500
CHUNK_OVERLAP = 200


@dataclass
class VectorStore:
    id: str
    object: str = "vector_store"
    created_at: int = 0
    name: Optional[str] = None
    status: str = "completed"
    file_counts: dict[str, int] = field(default_factory=lambda: {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0})
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding_model: str = "text-embedding-3-small"

    def _dir(self) -> Path:
        return VS_DIR / self.id

    def save(self) -> None:
        self._dir().mkdir(parents=True, exist_ok=True)
        (self._dir() / "store.json").write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, sid: str) -> Optional["VectorStore"]:
        p = VS_DIR / sid / "store.json"
        if not p.exists():
            return None
        return cls(**json.loads(p.read_text()))

    def chunks_path(self) -> Path:
        return self._dir() / "chunks.jsonl"

    def matrix_path(self) -> Path:
        return self._dir() / "matrix.npy"

    def files_path(self) -> Path:
        return self._dir() / "files.json"

    def list_files(self) -> list[dict]:
        if not self.files_path().exists():
            return []
        try:
            return json.loads(self.files_path().read_text())
        except Exception:
            return []

    def save_files(self, files: list[dict]) -> None:
        self.files_path().write_text(json.dumps(files))


def _chunk_text(text: str, chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    out = []
    i = 0
    n = len(text)
    step = max(1, chars - overlap)
    while i < n:
        out.append(text[i : i + chars])
        i += step
    return out or [""]


async def _embed(texts: list[str], model: str) -> list[list[float]]:
    from .routes_embeddings import embeddings as embed_route

    req = EmbeddingsRequest(input=texts, model=model)
    resp = await embed_route(req)
    body = resp.body if hasattr(resp, "body") else b""
    data = json.loads(body.decode() if isinstance(body, (bytes, bytearray)) else body)
    return [list(map(float, item["embedding"])) for item in data.get("data") or []]


def _load_matrix(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    try:
        import numpy as np  # type: ignore

        arr = np.load(path, allow_pickle=False)
        return arr.tolist()
    except Exception:
        try:
            return json.loads(path.read_text())
        except Exception:
            return []


def _save_matrix(path: Path, rows: list[list[float]]) -> None:
    try:
        import numpy as np  # type: ignore

        np.save(path, np.asarray(rows, dtype="float32"))
    except Exception:
        path.write_text(json.dumps(rows))


def _append_chunks(path: Path, entries: list[dict]) -> None:
    with path.open("a") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(x * x for x in b)) or 1.0
    return num / (da * db)


@router.post("/v1/vector_stores", dependencies=[Depends(auth_dependency)])
async def create_vector_store(req: VectorStoreCreateRequest):
    store = VectorStore(
        id=f"vs_{uuid.uuid4().hex[:24]}",
        created_at=int(time.time()),
        name=req.name,
        metadata=req.metadata or {},
    )
    store.save()
    if req.file_ids:
        for fid in req.file_ids:
            await _add_file(store, fid)
    return JSONResponse(content=asdict(store))


@router.get("/v1/vector_stores", dependencies=[Depends(auth_dependency)])
async def list_vector_stores(limit: int = 20):
    data = []
    for d in sorted(VS_DIR.glob("vs_*"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            data.append(json.loads((d / "store.json").read_text()))
        except Exception:
            continue
    return JSONResponse(content={"object": "list", "data": data, "has_more": False})


@router.get("/v1/vector_stores/{store_id}", dependencies=[Depends(auth_dependency)])
async def retrieve_vector_store(store_id: str):
    s = VectorStore.load(store_id)
    if s is None:
        raise HTTPException(status_code=404, detail="vector store not found")
    return JSONResponse(content=asdict(s))


@router.delete("/v1/vector_stores/{store_id}", dependencies=[Depends(auth_dependency)])
async def delete_vector_store(store_id: str):
    import shutil

    d = VS_DIR / store_id
    deleted = d.exists()
    if deleted:
        shutil.rmtree(d, ignore_errors=True)
    return JSONResponse(content={"id": store_id, "object": "vector_store.deleted", "deleted": deleted})


async def _add_file(store: VectorStore, file_id: str) -> dict:
    rec = await FILE_STORE.get(file_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"file {file_id} not found")
    blob = FILE_STORE.blob_path(rec)
    try:
        text = blob.read_text(encoding="utf-8")
    except Exception:
        text = blob.read_bytes().decode("utf-8", errors="replace")

    chunks = _chunk_text(text)
    vectors = await _embed(chunks, store.embedding_model)
    rows = _load_matrix(store.matrix_path())
    start = len(rows)
    rows.extend(vectors)
    _save_matrix(store.matrix_path(), rows)
    _append_chunks(
        store.chunks_path(),
        [
            {"row": start + i, "file_id": file_id, "filename": rec.filename, "text": c}
            for i, c in enumerate(chunks)
        ],
    )
    files = store.list_files()
    entry = {
        "id": f"vsfile_{uuid.uuid4().hex[:16]}",
        "object": "vector_store.file",
        "created_at": int(time.time()),
        "vector_store_id": store.id,
        "file_id": file_id,
        "status": "completed",
        "chunk_count": len(chunks),
    }
    files.append(entry)
    store.save_files(files)
    store.file_counts["completed"] += 1
    store.file_counts["total"] = len(files)
    store.save()
    return entry


@router.post("/v1/vector_stores/{store_id}/files", dependencies=[Depends(auth_dependency)])
async def vector_store_add_file(store_id: str, req: VectorStoreFileAdd):
    s = VectorStore.load(store_id)
    if s is None:
        raise HTTPException(status_code=404, detail="vector store not found")
    entry = await _add_file(s, req.file_id)
    return JSONResponse(content=entry)


@router.get("/v1/vector_stores/{store_id}/files", dependencies=[Depends(auth_dependency)])
async def vector_store_list_files(store_id: str):
    s = VectorStore.load(store_id)
    if s is None:
        raise HTTPException(status_code=404, detail="vector store not found")
    return JSONResponse(content={"object": "list", "data": s.list_files(), "has_more": False})


@router.delete("/v1/vector_stores/{store_id}/files/{vsfile_id}", dependencies=[Depends(auth_dependency)])
async def vector_store_delete_file(store_id: str, vsfile_id: str):
    s = VectorStore.load(store_id)
    if s is None:
        raise HTTPException(status_code=404, detail="vector store not found")
    files = s.list_files()
    before = len(files)
    files = [f for f in files if f.get("id") != vsfile_id]
    s.save_files(files)
    return JSONResponse(content={"id": vsfile_id, "object": "vector_store.file.deleted", "deleted": before != len(files)})


@router.post("/v1/vector_stores/{store_id}/search", dependencies=[Depends(auth_dependency)])
async def vector_store_search(store_id: str, req: VectorStoreSearchRequest):
    s = VectorStore.load(store_id)
    if s is None:
        raise HTTPException(status_code=404, detail="vector store not found")
    rows = _load_matrix(s.matrix_path())
    chunks = _load_chunks(s.chunks_path())
    if not rows or not chunks:
        return JSONResponse(content={"object": "vector_store.search_results.page", "data": [], "has_more": False})

    [qvec] = await _embed([req.query], s.embedding_model)
    scored: list[tuple[float, dict]] = []
    try:
        import numpy as np  # type: ignore

        mat = np.asarray(rows, dtype="float32")
        q = np.asarray(qvec, dtype="float32")
        mat_norms = np.linalg.norm(mat, axis=1) + 1e-9
        q_norm = np.linalg.norm(q) + 1e-9
        sims = (mat @ q) / (mat_norms * q_norm)
        order = sims.argsort()[::-1][: req.max_num_results]
        for idx in order:
            ch = chunks[int(idx)]
            scored.append((float(sims[int(idx)]), ch))
    except Exception:
        for ch in chunks:
            row = rows[ch["row"]]
            scored.append((_cosine(qvec, row), ch))
        scored.sort(key=lambda t: t[0], reverse=True)
        scored = scored[: req.max_num_results]

    data = [
        {
            "file_id": ch.get("file_id"),
            "filename": ch.get("filename"),
            "score": score,
            "content": [{"type": "text", "text": ch.get("text") or ""}],
        }
        for score, ch in scored
    ]
    return JSONResponse(content={"object": "vector_store.search_results.page", "data": data, "has_more": False})
