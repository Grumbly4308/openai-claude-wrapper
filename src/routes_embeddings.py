"""POST /v1/embeddings — compute dense vector embeddings.

Tries three backends in order:

1. ``fastembed`` (ONNX, preinstalled, fast, ~100MB default model).
2. ``sentence-transformers`` if the user has installed it.
3. Delegate to Claude Code: it installs sentence-transformers on the
   first call and writes vectors as JSON into the workspace.

A last-resort hashing embedding keeps the endpoint working even with no
backend at all (useful for tests and tiny deployments).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import struct
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from .deps import DELEGATE, auth_dependency
from .models import EmbeddingsRequest, Usage


log = logging.getLogger("claude_wrapper.embeddings")
router = APIRouter()


_EMBED_MODEL_SINGLETON: Optional[object] = None
_EMBED_LOCK = asyncio.Lock()


async def _load_fastembed():
    global _EMBED_MODEL_SINGLETON
    async with _EMBED_LOCK:
        if _EMBED_MODEL_SINGLETON is not None:
            return _EMBED_MODEL_SINGLETON
        try:
            from fastembed import TextEmbedding  # type: ignore

            def _init():
                return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

            _EMBED_MODEL_SINGLETON = await asyncio.to_thread(_init)
            return _EMBED_MODEL_SINGLETON
        except Exception as e:  # pragma: no cover
            log.info("fastembed not available: %s", e)
            return None


async def _embed_fastembed(texts: list[str]) -> Optional[list[list[float]]]:
    model = await _load_fastembed()
    if model is None:
        return None

    def _run():
        return [list(map(float, v)) for v in model.embed(texts)]  # type: ignore[attr-defined]

    return await asyncio.to_thread(_run)


async def _embed_sentence_transformers(texts: list[str]) -> Optional[list[list[float]]]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None

    def _run():
        m = SentenceTransformer("all-MiniLM-L6-v2")
        return [list(map(float, v)) for v in m.encode(texts, normalize_embeddings=True)]

    return await asyncio.to_thread(_run)


async def _embed_via_claude(texts: list[str]) -> Optional[list[list[float]]]:
    workspace, session_key = DELEGATE.new_workspace("embed")
    inputs_path = workspace / "uploads" / "texts.json"
    inputs_path.write_text(json.dumps(texts))
    prompt = (
        "You must compute sentence embeddings for the strings in "
        "`uploads/texts.json`.\n\n"
        "Use `sentence-transformers` (model `all-MiniLM-L6-v2`); install it "
        "with pip if it is not yet available. Normalize each vector to unit "
        "length.\n\n"
        "Write the result as a JSON array of arrays of floats to "
        "`outputs/embeddings.json`. The array length must equal the input "
        "array length. Do not write anything else. When done, reply with a "
        "single line confirming success."
    )
    result = await DELEGATE.run(prompt=prompt, kind="embed", workspace=workspace, session_key=session_key)
    out = workspace / "outputs" / "embeddings.json"
    if not out.exists():
        return None
    try:
        data = json.loads(out.read_text())
        if not isinstance(data, list) or len(data) != len(texts):
            return None
        return [[float(x) for x in v] for v in data]
    except Exception:
        return None


def _hash_embedding(text: str, dim: int = 384) -> list[float]:
    """Deterministic fallback so the endpoint works without ML."""
    buf = bytearray()
    counter = 0
    while len(buf) < dim * 4:
        counter += 1
        buf.extend(hashlib.sha256(f"{counter}:{text}".encode()).digest())
    vec = list(struct.unpack(f"{dim}f", bytes(buf[: dim * 4])))
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


@router.post("/v1/embeddings", dependencies=[Depends(auth_dependency)])
async def embeddings(req: EmbeddingsRequest):
    inputs = req.input if isinstance(req.input, list) else [req.input]
    # OpenAI allows token-id inputs; we coerce to text by stringifying.
    texts: list[str] = []
    for item in inputs:
        if isinstance(item, list):
            texts.append(" ".join(str(x) for x in item))
        else:
            texts.append(str(item))

    if not texts:
        raise HTTPException(status_code=400, detail="input is empty")

    vectors: Optional[list[list[float]]] = None
    for backend in (_embed_fastembed, _embed_sentence_transformers, _embed_via_claude):
        try:
            vectors = await backend(texts)
            if vectors is not None:
                break
        except Exception as e:
            log.warning("embedding backend %s failed: %s", backend.__name__, e)
    if vectors is None:
        log.warning("no embedding backend available; falling back to hashing")
        vectors = [_hash_embedding(t) for t in texts]

    if req.dimensions:
        vectors = [v[: req.dimensions] for v in vectors]

    def _encode(v: list[float]):
        if req.encoding_format == "base64":
            return base64.b64encode(struct.pack(f"{len(v)}f", *v)).decode("ascii")
        return v

    data = [
        {"object": "embedding", "embedding": _encode(v), "index": i}
        for i, v in enumerate(vectors)
    ]
    total_in = sum(len(t.split()) for t in texts)
    body = {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": Usage(prompt_tokens=total_in, completion_tokens=0, total_tokens=total_in).model_dump(),
    }
    return JSONResponse(content=body)
