"""POST /v1/moderations — content classification via Claude Code.

We ask Claude for a structured JSON verdict matching the OpenAI Moderation
response shape. If Claude returns malformed JSON we fall back to a safe
"flagged=false" default rather than 500.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from .deps import DELEGATE, auth_dependency
from .models import ModerationsRequest


router = APIRouter()


CATEGORIES: tuple[str, ...] = (
    "sexual",
    "sexual/minors",
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "self-harm",
    "self-harm/intent",
    "self-harm/instructions",
    "violence",
    "violence/graphic",
    "illicit",
    "illicit/violent",
)


def _empty_scores() -> dict[str, float]:
    return {c: 0.0 for c in CATEGORIES}


def _empty_flags() -> dict[str, bool]:
    return {c: False for c in CATEGORIES}


def _build_prompt(text: str) -> str:
    return (
        "You are a content moderation classifier. For the INPUT below, "
        "respond ONLY with a single JSON object (no prose, no code fences) "
        "of the form:\n"
        '  {"flagged": <bool>, '
        '"categories": {"sexual": <bool>, "hate": <bool>, ...}, '
        '"scores": {"sexual": <0..1>, "hate": <0..1>, ...}}\n\n'
        f"Use exactly these category keys: {', '.join(CATEGORIES)}.\n"
        'Set "flagged" = true if any category is true. Output valid JSON '
        "only.\n\n"
        f"INPUT:\n---\n{text}\n---"
    )


async def _classify(text: str) -> dict[str, Any]:
    prompt = _build_prompt(text)
    result = await DELEGATE.run(prompt=prompt, kind="moderate", cleanup=True)
    parsed = DELEGATE.extract_json_block(result.claude.final_text) or {}
    categories = _empty_flags()
    scores = _empty_scores()
    raw_cats = parsed.get("categories") or {}
    raw_scores = parsed.get("category_scores") or parsed.get("scores") or {}
    for c in CATEGORIES:
        if c in raw_cats:
            categories[c] = bool(raw_cats[c])
        if c in raw_scores:
            try:
                scores[c] = max(0.0, min(1.0, float(raw_scores[c])))
            except (TypeError, ValueError):
                pass
    flagged = bool(parsed.get("flagged") or any(categories.values()))
    return {
        "flagged": flagged,
        "categories": categories,
        "category_scores": scores,
    }


@router.post("/v1/moderations", dependencies=[Depends(auth_dependency)])
async def moderations(req: ModerationsRequest):
    inputs = req.input if isinstance(req.input, list) else [req.input]
    results = []
    for t in inputs:
        results.append(await _classify(str(t)))
    body = {
        "id": f"modr-{uuid.uuid4().hex[:24]}",
        "model": req.model,
        "results": results,
    }
    return JSONResponse(content=body)
