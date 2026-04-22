"""Image endpoints.

    POST /v1/images/generations  — generate an image from a prompt
    POST /v1/images/edits        — edit an image with a prompt (+ mask)
    POST /v1/images/variations   — produce variations of an image

Claude Code doesn't natively render raster images, so we have it draw
SVG (which is within the LLM's capability) and convert with imagemagick
to the requested size/format. For edits and variations we give Claude
the source image and let it decide how to apply the edit via imagemagick.

Heavier diffusion models are out of scope for this lean container; users
who want them can install them inside the container and ask Claude to
invoke them.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .deps import DELEGATE, FILE_STORE, auth_dependency
from .models import ImageGenRequest


log = logging.getLogger("claude_wrapper.images")
router = APIRouter()


def _parse_size(size: Optional[str]) -> tuple[int, int]:
    if not size:
        return (1024, 1024)
    try:
        w, h = size.lower().split("x")
        return (int(w), int(h))
    except Exception:
        return (1024, 1024)


def _gen_prompt(user_prompt: str, w: int, h: int, n: int, style: Optional[str]) -> str:
    style_hint = f"\nStyle: {style}." if style else ""
    return (
        f"Generate {n} image(s) from this prompt:\n"
        f"  PROMPT: {user_prompt}\n"
        f"  SIZE:   {w}x{h}{style_hint}\n\n"
        "Procedure:\n"
        f"1. For each of the {n} image(s), design an SVG that depicts the "
        "scene. Use vector shapes, gradients, text — whatever fits. The "
        f"SVG viewBox must be `0 0 {w} {h}`.\n"
        "2. Save each SVG as `outputs/gen-<i>.svg` (1-indexed).\n"
        f"3. Convert each to a {w}x{h} PNG. Prefer `rsvg-convert -w {w} -h {h} "
        "outputs/gen-<i>.svg -o outputs/gen-<i>.png`; fall back to imagemagick "
        "(`magick`) only if rsvg-convert is not installed.\n"
        "4. Confirm success with a single line reply."
    )


def _read_outputs(workspace: Path, n: int) -> list[Path]:
    outs = []
    for i in range(1, n + 1):
        p = workspace / "outputs" / f"gen-{i}.png"
        if p.exists():
            outs.append(p)
    # fall back to any PNGs that materialized
    if len(outs) < n:
        for p in sorted((workspace / "outputs").glob("*.png")):
            if p not in outs:
                outs.append(p)
            if len(outs) >= n:
                break
    return outs[:n]


def _to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


@router.post("/v1/images/generations", dependencies=[Depends(auth_dependency)])
async def image_generations(req: ImageGenRequest):
    if req.n < 1:
        raise HTTPException(status_code=400, detail="n must be >= 1")
    w, h = _parse_size(req.size)
    workspace, session_key = DELEGATE.new_workspace("imggen")
    prompt = _gen_prompt(req.prompt, w, h, req.n, req.style)
    result = await DELEGATE.run(prompt=prompt, kind="imggen", workspace=workspace, session_key=session_key)
    outs = _read_outputs(workspace, req.n)
    if not outs:
        raise HTTPException(
            status_code=502,
            detail=f"image generation failed; claude said: {result.claude.final_text[:400]}",
        )

    data = []
    for p in outs:
        rec = await FILE_STORE.ingest_path(
            src=p,
            filename=p.name,
            purpose="assistant_output",
            session_id=session_key,
            source="generated",
        )
        if req.response_format == "b64_json":
            data.append({"b64_json": _to_b64(p), "file_id": rec.id, "revised_prompt": req.prompt})
        else:
            data.append({"url": f"/v1/files/{rec.id}/content", "file_id": rec.id, "revised_prompt": req.prompt})

    return JSONResponse(content={"created": int(time.time()), "data": data})


@router.post("/v1/images/edits", dependencies=[Depends(auth_dependency)])
async def image_edits(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mask: Optional[UploadFile] = File(default=None),
    model: str = Form(default="claude-image-edit"),
    n: int = Form(default=1),
    size: Optional[str] = Form(default="1024x1024"),
    response_format: str = Form(default="b64_json"),
):
    workspace, session_key = DELEGATE.new_workspace("imgedit")
    w, h = _parse_size(size)
    src_name = Path(image.filename or "source.png").name
    src_path = workspace / "uploads" / src_name
    with src_path.open("wb") as f:
        while True:
            chunk = await image.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    mask_line = ""
    if mask is not None:
        mask_name = Path(mask.filename or "mask.png").name
        mask_path = workspace / "uploads" / mask_name
        with mask_path.open("wb") as f:
            while True:
                chunk = await mask.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        mask_line = f"An edit mask is provided at `uploads/{mask_name}`; black pixels are regions to edit.\n"

    task = (
        f"Edit the image at `uploads/{src_name}` to satisfy this instruction:\n"
        f"  {prompt}\n"
        f"{mask_line}"
        f"Produce {n} distinct edited result(s), each exactly {w}x{h}. Use "
        "imagemagick (`magick`) to perform the actual pixel operations "
        "(recolor, overlay, composite, blur, annotate, etc.). If the edit "
        "requires fresh content, generate an SVG overlay and composite it "
        "onto the source.\n\n"
        f"Write outputs to `outputs/edit-<i>.png` (1-indexed, 1..{n}). "
        "Confirm with a single line."
    )
    result = await DELEGATE.run(prompt=task, kind="imgedit", workspace=workspace, session_key=session_key)

    outs = [workspace / "outputs" / f"edit-{i}.png" for i in range(1, n + 1)]
    outs = [p for p in outs if p.exists()]
    if not outs:
        outs = sorted((workspace / "outputs").glob("*.png"))[:n]
    if not outs:
        raise HTTPException(
            status_code=502,
            detail=f"image edit failed; claude said: {result.claude.final_text[:400]}",
        )

    data = []
    for p in outs:
        rec = await FILE_STORE.ingest_path(
            src=p, filename=p.name, purpose="assistant_output", session_id=session_key, source="generated"
        )
        if response_format == "b64_json":
            data.append({"b64_json": _to_b64(p), "file_id": rec.id})
        else:
            data.append({"url": f"/v1/files/{rec.id}/content", "file_id": rec.id})
    return JSONResponse(content={"created": int(time.time()), "data": data})


@router.post("/v1/images/variations", dependencies=[Depends(auth_dependency)])
async def image_variations(
    image: UploadFile = File(...),
    model: str = Form(default="claude-image-var"),
    n: int = Form(default=1),
    size: Optional[str] = Form(default="1024x1024"),
    response_format: str = Form(default="b64_json"),
):
    workspace, session_key = DELEGATE.new_workspace("imgvar")
    w, h = _parse_size(size)
    src_name = Path(image.filename or "source.png").name
    src_path = workspace / "uploads" / src_name
    with src_path.open("wb") as f:
        while True:
            chunk = await image.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)

    task = (
        f"Produce {n} visual variation(s) of the image at `uploads/{src_name}`.\n"
        f"Each variation must be exactly {w}x{h}.\n\n"
        "Apply a different imagemagick transform to each: hue rotation, "
        "posterize, oil-paint, gaussian blur, charcoal, sketch, swirl, etc. "
        "Keep the subject recognizable but stylistically different.\n\n"
        f"Write outputs to `outputs/var-<i>.png` (1-indexed, 1..{n}). "
        "Confirm with a single line."
    )
    result = await DELEGATE.run(prompt=task, kind="imgvar", workspace=workspace, session_key=session_key)

    outs = [workspace / "outputs" / f"var-{i}.png" for i in range(1, n + 1)]
    outs = [p for p in outs if p.exists()]
    if not outs:
        outs = sorted((workspace / "outputs").glob("*.png"))[:n]
    if not outs:
        raise HTTPException(
            status_code=502,
            detail=f"image variation failed; claude said: {result.claude.final_text[:400]}",
        )

    data = []
    for p in outs:
        rec = await FILE_STORE.ingest_path(
            src=p, filename=p.name, purpose="assistant_output", session_id=session_key, source="generated"
        )
        if response_format == "b64_json":
            data.append({"b64_json": _to_b64(p), "file_id": rec.id})
        else:
            data.append({"url": f"/v1/files/{rec.id}/content", "file_id": rec.id})
    return JSONResponse(content={"created": int(time.time()), "data": data})
