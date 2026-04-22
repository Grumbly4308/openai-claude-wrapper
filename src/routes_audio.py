"""Audio endpoints.

    POST /v1/audio/transcriptions  — speech-to-text
    POST /v1/audio/translations    — speech → English text
    POST /v1/audio/speech          — text-to-speech

STT is delegated to Claude Code, which installs ``faster-whisper`` on
the first call and reuses it afterward. TTS uses ``espeak-ng``
(preinstalled in the Dockerfile) for a tiny, dependency-free default,
with ffmpeg for format conversion.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from .deps import DELEGATE, FILE_STORE, auth_dependency
from .models import SpeechRequest


log = logging.getLogger("claude_wrapper.audio")
router = APIRouter()


async def _save_upload(file: UploadFile, dest: Path) -> None:
    with dest.open("wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _transcribe_prompt(audio_rel: str, language: Optional[str], translate: bool, response_format: str) -> str:
    task = "translate" if translate else "transcribe"
    lang_hint = (
        f"The source language is `{language}`."
        if language and not translate
        else "Auto-detect the source language."
    )
    fmt_note = {
        "json": 'Write a single JSON object {"text": "..."} to outputs/transcript.json.',
        "verbose_json": 'Write a JSON object with keys "text", "language", "duration", "segments" (list of {id, start, end, text}) to outputs/transcript.json.',
        "text": "Write the plain text transcript to outputs/transcript.txt.",
        "srt": "Write a SubRip (.srt) subtitle file to outputs/transcript.srt.",
        "vtt": "Write a WebVTT subtitle file to outputs/transcript.vtt.",
    }.get(response_format, "Write the plain text transcript to outputs/transcript.txt.")
    return (
        f"You must {task} the audio file at `{audio_rel}`.\n\n"
        f"{lang_hint}\n\n"
        "Use `faster-whisper` (model `base` or `small`, whichever is already "
        "cached; otherwise download `base`). Install faster-whisper with pip "
        "if it is not yet available. Use ffmpeg for any format conversion.\n\n"
        f"{fmt_note}\n"
        "After writing the file, reply with a single line confirming success."
    )


async def _run_stt(
    file: UploadFile,
    language: Optional[str],
    prompt: Optional[str],
    response_format: str,
    translate: bool,
):
    workspace, session_key = DELEGATE.new_workspace("stt")
    safe_name = Path(file.filename or "audio.bin").name
    dest = workspace / "uploads" / safe_name
    await _save_upload(file, dest)

    task_prompt = _transcribe_prompt(
        audio_rel=f"uploads/{safe_name}",
        language=language,
        translate=translate,
        response_format=response_format,
    )
    if prompt:
        task_prompt += f"\n\nAdditional hint from caller: {prompt}"

    result = await DELEGATE.run(prompt=task_prompt, kind="stt", workspace=workspace, session_key=session_key)

    outputs = workspace / "outputs"
    file_map = {
        "json": outputs / "transcript.json",
        "verbose_json": outputs / "transcript.json",
        "text": outputs / "transcript.txt",
        "srt": outputs / "transcript.srt",
        "vtt": outputs / "transcript.vtt",
    }
    want = file_map.get(response_format, outputs / "transcript.txt")
    if not want.exists():
        for alt in outputs.glob("transcript.*"):
            want = alt
            break
    if not want.exists():
        raise HTTPException(
            status_code=502,
            detail=f"transcription failed; claude said: {result.claude.final_text[:400]}",
        )

    if response_format in ("json", "verbose_json"):
        try:
            return JSONResponse(content=json.loads(want.read_text()))
        except Exception:
            return JSONResponse(content={"text": want.read_text()})
    return JSONResponse(content=want.read_text(), media_type="text/plain")


@router.post("/v1/audio/transcriptions", dependencies=[Depends(auth_dependency)])
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: Optional[float] = Form(default=None),
):
    return await _run_stt(file, language, prompt, response_format, translate=False)


@router.post("/v1/audio/translations", dependencies=[Depends(auth_dependency)])
async def translations(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: Optional[float] = Form(default=None),
):
    return await _run_stt(file, None, prompt, response_format, translate=True)


@router.post("/v1/audio/speech", dependencies=[Depends(auth_dependency)])
async def speech(req: SpeechRequest):
    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="input must be non-empty")

    workspace, session_key = DELEGATE.new_workspace("tts")
    (workspace / "uploads" / "text.txt").write_text(text)
    target = workspace / "outputs" / f"speech.{req.response_format}"

    prompt = (
        f"You must synthesize speech from the text in `uploads/text.txt`.\n\n"
        f"Voice selector from caller: `{req.voice}` — interpret as an espeak-ng "
        f"voice name (e.g. `en`, `en-us`, `en+f3`). Speed multiplier: {req.speed} "
        f"(espeak-ng default is 175 wpm; multiply accordingly).\n\n"
        "1. Use `espeak-ng -v <voice> -s <wpm> -w outputs/raw.wav -f uploads/text.txt`.\n"
        f"2. Convert to `outputs/speech.{req.response_format}` using ffmpeg "
        "(choose the appropriate codec for the extension). Overwrite if necessary.\n"
        "3. Confirm success with a single line reply."
    )
    result = await DELEGATE.run(prompt=prompt, kind="tts", workspace=workspace, session_key=session_key)

    if not target.exists():
        # fallback: try any output the agent wrote
        alts = sorted((workspace / "outputs").glob(f"*.{req.response_format}"))
        if alts:
            target = alts[0]
    if not target.exists():
        raise HTTPException(
            status_code=502,
            detail=f"speech synthesis failed; claude said: {result.claude.final_text[:400]}",
        )

    media = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "opus": "audio/ogg",
        "aac": "audio/aac",
        "pcm": "audio/pcm",
    }.get(req.response_format, "application/octet-stream")

    async def _iter() -> AsyncIterator[bytes]:
        import aiofiles

        async with aiofiles.open(target, "rb") as f:
            while True:
                chunk = await f.read(1 << 16)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(_iter(), media_type=media)
