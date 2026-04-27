from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import httpx

from .config import SETTINGS
from .file_store import FileStore, FileRecord, _safe_filename
from .models import (
    ChatMessage,
    ContentPart,
    FileContent,
    ImageContent,
    InputAudioContent,
    TextContent,
)

if False:  # type-checking only
    from .claude_runner import SessionRegistry


DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;,]+)?(?P<b64>;base64)?,(?P<data>.*)$", re.DOTALL)


@dataclass
class PreparedMessage:
    role: str
    text: str
    attachments: list[Path]  # absolute paths Claude Code can read


class MessagePreparer:
    """Turns OpenAI-style chat messages into a flat text prompt + attachment paths.

    Binary parts (images, audio, generic files) are materialized into the
    session workspace, and a textual pointer is inserted into the prompt so
    Claude Code's Read tool can load them.
    """

    def __init__(
        self,
        file_store: FileStore,
        workspace_root: Path,
        http_client: Optional[httpx.AsyncClient] = None,
        registry: Optional["SessionRegistry"] = None,
    ):
        self.file_store = file_store
        self.workspace_root = workspace_root
        self._http = http_client
        self.registry = registry

    async def _http_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=60.0, follow_redirects=True)
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def session_workspace(self, session_id: str) -> Path:
        d = self.workspace_root / session_id
        (d / "uploads").mkdir(parents=True, exist_ok=True)
        (d / "outputs").mkdir(parents=True, exist_ok=True)
        return d

    def _kb_addendum(self) -> str:
        """System-prompt block teaching the model to query OpenWebUI's RAG.

        Returned only when OPENWEBUI_BASE_URL is configured. The api key and
        base url already pass through to the Claude subprocess via os.environ
        (see claude_runner._build_argv / env = os.environ.copy()), so the
        model can reference them as $OPENWEBUI_* in Bash.
        """
        base = SETTINGS.openwebui_base_url
        if not base:
            return ""
        default_collection = SETTINGS.openwebui_default_collection
        auth_hint = (
            ' -H "Authorization: Bearer $OPENWEBUI_API_KEY"'
            if SETTINGS.openwebui_api_key
            else ""
        )
        coll_hint = (
            f' (default collection: "{default_collection}")' if default_collection else ""
        )
        return (
            "## Knowledge base\n"
            f"An OpenWebUI knowledge base is reachable at $OPENWEBUI_BASE_URL{coll_hint}. "
            "When the user asks something that may be answered by stored documents, "
            "search the KB before answering. Use Bash + curl, e.g.:\n"
            "```\n"
            f"curl -sS{auth_hint} -H 'Content-Type: application/json' \\\n"
            "  -X POST \"$OPENWEBUI_BASE_URL/api/v1/retrieval/query/collection\" \\\n"
            "  -d '{\"collection_names\":[\"<collection>\"],\"query\":\"<query>\",\"k\":5}'\n"
            "```\n"
            "Read the JSON response, cite the most relevant chunks in your answer, "
            "and do not invent facts that are not in either the KB results or the "
            "user-supplied context."
        )

    async def _materialize_data_url(self, data_url: str, workspace: Path, hint_name: Optional[str]) -> tuple[Path, str]:
        m = DATA_URL_RE.match(data_url)
        if not m:
            raise ValueError("invalid data URL")
        mime = (m.group("mime") or "application/octet-stream").strip()
        is_b64 = bool(m.group("b64"))
        payload = m.group("data") or ""
        if is_b64:
            try:
                raw = base64.b64decode(payload, validate=False)
            except (binascii.Error, ValueError) as e:
                raise ValueError(f"invalid base64 in data URL: {e}")
        else:
            raw = payload.encode("utf-8")
        ext = _ext_for_mime(mime)
        name = _safe_filename(hint_name or f"upload-{uuid.uuid4().hex[:8]}{ext}")
        if ext and not name.endswith(ext):
            name = f"{name}{ext}"
        target = workspace / "uploads" / name
        target = _dedupe_path(target)
        target.write_bytes(raw)
        return target, mime

    async def _materialize_http(self, url: str, workspace: Path, hint_name: Optional[str]) -> tuple[Path, str]:
        client = await self._http_client()
        resp = await client.get(url)
        resp.raise_for_status()
        mime = resp.headers.get("content-type", "application/octet-stream").split(";")[0].strip()
        ext = _ext_for_mime(mime)
        name = hint_name or os.path.basename(httpx.URL(url).path) or f"download-{uuid.uuid4().hex[:8]}{ext}"
        name = _safe_filename(name)
        if ext and not name.endswith(ext):
            name = f"{name}{ext}"
        target = workspace / "uploads" / name
        target = _dedupe_path(target)
        target.write_bytes(resp.content)
        return target, mime

    async def _materialize_file_id(self, file_id: str, workspace: Path) -> tuple[Path, str, str]:
        rec: Optional[FileRecord] = await self.file_store.get(file_id)
        if rec is None:
            raise ValueError(f"unknown file_id: {file_id}")
        src = self.file_store.blob_path(rec)
        name = _safe_filename(rec.filename)
        target = workspace / "uploads" / name
        target = _dedupe_path(target)
        # Hard-link when possible, fall back to copy.
        try:
            os.link(src, target)
        except OSError:
            target.write_bytes(src.read_bytes())
        return target, rec.mime_type, rec.filename

    async def _materialize_file_data(self, b64: str, mime: Optional[str], filename: Optional[str], workspace: Path) -> tuple[Path, str]:
        try:
            raw = base64.b64decode(b64, validate=False)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"invalid base64 in file_data: {e}")
        mime = mime or "application/octet-stream"
        ext = _ext_for_mime(mime)
        name = _safe_filename(filename or f"upload-{uuid.uuid4().hex[:8]}{ext}")
        if ext and not name.endswith(ext):
            name = f"{name}{ext}"
        target = workspace / "uploads" / name
        target = _dedupe_path(target)
        target.write_bytes(raw)
        return target, mime

    async def prepare_messages(
        self,
        messages: Iterable[ChatMessage],
        session_id: str,
    ) -> tuple[str, list[Path]]:
        """Flatten OpenAI messages into a single prompt string.

        - On the first turn of a session, the full transcript is emitted
          (system header + earlier turns + trailing user prompt) so Claude
          gets the complete context.
        - On subsequent turns of a known session, Claude Code's own session
          log already holds prior turns, so we emit *only* the trailing
          user message (plus any new attachments). This avoids re-pasting
          history that Claude already has.
        - Binary attachments are saved into the session workspace and
          referenced by absolute path in the prompt.
        """
        workspace = self.session_workspace(session_id)
        attachments: list[Path] = []

        msgs = list(messages)
        last_user_idx = _last_user_index(msgs)

        resuming = False
        if self.registry is not None and self.registry.has(session_id):
            resuming = True

        if resuming and last_user_idx >= 0:
            # Replay-only mode: skip prior user/assistant turns (Claude Code
            # already has them in its session log) but re-feed every
            # system/developer message — these often carry per-turn dynamic
            # context (e.g. OpenWebUI's RAG-retrieved chunks) and must reach
            # the model on every request.
            system_parts: list[str] = []
            kb_addendum = self._kb_addendum()
            if kb_addendum:
                system_parts.append(kb_addendum)
            for msg in msgs:
                if msg.role in ("system", "developer"):
                    text, msg_attachments = await self._render_message(msg, workspace)
                    attachments.extend(msg_attachments)
                    if text.strip():
                        system_parts.append(text.strip())

            trailing_msg = msgs[last_user_idx]
            user_text, user_attachments = await self._render_message(trailing_msg, workspace)
            attachments.extend(user_attachments)

            chunks: list[str] = []
            if system_parts:
                chunks.append("\n\n".join(system_parts).strip())
            if user_text.strip():
                chunks.append(user_text.strip())
            prompt = "\n\n".join(chunks).strip()
            return prompt, attachments

        system_parts: list[str] = []
        transcript: list[str] = []

        for idx, msg in enumerate(msgs):
            role = msg.role
            text, msg_attachments = await self._render_message(msg, workspace)
            attachments.extend(msg_attachments)

            if role in ("system", "developer"):
                if text.strip():
                    system_parts.append(text.strip())
                continue

            label = {
                "user": "User",
                "assistant": "Assistant",
                "tool": "Tool",
            }.get(role, role.capitalize())

            is_last_user = idx == last_user_idx
            if is_last_user:
                # Emit as plain trailing user turn (most recent prompt).
                trailing = text
                transcript.append(f"{label}: {trailing}".rstrip())
            else:
                transcript.append(f"{label}: {text}".rstrip())

        preamble = (
            "You are running in a headless container. The current working "
            "directory is a private per-session scratch space. User-provided "
            "binary inputs live under ./uploads/ (referenced explicitly by "
            "absolute path in the transcript). Write any files you want to "
            "return to the caller into this directory (./outputs/ is "
            "conventional); every new file you create will be uploaded back "
            "to the caller as a file attachment."
        )
        header_chunks = [preamble]
        kb_addendum = self._kb_addendum()
        if kb_addendum:
            header_chunks.append(kb_addendum)
        if system_parts:
            header_chunks.append("\n\n".join(system_parts).strip())
        header = "\n\n".join(h for h in header_chunks if h.strip()) + "\n\n"

        body = "\n\n".join(t for t in transcript if t.strip())
        prompt = f"{header}{body}".strip()
        return prompt, attachments

    async def _render_message(self, msg: ChatMessage, workspace: Path) -> tuple[str, list[Path]]:
        if msg.content is None:
            return "", []
        if isinstance(msg.content, str):
            return msg.content, []

        pieces: list[str] = []
        attachments: list[Path] = []
        for part in msg.content:
            rendered, path = await self._render_part(part, workspace)
            if rendered:
                pieces.append(rendered)
            if path is not None:
                attachments.append(path)
        return "\n".join(p for p in pieces if p), attachments

    async def _render_part(self, part: ContentPart, workspace: Path) -> tuple[str, Optional[Path]]:
        if isinstance(part, TextContent):
            return part.text or "", None

        if isinstance(part, ImageContent):
            url = part.image_url.url
            path, mime = await self._resolve_media(url, workspace, hint_name=None)
            return f"[attached image: {path} (mime={mime})]", path

        if isinstance(part, InputAudioContent):
            payload = part.input_audio
            fmt = (payload.format or "").lower() or "wav"
            fake_url = f"data:audio/{fmt};base64,{payload.data}"
            path, mime = await self._resolve_media(fake_url, workspace, hint_name=f"audio.{fmt}")
            return f"[attached audio: {path} (mime={mime})]", path

        if isinstance(part, FileContent):
            payload = part.file
            if payload.file_id:
                path, mime, original = await self._materialize_file_id(payload.file_id, workspace)
                return f"[attached file: {path} (filename={original}, mime={mime})]", path
            if payload.file_data:
                path, mime = await self._materialize_file_data(
                    payload.file_data, payload.mime_type, payload.filename, workspace
                )
                return f"[attached file: {path} (mime={mime})]", path
            return "", None

        return "", None

    async def _resolve_media(self, url: str, workspace: Path, hint_name: Optional[str]) -> tuple[Path, str]:
        if url.startswith("data:"):
            return await self._materialize_data_url(url, workspace, hint_name)
        if url.startswith("file-"):  # shorthand "file-<id>"
            path, mime, _ = await self._materialize_file_id(url, workspace)
            return path, mime
        if url.startswith(("http://", "https://")):
            return await self._materialize_http(url, workspace, hint_name)
        raise ValueError(f"unsupported image/media URL scheme: {url[:40]}")


def _last_user_index(messages: list[ChatMessage]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            return i
    return -1


def _dedupe_path(p: Path) -> Path:
    if not p.exists():
        return p
    stem, suffix = p.stem, p.suffix
    for i in range(1, 10_000):
        candidate = p.with_name(f"{stem}-{i}{suffix}")
        if not candidate.exists():
            return candidate
    return p.with_name(f"{stem}-{uuid.uuid4().hex[:6]}{suffix}")


_MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/svg+xml": ".svg",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/aac": ".aac",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/webm": ".weba",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/x-msvideo": ".avi",
    "application/pdf": ".pdf",
    "application/json": ".json",
    "application/zip": ".zip",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/csv": ".csv",
    "application/octet-stream": "",
}


def _ext_for_mime(mime: str) -> str:
    m = (mime or "").lower().strip()
    if m in _MIME_EXT:
        return _MIME_EXT[m]
    import mimetypes

    guess = mimetypes.guess_extension(m) or ""
    return guess


def derive_session_id(messages: list[ChatMessage], explicit: Optional[str], user: Optional[str]) -> str:
    """Deterministically derive a session id for a chat completion.

    Priority: explicit request field > stable hash of the conversation's
    "anchor" (system messages + first user turn) > fresh UUID.

    Hashing only the anchor — not the full transcript — means every turn
    of the same OpenWebUI conversation lands on the same session key, so
    Claude Code's session can be reused via ``--resume`` indefinitely.
    """
    if explicit:
        return _clean_session_id(explicit)

    anchor: list[ChatMessage] = []
    seen_user = False
    for m in messages:
        role = m.role or ""
        if role in ("system", "developer"):
            anchor.append(m)
        elif role == "user":
            anchor.append(m)
            seen_user = True
            break

    if seen_user:
        h = hashlib.sha256()
        for m in anchor:
            h.update((m.role or "").encode())
            h.update(b"\x00")
            content = m.content
            if isinstance(content, str):
                h.update(content.encode("utf-8", errors="replace"))
            elif isinstance(content, list):
                for part in content:
                    h.update(part.model_dump_json().encode())
            h.update(b"\x00")
        digest = h.hexdigest()[:32]
        prefix = _clean_session_id(user) if user else "conv"
        return f"{prefix}-{digest[:24]}"

    return uuid.uuid4().hex


_SESSION_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _clean_session_id(value: str) -> str:
    cleaned = _SESSION_ID_RE.sub("-", value).strip("-._")
    return cleaned[:64] or uuid.uuid4().hex
