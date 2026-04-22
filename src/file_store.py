from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import AsyncIterator, Iterable, Optional

import aiofiles

try:
    import magic  # type: ignore
except Exception:  # pragma: no cover
    magic = None  # libmagic optional


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str) -> str:
    name = (name or "upload").strip().replace("\\", "/").split("/")[-1]
    name = SAFE_NAME_RE.sub("_", name).strip("._-")
    return name[:200] or "upload"


def _guess_mime(path: Path, fallback: Optional[str]) -> str:
    if fallback:
        return fallback
    if magic is not None:
        try:
            return magic.from_file(str(path), mime=True) or "application/octet-stream"
        except Exception:
            pass
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


@dataclass
class FileRecord:
    id: str
    filename: str
    bytes: int
    created_at: int
    purpose: str
    mime_type: str
    storage_path: str
    session_id: Optional[str] = None
    source: str = "upload"  # "upload" or "generated"

    def to_openai(self) -> dict:
        return {
            "id": self.id,
            "object": "file",
            "bytes": self.bytes,
            "created_at": self.created_at,
            "filename": self.filename,
            "purpose": self.purpose,
            "status": "processed",
            "mime_type": self.mime_type,
        }


class FileStore:
    """Filesystem-backed store for uploaded and generated binaries.

    Layout:
        files_dir/
            <file_id>/
                meta.json
                blob                 # the raw bytes
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._meta_lock = asyncio.Lock()

    def _dir_for(self, file_id: str) -> Path:
        return self.root / file_id

    async def save_bytes(
        self,
        data: bytes,
        filename: str,
        purpose: str = "user_data",
        mime_type: Optional[str] = None,
        session_id: Optional[str] = None,
        source: str = "upload",
    ) -> FileRecord:
        file_id = f"file-{uuid.uuid4().hex}"
        target_dir = self._dir_for(file_id)
        target_dir.mkdir(parents=True, exist_ok=False)
        blob_path = target_dir / "blob"
        async with aiofiles.open(blob_path, "wb") as f:
            await f.write(data)
        return await self._finalize(file_id, blob_path, filename, purpose, mime_type, session_id, source)

    async def save_stream(
        self,
        chunks: AsyncIterator[bytes],
        filename: str,
        purpose: str = "user_data",
        mime_type: Optional[str] = None,
        session_id: Optional[str] = None,
        source: str = "upload",
        max_bytes: Optional[int] = None,
    ) -> FileRecord:
        file_id = f"file-{uuid.uuid4().hex}"
        target_dir = self._dir_for(file_id)
        target_dir.mkdir(parents=True, exist_ok=False)
        blob_path = target_dir / "blob"
        written = 0
        async with aiofiles.open(blob_path, "wb") as f:
            async for chunk in chunks:
                if not chunk:
                    continue
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    await f.close()
                    shutil.rmtree(target_dir, ignore_errors=True)
                    raise ValueError(f"upload exceeds max size of {max_bytes} bytes")
                await f.write(chunk)
        return await self._finalize(file_id, blob_path, filename, purpose, mime_type, session_id, source)

    async def ingest_path(
        self,
        src: Path,
        filename: Optional[str] = None,
        purpose: str = "generated",
        mime_type: Optional[str] = None,
        session_id: Optional[str] = None,
        source: str = "generated",
        move: bool = False,
    ) -> FileRecord:
        file_id = f"file-{uuid.uuid4().hex}"
        target_dir = self._dir_for(file_id)
        target_dir.mkdir(parents=True, exist_ok=False)
        blob_path = target_dir / "blob"
        if move:
            shutil.move(str(src), str(blob_path))
        else:
            await asyncio.to_thread(shutil.copy2, str(src), str(blob_path))
        return await self._finalize(file_id, blob_path, filename or src.name, purpose, mime_type, session_id, source)

    async def _finalize(
        self,
        file_id: str,
        blob_path: Path,
        filename: str,
        purpose: str,
        mime_type: Optional[str],
        session_id: Optional[str],
        source: str,
    ) -> FileRecord:
        safe = _safe_filename(filename)
        size = blob_path.stat().st_size
        mime = _guess_mime(blob_path, mime_type)
        record = FileRecord(
            id=file_id,
            filename=safe,
            bytes=size,
            created_at=int(time.time()),
            purpose=purpose,
            mime_type=mime,
            storage_path=str(blob_path),
            session_id=session_id,
            source=source,
        )
        meta_path = blob_path.parent / "meta.json"
        async with aiofiles.open(meta_path, "w") as f:
            await f.write(json.dumps(asdict(record), indent=2))
        return record

    async def get(self, file_id: str) -> Optional[FileRecord]:
        meta_path = self._dir_for(file_id) / "meta.json"
        if not meta_path.exists():
            return None
        async with aiofiles.open(meta_path, "r") as f:
            raw = await f.read()
        try:
            return FileRecord(**json.loads(raw))
        except Exception:
            return None

    async def list(
        self,
        purpose: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10000,
    ) -> list[FileRecord]:
        out: list[FileRecord] = []
        for child in sorted(self.root.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            rec = await self.get(child.name)
            if rec is None:
                continue
            if purpose and rec.purpose != purpose:
                continue
            if session_id and rec.session_id != session_id:
                continue
            out.append(rec)
            if len(out) >= limit:
                break
        return out

    async def delete(self, file_id: str) -> bool:
        d = self._dir_for(file_id)
        if not d.exists():
            return False
        await asyncio.to_thread(shutil.rmtree, str(d), ignore_errors=True)
        return True

    async def open_stream(self, record: FileRecord, chunk_size: int = 1 << 16) -> AsyncIterator[bytes]:
        path = Path(record.storage_path)

        async def _iter() -> AsyncIterator[bytes]:
            async with aiofiles.open(path, "rb") as f:
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        return _iter()

    def blob_path(self, record: FileRecord) -> Path:
        return Path(record.storage_path)
