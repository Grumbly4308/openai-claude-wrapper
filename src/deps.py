"""Shared singletons and FastAPI dependencies used by every router."""

from __future__ import annotations

from typing import Optional

from fastapi import Header, HTTPException

from .claude_runner import ClaudeRunner, SessionRegistry
from .config import SETTINGS
from .converters import MessagePreparer
from .delegate import Delegator
from .file_store import FileStore


FILE_STORE = FileStore(SETTINGS.files_dir)
SESSIONS = SessionRegistry(SETTINGS.sessions_dir)
RUNNER = ClaudeRunner(
    registry=SESSIONS,
    workspace_root=SETTINGS.workspace_dir,
    claude_bin=SETTINGS.claude_bin,
    request_timeout_seconds=SETTINGS.request_timeout_seconds,
)
PREPARER = MessagePreparer(FILE_STORE, SETTINGS.workspace_dir)
DELEGATE = Delegator(RUNNER, SETTINGS.workspace_dir)


async def auth_dependency(authorization: Optional[str] = Header(default=None)) -> None:
    if not SETTINGS.require_auth:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="missing Authorization header")
    token = authorization
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if token not in SETTINGS.api_keys:
        raise HTTPException(status_code=401, detail="invalid API key")
