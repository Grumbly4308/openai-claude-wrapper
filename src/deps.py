"""Shared singletons and FastAPI dependencies used by every router."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Header, HTTPException, Query

from . import download_tokens
from .claude_runner import ClaudeRunner, RemoteAgentExecutor, SessionRegistry
from .config import SETTINGS
from .converters import MessagePreparer
from .delegate import Delegator
from .file_store import FileStore
from .usage import UsageLedger


FILE_STORE = FileStore(SETTINGS.files_dir)
SESSIONS = SessionRegistry(SETTINGS.sessions_dir)
# Per-conversation token accounting. Stored in a dedicated subdir so its
# "{key}.json" files never collide with SessionRegistry's, which live directly
# under sessions_dir. Disabled (no-op) unless a session token allowance is set.
USAGE_LEDGER = UsageLedger(SETTINGS.sessions_dir / "usage", SETTINGS.session_block_tokens)

if USAGE_LEDGER.enabled:
    logging.getLogger("claude_wrapper.usage").info(
        "per-conversation token cap on: plan=%s allowance=%d block=%d (%.3g%%)",
        SETTINGS.session_plan or "custom",
        SETTINGS.session_token_allowance,
        SETTINGS.session_block_tokens,
        SETTINGS.session_block_percent,
    )
RUNNER = ClaudeRunner(
    registry=SESSIONS,
    workspace_root=SETTINGS.workspace_dir,
    claude_bin=SETTINGS.claude_bin,
    request_timeout_seconds=SETTINGS.request_timeout_seconds,
    effort=SETTINGS.effort,
    # Empty when CLAUDE_WRAPPER_CLARIFY=off, which makes clarify=True a no-op.
    clarify_system_prompt=SETTINGS.clarify_system_prompt if SETTINGS.clarify_enabled else "",
    clarify_disallowed_tools=SETTINGS.clarify_disallowed_tools if SETTINGS.clarify_enabled else (),
    # Empty when CLAUDE_WRAPPER_WORKSPACE_HINT=off, which makes
    # workspace_hint=True a no-op.
    workspace_system_prompt=SETTINGS.workspace_system_prompt if SETTINGS.workspace_hint_enabled else "",
    stream_partial_messages=SETTINGS.stream_partial_messages,
    # Sandboxed topology: send runs to the agent shim instead of spawning the
    # CLI here. None keeps the classic local subprocess.
    executor=(
        RemoteAgentExecutor(SETTINGS.agent_url, SETTINGS.agent_token)
        if SETTINGS.agent_url
        else None
    ),
)
PREPARER = MessagePreparer(FILE_STORE, SETTINGS.workspace_dir, registry=SESSIONS)
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


async def download_auth_dependency(
    file_id: str,
    exp: Optional[str] = Query(default=None),
    sig: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None),
) -> None:
    """Auth for the one route a browser opens directly.

    Accepts either a normal API key (SDK clients, unchanged) or a valid,
    unexpired per-file capability minted by _file_download_url. The signature is
    an ADDITION to the header check, never a replacement, and it grants read of
    exactly one blob — listing, metadata and delete still require the key.
    """
    if not SETTINGS.require_auth:
        return
    if download_tokens.verify(file_id, exp, sig, SETTINGS.download_signing_key):
        return
    # A browser following a link sends no header, so "bad signature" is the real
    # diagnosis there; falling through would report a missing Authorization
    # header the user cannot supply.
    if authorization is None and sig is not None:
        raise HTTPException(status_code=401, detail="invalid or expired download link")
    await auth_dependency(authorization)
