"""Per-conversation token accounting and budget gating.

The wrapper spends the operator's Anthropic session/subscription quota every time
it runs ``claude -p``. ``UsageLedger`` accumulates the tokens spent by each
conversation (keyed by the same ``session_key`` the runner uses) and decides when
a conversation has consumed its current allowance — at which point the API layer
pauses and asks the user to confirm before spending more.

Budgeting is expressed in *blocks*: a block is a fixed number of tokens, computed
upstream as ``session_token_allowance × block_percent`` (see ``config.py``). A
conversation starts with one block of allowance; replying "continue" grants one
more block (``grants += 1``). The ledger itself only knows the block size — the
percent/allowance arithmetic stays in config.

State is persisted one JSON file per session key, mirroring ``SessionRegistry``,
so budgets survive restarts. A new conversation gets a fresh ``session_key`` and
therefore a fresh, empty ledger entry — i.e. starting a new chat resets the cap.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import aiofiles

log = logging.getLogger("claude_wrapper.usage")


@dataclass
class UsageState:
    """A snapshot of one conversation's token accounting.

    ``block_tokens`` is carried alongside the persisted counters so the computed
    properties below are self-contained (callers can format messages from a single
    object). ``grants`` starts at 1 — every conversation gets one block for free.
    """

    spent_tokens: int = 0
    grants: int = 1
    requests: int = 0
    block_tokens: int = 0

    @property
    def allowance_tokens(self) -> int:
        return self.block_tokens * self.grants

    @property
    def over_budget(self) -> bool:
        return self.block_tokens > 0 and self.spent_tokens >= self.allowance_tokens


class UsageLedger:
    """Per-session token accounting persisted to disk.

    All mutating operations are serialized by a single lock: the read-modify-write
    of a state file must be atomic with respect to concurrent requests to the same
    conversation. Budget enforcement runs before the runner's own per-session lock,
    so without this two in-flight turns could race the same file.
    """

    def __init__(self, root: Path, block_tokens: int):
        self.root = root
        self.block_tokens = max(0, int(block_tokens))
        if self.block_tokens > 0:
            self.root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self.block_tokens > 0

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def _state_from_disk(self, key: str) -> UsageState:
        path = self._path(key)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return UsageState(
                    spent_tokens=int(data.get("spent_tokens") or 0),
                    grants=int(data.get("grants") or 1),
                    requests=int(data.get("requests") or 0),
                    block_tokens=self.block_tokens,
                )
            except Exception:  # pragma: no cover - corrupt/partial file
                log.warning("usage ledger %s unreadable; treating as empty", path)
        return UsageState(block_tokens=self.block_tokens)

    async def _write(self, key: str, state: UsageState) -> None:
        payload = json.dumps(
            {
                "spent_tokens": state.spent_tokens,
                "grants": state.grants,
                "requests": state.requests,
            }
        )
        async with aiofiles.open(self._path(key), "w") as f:
            await f.write(payload)

    async def snapshot(self, key: str) -> UsageState:
        """Return the conversation's current accounting (empty if unseen)."""
        if not self.enabled:
            return UsageState(block_tokens=self.block_tokens)
        async with self._lock:
            return self._state_from_disk(key)

    async def record(self, key: str, tokens: int) -> UsageState:
        """Add a completed request's token spend to the conversation's total."""
        if not self.enabled:
            return UsageState(block_tokens=self.block_tokens)
        async with self._lock:
            state = self._state_from_disk(key)
            state.spent_tokens += max(0, int(tokens))
            state.requests += 1
            await self._write(key, state)
            return state

    async def grant(self, key: str) -> UsageState:
        """Extend the conversation's allowance by one more block (a "continue")."""
        if not self.enabled:
            return UsageState(block_tokens=self.block_tokens)
        async with self._lock:
            state = self._state_from_disk(key)
            state.grants += 1
            await self._write(key, state)
            return state
