"""Run Claude Code as a one-shot worker for endpoint-specific subtasks.

Each OpenAI endpoint that isn't pure text (audio, images, embeddings, etc.)
delegates execution to Claude Code running in an isolated workspace:

    1. Caller writes input files into ``workspace/uploads/`` (or inline).
    2. Caller describes the task as a natural-language prompt that tells
       Claude exactly which output files to produce and where.
    3. Claude runs, uses its Bash / Read / Write tools to do the work
       (install whisper, invoke ffmpeg, write a vector file, etc.).
    4. Caller reads the named output files back and returns them in the
       endpoint's OpenAI-shaped response.

The workspace is a per-request scratch directory rooted under the main
session workspace, so concurrent requests don't collide.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .claude_runner import ClaudeResult, ClaudeRunner


log = logging.getLogger("claude_wrapper.delegate")


@dataclass
class DelegationResult:
    workspace: Path
    session_key: str
    claude: ClaudeResult
    outputs: list[Path]


class Delegator:
    """Helper around ClaudeRunner for one-shot endpoint subtasks."""

    def __init__(self, runner: ClaudeRunner, workspace_root: Path):
        self.runner = runner
        self.workspace_root = workspace_root

    def new_workspace(self, kind: str) -> tuple[Path, str]:
        """Create a fresh per-request workspace with uploads/ + outputs/."""
        key = f"{kind}-{uuid.uuid4().hex[:16]}"
        cwd = self.workspace_root / key
        (cwd / "uploads").mkdir(parents=True, exist_ok=True)
        (cwd / "outputs").mkdir(parents=True, exist_ok=True)
        return cwd, key

    async def run(
        self,
        prompt: str,
        kind: str,
        model: Optional[str] = None,
        session_key: Optional[str] = None,
        workspace: Optional[Path] = None,
        cleanup: bool = False,
    ) -> DelegationResult:
        """Execute a task prompt. Create a fresh workspace if none provided."""
        if workspace is None or session_key is None:
            workspace, session_key = self.new_workspace(kind)
        started = time.time()
        result = await self.runner.run_collect(
            prompt=prompt,
            session_key=session_key,
            model=model,
        )
        outputs_dir = workspace / "outputs"
        outputs = sorted(p for p in outputs_dir.rglob("*") if p.is_file()) if outputs_dir.exists() else []
        log.info(
            "delegated %s session=%s secs=%.2f outputs=%d err=%s",
            kind,
            session_key,
            time.time() - started,
            len(outputs),
            result.error,
        )
        if cleanup:
            shutil.rmtree(workspace, ignore_errors=True)
        return DelegationResult(workspace=workspace, session_key=session_key, claude=result, outputs=outputs)

    @staticmethod
    def extract_json_block(text: str) -> Optional[dict]:
        """Pull the first ``{...}`` JSON object from a chat response.

        Claude tends to wrap JSON in fences or preamble; this strips that.
        """
        if not text:
            return None
        s = text.strip()
        # Fenced block: ```json ... ```
        if "```" in s:
            parts = s.split("```")
            for chunk in parts:
                chunk = chunk.strip()
                if chunk.startswith("json"):
                    chunk = chunk[4:].strip()
                if chunk.startswith("{") and chunk.rstrip().endswith("}"):
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
        # Bare: first { ... matching } at the top level
        start = s.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(s)):
            c = s[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        return None
        return None
