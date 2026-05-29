from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    workspace_dir: Path
    files_dir: Path
    sessions_dir: Path
    api_keys: frozenset[str]
    require_auth: bool
    default_model: str
    claude_bin: str
    max_upload_bytes: int
    request_timeout_seconds: int
    public_base_url: str
    openwebui_base_url: str
    openwebui_api_key: str
    openwebui_default_collection: str
    pdf_inline_max_chars: int
    effort: str

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.environ.get("CLAUDE_WRAPPER_DATA", "/data"))
        workspace = Path(os.environ.get("CLAUDE_WRAPPER_WORKSPACE", str(data_dir / "workspace")))
        files = Path(os.environ.get("CLAUDE_WRAPPER_FILES", str(data_dir / "files")))
        sessions = Path(os.environ.get("CLAUDE_WRAPPER_SESSIONS", str(data_dir / "sessions")))

        raw_keys = os.environ.get("CLAUDE_WRAPPER_API_KEYS", "").strip()
        keys = frozenset(k.strip() for k in raw_keys.split(",") if k.strip())
        require = bool(keys)

        for d in (data_dir, workspace, files, sessions):
            d.mkdir(parents=True, exist_ok=True)

        return cls(
            data_dir=data_dir,
            workspace_dir=workspace,
            files_dir=files,
            sessions_dir=sessions,
            api_keys=keys,
            require_auth=require,
            default_model=os.environ.get("CLAUDE_WRAPPER_DEFAULT_MODEL", "claude-opus-4-8"),
            claude_bin=os.environ.get("CLAUDE_WRAPPER_CLAUDE_BIN", "claude"),
            max_upload_bytes=int(os.environ.get("CLAUDE_WRAPPER_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024))),
            request_timeout_seconds=int(os.environ.get("CLAUDE_WRAPPER_REQUEST_TIMEOUT", "1800")),
            public_base_url=os.environ.get("CLAUDE_WRAPPER_PUBLIC_BASE_URL", "").rstrip("/"),
            openwebui_base_url=os.environ.get("OPENWEBUI_BASE_URL", "").rstrip("/"),
            openwebui_api_key=os.environ.get("OPENWEBUI_API_KEY", ""),
            openwebui_default_collection=os.environ.get("OPENWEBUI_DEFAULT_COLLECTION", ""),
            pdf_inline_max_chars=int(os.environ.get("CLAUDE_WRAPPER_PDF_INLINE_MAX_CHARS", "0")),
            # Reasoning effort forwarded to `claude --effort`. Empty means
            # "don't pass the flag" (use the CLI's own default).
            effort=os.environ.get("CLAUDE_WRAPPER_EFFORT", "").strip(),
        )


SETTINGS = Settings.from_env()


SUPPORTED_MODELS: tuple[str, ...] = (
    "claude-opus-4-8",
    "claude-opus-4-8[1m]",
    "claude-opus-4-7",
    "claude-opus-4-7[1m]",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
)


# Reasoning-effort levels accepted by `claude --effort`. The wrapper advertises
# one model variant per level for every effort-capable (Opus) model, e.g.
# "claude-opus-4-8 (max)", so clients can switch effort from a model dropdown.
EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

_EFFORT_LEVEL_SET = frozenset(EFFORT_LEVELS)


def is_effort_capable(model: str) -> bool:
    """Whether `claude --effort` applies to this model. Effort is an Opus 4.5+
    feature; other families ignore or reject the flag."""
    return model.startswith("claude-opus-")


def advertised_models() -> list[str]:
    """SUPPORTED_MODELS plus one '<model> (<level>)' variant per effort level
    for each effort-capable model. This is what /v1/models exposes."""
    out: list[str] = []
    for base in SUPPORTED_MODELS:
        out.append(base)
        if is_effort_capable(base):
            out.extend(f"{base} ({lvl})" for lvl in EFFORT_LEVELS)
    return out


def split_model_effort(model: str) -> tuple[str, str | None]:
    """Split an advertised model id into (base_model, effort).

    Accepts the '<base> (<level>)' form shown in /v1/models and a '<base>:<level>'
    shorthand. Returns (model, None) when no recognized effort suffix is present,
    so plain model ids keep using the server-default effort.
    """
    m = (model or "").strip()
    paren = re.match(r"^(?P<base>.+?)\s*\((?P<lvl>[A-Za-z]+)\)\s*$", m)
    if paren and paren.group("lvl").lower() in _EFFORT_LEVEL_SET:
        return paren.group("base").strip(), paren.group("lvl").lower()
    if ":" in m:
        base, _, lvl = m.rpartition(":")
        if base.strip() and lvl.strip().lower() in _EFFORT_LEVEL_SET:
            return base.strip(), lvl.strip().lower()
    return m, None
