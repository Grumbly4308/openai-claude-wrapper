"""Discover the model ids the installed Claude Code CLI knows about.

Claude Code ships no command to enumerate models (``--model`` only *sets* one),
and the on-disk option caches are empty unless a 3P provider populates them. The
one place the full set lives is the CLI binary itself: a ~250 MB bundled
executable whose embedded JS contains every model id it understands.

We scan that binary once at startup, keep only the canonical
``<family>-<major>-<minor>`` ids (optionally the ``[1m]`` long-context variant),
and hand the result to ``config.supported_models()``. The bundle also contains
dated snapshots (``…-20251101``), internal routing ids (``…-v1``), deployment
ids (``…-fast``), dotted aliases (``sonnet-4.6``), and bare family aliases
(``claude-opus-4``) — all of which we drop here.

Discovery is best-effort: any failure returns an empty list so the caller falls
back to a static set.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

log = logging.getLogger("claude_wrapper.models")

# Broad matcher for any claude-* model token embedded in the bundle. Deliberately
# permissive (dates, suffixes, [1m], dotted) — narrowing happens in filter_canonical.
_MODEL_TOKEN = re.compile(rb"claude-(?:opus|sonnet|haiku)-[0-9][0-9a-z.\-]*(?:\[1m\])?")

# Keep only canonical "<family>-<major>-<minor>" ids (optionally the [1m]
# variant). Minor is bounded to 1-2 digits so an 8-digit dated snapshot like
# "claude-opus-4-20250514" (date read as a "minor") doesn't slip through.
_CANONICAL = re.compile(r"^claude-(opus|sonnet|haiku)-(\d+)-(\d{1,2})(\[1m\])?$")

# Families below this major are retired (sonnet-3-7, haiku-3-5, opus-3, …).
_MIN_MAJOR = 4

# Deprecated (retiring) models to hide even though the bundle still knows them.
# Deprecation is external metadata — it isn't encoded in the binary — so this is
# necessarily a maintained denylist; update it as models are deprecated. Compared
# against the base id (any trailing "[1m]" stripped).
DEPRECATED_MODELS = frozenset(
    {
        "claude-opus-4-0",
        "claude-sonnet-4-0",
    }
)

_READ_CHUNK = 8 * 1024 * 1024
# Carry a few bytes between chunks so a token straddling a read boundary still matches.
_OVERLAP = 64


def _resolve_bin(claude_bin: str) -> Path | None:
    """Resolve a ``claude_bin`` setting (name or path) to the real executable."""
    p = Path(claude_bin)
    if not p.is_absolute():
        found = shutil.which(claude_bin)
        if not found:
            return None
        p = Path(found)
    try:
        p = p.resolve()  # follow the version symlink to the actual binary
    except OSError:
        return None
    return p if p.is_file() else None


def _scan_tokens(path: Path) -> set[str]:
    found: set[str] = set()
    try:
        with path.open("rb") as f:
            tail = b""
            while True:
                chunk = f.read(_READ_CHUNK)
                if not chunk:
                    break
                buf = tail + chunk
                for m in _MODEL_TOKEN.finditer(buf):
                    found.add(m.group(0).decode("ascii", "ignore"))
                tail = buf[-_OVERLAP:]
    except OSError as exc:
        log.warning("model scan failed for %s: %s", path, exc)
        return set()
    return found


def filter_canonical(ids: set[str]) -> list[str]:
    """Reduce raw bundle tokens to the canonical, current-family ids we expose."""
    keep: list[str] = []
    for mid in ids:
        m = _CANONICAL.match(mid)
        if not m:
            continue
        if int(m.group(2)) < _MIN_MAJOR:
            continue
        if mid.removesuffix("[1m]") in DEPRECATED_MODELS:
            continue
        keep.append(mid)
    return sorted(keep)


def discover_models(claude_bin: str) -> list[str]:
    """Return canonical model ids embedded in the installed Claude Code binary.

    Returns an empty list on any failure (binary not found, unreadable, or no
    matches) so the caller can fall back to a static list.
    """
    path = _resolve_bin(claude_bin)
    if path is None:
        log.warning("claude binary %r not found; skipping model discovery", claude_bin)
        return []
    models = filter_canonical(_scan_tokens(path))
    log.info("discovered %d models from %s", len(models), path)
    return models
