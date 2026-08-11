"""Per-model capability profiles.

The wrapper is the single source of truth for what each advertised model can
do (see TOOL-PROFILES-ROADMAP.md). A profile is a set of capabilities,
resolved per model id in three layers:

  built-in default  →  profile file (CLAUDE_WRAPPER_MODEL_PROFILES, JSON)
                    →  inline overrides (CLAUDE_WRAPPER_MODEL_PROFILE_OVERRIDES)

Absent any configuration the resolved profile reproduces today's behavior
exactly: the CLI loop keeps its built-in tools (terminal, web search,
sub-agents), clients may declare their own tools, and vision/file input work
everywhere. Capabilities that require wrapper machinery that doesn't exist
yet (server-side code interpreter, memory, citations, image generation)
default off until their phases land.

Profile file schema::

    {
      "default": {"capabilities": ["vision", "client_tools"]},
      "models": [
        {"match": "claude-haiku-*", "remove": ["terminal", "sub_agents"]},
        {"match": "claude-opus-5", "add": ["memory"]},
        {"match": "claude-sonnet-5", "capabilities": ["vision", "client_tools"]}
      ]
    }

`default` (optional) replaces the built-in default set. `models` entries are
applied in order; every entry whose `match` fits the model applies. An entry
carries either `capabilities` (replace the set) or `add`/`remove` (deltas),
never both. `match` is compared literally first, then as an fnmatch glob.
Matching is done against the base model id — effort suffixes (``"claude-opus-5
(high)"``) are stripped, and a ``[1m]`` id also matches its base id's entries.

One capability is additionally gated by a hard env toggle: ``terminal``.
Exposing a shell to a chat UI is consequential enough that a profile grant
alone must not be able to switch it on — CLAUDE_WRAPPER_EXPOSE_TERMINAL
(default off) masks ``terminal`` out of every resolved profile until an
operator sets it. Note this gates the *UI-facing* capability only; the
wrapper's internal delegation runs (audio/images/embeddings doing their work
through Claude Code's Bash) are not chat-path capability enforcement and are
unaffected.

Configuration errors raise ProfileConfigError naming the offending entry.
Resolution is memoized per base model for the process lifetime (same pattern
as config.supported_models); tests reset via reset_profile_cache().
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .config import _bool_env, split_model_effort

log = logging.getLogger("claude_wrapper.capabilities")

PROFILE_FILE_ENV = "CLAUDE_WRAPPER_MODEL_PROFILES"
PROFILE_OVERRIDES_ENV = "CLAUDE_WRAPPER_MODEL_PROFILE_OVERRIDES"
# Hard gate for the terminal capability: profile grants are masked while off.
TERMINAL_TOGGLE_ENV = "CLAUDE_WRAPPER_EXPOSE_TERMINAL"


class Capability(str, Enum):
    VISION = "vision"
    FILE_UPLOAD = "file_upload"
    WEB_SEARCH = "web_search"
    CODE_INTERPRETER = "code_interpreter"
    TERMINAL = "terminal"
    MEMORY = "memory"
    CITATIONS = "citations"
    IMAGE_GENERATION = "image_generation"
    SUB_AGENTS = "sub_agents"
    CLIENT_TOOLS = "client_tools"


_BY_VALUE = {c.value: c for c in Capability}

# What every model does today, so an absent profile file changes nothing.
DEFAULT_CAPABILITIES: frozenset[Capability] = frozenset(
    {
        Capability.VISION,
        Capability.FILE_UPLOAD,
        Capability.WEB_SEARCH,
        Capability.TERMINAL,
        Capability.SUB_AGENTS,
        Capability.CLIENT_TOOLS,
    }
)


class ProfileConfigError(ValueError):
    """A profile file or overrides document failed validation."""


@dataclass(frozen=True)
class _Entry:
    match: str
    replace: frozenset[Capability] | None
    add: frozenset[Capability]
    remove: frozenset[Capability]


@dataclass(frozen=True)
class _ProfileConfig:
    default: frozenset[Capability] | None
    entries: tuple[_Entry, ...]


def _parse_caps(value: object, where: str) -> frozenset[Capability]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ProfileConfigError(f"{where}: expected a list of capability names")
    caps = set()
    for name in value:
        cap = _BY_VALUE.get(name)
        if cap is None:
            raise ProfileConfigError(
                f"{where}: unknown capability {name!r} "
                f"(valid: {', '.join(sorted(_BY_VALUE))})"
            )
        caps.add(cap)
    return frozenset(caps)


def _parse_entry(raw: object, where: str) -> _Entry:
    if not isinstance(raw, dict):
        raise ProfileConfigError(f"{where}: expected an object")
    unknown = set(raw) - {"match", "capabilities", "add", "remove"}
    if unknown:
        raise ProfileConfigError(f"{where}: unknown keys {sorted(unknown)}")
    match = raw.get("match")
    if not isinstance(match, str) or not match.strip():
        raise ProfileConfigError(f"{where}: 'match' must be a non-empty string")
    has_replace = "capabilities" in raw
    has_delta = "add" in raw or "remove" in raw
    if has_replace and has_delta:
        raise ProfileConfigError(
            f"{where}: 'capabilities' cannot be combined with 'add'/'remove'"
        )
    if not has_replace and not has_delta:
        raise ProfileConfigError(
            f"{where}: entry must carry 'capabilities' or 'add'/'remove'"
        )
    return _Entry(
        match=match.strip(),
        replace=_parse_caps(raw["capabilities"], f"{where}.capabilities") if has_replace else None,
        add=_parse_caps(raw["add"], f"{where}.add") if "add" in raw else frozenset(),
        remove=_parse_caps(raw["remove"], f"{where}.remove") if "remove" in raw else frozenset(),
    )


def _parse_document(doc: object, source: str) -> _ProfileConfig:
    if not isinstance(doc, dict):
        raise ProfileConfigError(f"{source}: top level must be an object")
    unknown = set(doc) - {"default", "models"}
    if unknown:
        raise ProfileConfigError(f"{source}: unknown keys {sorted(unknown)}")
    default = None
    if "default" in doc:
        raw = doc["default"]
        if not isinstance(raw, dict) or set(raw) != {"capabilities"}:
            raise ProfileConfigError(
                f"{source}.default: expected an object with only 'capabilities'"
            )
        default = _parse_caps(raw["capabilities"], f"{source}.default.capabilities")
    entries: list[_Entry] = []
    raw_models = doc.get("models", [])
    if not isinstance(raw_models, list):
        raise ProfileConfigError(f"{source}.models: expected a list")
    for i, raw in enumerate(raw_models):
        entries.append(_parse_entry(raw, f"{source}.models[{i}]"))
    return _ProfileConfig(default=default, entries=tuple(entries))


def _load_config() -> _ProfileConfig:
    default: frozenset[Capability] | None = None
    entries: list[_Entry] = []

    path = (os.environ.get(PROFILE_FILE_ENV) or "").strip()
    if path:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise ProfileConfigError(f"{PROFILE_FILE_ENV}={path}: cannot read: {exc}") from exc
        try:
            doc = json.loads(text)
        except ValueError as exc:
            raise ProfileConfigError(f"{PROFILE_FILE_ENV}={path}: invalid JSON: {exc}") from exc
        cfg = _parse_document(doc, path)
        default = cfg.default
        entries.extend(cfg.entries)

    inline = (os.environ.get(PROFILE_OVERRIDES_ENV) or "").strip()
    if inline:
        try:
            doc = json.loads(inline)
        except ValueError as exc:
            raise ProfileConfigError(f"{PROFILE_OVERRIDES_ENV}: invalid JSON: {exc}") from exc
        cfg = _parse_document(doc, PROFILE_OVERRIDES_ENV)
        if cfg.default is not None:
            default = cfg.default
        # Override entries run after the file's, so they win on conflict.
        entries.extend(cfg.entries)

    return _ProfileConfig(default=default, entries=tuple(entries))


def _matches(entry: _Entry, names: tuple[str, ...]) -> bool:
    return any(n == entry.match or fnmatch.fnmatchcase(n, entry.match) for n in names)


def _resolve(base: str, cfg: _ProfileConfig) -> frozenset[Capability]:
    caps = set(cfg.default if cfg.default is not None else DEFAULT_CAPABILITIES)
    # A [1m] id also inherits its base id's entries. The literal id goes first
    # so an exact "claude-opus-5[1m]" entry matches without fnmatch escaping
    # (fnmatch would read the brackets as a character class).
    names = (base, base[: -len("[1m]")]) if base.endswith("[1m]") else (base,)
    for entry in cfg.entries:
        if not _matches(entry, names):
            continue
        if entry.replace is not None:
            caps = set(entry.replace)
        caps |= entry.add
        caps -= entry.remove
    return frozenset(caps)


_config_cache: _ProfileConfig | None = None
_resolved_cache: dict[str, frozenset[Capability]] = {}


def resolve_profile(model: str) -> frozenset[Capability]:
    """Resolved capability set for an advertised model id.

    Accepts any id the wrapper advertises, including effort variants
    ("claude-opus-5 (high)") and [1m] ids. Raises ProfileConfigError if the
    configured profile file or overrides are invalid.
    """
    global _config_cache
    base, _ = split_model_effort(model)
    cached = _resolved_cache.get(base)
    if cached is not None:
        return cached
    if _config_cache is None:
        _config_cache = _load_config()
    caps = _resolve(base, _config_cache)
    if Capability.TERMINAL in caps and not _bool_env(TERMINAL_TOGGLE_ENV, False):
        log.debug(
            "terminal capability masked for %s (%s not enabled)", base, TERMINAL_TOGGLE_ENV
        )
        caps = caps - {Capability.TERMINAL}
    _resolved_cache[base] = caps
    return caps


def has_capability(model: str, cap: Capability) -> bool:
    return cap in resolve_profile(model)


def reset_profile_cache() -> None:
    """Drop the memoized config and resolutions (tests / config reload)."""
    global _config_cache
    _config_cache = None
    _resolved_cache.clear()
