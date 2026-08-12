from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("claude_wrapper.config")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off", "disabled")


def _keyword_set(raw: str) -> frozenset[str]:
    return frozenset(k.strip().lower() for k in raw.split(",") if k.strip())


# Injected (via `claude --append-system-prompt`) on interactive chat/responses
# requests so Claude pauses for clarification at a real turn boundary the user
# can answer, instead of either firing the headless-dead AskUserQuestion card or
# asking-then-proceeding in one shot. Override with CLAUDE_WRAPPER_CLARIFY_PROMPT;
# turn the whole behavior off with CLAUDE_WRAPPER_CLARIFY=off.
DEFAULT_CLARIFY_SYSTEM_PROMPT = (
    "Clarification protocol (you are an interactive chat assistant reached over a "
    "headless API; there is no interactive question UI, so the only way to ask the "
    "user something is in plain text). When a genuine ambiguity would materially "
    "change what you build or answer, do NOT guess and proceed. Instead make your "
    "ENTIRE reply a short list of only the blocking questions — at most 2-3, each a "
    "numbered question with 2-4 lettered options and a recommended default — then "
    "STOP and end your turn so the user can answer. Do not begin the work in that "
    "same turn. End with a line like: \"Reply e.g. `1a 2b`, or in your own words — "
    "if you don't answer I'll proceed with the defaults.\" Treat the user's next "
    "message as the answers and continue from there. If the ambiguity is minor or "
    "the request is already clear, just proceed, stating any assumptions in one "
    "short line. Never ask more questions than necessary."
)


# Injected (via `claude --append-system-prompt`) on interactive chat/responses
# requests so Claude knows the files it writes are actually delivered to the
# user. Without it Claude has no reason to prefer writing a file over pasting
# the contents inline, and the download-link feature mostly never fires.
# Override with CLAUDE_WRAPPER_WORKSPACE_PROMPT; turn it off with
# CLAUDE_WRAPPER_WORKSPACE_HINT=off.
#
# Keep this text free of "json schema" markers. It travels as a CLI argument
# and is never concatenated into the prompt, so nothing reads it as a
# structured-output declaration today — but staying schema-free keeps it inert
# if a future refactor ever does concatenate it.
DEFAULT_WORKSPACE_SYSTEM_PROMPT = (
    "Workspace protocol. Your current working directory is a private, "
    "per-conversation workspace, and it is wired to the user: any file you "
    "create or modify there is captured after your turn and handed back to them "
    "as a download link appended to your reply. So when the user asks for a "
    "document, spreadsheet, dataset, script, diagram or image, WRITE IT TO A "
    "FILE in the working directory with a short descriptive filename and a "
    "correct extension, and do not also paste the full contents into your reply "
    "— summarize what you produced in a sentence or two and let the link carry "
    "the artifact. Short snippets the user clearly wants to read inline (a few "
    "lines of code, a command, a brief answer) stay in the reply as usual. Keep "
    "scratch and intermediate files out of the delivery: put them under a "
    "dot-prefixed directory such as `.scratch/` (dot-prefixed paths are never "
    "delivered), and never write into `uploads/`, which holds the user's own "
    "attachments."
)


# Subscription-plan → per-session token allowance.
#
# Anthropic does NOT publish a token figure for the Pro/Max session windows, so
# the absolute number is an estimate the operator can tune. What *is* defined is
# the relationship: Max is marketed as "5×" ($100) and "20×" ($200) of Pro. We
# therefore anchor on one tunable "Pro session tokens" value
# (CLAUDE_WRAPPER_PRO_SESSION_TOKENS) and derive each plan as a multiple of it.
#
# The anchor default (1,500,000) is calibrated from real Claude Code usage: a
# heavy ~2h session measured ~1.54M billable tokens (input + cache-creation +
# output, excluding near-free cache reads) = 21% of a Max-5x window, which back-
# solves to a Max-5x allowance of ~7.5M -> Pro anchor ~1.5M.
PRO_SESSION_TOKENS_DEFAULT = 1_500_000

# The cap is ON by default at Max 5x ($100). Set CLAUDE_WRAPPER_SESSION_PLAN to
# "off" (or "none"/"disabled") to turn it off.
DEFAULT_SESSION_PLAN = "max 5x"
PLAN_MULTIPLIERS: dict[str, int] = {
    "pro": 1,        # Claude Pro ($20/mo)
    "max_5x": 5,     # Claude Max 5× ($100/mo)
    "max_20x": 20,   # Claude Max 20× ($200/mo)
}


def _normalize_plan(raw: str) -> str:
    """Map a free-form plan string to a PLAN_MULTIPLIERS key (or "").

    Accepts forms like: "pro", "pro $20", "max 5x", "max $100",
    "max 20x", "max $200" (case-insensitive). Bare "max" defaults to 5× ($100).
    """
    s = (raw or "").strip().lower()
    if not s:
        return ""
    if "pro" in s:
        return "pro"
    if "20x" in s or "200" in s:
        return "max_20x"
    if "5x" in s or "100" in s or "max" in s:
        return "max_5x"
    return ""


def _resolve_session_allowance() -> tuple[int, str]:
    """Resolve (allowance_tokens, plan_label) from env.

    Precedence: an explicit CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE > 0 always
    wins (label "custom"); otherwise CLAUDE_WRAPPER_SESSION_PLAN derives
    allowance = pro_anchor × plan_multiplier. The plan defaults to Max 5x when
    unset, so the cap is ON out of the box; set the plan to "off" (→ unrecognized
    → multiplier None) to disable.
    """
    explicit = _int_env("CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE", 0)
    raw_plan = os.environ.get("CLAUDE_WRAPPER_SESSION_PLAN")
    if raw_plan is None or not raw_plan.strip():
        raw_plan = DEFAULT_SESSION_PLAN
    plan = _normalize_plan(raw_plan)
    if explicit > 0:
        # An explicit number isn't a plan's figure — label it "custom" so the
        # startup log doesn't pair a plan name with a mismatched allowance.
        return explicit, "custom"
    mult = PLAN_MULTIPLIERS.get(plan)
    if mult:
        pro = _int_env("CLAUDE_WRAPPER_PRO_SESSION_TOKENS", PRO_SESSION_TOKENS_DEFAULT)
        return max(0, pro) * mult, plan
    return 0, ""


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
    derive_base_url: bool
    download_signing_key: str
    download_url_ttl_seconds: int
    openwebui_base_url: str
    openwebui_api_key: str
    openwebui_default_collection: str
    pdf_inline_max_chars: int
    effort: str
    session_token_allowance: int
    session_block_percent: float
    session_plan: str
    budget_continue_keywords: frozenset[str]
    clarify_enabled: bool
    clarify_system_prompt: str
    clarify_disallowed_tools: tuple[str, ...]
    workspace_hint_enabled: bool
    workspace_system_prompt: str
    stream_partial_messages: bool
    agent_url: str
    agent_token: str

    @property
    def download_links_signed(self) -> bool:
        """Whether download-link capabilities are actually *enforced*.

        deps.download_auth_dependency returns before any signature check when
        require_auth is off, so a signing key alone only stamps exp/sig onto
        links nobody verifies. Anything reporting on link security must key off
        this, not off download_signing_key.
        """
        return self.require_auth and bool(self.download_signing_key)

    @property
    def session_block_tokens(self) -> int:
        """Tokens a conversation may spend per checkpoint = allowance × percent.

        Zero (the default) disables the per-conversation cap entirely.
        """
        if self.session_token_allowance <= 0 or self.session_block_percent <= 0:
            return 0
        return int(self.session_token_allowance * self.session_block_percent / 100)

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.environ.get("CLAUDE_WRAPPER_DATA", "/data"))
        workspace = Path(os.environ.get("CLAUDE_WRAPPER_WORKSPACE", str(data_dir / "workspace")))
        files = Path(os.environ.get("CLAUDE_WRAPPER_FILES", str(data_dir / "files")))
        sessions = Path(os.environ.get("CLAUDE_WRAPPER_SESSIONS", str(data_dir / "sessions")))

        raw_keys = os.environ.get("CLAUDE_WRAPPER_API_KEYS", "").strip()
        keys = frozenset(k.strip() for k in raw_keys.split(",") if k.strip())
        require = bool(keys)

        # Signing key for download links. An explicit key wins; otherwise derive
        # one deterministically from the configured API keys, so it is stable
        # across workers (compose exposes CLAUDE_WRAPPER_WORKERS) and restarts
        # with zero operator configuration and no state on disk. Empty when no
        # API keys are set: auth is off in that deployment, so links need no
        # signature.
        #
        # The derivation is deliberately EXPENSIVE. A signed link is a
        # (file_id, exp, sig) triple published into chat text, shared-chat
        # exports, browser history and proxy logs — i.e. a known MAC pair over
        # this key. A single cheap hash would make any leaked link an offline
        # oracle for CLAUDE_WRAPPER_API_KEYS itself, at millions of guesses per
        # second, which is exactly the full-privilege escalation the per-file
        # capability exists to avoid. scrypt runs once at boot (~50ms, unnoticed
        # by the operator) and is memory-hard, so it does not parallelise onto a
        # GPU the way SHA-256 does.
        #
        # This raises the cost of a weak API key; it does not make one safe. The
        # salt is a fixed domain string (there is no disk state to hold a random
        # one), so set CLAUDE_WRAPPER_DOWNLOAD_SIGNING_KEY to 32 random bytes if
        # you want the link secret fully independent of your API keys — that
        # also stops an API-key rotation invalidating every outstanding link.
        dl_key = os.environ.get("CLAUDE_WRAPPER_DOWNLOAD_SIGNING_KEY", "").strip()
        if not dl_key and keys:
            dl_key = hashlib.scrypt(
                "\n".join(sorted(keys)).encode(),
                salt=b"claude-wrapper/download-key/v2",
                n=2**15,
                r=8,
                p=1,
                maxmem=64 * 1024 * 1024,
                dklen=32,
            ).hex()

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
            public_base_url=os.environ.get("CLAUDE_WRAPPER_PUBLIC_BASE_URL", "").strip().rstrip("/"),
            # Generated-file links must be absolute to be clickable in a chat
            # UI. public_base_url stays authoritative when set; otherwise the
            # inbound request's own origin is used. Set
            # CLAUDE_WRAPPER_DERIVE_BASE_URL=off to restore the old plain-text
            # "→ file_id=…" trailer on deployments that name themselves badly.
            derive_base_url=_bool_env("CLAUDE_WRAPPER_DERIVE_BASE_URL", True),
            download_signing_key=dl_key,
            # Lifetime of a download link's capability, in seconds (30 days).
            # Expiry is the only revocation short of deleting the blob, and
            # these links live forever in the chat client's message history.
            # 0 = never expires (still signed). Clamped so a negative value —
            # a plausible "disable" idiom — is the same 0 that mint() would
            # treat it as anyway, and everything downstream (the boot log
            # included) sees one consistent "never expires" instead of "-1s".
            download_url_ttl_seconds=max(0, _int_env("CLAUDE_WRAPPER_DOWNLOAD_URL_TTL", 2592000)),
            openwebui_base_url=os.environ.get("OPENWEBUI_BASE_URL", "").rstrip("/"),
            openwebui_api_key=os.environ.get("OPENWEBUI_API_KEY", ""),
            openwebui_default_collection=os.environ.get("OPENWEBUI_DEFAULT_COLLECTION", ""),
            pdf_inline_max_chars=int(os.environ.get("CLAUDE_WRAPPER_PDF_INLINE_MAX_CHARS", "0")),
            # Reasoning effort forwarded to `claude --effort`. Empty means
            # "don't pass the flag" (use the CLI's own default).
            effort=os.environ.get("CLAUDE_WRAPPER_EFFORT", "").strip(),
            # Per-conversation token cap. The session allowance can be set
            # directly (CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE) or derived from a
            # named subscription plan (CLAUDE_WRAPPER_SESSION_PLAN=pro|max_5x|
            # max_20x); an explicit allowance wins. A conversation may spend
            # `allowance × percent` tokens before the wrapper pauses to ask
            # whether to continue. Allowance 0 disables the cap.
            session_token_allowance=_resolve_session_allowance()[0],
            session_block_percent=_float_env("CLAUDE_WRAPPER_SESSION_BLOCK_PERCENT", 5.0),
            session_plan=_resolve_session_allowance()[1],
            budget_continue_keywords=_keyword_set(
                os.environ.get(
                    "CLAUDE_WRAPPER_BUDGET_CONTINUE_KEYWORD",
                    "continue,proceed,keep going,go on,yes",
                )
            ),
            # Interactive clarification protocol. On by default for chat/responses
            # so Claude asks answerable questions and pauses, rather than firing a
            # dead AskUserQuestion card or asking-then-proceeding. Delegated task
            # endpoints (audio/images/etc.) never opt in.
            clarify_enabled=_bool_env("CLAUDE_WRAPPER_CLARIFY", True),
            clarify_system_prompt=(
                os.environ.get("CLAUDE_WRAPPER_CLARIFY_PROMPT", "").strip()
                or DEFAULT_CLARIFY_SYSTEM_PROMPT
            ),
            clarify_disallowed_tools=tuple(
                t.strip()
                for t in os.environ.get(
                    "CLAUDE_WRAPPER_CLARIFY_DISALLOWED_TOOLS", "AskUserQuestion"
                ).split(",")
                if t.strip()
            ),
            # Tell Claude its cwd is a workspace whose new files are delivered
            # to the user as download links.
            #
            # OFF by default, because it changes the SHAPE of the reply: it asks
            # Claude to write the deliverable to a file and summarise instead of
            # pasting it inline. That is right for a chat UI and wrong for every
            # existing programmatic caller — run_chat_completion is shared by
            # /v1/completions, /v1/assistants runs and the batches worker, so a
            # script doing completions.create("write me a python script…") and
            # reading choices[0].text would silently start getting "I wrote it
            # to script.py" plus a link. Turn it on for chat-UI deployments.
            workspace_hint_enabled=_bool_env("CLAUDE_WRAPPER_WORKSPACE_HINT", False),
            workspace_system_prompt=(
                os.environ.get("CLAUDE_WRAPPER_WORKSPACE_PROMPT", "").strip()
                or DEFAULT_WORKSPACE_SYSTEM_PROMPT
            ),
            # Add `--include-partial-messages` so Claude Code emits incremental
            # text/thinking deltas (live token-by-token streaming) instead of one
            # whole block per step. On by default; set CLAUDE_WRAPPER_STREAM_PARTIAL
            # =off to fall back to whole-block events.
            stream_partial_messages=_bool_env("CLAUDE_WRAPPER_STREAM_PARTIAL", True),
            # Sandboxed topology (docker-compose.sandbox.yml): when set, the CLI
            # is not spawned in this container — every run is sent to the agent
            # shim (src.agent_shim) at this base URL, which lives on an internal
            # network whose only egress is the allowlisting proxy. Empty (the
            # default) keeps the classic single-container local subprocess.
            agent_url=os.environ.get("CLAUDE_WRAPPER_AGENT_URL", "").strip().rstrip("/"),
            # Optional shared secret the shim requires as a bearer token. The
            # shim sits on an internal network either way; this guards against
            # anything else that can reach that network.
            agent_token=os.environ.get("CLAUDE_WRAPPER_AGENT_TOKEN", "").strip(),
        )


SETTINGS = Settings.from_env()


# Static fallback used only when binary discovery yields nothing (binary
# missing, unreadable, or its bundle format changed). Mirrors what the wrapper
# shipped with so /v1/models is never empty.
FALLBACK_MODELS: tuple[str, ...] = (
    "claude-opus-4-8",
    "claude-opus-4-8[1m]",
    "claude-opus-4-7",
    "claude-opus-4-7[1m]",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
)


# Reasoning-effort levels accepted by `claude --effort`. This stays the Opus
# ladder for back-compat with callers/tests that import EFFORT_LEVELS.
EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

# Sonnet 4.6+ accepts effort but not `max`; earlier Sonnet rejects effort entirely.
_SONNET_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh")

# "ultracode" is exposed as an effort choice too, but it is NOT a `--effort`
# value — the CLI ignores it and falls back to the default effort. It is
# requested via `--settings '{"ultracode": true}'` instead, which the CLI
# resolves to xhigh effort plus ultracode's dynamic-workflow orchestration
# opt-in (the exact behavior is the CLI's to decide). The runner special-cases
# it when building argv (see claude_runner._build_argv). Ultracode is Opus-only.
ULTRACODE_EFFORT = "ultracode"

# Union of every effort token a client may select, used to *recognize* a
# suffix while parsing a model id. Whether a given token actually applies to a
# given model is decided per-model by effort_choices_for().
EFFORT_CHOICES: tuple[str, ...] = EFFORT_LEVELS + (ULTRACODE_EFFORT,)

_EFFORT_CHOICE_SET = frozenset(EFFORT_CHOICES)

# Family-rule version boundaries for effort support (from the model docs):
# effort landed on Opus 4.5 and on Sonnet 4.6.
_OPUS_EFFORT_MIN = (4, 5)
_SONNET_EFFORT_MIN = (4, 6)

# Minor is 1-2 digits and must not be followed by another digit, so a dated
# snapshot ("claude-opus-4-20250514") isn't misread as version (4, 20).
_MODEL_FAMILY_RE = re.compile(r"^claude-(opus|sonnet|haiku)-(\d+)-(\d{1,2})(?!\d)")

# Codename families (fable/mythos) carry a single version, e.g. claude-fable-5,
# and are treated as Opus-tier for effort (full ladder + ultracode).
_CODENAME_RE = re.compile(r"^claude-(fable|mythos)-(\d{1,2})(?!\d)")


def _family_version(model: str) -> tuple[str | None, tuple[int, int] | None]:
    """Parse (family, (major, minor)) from a model id; (None, None) if unparseable.

    Tolerates trailing suffixes like ``[1m]`` — only the leading
    ``claude-<family>-<major>-<minor>`` is needed.
    """
    m = _MODEL_FAMILY_RE.match(model or "")
    if not m:
        return None, None
    return m.group(1), (int(m.group(2)), int(m.group(3)))


def effort_choices_for(model: str) -> tuple[str, ...]:
    """Effort choices a given model accepts, by the family rule.

    Opus 4.5+ and the codename families (fable/mythos): low/medium/high/xhigh/max
    + ultracode. Sonnet 4.6+: low/medium/high/xhigh (no max, no ultracode).
    Haiku, older Opus/Sonnet, and anything unrecognized: none.
    """
    fam, ver = _family_version(model)
    if ver is not None:
        if fam == "opus" and ver >= _OPUS_EFFORT_MIN:
            return EFFORT_CHOICES
        if fam == "sonnet" and ver >= _SONNET_EFFORT_MIN:
            return _SONNET_EFFORT_LEVELS
        return ()
    # Codename families (fable/mythos) are Opus-tier: full ladder + ultracode.
    if _CODENAME_RE.match(model or ""):
        return EFFORT_CHOICES
    return ()


def is_effort_capable(model: str) -> bool:
    """Whether `claude --effort` / the ultracode settings overlay apply here."""
    return bool(effort_choices_for(model))


_supported_models_cache: tuple[str, ...] | None = None


def _discovery_mode() -> str:
    """`auto` (default) scans the installed binary; `off` uses FALLBACK_MODELS."""
    return (os.environ.get("CLAUDE_WRAPPER_MODEL_DISCOVERY", "auto") or "auto").strip().lower()


def _build_supported_models() -> tuple[str, ...]:
    discovered: list[str] = []
    if _discovery_mode() != "off":
        try:
            from .model_discovery import discover_models

            discovered = discover_models(SETTINGS.claude_bin)
        except Exception:  # never let discovery break startup
            log.exception("model discovery failed; falling back to static list")
            discovered = []
    models = discovered or list(FALLBACK_MODELS)
    # The configured default must always be selectable, even if discovery missed it.
    if SETTINGS.default_model and SETTINGS.default_model not in models:
        models.append(SETTINGS.default_model)
    return tuple(dict.fromkeys(models))  # de-dupe, preserve order


def supported_models() -> tuple[str, ...]:
    """Models the wrapper accepts.

    Built once on first call by scanning the installed Claude Code binary (see
    model_discovery), then memoized for the process lifetime. Falls back to
    FALLBACK_MODELS when discovery is disabled or yields nothing.
    """
    global _supported_models_cache
    if _supported_models_cache is None:
        _supported_models_cache = _build_supported_models()
    return _supported_models_cache


def advertised_models() -> list[str]:
    """Each supported base model, plus one '<model> (<choice>)' variant per
    effort choice the model accepts. This is what /v1/models exposes."""
    out: list[str] = []
    for base in supported_models():
        out.append(base)
        out.extend(f"{base} ({choice})" for choice in effort_choices_for(base))
    return out


def split_model_effort(model: str) -> tuple[str, str | None]:
    """Split an advertised model id into (base_model, effort).

    Accepts the '<base> (<choice>)' form shown in /v1/models and a '<base>:<choice>'
    shorthand. Returns (model, None) when no recognized effort suffix is present,
    so plain model ids keep using the server-default effort.
    """
    m = (model or "").strip()
    paren = re.match(r"^(?P<base>.+?)\s*\((?P<lvl>[A-Za-z]+)\)\s*$", m)
    if paren and paren.group("lvl").lower() in _EFFORT_CHOICE_SET:
        return paren.group("base").strip(), paren.group("lvl").lower()
    if ":" in m:
        base, _, lvl = m.rpartition(":")
        if base.strip() and lvl.strip().lower() in _EFFORT_CHOICE_SET:
            return base.strip(), lvl.strip().lower()
    return m, None


# ---------- Claude Code credentials ----------

# Where the CLI persists an interactive login. The tool bridge reads the same
# file for its direct Messages API calls, which is why it is configurable.
CREDENTIALS_FILE = Path(
    os.environ.get(
        "CLAUDE_WRAPPER_CREDENTIALS_FILE",
        str(Path.home() / ".claude" / ".credentials.json"),
    )
)

# A login that expires within a month is treated as short-lived. The CLI renews
# its token whenever it runs, so that kind of credential only stays valid while
# the wrapper is *used* and its egress works — an idle deployment, or one whose
# proxy is down, drifts past expiry with nothing in the logs until every turn
# starts failing with a 401 the CLI reports as its own error. `claude
# setup-token` mints a credential measured in months instead, which is what a
# headless deployment wants.
_LONG_LIVED_SECONDS = 30 * 24 * 3600

# A login renews itself, so its refresh window only deserves a warning once
# re-authenticating is actually near. A fresh login is ~30 days, and warning
# about that at every boot would train you to ignore the line.
_RELOGIN_SOON_SECONDS = 7 * 24 * 3600


def _describe_duration(seconds: float) -> str:
    """Coarse, human-readable duration: the boot log wants '12d', not '12.4d'."""
    seconds = abs(int(seconds))
    if seconds >= 86400:
        return f"{seconds // 86400}d"
    if seconds >= 3600:
        return f"{seconds // 3600}h"
    if seconds >= 60:
        return f"{seconds // 60}m"
    return f"{seconds}s"


@dataclass(frozen=True)
class CredentialStatus:
    """The credential this container will actually authenticate with.

    ``kind`` follows tool_bridge.resolve_auth's precedence exactly, so what gets
    reported at boot is what a request would really use — not merely what is
    present. ``expires_in`` is None when there is no expiry to read: an API key,
    an opaque env token, or a credentials file without an ``expiresAt``.
    """

    kind: str  # "api-key" | "env-token" | "oauth-file" | "none"
    expires_in: Optional[float] = None
    # The file actually inspected. Carried on the status rather than read from
    # the module constant so the log names the path that was checked, which is
    # the whole point of printing it.
    path: Optional[Path] = None
    # Seconds until the REFRESH token dies. This is the number that decides
    # whether you ever have to log in again — expires_in only describes the
    # access token, which the CLI renews by itself.
    refresh_in: Optional[float] = None

    @property
    def expired(self) -> bool:
        return self.expires_in is not None and self.expires_in <= 0

    @property
    def short_lived(self) -> bool:
        """Whether this credential runs out soon in the way that matters.

        Keyed off the refresh window when there is one: an access token good for
        8 hours is not short-lived if a 30-day refresh token renews it, and
        warning about that is noise. Only when no refresh token is present does
        the access token's own expiry become the deadline.
        """
        if self.refresh_in is not None:
            return 0 < self.refresh_in < _RELOGIN_SOON_SECONDS
        return self.expires_in is not None and 0 < self.expires_in < _LONG_LIVED_SECONDS

    def describe(self) -> str:
        where = self.path or CREDENTIALS_FILE
        if self.kind == "api-key":
            return "ANTHROPIC_API_KEY (no expiry)"
        if self.kind == "env-token":
            if self.expires_in is None:
                return (
                    "CLAUDE_CODE_OAUTH_TOKEN from the environment (expiry not "
                    "readable here — set CLAUDE_CODE_OAUTH_TOKEN_MINTED to track it)"
                )
            if self.expired:
                return (
                    "CLAUDE_CODE_OAUTH_TOKEN from the environment, presumed EXPIRED "
                    f"~{_describe_duration(self.expires_in)} ago (minted "
                    "CLAUDE_CODE_OAUTH_TOKEN_MINTED + ~1y)"
                )
            return (
                "CLAUDE_CODE_OAUTH_TOKEN from the environment, "
                f"~{_describe_duration(self.expires_in)} of its ~1y left "
                "(estimated from CLAUDE_CODE_OAUTH_TOKEN_MINTED)"
            )
        if self.kind == "none":
            return f"NONE — no API key, no env token, and no usable login in {where}"
        if self.expires_in is None:
            return f"Claude Code login ({where}, no expiry recorded)"
        renew = (
            f", re-login needed in {_describe_duration(self.refresh_in)}"
            if self.refresh_in is not None and self.refresh_in > 0
            else ""
        )
        if self.expired:
            return (
                f"Claude Code login ({where}) EXPIRED "
                f"{_describe_duration(self.expires_in)} ago{renew}"
            )
        return (
            f"Claude Code login ({where}), access valid for "
            f"{_describe_duration(self.expires_in)}{renew}"
        )


def _epoch_seconds(raw: object) -> Optional[float]:
    """Milliseconds-since-epoch -> seconds from now, or None when unreadable."""
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return None
    return raw / 1000 - time.time()


def read_file_credential(path: Optional[Path] = None) -> CredentialStatus:
    """Inspect the on-disk login, ignoring the environment entirely.

    Separate from read_credential_status because an env credential wins but does
    NOT renew this file — so the boot report has to be able to look underneath.
    """
    where = path or CREDENTIALS_FILE
    try:
        data = json.loads(where.read_text())
    except (OSError, json.JSONDecodeError):
        return CredentialStatus("none", path=where)
    oauth = data.get("claudeAiOauth") or {}
    if not str(oauth.get("accessToken") or ""):
        return CredentialStatus("none", path=where)
    return CredentialStatus(
        "oauth-file",
        _epoch_seconds(oauth.get("expiresAt")),
        where,
        _epoch_seconds(oauth.get("refreshTokenExpiresAt")),
    )


def _minted_token_expiry() -> Optional[float]:
    """Estimated seconds until the env token dies, from its recorded mint date.

    `setup-token` prints an opaque credential: nothing can read its expiry
    afterwards, so the only warning anyone will ever get is arithmetic on the
    date it was minted. The CLI describes the token as good for about a year;
    the estimate is deliberately conservative in wording, not in math.
    """
    raw = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN_MINTED", "").strip()
    if not raw:
        return None
    try:
        minted = datetime.date.fromisoformat(raw)
    except ValueError:
        return None
    return (
        time.mktime(minted.timetuple()) + 365 * 86400 - time.time()
    )


def read_credential_status(path: Optional[Path] = None) -> CredentialStatus:
    """Inspect the credential in force, without validating it against the API."""
    where = path or CREDENTIALS_FILE
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return CredentialStatus("api-key", path=where)
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip():
        return CredentialStatus("env-token", _minted_token_expiry(), path=where)
    return read_file_credential(where)


def log_credential_status(where: str, path: Optional[Path] = None) -> CredentialStatus:
    """Report the credential state at boot. Called by every role that needs one.

    Escalates deliberately: an expired credential is an error because nothing
    will work until it is replaced, and a short-lived one is a warning because
    it works now and fails later, which is the case that reaches production.
    """
    status = read_credential_status(path)
    if status.kind == "none":
        log.warning(
            "%s credentials: %s — run `setup-token` (see README 'First-time login')",
            where,
            status.describe(),
        )
    elif status.expired:
        log.error(
            "%s credentials: %s. Every turn will fail with a 401 until this is "
            "replaced — log in again with the CLI's /login, or set "
            "CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`.",
            where,
            status.describe(),
        )
    elif status.short_lived:
        if status.kind == "env-token":
            # Nothing renews a minted token — the only remedy is a new one,
            # and the point of warning a month out is having time to mint it.
            log.warning(
                "%s credentials: %s. Nothing renews this token — mint a "
                "replacement with `setup-token` and update "
                "CLAUDE_CODE_OAUTH_TOKEN (and its _MINTED date) before it dies.",
                where,
                status.describe(),
            )
        else:
            log.warning(
                "%s credentials: %s. Renewal depends on the CLI running with working "
                "egress before then; if it lapses you need a fresh interactive login "
                "(the CLI's /login — `claude login` is not an auth command).",
                where,
                status.describe(),
            )
    else:
        log.info("%s credentials: %s", where, status.describe())

    # An env credential wins, and the CLI then stops touching the on-disk login
    # entirely — so a perfectly good volume credential sits there decaying, and
    # the day the env token is removed you fall through to something that
    # expired weeks ago. Nothing else surfaces that, so say it here.
    if status.kind in ("api-key", "env-token"):
        shadowed = read_file_credential(path)
        if shadowed.kind != "none":
            log.warning(
                "%s credentials: a login also exists (%s) but is SHADOWED by the "
                "environment credential — the CLI will not refresh it while that "
                "is set, so it decays until it needs a fresh login.",
                where,
                shadowed.describe(),
            )
    return status
