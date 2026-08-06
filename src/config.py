from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

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
    tools_mode: str
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

        # Who owns the agent loop for a tools-carrying chat request. "bridge"
        # (the default) is the historical behavior: any request declaring tools
        # goes to the tool bridge. "agentic" keeps only the turns that are owed
        # a tool call on the bridge and runs the CLI for the rest, so a chat UI
        # in native function-calling mode can still produce files. Resolved here
        # rather than at the call site so an unknown value warns once at boot.
        tools_mode = os.environ.get("CLAUDE_WRAPPER_TOOLS_MODE", "bridge").strip().lower() or "bridge"
        if tools_mode not in ("bridge", "agentic"):
            log.warning("unknown CLAUDE_WRAPPER_TOOLS_MODE=%r; using 'bridge'", tools_mode)
            tools_mode = "bridge"

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
            tools_mode=tools_mode,
            claude_bin=os.environ.get("CLAUDE_WRAPPER_CLAUDE_BIN", "claude"),
            max_upload_bytes=int(os.environ.get("CLAUDE_WRAPPER_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024))),
            request_timeout_seconds=int(os.environ.get("CLAUDE_WRAPPER_REQUEST_TIMEOUT", "1800")),
            public_base_url=os.environ.get("CLAUDE_WRAPPER_PUBLIC_BASE_URL", "").rstrip("/"),
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
            # 0 = never expires (still signed).
            download_url_ttl_seconds=_int_env("CLAUDE_WRAPPER_DOWNLOAD_URL_TTL", 2592000),
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
