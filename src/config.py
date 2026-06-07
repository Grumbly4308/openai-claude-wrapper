from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


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


def _keyword_set(raw: str) -> frozenset[str]:
    return frozenset(k.strip().lower() for k in raw.split(",") if k.strip())


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
    openwebui_base_url: str
    openwebui_api_key: str
    openwebui_default_collection: str
    pdf_inline_max_chars: int
    effort: str
    session_token_allowance: int
    session_block_percent: float
    session_plan: str
    budget_continue_keywords: frozenset[str]

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


# Reasoning-effort levels accepted by `claude --effort`.
EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

# "ultracode" is exposed as an effort choice too, but it is NOT a `--effort`
# value — the CLI ignores it and falls back to the default effort. It is
# requested via `--settings '{"ultracode": true}'` instead, which the CLI
# resolves to xhigh effort plus ultracode's dynamic-workflow orchestration
# opt-in (the exact behavior is the CLI's to decide). The runner special-cases
# it when building argv (see claude_runner._build_argv).
ULTRACODE_EFFORT = "ultracode"

# Every effort token a client may select. The wrapper advertises one model
# variant per choice for each effort-capable (Opus) model, e.g.
# "claude-opus-4-8 (max)" / "claude-opus-4-8 (ultracode)", so clients can
# switch effort straight from a model dropdown.
EFFORT_CHOICES: tuple[str, ...] = EFFORT_LEVELS + (ULTRACODE_EFFORT,)

_EFFORT_CHOICE_SET = frozenset(EFFORT_CHOICES)


def is_effort_capable(model: str) -> bool:
    """Whether `claude --effort` applies to this model. Effort is an Opus 4.5+
    feature; other families ignore or reject the flag."""
    return model.startswith("claude-opus-")


def advertised_models() -> list[str]:
    """SUPPORTED_MODELS plus one '<model> (<choice>)' variant per effort choice
    for each effort-capable model. This is what /v1/models exposes."""
    out: list[str] = []
    for base in SUPPORTED_MODELS:
        out.append(base)
        if is_effort_capable(base):
            out.extend(f"{base} ({lvl})" for lvl in EFFORT_CHOICES)
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
