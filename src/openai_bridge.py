"""OpenAI function calling served by calling the OpenAI API directly.

The codex-agent counterpart of tool_bridge.py, speaking the same public
surface (resolve_auth / build_payload / complete / stream / aclose) so
main._tool_bridge_completion can switch modules on SETTINGS.agent with zero
shape changes. The design point that makes this module small: **the inbound
request is already OpenAI-shaped, so this bridge is a near-pure proxy to
``POST {base}/v1/chat/completions``** — no history translation, no tool-name
sanitization, and no wrapper-owned tool loop (memory/time_calc are Anthropic
hybrid-loop machinery and are never injected here). Upstream is the
authoritative validator; its errors are already OpenAI-shaped, so they pass
through where safe (see complete()'s status ladder).

Auth resolution order (per request):
  1. OPENAI_API_KEY (environment)
  2. auth.json in API-key mode (``codex login --with-api-key``), re-read per
     request via the codex-home read-only mount when the operator opts in.
ChatGPT-plan OAuth tokens are DELIBERATELY not a fallback: they authenticate
against the Codex backend, not the OpenAI Platform API — see resolve_auth().
This is the one asymmetry vs the Claude bridge (whose plan login IS
bridge-usable) and is called out in the README's Auth section.

These requests leave the API container and flow through HTTPS_PROXY → squid,
so ``api.openai.com`` must be on the allowlist for tools-carrying requests to
work (sandbox/allowlist.txt).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import httpx
from fastapi import HTTPException

from .capabilities import Capability, resolve_profile
from .config import CODEX_CREDENTIALS_FILE, CODEX_EFFORT_CHOICES, SETTINGS, split_model_effort
from .models import ChatCompletionRequest
# Shared error envelope + result shape — reused, not duplicated, so main.py's
# rendering code is provably identical for both bridges.
from .tool_bridge import BridgeResult, openai_error

log = logging.getLogger("claude_wrapper.openai_bridge")

# `or` pattern, not a .get default: the codex compose file delivers this var
# as "" (the ${VAR:-} interpolation trap) — a bare .get() default would send
# every bridge call to a relative URL and 502 permanently.
OPENAI_BASE_URL = (
    os.environ.get("CLAUDE_WRAPPER_OPENAI_BASE_URL", "").strip() or "https://api.openai.com"
).rstrip("/")
if OPENAI_BASE_URL.endswith("/v1"):
    # Accept the OpenAI SDK's convention too (OPENAI_BASE_URL=https://host/v1
    # is what every compatible backend documents): the request paths below
    # append /v1/..., so a suffixed value would 404 at /v1/v1/chat/completions
    # with an upstream error naming neither the URL nor the doubled path.
    OPENAI_BASE_URL = OPENAI_BASE_URL[: -len("/v1")]

# NOTE: no CLAUDE_WRAPPER_TOOLS_MAX_TOKENS equivalent here — unlike the
# Messages API, OpenAI does not require max_tokens; absent means model default.

_CODEX_CREDENTIALS_FILE = CODEX_CREDENTIALS_FILE

# Same proxy-buffer-flushing preamble the main chat stream uses (see main.py).
_STREAM_PREAMBLE_BYTES = int(os.environ.get("CLAUDE_WRAPPER_SSE_PREAMBLE_BYTES", "2048"))

# What a tenant sees when upstream rejects OUR credential. Fixed on purpose:
# OpenAI 401 bodies echo a partially-redacted rendering of the presented key,
# and those fragments belong to the operator (the server log), not to wrapper
# tenants. The status is wrapped into 502 at all because a passed-through 401
# would send the client chasing its own (valid) wrapper API key.
_CREDENTIAL_REJECTED_MSG = (
    "openai rejected the wrapper's upstream credential — check OPENAI_API_KEY / "
    "auth.json (details in the server log)"
)

# Upstream statuses whose bodies are safe and useful to the client verbatim
# (the "honor contracts" rule from the tool-bridge): request-shape errors,
# unknown model, oversized payload, rate limits. 401/403 are deliberately NOT
# here (operator credential, see above) and 5xx are wrapped as upstream errors.
_PASSTHROUGH_STATUSES = frozenset({400, 404, 413, 422, 429})

# The wrapper-extension keys on ChatCompletionRequest, enumerated and pinned
# (tests/test_openai_bridge.py pins the outbound key set). exclude_none alone
# is NOT enough: inline_generated_files has a non-None default and would reach
# OpenAI as an unknown top-level argument → 400 on every request.
_WRAPPER_EXTENSION_KEYS = ("session_id", "inline_generated_files", "clarify")


# Shared client; tests replace it with one built on httpx.MockTransport.
_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, read=float(SETTINGS.request_timeout_seconds))
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ---------- auth ----------


def _codex_file_api_key() -> str:
    """The API key persisted by ``codex login --with-api-key``, if any."""
    try:
        data = json.loads(_CODEX_CREDENTIALS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(data, dict) or data.get("auth_mode") != "apikey":
        return ""
    return str(data.get("OPENAI_API_KEY") or "").strip()


def _codex_file_has_chatgpt_login() -> bool:
    try:
        data = json.loads(_CODEX_CREDENTIALS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    tokens = data.get("tokens") if isinstance(data, dict) else None
    return bool(isinstance(tokens, dict) and tokens.get("access_token"))


def has_platform_credential() -> bool:
    """Whether a Platform API key resolves — the `auto` tool-mode predicate.

    Deliberately the same lookup resolve_auth() performs, minus the raising:
    routing must not send a request down the bridge only for it to 502 on the
    credential it just failed to find.
    """
    return bool(os.environ.get("OPENAI_API_KEY", "").strip() or _codex_file_api_key())


def resolve_auth() -> dict[str, str]:
    """Bearer header for api.openai.com. Precedence:

      1. OPENAI_API_KEY (environment) — .strip()'d; the compose env always
         carries the var, usually as "", which must fall through (mirrors the
         claude bridge's handling of ANTHROPIC_API_KEY).
      2. auth.json in API-key mode (``codex login --with-api-key`` writes
         {"auth_mode":"apikey","OPENAI_API_KEY":"sk-..."}), re-read per request
         via the codex-home read-only mount — which SHIPS COMMENTED OUT on the
         API container (see docker-compose.codex.yml): a ChatGPT-login
         deployment gains nothing from exposing auth.json's OAuth refresh
         token to the internet-facing container, so only operators who
         actually persist an API key to the volume opt in.

    ChatGPT-plan OAuth tokens (auth.json "tokens") are DELIBERATELY not used:
    they authenticate against chatgpt.com/backend-api/codex (the Codex
    backend), not the OpenAI Platform API — sending one to /v1/chat/completions
    is both non-functional and a plan-terms violation. There is no fallback: a
    ChatGPT-plan-only deployment gets a 502 naming the two working options.
    (This is the asymmetry vs the Claude bridge, whose plan login IS usable.)
    """
    key = os.environ.get("OPENAI_API_KEY", "").strip() or _codex_file_api_key()
    if key:
        return {"authorization": f"Bearer {key}"}
    message = (
        "function calling under CLAUDE_WRAPPER_AGENT=codex requires OpenAI "
        "Platform API access: set OPENAI_API_KEY, or run `codex login "
        "--with-api-key` and uncomment the codex-home mount on the "
        "claude-wrapper service; a ChatGPT-plan login cannot call the OpenAI API"
    )
    if _codex_file_has_chatgpt_login():
        message += (
            f" (a ChatGPT-plan login was found in {_CODEX_CREDENTIALS_FILE} "
            "but is not usable here)"
        )
    raise openai_error(502, message, err_type="api_error", code="no_upstream_credential")


# ---------- payload ----------


def _request_effort(req: ChatCompletionRequest) -> Optional[str]:
    """Effort suffix of the model this request actually RESOLVES to — the
    same computation main performs before building the envelope's `effort`
    claim. Deriving from req.model alone lied for "auto": with a suffixed
    default model the envelope claimed the effort was applied while the
    payload never carried it."""
    model = req.model if req.model and req.model != "auto" else SETTINGS.default_model
    _, effort = split_model_effort(model)
    return effort


def build_payload(
    req: ChatCompletionRequest, run_model: str, effort: Optional[str], stream: bool
) -> dict[str, Any]:
    """The outbound chat.completions payload: the request minus the wrapper's
    own extension fields, with model/stream/reasoning_effort overwritten."""
    payload = req.model_dump(exclude_none=True)
    for key in _WRAPPER_EXTENSION_KEYS:
        payload.pop(key, None)
    if req.legacy_functions_shape:
        # Legacy `functions` clients: models._adopt_legacy_functions synthesized
        # tools/tool_choice while leaving the legacy fields set, so a verbatim
        # dump would send BOTH parameter families upstream. Forward the legacy
        # pair as received instead — upstream then answers in the legacy shape
        # (message.function_call, delta.function_call, finish_reason
        # "function_call") for both stream and non-stream, which is why this
        # bridge needs no legacy down-conversion of its own.
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
    else:
        # Modern contract (including a migration-era client that sent BOTH
        # families): tools/tool_choice win, the legacy pair is the redundant
        # copy — upstream rejects requests carrying both.
        payload.pop("functions", None)
        payload.pop("function_call", None)
    payload["model"] = run_model
    payload["stream"] = stream
    if "-codex" in run_model and OPENAI_BASE_URL == "https://api.openai.com":
        # The codex-tuned ids are served only by the Responses API — the real
        # api.openai.com rejects them at /v1/chat/completions, so without this
        # gate every tools request on an advertised *-codex model dies as an
        # opaque upstream-404 passthrough. Fail loud with the fix instead.
        # Scoped to the default base URL on purpose: a custom
        # CLAUDE_WRAPPER_OPENAI_BASE_URL backend (vLLM, a proxy) may well
        # serve these ids on chat.completions, and that must keep working.
        raise openai_error(
            400,
            f"model '{run_model}' is a codex-tuned id, which OpenAI serves only "
            "via the Responses API — function-calling (tools) requests go "
            "through /v1/chat/completions and cannot use it. Pick a non-codex "
            "model for tools requests, or drop `tools` to run this model "
            "through the codex CLI",
            param="model",
            code="model_not_bridge_capable",
        )
    if effort and effort in CODEX_EFFORT_CHOICES:
        # Only the explicit per-request suffix maps. SETTINGS.effort (server
        # default, ships "medium") is NOT injected: non-reasoning OpenAI models
        # 400 on the parameter, and a server default would poison every bridge
        # call. Trade-off: CLI turns honor the server default, bridge turns
        # take the model's own default unless the client asks.
        payload["reasoning_effort"] = effort
    # The one non-proxy gate kept: capability parity with the Claude bridge.
    # req.tools is also set for legacy-`functions` requests (via the adopter),
    # so this covers both parameter families. Everything else — tool shapes,
    # tool_choice, response_format, parallel_tool_calls, non-`function` tool
    # types (OpenAI built-ins) — passes through verbatim; upstream is the
    # authoritative validator and its errors are already OpenAI-shaped.
    if req.tools and Capability.CLIENT_TOOLS not in resolve_profile(run_model):
        declared = ", ".join(
            sorted({t.function.name for t in req.tools if t.type == "function"})
        )
        raise openai_error(
            400,
            f"model '{run_model}' does not accept client-declared tools "
            f"(capability 'client_tools' is not in its profile); declared: {declared}",
            param="tools",
        )
    return payload


# ---------- non-streaming ----------


def _raise_for_status(resp: httpx.Response, body: str) -> None:
    """The status-mapping ladder for a non-200 upstream answer."""
    if resp.status_code in (401, 403):
        # Operator credential problem; body stays server-side (see the fixed
        # message's rationale above).
        log.error(
            "openai rejected the upstream credential (%s): %s", resp.status_code, body[:500]
        )
        raise openai_error(
            502, _CREDENTIAL_REJECTED_MSG, err_type="api_error", code="upstream_error"
        )
    if resp.status_code in _PASSTHROUGH_STATUSES:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            # Already the OpenAI envelope — main.py's handler returns it
            # verbatim, status and all.
            raise HTTPException(status_code=resp.status_code, detail=parsed)
        raise openai_error(resp.status_code, body[:500])
    # Everything else (5xx outages foremost): outage bodies carry no
    # credentials, so quoting them is safe and useful.
    raise openai_error(
        502,
        f"openai api error {resp.status_code}: {body[:500]}",
        err_type="api_error",
        code="upstream_error",
    )


async def complete(
    req: ChatCompletionRequest, run_model: str, session_key: str = ""
) -> BridgeResult:
    """Non-streaming tools request: one upstream call, no loop.

    Returning BridgeResult (not the raw body) is deliberate: main's existing
    envelope code then re-adds session_id, effort, the legacy functions→
    function_call down-conversion, and the explicit-null-content fix — full
    parity with the Claude bridge for free.
    """
    # The signature carries no effort (parity with the Claude bridge), so it is
    # re-derived here — the same computation main._tool_bridge_completion
    # performs, so the response envelope's `effort` claim and the payload agree.
    effort = _request_effort(req)
    payload = build_payload(req, run_model, effort, stream=False)
    headers = resolve_auth()
    client = _get_client()
    try:
        resp = await client.post(
            f"{OPENAI_BASE_URL}/v1/chat/completions", json=payload, headers=headers
        )
    except httpx.HTTPError as e:
        raise openai_error(
            502, f"openai api unreachable: {e}", err_type="api_error", code="upstream_error"
        )
    if resp.status_code != 200:
        _raise_for_status(resp, resp.text)

    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    # tool_calls pass through as plain dicts, verbatim — ids/names/arguments
    # untouched. A legacy-`functions` request comes back as
    # message.function_call instead; re-wrap it in the tool_calls shape so
    # main's legacy envelope path (which reads tool_calls[0]["function"])
    # renders it back down to the wire shape the client expects.
    tool_calls = message.get("tool_calls")
    if not tool_calls and message.get("function_call"):
        tool_calls = [{"id": "", "type": "function", "function": message["function_call"]}]
    usage = data.get("usage") or {}
    return BridgeResult(
        content=message.get("content"),
        tool_calls=tool_calls or None,
        finish_reason=choice.get("finish_reason") or "stop",
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
    )


# ---------- streaming ----------


async def stream(
    req: ChatCompletionRequest,
    run_model: str,
    model_label: str,
    session_key: str,
    effort_info: dict[str, Any],
    on_usage: Optional[Callable[[int, int], Awaitable[None]]] = None,
) -> AsyncIterator[bytes]:
    """Streaming tools request: upstream SSE forwarded verbatim.

    NO synthetic first chunk: upstream chunks carry their own id/model, and
    interleaving a wrapper-minted chunk with a different id ahead of them
    breaks strict clients. Consequence (documented in the README): under codex
    the tools-path stream does not carry the session_id/effort extension
    fields — model_label/effort_info are accepted only for signature parity
    with the Claude bridge.

    Two interceptions, both bookkeeping: usage is harvested from chunks that
    carry it (suppressing the chunk only when the wrapper injected the
    request for it), and upstream's [DONE] is swallowed so ours goes out
    after on_usage runs. finish_reason/tool-call deltas stream untouched;
    legacy-`functions` requests need no delta rewriting because build_payload
    forwards the legacy parameter family, so upstream already streams
    delta.function_call.
    """
    if _STREAM_PREAMBLE_BYTES > 0:
        yield b": " + b" " * _STREAM_PREAMBLE_BYTES + b"\n\n"

    input_tokens = output_tokens = 0
    errored: Optional[str] = None
    errored_type = "upstream_error"

    try:
        # Same re-derivation as complete() — the payload must carry the same
        # effort the envelope main built from the resolved model does.
        effort = _request_effort(req)
        payload = build_payload(req, run_model, effort, stream=True)
        # Usage capture: the ledger can only record what upstream reports, so
        # when the caller wired on_usage and the client didn't ask for the
        # usage chunk itself, ask on its behalf — and consume the synthetic
        # chunk instead of forwarding a frame the client never requested.
        suppress_usage_chunk = False
        if on_usage is not None and not (payload.get("stream_options") or {}).get(
            "include_usage"
        ):
            # Merged, not replaced: the client's other stream_options keys
            # (present or future — e.g. include_obfuscation) must survive the
            # wrapper's piggybacked usage ask.
            payload["stream_options"] = {
                **(payload.get("stream_options") or {}),
                "include_usage": True,
            }
            suppress_usage_chunk = True
        headers = resolve_auth()
        client = _get_client()
        async with client.stream(
            "POST", f"{OPENAI_BASE_URL}/v1/chat/completions", json=payload, headers=headers
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", errors="replace")
                if resp.status_code in (401, 403):
                    # Same fixed-message rule as complete(): the body may echo
                    # the operator's key, so it goes to the server log only.
                    log.error(
                        "openai rejected the upstream credential (%s): %s",
                        resp.status_code,
                        body[:500],
                    )
                    errored = _CREDENTIAL_REJECTED_MSG
                    errored_type = "api_error"
                else:
                    errored = f"openai api error {resp.status_code}: {body[:500]}"
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        # Swallowed: ours goes out after the usage bookkeeping.
                        break
                    # Cheap parse purely to harvest usage; the forwarded bytes
                    # are the upstream line, never a re-serialization.
                    try:
                        evt = json.loads(data)
                    except json.JSONDecodeError:
                        evt = None
                    if isinstance(evt, dict) and isinstance(evt.get("usage"), dict):
                        usage = evt["usage"]
                        input_tokens = int(usage.get("prompt_tokens") or 0)
                        output_tokens = int(usage.get("completion_tokens") or 0)
                        if suppress_usage_chunk and not evt.get("choices"):
                            continue  # the injected chunk: record, don't forward
                    yield (line + "\n\n").encode("utf-8")
    except HTTPException as e:
        # openai_error details carry the envelope; surface its message/type on
        # the SSE error channel instead of a stringified dict.
        if isinstance(e.detail, dict) and isinstance(e.detail.get("error"), dict):
            errored = str(e.detail["error"].get("message") or "request failed")
            errored_type = str(e.detail["error"].get("type") or errored_type)
        else:
            errored = str(e.detail)
    except httpx.HTTPError as e:
        errored = f"openai api stream failed: {e}"
    except Exception as e:  # pragma: no cover - defensive
        log.exception("openai-bridge stream failed (session=%s)", session_key)
        errored = f"internal wrapper error: {e}"

    if on_usage is not None and (input_tokens or output_tokens):
        try:
            await on_usage(input_tokens, output_tokens)
        except Exception:  # pragma: no cover
            log.exception("usage recording failed (session=%s)", session_key)

    if errored:
        err_payload = {"error": {"message": errored, "type": errored_type}}
        yield f"data: {json.dumps(err_payload)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"
