#!/usr/bin/env bash
set -euo pipefail

CLAUDE_HOME="${CLAUDE_CONFIG_DIR:-${HOME:-/home/claude}/.claude}"
CODEX_HOME_DIR="${CODEX_HOME:-${HOME:-/home/claude}/.codex}"
mkdir -p "${CLAUDE_WRAPPER_WORKSPACE}" "${CLAUDE_WRAPPER_FILES}" "${CLAUDE_WRAPPER_SESSIONS}" "${CLAUDE_HOME}" "${CODEX_HOME_DIR}"

agent_kind() { echo "${CLAUDE_WRAPPER_AGENT:-claude}"; }

has_saved_login() {
    # Claude Code stores credentials in ~/.claude/ — but presence alone is not
    # enough: the file survives its own expiry, and once the refresh window is
    # past, re-login is needed regardless of what is on disk. Treating a dead
    # file as "logged in" is how the no-auth warning never fired on a
    # deployment that was already failing every turn.
    if [[ -f "${CLAUDE_HOME}/.credentials.json" ]]; then
        python3 - "${CLAUDE_HOME}/.credentials.json" <<'PY' 2>/dev/null
import json, sys, time
try:
    o = json.load(open(sys.argv[1])).get('claudeAiOauth') or {}
except Exception:
    raise SystemExit(1)
if not o.get('accessToken'):
    raise SystemExit(1)
# The refresh window decides viability when recorded (an expired access token
# with live refresh is renewable); a file with no readable expiry counts as
# viable rather than dead — opaque is not the same as expired.
exp = o.get('refreshTokenExpiresAt') or o.get('expiresAt')
if isinstance(exp, (int, float)) and exp / 1000 <= time.time():
    raise SystemExit(1)
PY
        return
    fi
    [[ -f "${CLAUDE_HOME}/credentials.json" ]] \
        || [[ -f "${CLAUDE_HOME}/auth.json" ]] \
        || [[ -d "${CLAUDE_HOME}/projects" && -n "$(ls -A "${CLAUDE_HOME}/projects" 2>/dev/null || true)" ]]
}

has_env_auth() {
    [[ -n "${ANTHROPIC_API_KEY:-}" ]] || [[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]
}

has_codex_saved_login() {
    # Codex keeps auth in ${CODEX_HOME}/auth.json — either an API key
    # (auth_mode=apikey) or ChatGPT-plan OAuth tokens. Unlike the claude file,
    # token expiry is not recorded readably, so presence of tokens counts as
    # viable — opaque is not the same as expired (same philosophy as the
    # claude helper above).
    [[ -f "${CODEX_HOME_DIR}/auth.json" ]] || return 1
    python3 - "${CODEX_HOME_DIR}/auth.json" <<'PY' 2>/dev/null
import json, sys
try:
    o = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(1)
if o.get('auth_mode') == 'apikey' and o.get('OPENAI_API_KEY'):
    raise SystemExit(0)
if (o.get('tokens') or {}).get('access_token'):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

has_codex_env_auth() {
    [[ -n "${OPENAI_API_KEY:-}" ]] || [[ -n "${CODEX_API_KEY:-}" ]]
}

warn_if_no_codex_auth() {
    if ! has_codex_saved_login && ! has_codex_env_auth; then
        cat >&2 <<'MSG'
================================================================
  claude-wrapper: no usable Codex credential. No OPENAI_API_KEY is
  set and the volume login is missing.

  Bootstrap a ChatGPT-plan login into the codex-home volume from a
  container with ordinary networking (README "First-time login (Codex)"):

      docker compose -f docker-compose.codex.yml run --rm -it codex-refresher codex-login
      # follow the device-code flow in your browser

  …or use API-key auth instead: set OPENAI_API_KEY in .env, or persist
  it to the volume:

      printenv OPENAI_API_KEY | docker compose -f docker-compose.codex.yml \
          run --rm -T codex-refresher codex login --with-api-key

  (API requests will fail until one of the above is done.)
================================================================
MSG
    fi
}

warn_if_no_auth() {
    # Dispatch on the wrapped agent: each stack warns about its own vendor's
    # credential, not the other one's.
    if [[ "$(agent_kind)" == codex ]]; then
        warn_if_no_codex_auth
        return
    fi
    if ! has_saved_login && ! has_env_auth; then
        cat >&2 <<'MSG'
================================================================
  claude-wrapper: no usable Claude Code credential. No
  ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN env var is set,
  and the volume login is missing or past its refresh window.

  Bootstrap a login into the volume from a container with
  ordinary networking (see README "First-time login"):

      docker compose run --rm -it claude-wrapper claude
      # type /login at the prompt, complete the flow, /exit

  After that the CLI renews the login itself whenever it runs,
  and the claude-refresher service covers idle stretches — no
  further action needed.

  …or mint a ~1-year token and set it in .env (it is PRINTED,
  not saved to the volume; also record the date so the boot
  report can warn before it dies):

      docker compose run --rm -it claude-wrapper setup-token
      # CLAUDE_CODE_OAUTH_TOKEN=<token>
      # CLAUDE_CODE_OAUTH_TOKEN_MINTED=<today, YYYY-MM-DD>

  Then `docker compose up -d` as normal.

  (API requests will fail until one of the above is done.)
================================================================
MSG
    fi
}

cmd_serve() {
    warn_if_no_auth
    exec uvicorn src.main:app \
        --host "${CLAUDE_WRAPPER_HOST}" \
        --port "${CLAUDE_WRAPPER_PORT}" \
        --workers "${CLAUDE_WRAPPER_WORKERS:-1}" \
        --proxy-headers \
        --forwarded-allow-ips='*'
}

cmd_agent() {
    # Sandboxed topology: the agent shim (src/agent_shim.py) is this
    # container's only inbound surface; the API container sends it runs via
    # CLAUDE_WRAPPER_AGENT_URL. The CLI executes HERE, so credentials must
    # live here too — the no-auth warning applies to this role, not serve.
    warn_if_no_auth
    exec uvicorn src.agent_shim:app \
        --host "${CLAUDE_WRAPPER_HOST}" \
        --port "${CLAUDE_WRAPPER_AGENT_PORT:-8791}"
}

cmd_refresher() {
    # Keeps the volume login alive regardless of traffic. Renewal happens only
    # when the CLI runs, so an idle deployment drifts past expiry no matter
    # how healthy its network path is (CREDENTIALS-FIX.md, Round 3: the
    # refresh client itself is proxy-capable). This role runs the same CLI
    # against the same claude-home volume from a container with ordinary
    # networking — the position the first-time /login already works from —
    # and spends one minimal turn whenever the access token nears expiry,
    # which makes the CLI rewrite .credentials.json for every reader of the
    # volume.
    #
    # An environment credential would make the CLI ignore the file entirely,
    # so this loop would burn turns renewing nothing. Refuse rather than
    # silently measure the wrong thing.
    if [[ -n "${ANTHROPIC_API_KEY:-}" || -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]; then
        echo "refresher: ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN is set — the" >&2
        echo "CLI ignores the volume login while one is present, so this loop" >&2
        echo "cannot renew anything. Compose pins both empty for this service;" >&2
        echo "find what overrode that." >&2
        exit 1
    fi

    local check="${CLAUDE_REFRESH_CHECK_SECONDS:-900}"
    local below="${CLAUDE_REFRESH_BELOW_SECONDS:-14400}"
    local retry="${CLAUDE_REFRESH_RETRY_SECONDS:-3600}"

    # Prints "<access-remaining-s> <refresh-remaining-s|none>", or "none" when
    # there is no readable login at all.
    cred_expiry() {
        python3 - "${CLAUDE_HOME}/.credentials.json" <<'PY' 2>/dev/null || echo none
import json, sys, time
o = json.load(open(sys.argv[1]))['claudeAiOauth']
n = time.time()
def rem(k):
    v = o.get(k)
    return str(int(v / 1000 - n)) if isinstance(v, (int, float)) else 'none'
print(rem('expiresAt'), rem('refreshTokenExpiresAt'))
PY
    }

    echo "refresher: watching ${CLAUDE_HOME}/.credentials.json" >&2
    echo "refresher: check every ${check}s, renew below ${below}s, retry after ${retry}s" >&2
    while :; do
        local access refresh
        read -r access refresh <<<"$(cred_expiry)"
        if [[ "${access}" == "none" ]]; then
            echo "refresher: no readable login in the volume — bootstrap one" >&2
            echo "refresher: (README 'First-time login'); checking again in ${check}s" >&2
            sleep "${check}"; continue
        fi
        # Both expiries logged every pass on purpose: whether refreshTokenExpiresAt
        # rolls forward on renewal is an open question this log answers.
        echo "refresher: access $((access / 60))m, refresh $([[ "${refresh}" == none ]] && echo none || echo "$((refresh / 86400))d ($((refresh / 60))m)")" >&2
        if (( access < below )); then
            echo "refresher: inside the renewal window — spending one CLI turn" >&2
            claude -p ok --output-format json >/dev/null 2>&1 || true
            local after after_refresh
            read -r after after_refresh <<<"$(cred_expiry)"
            if [[ "${after}" != "none" ]] && (( after > access )); then
                echo "refresher: renewed — access $((after / 60))m, refresh $([[ "${after_refresh}" == none ]] && echo none || echo "$((after_refresh / 86400))d")" >&2
            elif (( access > 0 )); then
                # The CLI may decline to refresh a still-valid token; back off
                # rather than burning a turn every pass until it does.
                echo "refresher: turn ran but the token did not renew; retrying in ${retry}s" >&2
                sleep "${retry}"; continue
            else
                # Expired is an outage — every agent turn is failing — so keep
                # the short cadence and close it as fast as possible.
                echo "refresher: token EXPIRED and the turn did not renew it; retrying in ${check}s" >&2
            fi
        fi
        sleep "${check}"
    done
}

cmd_login() {
    echo "launching interactive Claude Code login..." >&2
    echo "credentials will be written to ${CLAUDE_HOME} (persisted on the claude-home volume)." >&2
    echo >&2
    # Do NOT probe with `claude login --help`. On a CLI with no `login`
    # subcommand — 2.1.226 has none — the word is taken as a prompt, --help
    # still exits 0, and the probe passes; `exec claude login` then opens a
    # chat session that writes history and sessions but never a credential,
    # so the command silently "logs in" by starting a conversation. Ask the
    # help output whether the subcommand exists instead.
    if claude --help 2>/dev/null | grep -qE '^[[:space:]]*login([[:space:]]|$)'; then
        exec claude login
    fi
    echo "this CLI has no 'login' subcommand — opening the interactive TUI." >&2
    echo "type /login at the prompt, complete the OAuth flow, then /exit." >&2
    exec claude
}

cmd_codex_refresher() {
    # Keeps the codex volume login alive regardless of traffic. Codex renews
    # its ChatGPT tokens only when it runs, and the refresh token's lifetime
    # is finite (~28 days observed historically — an estimate, not a
    # published figure), so an idle deployment drifts past it no matter how
    # healthy its network path is. This role runs the same CLI against the
    # same codex-home volume from a container with ordinary networking — the
    # position the device-code login bootstrap already works from — and
    # spends one throwaway turn whenever last_refresh goes stale, which makes
    # codex rewrite auth.json for every reader of the volume.
    #
    # An environment credential would make codex ignore the file entirely,
    # so this loop would burn turns renewing nothing. Refuse rather than
    # silently measure the wrong thing.
    if [[ -n "${OPENAI_API_KEY:-}" || -n "${CODEX_API_KEY:-}" ]]; then
        echo "codex-refresher: OPENAI_API_KEY / CODEX_API_KEY is set — codex" >&2
        echo "ignores the volume login while one is present, so this loop" >&2
        echo "cannot renew anything. Compose pins both empty for this service;" >&2
        echo "find what overrode that." >&2
        exit 1
    fi

    local check="${CODEX_REFRESH_CHECK_SECONDS:-3600}"
    local stale="${CODEX_REFRESH_STALE_SECONDS:-86400}"
    local retry="${CODEX_REFRESH_RETRY_SECONDS:-3600}"

    # Prints "none" (no readable login), "apikey", or
    # "tokens <seconds-since-last_refresh|none>".
    cred_state() {
        python3 - "${CODEX_HOME_DIR}/auth.json" <<'PY' 2>/dev/null || echo none
import datetime, json, sys, time
try:
    o = json.load(open(sys.argv[1]))
except Exception:
    print('none'); raise SystemExit
if o.get('auth_mode') == 'apikey' and o.get('OPENAI_API_KEY'):
    print('apikey'); raise SystemExit
if not (o.get('tokens') or {}).get('access_token'):
    print('none'); raise SystemExit
try:
    lr = datetime.datetime.fromisoformat(str(o['last_refresh']).replace('Z', '+00:00'))
    print('tokens', int(time.time() - lr.timestamp()))
except Exception:
    print('tokens', 'none')
PY
    }

    echo "codex-refresher: watching ${CODEX_HOME_DIR}/auth.json" >&2
    echo "codex-refresher: check every ${check}s, refresh past ${stale}s, retry after ${retry}s" >&2
    while :; do
        local kind age
        read -r kind age <<<"$(cred_state)"
        if [[ "${kind}" == "none" ]]; then
            echo "codex-refresher: no readable login in the volume — bootstrap one" >&2
            echo "codex-refresher: (README 'First-time login (Codex)'); checking again in ${check}s" >&2
            sleep "${check}"; continue
        fi
        if [[ "${kind}" == "apikey" ]]; then
            echo "codex-refresher: API-key auth needs no refresh; sleeping $((check * 24))s" >&2
            sleep "$((check * 24))"; continue
        fi
        # The age is logged every pass on purpose: how often codex actually
        # rolls last_refresh forward is an open question this log answers.
        echo "codex-refresher: last_refresh $([[ "${age}" == none ]] && echo unreadable || echo "$((age / 3600))h ($((age / 60))m) ago")" >&2
        if [[ "${age}" == "none" ]] || (( age > stale )); then
            echo "codex-refresher: stale — spending one CLI turn" >&2
            # Sandboxed on purpose: this container has ordinary egress and no
            # container-level sandbox (the documented refresher posture is
            # "runs no agent code, executes no model-driven tool use" —
            # docker-compose.yml claude-refresher header). Token refresh needs
            # auth traffic, not tool use, so the model turn runs read-only
            # with approvals left at their exec defaults. The bypass flag
            # exists ONLY in CodexRunner._build_argv, where the
            # network-isolated agent container is the boundary.
            printf 'ok' | codex exec --json --skip-git-repo-check \
                --sandbox read-only --ephemeral - >/dev/null 2>&1 || true
            # (--ephemeral so refresher turns never enter the resumable
            # thread store; codex refreshes stale tokens on use as a side
            # effect.)
            local kind_after age_after
            read -r kind_after age_after <<<"$(cred_state)"
            if [[ "${kind_after}" == "tokens" && "${age_after}" != "none" ]] \
                && { [[ "${age}" == "none" ]] || (( age_after < age )); }; then
                echo "codex-refresher: renewed — last_refresh $((age_after / 60))m ago" >&2
            else
                # Codex may decline to refresh a still-valid token; back off
                # rather than burning a turn every pass until it does.
                echo "codex-refresher: turn ran but the token did not renew; retrying in ${retry}s" >&2
                sleep "${retry}"; continue
            fi
        fi
        sleep "${check}"
    done
}

cmd_codex_login() {
    echo "launching codex device-code login..." >&2
    echo "credentials will be written to ${CODEX_HOME_DIR}/auth.json (codex-home volume)." >&2
    # Device-auth needs no localhost callback, so it works from `compose run`
    # (the default browser flow's localhost:1455 callback server is
    # unreachable in a run container unless you publish the port:
    #   docker compose ... run --rm -it -p 1455:1455 codex-refresher codex login).
    exec codex login --device-auth "$@"
}

cmd_setup_token() {
    echo "launching 'claude setup-token' — follow the prompts to generate" >&2
    echo "a long-lived OAuth token. It is PRINTED, not saved: set it as" >&2
    echo "CLAUDE_CODE_OAUTH_TOKEN. Note that doing so stops the CLI from" >&2
    echo "refreshing any login in ${CLAUDE_HOME}, which then decays." >&2
    echo >&2
    exec claude setup-token "$@"
}

cmd_shell() {
    exec /bin/bash "$@"
}

cmd_claude() {
    exec claude "$@"
}

case "${1:-serve}" in
    ""|serve|start|run)
        shift || true
        cmd_serve
        ;;
    agent|shim)
        shift
        cmd_agent
        ;;
    refresher|refresh)
        shift
        cmd_refresher
        ;;
    login|init)
        shift
        cmd_login "$@"
        ;;
    setup-token|token)
        shift
        cmd_setup_token "$@"
        ;;
    shell|bash)
        shift
        cmd_shell "$@"
        ;;
    claude)
        shift
        cmd_claude "$@"
        ;;
    codex-refresher)
        shift
        cmd_codex_refresher
        ;;
    codex-login)
        shift
        cmd_codex_login "$@"
        ;;
    codex)
        shift
        exec codex "$@"
        ;;
    *)
        # Unknown command — treat as a raw exec so advanced users can run
        # arbitrary binaries inside the container. Checked first so a role
        # name this image doesn't know yet fails with a diagnosis instead of
        # a bare `exec: not found` crash-loop: that is what a stale image
        # looks like when compose comes from a newer checkout.
        if ! command -v "$1" >/dev/null 2>&1; then
            echo "entrypoint: '$1' is neither a role this image knows nor an" >&2
            echo "executable on PATH. If it is a role from a newer checkout," >&2
            echo "this IMAGE is stale — rebuild it:" >&2
            echo "    docker compose -f <compose file> build --no-cache && docker compose -f <compose file> up -d --force-recreate" >&2
            exit 127
        fi
        exec "$@"
        ;;
esac
