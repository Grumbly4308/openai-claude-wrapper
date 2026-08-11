#!/usr/bin/env bash
set -euo pipefail

CLAUDE_HOME="${CLAUDE_CONFIG_DIR:-${HOME:-/home/claude}/.claude}"
mkdir -p "${CLAUDE_WRAPPER_WORKSPACE}" "${CLAUDE_WRAPPER_FILES}" "${CLAUDE_WRAPPER_SESSIONS}" "${CLAUDE_HOME}"

has_saved_login() {
    # Claude Code stores credentials in ~/.claude/ — presence of any of
    # these files means a previous login is persisted.
    [[ -f "${CLAUDE_HOME}/.credentials.json" ]] \
        || [[ -f "${CLAUDE_HOME}/credentials.json" ]] \
        || [[ -f "${CLAUDE_HOME}/auth.json" ]] \
        || [[ -d "${CLAUDE_HOME}/projects" && -n "$(ls -A "${CLAUDE_HOME}/projects" 2>/dev/null || true)" ]]
}

has_env_auth() {
    [[ -n "${ANTHROPIC_API_KEY:-}" ]] || [[ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]]
}

warn_if_no_auth() {
    if ! has_saved_login && ! has_env_auth; then
        cat >&2 <<'MSG'
================================================================
  claude-wrapper: no saved Claude Code login found and no
  ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN env var set.

  Mint a long-lived token once — it persists to the mounted
  volume and does not depend on the CLI refreshing it:

      docker compose run --rm -it claude-wrapper setup-token

  …or, for a desktop-style interactive login (shorter-lived, and
  its OAuth callback cannot complete inside the sandbox topology):

      docker compose run --rm -it claude-wrapper login

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
    *)
        # Unknown command — treat as a raw exec so advanced users can run
        # arbitrary binaries inside the container.
        exec "$@"
        ;;
esac
