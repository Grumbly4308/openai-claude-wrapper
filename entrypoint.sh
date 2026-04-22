#!/usr/bin/env bash
set -euo pipefail

CLAUDE_HOME="${HOME:-/home/claude}/.claude"
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

cmd_serve() {
    if ! has_saved_login && ! has_env_auth; then
        cat >&2 <<'MSG'
================================================================
  claude-wrapper: no saved Claude Code login found and no
  ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN env var set.

  Run the interactive login once to persist credentials to the
  mounted volume:

      docker compose run --rm -it claude-wrapper login

  …or for a long-lived headless token:

      docker compose run --rm -it claude-wrapper setup-token

  Then `docker compose up -d` as normal.

  (API requests will fail until one of the above is done.)
================================================================
MSG
    fi
    exec uvicorn src.main:app \
        --host "${CLAUDE_WRAPPER_HOST}" \
        --port "${CLAUDE_WRAPPER_PORT}" \
        --workers "${CLAUDE_WRAPPER_WORKERS:-1}" \
        --proxy-headers \
        --forwarded-allow-ips='*'
}

cmd_login() {
    echo "launching interactive Claude Code login..." >&2
    echo "credentials will be written to ${CLAUDE_HOME} (persisted on the claude-home volume)." >&2
    echo >&2
    # Try the dedicated subcommand first; fall back to TUI /login.
    if claude login --help >/dev/null 2>&1; then
        exec claude login
    fi
    if claude /login --help >/dev/null 2>&1; then
        exec claude /login
    fi
    echo "no 'claude login' subcommand detected — opening the interactive TUI." >&2
    echo "type /login at the prompt, complete the OAuth flow, then /exit." >&2
    exec claude
}

cmd_setup_token() {
    echo "launching 'claude setup-token' — follow the prompts to generate" >&2
    echo "a long-lived OAuth token. It will be saved to ${CLAUDE_HOME}." >&2
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
