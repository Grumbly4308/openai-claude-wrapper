#!/bin/bash
# Does the Claude Code CLI's token refresh actually work in this deployment?
#
# The question matters because an interactive login only sustains itself if the
# CLI can renew its access token, and under the sandboxed topology the CLI's
# only egress is an allowlisting proxy. A deployment where the refresh silently
# fails looks completely healthy until the token lapses, at which point every
# turn fails with `claude exited 1:` and an empty stderr.
#
# Waiting for a real expiry to find out takes hours, and editing `expiresAt` to
# fake one proves nothing — that field is local metadata the API never sees, so
# the old token keeps working and no refresh is attempted. This forces the
# issue properly: it corrupts the access token so the API returns 401, leaving
# the refresh token intact, and then makes one real request. If the CLI can
# refresh, the request succeeds and the credential file is rewritten.
#
# The original credential is backed up first and the refresh token is never
# touched, so this cannot lock you out. Restore is printed at the end.
#
#   tools/credential_refresh_test.sh
#   RUNTIME=docker CONTAINER=claude-wrapper tools/credential_refresh_test.sh
#
# CONTAINER defaults to the sandboxed topology's agent, which is where the CLI
# runs and where claude-home is mounted writable. For the single-container
# layout, point it at claude-wrapper.
set -u

RUNTIME=${RUNTIME:-podman}
CONTAINER=${CONTAINER:-claude-agent}
BASE_URL=${BASE_URL:-http://localhost:8000}
MODEL=${MODEL:-claude-opus-4-8}
CRED=${CRED:-/home/claude/.claude/.credentials.json}
BACKUP=${BACKUP:-$HOME/creds-backup-refreshtest.json}

show() {
"$RUNTIME" exec -i "$CONTAINER" python3 - "$CRED" <<'PY'
import json, os, sys, time
p = sys.argv[1]
try:
    d = json.load(open(p))['claudeAiOauth']
except Exception as e:
    raise SystemExit('  unreadable: %s' % e)
n = time.time()
print('  mtime   :', time.ctime(os.path.getmtime(p)))
print('  access  : %.2f h' % ((d['expiresAt'] / 1000 - n) / 3600))
if isinstance(d.get('refreshTokenExpiresAt'), (int, float)):
    print('  refresh : %.1f d' % ((d['refreshTokenExpiresAt'] / 1000 - n) / 86400))
else:
    print('  refresh : none recorded — this credential cannot renew itself')
PY
}

# An env credential wins over the file and stops the CLI renewing it, so the
# test would measure nothing. Refuse rather than report a false negative.
in_force=$("$RUNTIME" exec "$CONTAINER" sh -c 'echo "${ANTHROPIC_API_KEY:-}${CLAUDE_CODE_OAUTH_TOKEN:-}"' 2>/dev/null)
if [ -n "$in_force" ]; then
    echo "ABORT: $CONTAINER has ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN set."
    echo "An environment credential overrides the file, and the CLI then never"
    echo "refreshes it — this test can only produce a false negative. Blank it"
    echo "and recreate the container first."
    exit 1
fi

echo "== backup =="
"$RUNTIME" exec "$CONTAINER" cat "$CRED" > "$BACKUP" || { echo "backup FAILED — aborting"; exit 1; }
chmod 600 "$BACKUP"
echo "  saved to $BACKUP"

echo "== before =="
show || exit 1

echo "== invalidating the access token (refresh token left intact) =="
"$RUNTIME" exec -i "$CONTAINER" python3 - "$CRED" <<'PY'
import json, sys, time
p = sys.argv[1]
d = json.load(open(p))
o = d['claudeAiOauth']
o['accessToken'] = o['accessToken'] + 'INVALID'
o['expiresAt'] = int((time.time() - 60) * 1000)
json.dump(d, open(p, 'w'))
print('  done')
PY

# Captured AFTER the edit: the edit moves mtime itself, so this is the baseline
# a genuine refresh has to move again.
before=$("$RUNTIME" exec "$CONTAINER" stat -c %y "$CRED")
sleep 1

echo "== one turn =="
resp=$(curl -s "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}")
echo "$resp" | head -c 400; echo

echo "== after =="
show
after=$("$RUNTIME" exec "$CONTAINER" stat -c %y "$CRED")

echo "== verdict =="
if echo "$resp" | grep -q '"choices"'; then
    if [ "$before" != "$after" ]; then
        echo "  REFRESH WORKS — the turn succeeded and the credential was rewritten."
        echo "  An interactive login sustains itself here; no long-lived token needed."
    else
        echo "  Turn succeeded but the credential was NOT rewritten, which should not"
        echo "  happen with an invalid access token. Check whether something else is"
        echo "  answering (the tool-bridge path does not spawn the CLI)."
    fi
else
    echo "  NO REFRESH — the turn failed, so the CLI could not renew the token"
    echo "  from inside this deployment. The refresh client itself is proven"
    echo "  proxy-capable (CREDENTIALS-FIX.md, Round 3), so check Squid's log"
    echo "  for CONNECT platform.claude.com — TUNNEL/200 means the server"
    echo "  rejected this refresh token (re-login and rerun); TCP_DENIED means"
    echo "  the running squid is stale; no line means the CONNECT never left"
    echo "  the agent. The claude-refresher service renews from outside the"
    echo "  sandbox either way; CLAUDE_CODE_OAUTH_TOKEN is the static fallback."
fi

echo
echo "restore the real credential with:"
echo "  $RUNTIME exec -i $CONTAINER sh -c 'cat > $CRED' < $BACKUP"
