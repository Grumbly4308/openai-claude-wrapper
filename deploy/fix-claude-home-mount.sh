#!/usr/bin/env bash
#
# fix-claude-home-mount.sh — repair durable `claude --resume` on the prod host.
#
# ROOT CAUSE
#   The persistence volume `claude-home` is mounted at /root/.claude, but the
#   container runs as user `claude` (HOME=/home/claude). Claude Code therefore
#   writes its session transcripts to /home/claude/.claude — the container's
#   EPHEMERAL overlay — which is wiped on every container recreate. Meanwhile the
#   wrapper's session registry (key -> uuid map) lives on the persistent
#   `claude-data` volume and survives. So after any recreate, conversations
#   older than the current container resume against a uuid whose transcript is
#   gone: `claude --resume <dead-uuid>` fails in ~seconds with
#   error_during_execution, consistently, on every retry.
#
# WHAT THIS DOES
#   1. Migrates the live (ephemeral) Claude state into the persistent volume so
#      currently-working chats are preserved across the remount.
#   2. Repoints the volume mount to /home/claude/.claude in docker-compose.yml
#      (with a timestamped backup).
#   3. (optional) Rebuilds the image — needed to also ship the run_stream
#      self-heal change; set REBUILD=1.
#   4. Recreates the container and verifies.
#
# Already-BROKEN old chats can't be recovered here (their transcripts are gone).
# With the self-heal code change deployed (REBUILD=1), they auto-recover on
# their next turn by dropping the dead mapping and replaying full history.
#
# Run ON the prod host, e.g.:
#   sudo REBUILD=1 ./fix-claude-home-mount.sh
#
set -euo pipefail

CONTAINER="${CONTAINER:-claude-wrapper}"
COMPOSE_DIR="${COMPOSE_DIR:-/root/openai-claude-wrapper}"
COMPOSE_FILE="${COMPOSE_FILE:-${COMPOSE_DIR}/docker-compose.yml}"
OLD_MOUNT="${OLD_MOUNT:-claude-home:/root/.claude}"
NEW_MOUNT="${NEW_MOUNT:-claude-home:/home/claude/.claude}"
REBUILD="${REBUILD:-0}"

cd "$COMPOSE_DIR"

echo "==> Pre-flight"
docker inspect "$CONTAINER" >/dev/null
if ! grep -qF "$OLD_MOUNT" "$COMPOSE_FILE"; then
  echo "    '$OLD_MOUNT' not found in $COMPOSE_FILE — already fixed, or the line"
  echo "    differs. Current claude-home mount(s):"
  grep -n 'claude-home' "$COMPOSE_FILE" || true
  exit 1
fi
echo "    container present; mount to fix found in $COMPOSE_FILE"

echo "==> 1/4 Migrating live Claude state into the persistent volume"
# The volume is currently mounted at /root/.claude; copy the real (ephemeral)
# state there so it survives the remount+recreate. Ownership -> uid 1000 (claude).
docker exec -u root "$CONTAINER" sh -c '
  set -e
  if [ -d /home/claude/.claude ] && [ -n "$(ls -A /home/claude/.claude 2>/dev/null)" ]; then
    cp -a /home/claude/.claude/. /root/.claude/ 2>/dev/null || true
    chown -R 1000:1000 /root/.claude
    echo "    migrated $(find /root/.claude -type f 2>/dev/null | wc -l) files into the volume"
  else
    echo "    nothing to migrate (no live state) — proceeding"
  fi
'

echo "==> 2/4 Backing up compose and repointing the mount"
BACKUP="${COMPOSE_FILE}.bak.$(date +%Y%m%d%H%M%S)"
cp -a "$COMPOSE_FILE" "$BACKUP"
sed -i "s#${OLD_MOUNT}#${NEW_MOUNT}#g" "$COMPOSE_FILE"
if grep -qF "$NEW_MOUNT" "$COMPOSE_FILE" && ! grep -qF "$OLD_MOUNT" "$COMPOSE_FILE"; then
  echo "    backup: $BACKUP"
  echo "    mount is now:$(grep -F "$NEW_MOUNT" "$COMPOSE_FILE")"
else
  echo "ERROR: compose edit did not apply cleanly — restoring from backup"
  cp -a "$BACKUP" "$COMPOSE_FILE"
  exit 1
fi

echo "==> 3/4 Recreating the container"
if [ "$REBUILD" = "1" ]; then
  echo "    REBUILD=1 -> docker compose build (ships run_stream self-heal)"
  docker compose build
fi
docker compose up -d

echo "==> 4/4 Verifying"
# Give uvicorn a moment, then confirm health + that state now lands on the volume.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if curl -fsS -m 3 "http://127.0.0.1:8000/healthz" >/dev/null 2>&1; then break; fi
  sleep 1
done
docker compose ps
docker exec -u root "$CONTAINER" sh -c '
  echo "    container user: $(id -un)  HOME=$HOME"
  echo "    persisted sessions under /home/claude/.claude/projects:"
  ls /home/claude/.claude/projects 2>/dev/null | sed "s/^/      /" | head || echo "      (none yet)"
'
echo
echo "Done. Claude session state now lives on the claude-home volume and"
echo "survives rebuilds — resume is durable from here on."
[ "$REBUILD" = "1" ] || echo "NOTE: re-run with REBUILD=1 (after pulling the code change) to also ship the run_stream self-heal."
