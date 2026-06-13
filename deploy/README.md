# Prod fix: durable `claude --resume`

Symptom: Open WebUI shows `TransferEncodingError: Not enough data to satisfy
transfer length header` on **older** conversations — fast (a few seconds) and
consistent for that chat, while new chats work.

## Root cause

The container runs as user `claude` (`HOME=/home/claude`), so Claude Code stores
its session transcripts under `/home/claude/.claude/projects/`. But prod's
`docker-compose.yml` mounts the persistence volume at the **wrong path**:

```yaml
# running container (from `docker inspect`):
- claude-home:/root/.claude                                   # ← volume here; claude can't even write /root
- /root/.claude/.credentials.json:/home/claude/.claude/.credentials.json   # only the login file lands correctly
```

So `claude-home` is effectively unused. Claude's real state lives in the
**ephemeral** `/home/claude/.claude`, which is wiped whenever the container is
recreated. The wrapper's session registry (`session_key → uuid`) lives on the
*persistent* `claude-data` volume (`/data/sessions`) and survives. Result: after
any recreate, an old chat resumes a uuid whose transcript is gone →
`claude --resume <dead-uuid>` fails with `error_during_execution` in seconds,
every retry, forever (the failure string isn't "session not found", so the old
self-heal path didn't fire). New chats work because their transcripts still
exist in the *current* ephemeral layer.

Confirmed via `docker inspect`: `OOMKilled: false`, `RestartCount: 0` (not a
crash), volume mounted at `/root/.claude`, `User: claude`.

## Fix (two parts — apply both)

### a. Correct the mount  →  `fix-claude-home-mount.sh`

Repoints the volume to `/home/claude/.claude` and recreates the container. It
first migrates the live (ephemeral) state into the volume so currently-working
chats survive the cutover.

```bash
# on the prod host (10.160.2.11):
sudo REBUILD=1 deploy/fix-claude-home-mount.sh
```

The one-line change it makes to `docker-compose.yml`:

```diff
-      - claude-home:/root/.claude
+      - claude-home:/home/claude/.claude
```

After this, Claude session state persists across rebuilds and resume is durable.
Already-broken old chats are **not** recoverable (their transcripts are gone) —
but part (b) makes them self-recover on the next turn.

### b. Self-heal a broken resume (code change in `src/claude_runner.py`)

`run_stream` now drops the `key → uuid` mapping whenever a `--resume` turn fails
without producing any assistant text (non-zero exit *or* an error result subtype
like `error_during_execution`). The next turn then mints a fresh uuid, switches
to `--session-id`, and replays the full transcript — so a missing/dead session
recovers gracefully instead of bricking the chat. A resume that streamed a real
answer before a late error is left intact. Covered by
`tests/test_resume_selfheal.py`.

Ship it by rebuilding the image (`REBUILD=1` above, after the code is pulled
onto prod).
