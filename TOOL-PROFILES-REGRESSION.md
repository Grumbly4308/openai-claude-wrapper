# Per-model tool profiles — manual regression pass

Manual acceptance run for the `feat/per-model-tool-profiles` branch against a
live OpenWebUI. Work top to bottom — groups build on each other. Prompts in
quotes go into OpenWebUI chat; commands run where indicated. Record failures
inline; RT-8's raw output is needed verbatim to close the open checklist items
in [TOOL-PROFILES-TASKS.md](TOOL-PROFILES-TASKS.md).

Throughout: `WRAPPER=http://<dev-server>:8000` (adjust host/port).

---

## Setup — deploy the branch on the dev server

Credentials live in the `claude-home` Docker volume (or your
`docker-compose.host-credentials.yml` overlay / `.env` auth vars). None of
that is touched by a branch switch or rebuild — **no re-login needed**. Your
existing `.env` is untracked and survives too.

```bash
cd /path/to/claude-wrapper          # the existing dev checkout
docker compose ps                    # note current state
git fetch origin
git checkout feat/per-model-tool-profiles
git pull --ff-only origin feat/per-model-tool-profiles

# Rebuild + restart. Add `-f docker-compose.host-credentials.yml` after
# `-f docker-compose.yml` if the dev server uses the host-credentials overlay.
docker compose up -d --build

curl -fsS $WRAPPER/healthz           # {"status":"ok"}
docker compose logs claude-wrapper | grep "capabilities\["   # per-model boot log
```

Between test groups, env changes go in `.env`; apply with
`docker compose up -d` (recreate, no rebuild needed). The profile file must be
visible **inside** the container — the inbox mount is the easy path:

```bash
cp deploy/model-profiles.example.json ./inbox/profiles.json   # or your CLAUDE_INBOX_DIR
# .env: CLAUDE_WRAPPER_MODEL_PROFILES=/data/inbox/profiles.json
```

Rollback at any point: `git checkout main && docker compose up -d --build`.

---

## A. Baseline — absent config must be a no-op (plus the one intended change)

Run group A with **no profile file and none of the new env vars set**.

- [ ] **RT-1 — model list carries capabilities**
  ```bash
  curl -s $WRAPPER/v1/models | python3 -c "import json,sys; [print(m['id'], m['capabilities']) for m in json.load(sys.stdin)['data']]"
  ```
  Every entry lists `vision, file_upload, web_search, sub_agents,
  client_tools`; **no entry contains `terminal`** (gate unset).

- [ ] **RT-2 — plain chat unchanged** (Function Calling = *Default*)
  > "Summarize what a reverse proxy does."

  Streams normally, usage populated.

- [ ] **RT-3 — effort variant unchanged** — pick a `… (high)` model, any prompt.

- [ ] **RT-4 — terminal off by default** (the one intended behavior change)
  > "Use your bash tool to run `uname -a` and show me the output."

  Model states it cannot run shell commands.

- [ ] **RT-5 — delegation exempt from the gate** — with terminal still off,
  generate an image from OpenWebUI (or any audio/TTS request). Still works —
  internal delegation runs are never gated.

- [ ] **RT-6 — generated-file downloads intact** (Function Calling = *Default*)
  > "Write a 3-line CSV of fruit prices and give it to me as a file."

  Clickable download link appears.

- [ ] **RT-7 — gate opt-in restores classic behavior** — set
  `CLAUDE_WRAPPER_EXPOSE_TERMINAL=true`, `docker compose up -d`, repeat RT-4
  (now returns `uname -a` output) and RT-1 (`terminal` now advertised).

## B. Puller / OpenWebUI toggles (run on the OpenWebUI host)

- [ ] **RT-8 — dry run** (validates the OpenWebUI admin endpoint shapes —
  capture the output verbatim if anything errors):
  ```bash
  WRAPPER_BASE_URL=$WRAPPER OPENWEBUI_API_KEY=<admin-api-key> SYNC_DRY_RUN=1 \
    python3 openwebui_capability_sync.py
  ```
  Prints planned toggles per model, exit 0.

- [ ] **RT-9 — real sync** — same line without `SYNC_DRY_RUN`. Reports
  updated/created counts; OpenWebUI Admin → Models now shows toggles matching
  RT-1 (Vision/File Upload on, Terminal per your RT-7 state, Code Interpreter
  off).

## C. Profiles

Install this as `./inbox/profiles.json`, set
`CLAUDE_WRAPPER_MODEL_PROFILES=/data/inbox/profiles.json`, restart:

```json
{"models": [
  {"match": "claude-haiku-*", "remove": ["client_tools", "web_search"]},
  {"match": "claude-sonnet-*", "add": ["time_calc", "memory"]}
]}
```

- [ ] **RT-10 — boot validation fails loudly** — boot logs show per-model
  `capabilities[...]` lines. Then temporarily add `"warp_drive"` to an entry:
  startup **fails naming the entry**. Restore the file.

- [ ] **RT-11 — CLI web-search gating** — Haiku, Function Calling = *Default*:
  > "Search the web for today's top tech headline."

  Haiku says it can't browse; the same prompt on Sonnet searches.

- [ ] **RT-12 — bridge denial is a hard 400** — Haiku, Function Calling =
  *Native*, send any message. OpenWebUI surfaces an error naming
  `client_tools` and the declared tool names — not silent prose. (Designed
  behavior; note UX feedback here if 400 feels wrong in practice.)

- [ ] **RT-13 — wrapper calculator, streaming, invisible** — Sonnet, *Native*:
  > "What is 234.7 * 89.3 / 1.7, exactly? Don't estimate."

  Exact answer streams as plain text (≈ 12329.48…); **no tool-call card**
  for `calculate` appears; OpenWebUI's own tools still work in the same chat.

- [ ] **RT-14 — wrapper memory, per conversation** — Sonnet, *Native*, one
  conversation:
  > "Remember that my favorite deployment target is Podman."

  …then a few turns later:
  > "What's my favorite deployment target?"

  Recalled. On the wrapper host, `docker compose exec claude-wrapper ls
  /data/memory/` shows the conversation dir. A **new** conversation asking
  the same question does *not* recall it (memory is per-conversation).

- [ ] **RT-15 — collision guard** — enable an OpenWebUI tool literally named
  `calculate`, Sonnet, *Native*: request 400s naming the collision.

## D. Opt-in server tools (each needs its env/profile; restart between)

- [ ] **RT-16 — bridge server web search** — set
  `CLAUDE_WRAPPER_BRIDGE_WEB_SEARCH=true`, Sonnet, *Native*:
  > "What was yesterday's closing price of the S&P 500? Search for it."

  Answer reflects a live search (bills per-search on the Anthropic account).

- [ ] **RT-17 — code interpreter** — add `"add": ["code_interpreter"]` for one
  model, *Native*:
  > "Compute the 50th Fibonacci number by running code."

  Returns 12586269025 via Anthropic's sandbox.

- [ ] **RT-18 — image backend** *(only once a backend is chosen)* — set
  `CLAUDE_WRAPPER_IMAGE_BACKEND_URL/_KEY`, generate an image from OpenWebUI;
  the image comes from the backend, not the SVG path.

## E. Loop safety

- [ ] **RT-19 — round cap never hangs** — Sonnet with `time_calc`, *Native*:
  > "Call your calculator tool 20 separate times, once per number 1–20, doubling each."

  Either a sensible answer, or a clean 502 mentioning "rounds" — never a hang.
