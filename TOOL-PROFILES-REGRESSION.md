# Per-model tool profiles — manual regression pass (sandboxed wrapper)

Manual acceptance run for the `feat/sandbox-tool-profiles` branch against a
live OpenWebUI. **The sandboxed topology (`docker-compose.sandbox.yml`) is the
deployment of record** — the single-container layout is sunset and no longer
regression-tested. Work top to bottom; prompts in quotes go into OpenWebUI
chat; commands run where indicated. RT-8's raw output is needed verbatim to
close the open checklist items in [TOOL-PROFILES-TASKS.md](TOOL-PROFILES-TASKS.md).

Throughout: `WRAPPER=http://<dev-server>:8000` and
`DC="docker compose -f docker-compose.sandbox.yml"` (rootless podman:
`DC="podman-compose -f docker-compose.sandbox.yml"` — see the README's
podman caveats).

---

## Setup — deploy the branch on the dev server

Credentials live in the shared `claude-home` volume, owned by the agent
container and kept fresh by the `claude-refresher` service. A branch switch or
rebuild touches none of that — **no re-login needed**. Your untracked `.env`
survives too.

```bash
cd /path/to/claude-wrapper           # the existing dev checkout
$DC ps                                # note current state
git fetch origin
git checkout feat/sandbox-tool-profiles
git pull --ff-only origin feat/sandbox-tool-profiles

# Build with podman itself — NOT docker/buildx. A buildx build lands in the
# wrong image store and podman keeps serving the stale localhost/
# claude-wrapper:latest (symptom: rebuilds "succeed" but /v1/models never
# changes). Same fix as the historical stale-image loop.
podman build -t claude-wrapper:latest \
  --build-arg CLAUDE_UID=$(id -u) --build-arg CLAUDE_GID=$(id -g) .
$DC up -d

curl -fsS $WRAPPER/healthz            # {"status":"ok"}
$DC logs claude-wrapper | grep "capabilities\["    # per-model boot log
$DC logs claude-wrapper | grep "agent execution"   # REMOTE via http://claude-agent:8791
```

Only a **fresh** deployment (empty `claude-home` volume) needs the one-time
login bootstrap — see README "Sandboxed deployment"; an existing dev volume
does not.

Two knobs you'll flip during the run:

- **Env changes** go in `.env`, applied with `$DC up -d` (recreate, no
  rebuild).
- **The profile file is `sandbox/profiles.json`** in the checkout — mounted
  read-only into the wrapper at `/etc/claude-wrapper/profiles.json` (the
  compose default). It ships as a no-op `{}`. Edit it, then
  `$DC restart claude-wrapper`. Same workflow as `sandbox/allowlist.txt`.
  (There is **no** inbox path for profiles in this topology — the inbox
  mounts on the agent, not the wrapper.)

Rollback at any point: `git checkout <previous branch> && $DC up -d --build`.

---

## A. Baseline — absent config must be a no-op (plus the one intended change)

Run group A with `sandbox/profiles.json` still `{}` and none of the new env
vars set — and with the model's **Function Calling = *Default*** in OpenWebUI
for the whole group. In Native mode OpenWebUI attaches its tool roster to
every message and the request routes to the tool bridge, which never has a
Bash tool regardless of the gate — the test proves nothing there. Verify the
path per turn in the logs: `$DC logs claude-wrapper | grep chat/completions |
tail -1` — a CLI-path turn has **no** `[tool-bridge]` tag and `tools=0`.
(Native mode is exercised deliberately in groups C–E.)

Config note: if `.env` sets `CLAUDE_WRAPPER_DEFAULT_MODEL=auto`, an `auto`
row appears in `/v1/models` — the wrapper always advertises its configured
default. Set a real model id there.

- [ ] **RT-1 — model list carries capabilities**
  ```bash
  curl -s $WRAPPER/v1/models | python3 -c "import json,sys; [print(m['id'], m['capabilities']) for m in json.load(sys.stdin)['data']]"
  ```
  Every entry lists `vision, file_upload, web_search, sub_agents,
  client_tools`; **no entry contains `terminal`** (gate unset).

- [ ] **RT-2 — plain chat unchanged** (Function Calling = *Default*)
  > "Summarize what a reverse proxy does."

  Streams normally, usage populated. Runs remotely in the agent container as
  before.

- [ ] **RT-3 — effort variant unchanged** — pick a `… (high)` model, any prompt.

- [ ] **RT-4 — terminal off by default** (the one intended behavior change)
  > "Use your bash tool to run `uname -a` and show me the output."

  Model states it cannot run shell commands — the profile's
  `--disallowedTools Bash` crossed the shim into the agent's CLI.

- [ ] **RT-5 — delegation exempt from the gate** — with terminal still off,
  hit the wrapper's image endpoint (do NOT ask the chat model to draw — chat
  has no image tool on any path; the endpoint does its work via a delegation
  session that needs Bash):
  ```bash
  curl -s -X POST $WRAPPER/v1/images/generations \
    -H 'content-type: application/json' \
    -d '{"prompt":"a red circle on a white background","n":1,"size":"256x256","response_format":"url"}'
  ```
  JSON with `data[0].url` after ~30–90s. Passing while RT-4 shows chat-Bash
  gated proves delegation is exempt.

- [ ] **RT-6 — generated-file downloads intact** (Function Calling = *Default*)
  > "Write a 3-line CSV of fruit prices and give it to me as a file."

  Clickable download link appears (shared workspace volume unaffected).

- [ ] **RT-7 — gate opt-in restores classic behavior** — set
  `CLAUDE_WRAPPER_EXPOSE_TERMINAL=true` in `.env`, `$DC up -d`, repeat RT-4
  (now returns `uname -a` output) and RT-1 (`terminal` now advertised).

- [ ] **RT-8s — terminal exposure stays inside the sandbox** *(sandbox
  marquee test — with the gate from RT-7 still open)*
  > "Use bash to run: curl -sS --max-time 10 https://example.com and show me what you get."

  The request **fails** (squid denies CONNECT — example.com is not on
  `sandbox/allowlist.txt`). An exposed terminal still has no egress beyond
  the allowlist; that's the point of running profiles inside the sandbox.

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

## C. Profiles (edit `sandbox/profiles.json`, `$DC restart claude-wrapper`)

Use this test profile:

```json
{"models": [
  {"match": "claude-haiku-*", "remove": ["client_tools", "web_search"]},
  {"match": "claude-sonnet-*", "add": ["time_calc", "memory"]}
]}
```

- [ ] **RT-10 — boot validation fails loudly** — after the restart, wrapper
  logs show per-model `capabilities[...]` lines reflecting the profile. Then
  temporarily add `"warp_drive"` to an entry and restart: the wrapper
  **fails to start, naming the entry** (`$DC logs claude-wrapper`). Restore
  the file.

- [ ] **RT-11 — CLI web-search gating** — Haiku, Function Calling = *Default*:
  > "Search the web for today's top tech headline."

  Haiku says it can't browse; the same prompt on Sonnet searches (via squid —
  no allowlist change involved).

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

  Recalled. On the dev server: `$DC exec claude-wrapper ls /data/memory/`
  shows the conversation dir (memory lives in the wrapper's `claude-data`
  volume — the agent container never sees it). A **new** conversation asking
  the same question does *not* recall it (memory is per-conversation).

- [ ] **RT-15 — collision guard** — enable an OpenWebUI tool literally named
  `calculate`, Sonnet, *Native*: request 400s naming the collision.

## D. Opt-in server tools (each needs its env/profile; restart between)

- [ ] **RT-16 — bridge server web search** — set
  `CLAUDE_WRAPPER_BRIDGE_WEB_SEARCH=true`, Sonnet, *Native*:
  > "What was yesterday's closing price of the S&P 500? Search for it."

  Answer reflects a live search. Executes on Anthropic's side through
  `api.anthropic.com` (already allowlisted — no squid change), and bills
  per-search on the Anthropic account.

- [ ] **RT-17 — code interpreter** — add `"add": ["code_interpreter"]` for one
  model, *Native*:
  > "Compute the 50th Fibonacci number by running code."

  Returns 12586269025 via Anthropic's sandbox (server-side; no egress change).

- [ ] **RT-18 — image backend** *(only once a backend is chosen)* — set
  `CLAUDE_WRAPPER_IMAGE_BACKEND_URL/_KEY`, **and allowlist the backend
  host**: external backends go in `sandbox/allowlist.txt` (edit + reload, or
  `./sandbox allow <domain>` if you use the helper); an internal backend goes
  in `SANDBOX_EXTRA_NO_PROXY` instead. Then generate an image from OpenWebUI;
  it comes from the backend, not the SVG path. Without the allowlist step the
  expected failure is a 502 "image backend unreachable" — that's squid doing
  its job, not a wrapper bug.

## E. Loop safety

- [ ] **RT-19 — round cap never hangs** — Sonnet with `time_calc`, *Native*:
  > "Call your calculator tool 20 separate times, once per number 1–20, doubling each."

  Either a sensible answer, or a clean 502 mentioning "rounds" — never a hang.
