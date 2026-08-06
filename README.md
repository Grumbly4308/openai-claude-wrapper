# claude-wrapper

OpenAI-compatible HTTP API in front of [Claude Code](https://docs.claude.com/en/docs/claude-code), packaged as a Docker container.

- Drop-in replacement for `https://api.openai.com` in any OpenAI client.
- Handles text, images, audio, video, and arbitrary binary files through
  `chat/completions` multimodal content and a full `/v1/files` API.
- Serves many concurrent clients in parallel; requests that target the
  same conversation are serialized automatically to keep Claude Code's
  session log consistent.

## Requirements

- Docker Engine 24+ with Compose v2 (`docker compose`, not `docker-compose`).
- An Anthropic account that can log into Claude Code, **or** an
  `ANTHROPIC_API_KEY` / `CLAUDE_CODE_OAUTH_TOKEN`.
- ~3 GB free disk for the image and persisted volumes.

## 1. Configure

```bash
git clone <this-repo>
cd claude-wrapper
cp .env.example .env
```

Open `.env` and set anything you need:

| Variable | Purpose |
| --- | --- |
| `CLAUDE_UID` / `CLAUDE_GID` | Uid/gid the container runs as. Must match the host account that owns any bind-mounted path. Defaults to `1000`. |
| `CLAUDE_INBOX_DIR` | Host drop folder, bind-mounted read-only at `/data/inbox`. Defaults to `./inbox`. |
| `CLAUDE_WRAPPER_API_KEYS` | Comma-separated bearer tokens clients must send. Leave blank on a trusted network. |
| `CLAUDE_WRAPPER_PORT` | Host + container port. Defaults to `8000`. |
| `CLAUDE_WRAPPER_DEFAULT_MODEL` | Used when a request sets `"model": "auto"`. |
| `ANTHROPIC_API_KEY` | Skip the interactive login — use API key auth instead. |
| `CLAUDE_CODE_OAUTH_TOKEN` | Skip the interactive login — use a pre-minted OAuth token. |

Set `CLAUDE_UID` / `CLAUDE_GID` first, before you build. The container
runs unprivileged as an in-image `claude` user, and that uid has to
match the host account owning the inbox (and the credentials file, if
you use the overlay in [Auth](#auth)) — otherwise the container reads
those paths as a stranger and mode-600 files are simply unreadable:

```bash
id -u   # -> CLAUDE_UID
id -g   # -> CLAUDE_GID
```

These are baked into the image at build time, so changing them later
needs `docker compose up -d --build`, not just a restart.

If both `ANTHROPIC_API_KEY` and a persisted OAuth login are set, the
env var wins. Most users should leave the auth vars blank and run the
interactive login in step 3.

## 2. Build the image

```bash
docker compose build
```

Builds `claude-wrapper:latest` from the pinned `node:22-bookworm-slim`
base. Takes ~3 min on a cold cache. Re-runs are cached at the layer
level, so only the layer holding your source changes rebuilds after
you edit `src/`.

Verify the build:

```bash
docker images claude-wrapper:latest
```

## 3. Initialize Claude Code credentials (one time)

This runs Claude's OAuth flow inside the container and stores the
resulting credentials in the `claude-home` Docker volume. They survive
restarts, rebuilds, and `docker compose down`.

```bash
# Interactive — opens a browser-based OAuth flow (recommended):
docker compose run --rm -it claude-wrapper login

# OR, for headless / CI — prints a URL + code, accepts a long-lived token:
docker compose run --rm -it claude-wrapper setup-token
```

Skip this step entirely if you set `ANTHROPIC_API_KEY` or
`CLAUDE_CODE_OAUTH_TOKEN` in `.env`.

## 4. Run the server

```bash
docker compose up -d
```

The container binds `0.0.0.0:${CLAUDE_WRAPPER_PORT:-8000}` and is
reachable from loopback, LAN, or any peer container.

Confirm it's up:

```bash
curl -fsS http://localhost:8000/healthz
# {"status":"ok"}

docker compose ps
# claude-wrapper  Up (healthy)
```

Tail logs:

```bash
docker compose logs -f claude-wrapper
```

## 5. First request

Point any OpenAI-compatible client at `http://<host>:8000/v1`:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "claude-sonnet-4-6",
      "messages": [{"role": "user", "content": "Say hello."}]
    }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-anything")

resp = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Say hello."}],
)
print(resp.choices[0].message.content)
```

If `CLAUDE_WRAPPER_API_KEYS` is set, pass one of those tokens as the
OpenAI `api_key` / `Authorization: Bearer …` header instead of
`sk-anything`.

## Lifecycle cheatsheet

```bash
# stop the server (keeps volumes + credentials)
docker compose down

# stop and wipe everything (uploads, sessions, credentials, batches)
docker compose down -v

# rebuild after a code change, keeping credentials
docker compose build && docker compose up -d

# drop into a shell inside the running container
docker compose exec claude-wrapper bash

# run arbitrary claude CLI commands
docker compose run --rm -it claude-wrapper claude --help

# view what Claude wrote to disk (per-session)
docker compose exec claude-wrapper ls /data/workspace
```

## Running the endpoint tests

There's an ASGI-level smoke test that stubs Claude Code and hits every
OpenAI-shaped route:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_endpoints.py
# RESULT pass=45 fail=0
```

No Docker or Anthropic credentials required — the test monkey-patches
the subprocess runner.

## Endpoints

The wrapper implements the full OpenAI surface one-for-one. Endpoints
that aren't naturally served by an LLM (audio, images, embeddings) are
implemented by having Claude Code do the work inside its per-request
workspace — it installs whatever tools it needs via Bash the first
time and reuses them afterwards. See "Delegation design" below.

### Text

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | Streaming + non-streaming multimodal chat |
| `POST` | `/v1/completions` | Legacy text-prompt completion |
| `POST` | `/v1/responses` | OpenAI Responses API — streaming + multi-turn chaining |
| `POST` | `/v1/embeddings` | Dense vectors via fastembed / sentence-transformers |
| `POST` | `/v1/moderations` | Content classification via Claude |

`/v1/responses` is the modern "ask and response" primitive. It accepts a
string or structured `input` (plus optional `instructions`), and returns a
`response` object whose `output_text` flattens the assistant message.

- **Streaming** (`"stream": true`) emits the typed Responses event protocol —
  `response.created` → `response.output_text.delta` … → `response.completed` —
  not `chat.completion.chunk`s. There is no `[DONE]` sentinel; the stream ends
  on the terminal `response.completed`/`response.failed` event.
- **Multi-turn chaining**: pass a prior response's `id` back as
  `previous_response_id` to continue the same Claude session. The id is derived
  from the session key, so the thread deterministically reattaches rather than
  forking a new session.

#### Structured output (`response_format`)

`/v1/chat/completions` honors the OpenAI `response_format` parameter. With
`{"type": "json_object"}` or `{"type": "json_schema", "json_schema": {…}}` the
wrapper appends a raw-JSON-only instruction to the prompt (including the schema
for `json_schema`) and reduces the reply to the bare JSON value before it
leaves the wrapper — markdown fences and any surrounding prose are stripped,
and the file-reference trailer is suppressed. Streamed JSON-mode requests
buffer the answer (fences can span chunk deltas) and deliver the cleaned JSON
as a single content chunk before the terminator; reasoning/progress frames are
suppressed so the concatenated content is pure JSON.

This is what clients built on the Vercel AI SDK's `generateObject` expect —
e.g. Vane, whose `JSON.parse` chokes on ` ```json `-fenced replies
([Vane#959](https://github.com/ItzCrazyKns/Vane/issues/959)). Requests without
`response_format` (or with `{"type": "text"}`) are completely unaffected. If
the model produces no parseable JSON at all, the reply passes through
unchanged rather than masking what it said.

#### Streaming feedback on long runs

On a hard problem at high/max/ultracode effort, Claude may think or run
tools for many minutes before the first answer token. To keep
`/v1/chat/completions` streams alive and visibly *working* (rather than
looking stalled — or being severed by a buffering proxy before any headers
reach the client), the wrapper:

- sends a one-time comment preamble to flush the response head past buffering
  proxies immediately;
- emits lightweight keep-alive comments, plus periodic **visible** "⏳ Still
  working… (elapsed)" ticks on the `reasoning_content` channel during silence;
- surfaces **tool/subagent activity** (`🔧 Bash: …`, `🔧 Read: …`) on the same
  channel, so a tool-heavy phase shows real progress.

Progress rides `reasoning_content` (Open WebUI's collapsible "Thinking"
section), never the answer content. Tune via `CLAUDE_WRAPPER_SSE_*` (see
`.env.example`); set `CLAUDE_WRAPPER_SSE_PROGRESS_SECONDS=0` /
`CLAUDE_WRAPPER_SSE_SHOW_ACTIVITY=false` to quiet it.

#### Clarifying questions (interactive)

Headless Claude Code has no interactive question UI: its `AskUserQuestion`
card gets auto-dismissed, so Claude proceeds on assumptions and you never get
to answer. With the **clarification protocol** on (default), the wrapper:

- injects a system prompt (`--append-system-prompt`) teaching Claude that, when
  a decision genuinely changes the result, it should ask its questions as plain
  numbered text **and stop** — making the questions the whole turn — so you
  answer in your next message and the session resumes automatically;
- disables the dead interactive tool (`--disallowedTools AskUserQuestion`) so
  questions always arrive as answerable text.

Only chat/responses opt in; delegated task endpoints (audio, images, …) never
pause. Disable globally with `CLAUDE_WRAPPER_CLARIFY=off`, override the injected
instruction with `CLAUDE_WRAPPER_CLARIFY_PROMPT`, or opt a single request out
with `"clarify": false` in the request body. (No clickable option buttons —
Open WebUI has no way to feed those back; this is well-formatted text Q&A.)

### Audio

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text (whisper installed on first call) |
| `POST` | `/v1/audio/translations` | Speech → English text |
| `POST` | `/v1/audio/speech` | Text-to-speech via espeak-ng |

### Images

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/images/generations` | Prompt → SVG → PNG (rsvg-convert) |
| `POST` | `/v1/images/edits` | Apply a prompt + optional mask to an image |
| `POST` | `/v1/images/variations` | Produce N imagemagick-driven variations |

### Files, batches, fine-tuning

| Method | Path | Purpose |
| --- | --- | --- |
| `POST/GET/DELETE` | `/v1/files`, `/v1/files/{id}`, `/v1/files/{id}/content` | Binary storage |
| `POST/GET` | `/v1/batches`, `/v1/batches/{id}`, `/v1/batches/{id}/cancel` | JSONL batch submission (local queue) |
| `GET/POST` | `/v1/fine_tuning/jobs` | Stubs — Claude isn't user-fine-tunable (returns 501) |

### Assistants + threads

| Method | Path | Purpose |
| --- | --- | --- |
| `POST/GET/DELETE` | `/v1/assistants`, `/v1/assistants/{id}` | Saved (model, instructions) configs |
| `POST/GET/DELETE` | `/v1/threads`, `/v1/threads/{id}` | Persistent conversation threads |
| `POST/GET` | `/v1/threads/{id}/messages` | Append / list thread messages |
| `POST/GET` | `/v1/threads/{id}/runs`, `/v1/threads/{id}/runs/{id}` | Execute an assistant on a thread |

### Vector stores

| Method | Path | Purpose |
| --- | --- | --- |
| `POST/GET/DELETE` | `/v1/vector_stores`, `/v1/vector_stores/{id}` | Embedding-backed vector collections |
| `POST/GET/DELETE` | `/v1/vector_stores/{id}/files`, `/v1/vector_stores/{id}/files/{id}` | Manage indexed files |
| `POST` | `/v1/vector_stores/{id}/search` | Cosine similarity search |

### Realtime

| Method | Path | Purpose |
| --- | --- | --- |
| `WS` | `/v1/realtime` | Text-only WebSocket bridge to chat completions |
| `GET` | `/v1/realtime/sessions` | Discovery helper for OpenAI SDKs |

### Models + health

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/v1/models`, `/v1/models/{id}` | Advertise supported Claude models |
| `GET`  | `/v1/usage/{session_id}` | Per-conversation token spend + remaining allowance (see usage cap) |
| `GET`  | `/healthz` | Liveness probe |

## Delegation design

Non-text endpoints delegate to Claude Code through a per-request
workspace. The pattern:

1. Caller writes any input bytes into `workspace/uploads/`.
2. Caller invokes Claude with a structured prompt that names the exact
   output file(s) to produce under `workspace/outputs/`.
3. Claude uses its Bash / Read / Write tools to do the work — install
   faster-whisper, invoke ffmpeg or espeak-ng, render an SVG with
   rsvg-convert, etc.
4. Caller reads the named output file(s) and packages them in the
   endpoint's OpenAI-shaped response.

This keeps the image small — only `ffmpeg`, `espeak-ng`, `imagemagick`,
`librsvg2-bin` and a couple of fast Python packages (`fastembed`,
`numpy`) are preinstalled. Heavier dependencies (faster-whisper, etc.)
are installed lazily by Claude on first use and cached for subsequent
calls.

## Multimodal input

The chat endpoint accepts the standard OpenAI content-part union:

```jsonc
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Summarise this video and extract the audio."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0K..."}},
    {"type": "input_audio", "input_audio": {"data": "UklGR...", "format": "wav"}},
    {"type": "file", "file": {"file_id": "file-abc123"}}
  ]
}
```

Accepted URL schemes inside `image_url.url`:

- `data:<mime>;base64,<payload>` — inline bytes.
- `https://…` / `http://…` — fetched by the server.
- `file-<id>` — reference to something previously uploaded to `/v1/files`.

Uploaded and inlined binaries are written into the per-session workspace
before Claude Code is invoked, so Claude can open them with its `Read`
tool (including images, PDFs, audio, and video — use ffmpeg inside the
container for the latter).

### Host drop folder (`/data/inbox`)

Some clients (Open WebUI in particular) extract a document to text
before it ever reaches the wrapper, so Claude sees the client's
transcription rather than the file. The inbox sidesteps that: whatever
you put in `CLAUDE_INBOX_DIR` on the host appears read-only inside the
container at `/data/inbox`, and Claude reads the real bytes.

```bash
cp report.pdf ./inbox/          # CLAUDE_INBOX_DIR on the host
```

Then just name the path in chat:

> "Read /data/inbox/report.pdf and summarise section 3."

Two requirements, both easy to get wrong:

- **The host path must already exist.** Docker creates a missing bind
  source as a *root-owned* directory, which the container then can't
  read.
- **It must be readable by `CLAUDE_UID`** (see [1. Configure](#1-configure)).

The mount is read-only, so Claude can't modify or delete anything you
drop there; it writes results to the session workspace as usual.

### Upload a file

```bash
curl -X POST http://localhost:8000/v1/files \
    -F 'file=@clip.mp4' \
    -F 'purpose=user_data'
```

Then reference the returned `id` in a subsequent chat turn:

```python
client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this clip frame by frame."},
            {"type": "file", "file": {"file_id": "file-abc123"}},
        ],
    }],
)
```

## Generated files (Claude writes binaries back)

When Claude writes a file into the session workspace, the wrapper detects it
and:

1. Registers each one with `/v1/files` (`purpose=assistant_output`).
2. Appends a "Generated files" block to the assistant message listing
   each file as a markdown download link (mime, size, and `file_id`).
3. If the request body sets `"inline_generated_files": true`, the raw
   bytes are included inline as base64 under `message.attachments[*].content_base64`.
   (Non-streaming chat only — there is no field to carry them while streaming.)

A typical prompt:

> "Transcode the attached `.wav` to MP3 and write it to
> `outputs/result.mp3`."

…produces a response with a link the user can click, plus a `file_id` the client
can download with `GET /v1/files/{id}/content`.

Files are excluded from delivery if they live under `uploads/` (the user's own
attachments) or under any dot-prefixed path — so `.scratch/` is a free channel
for intermediate work that shouldn't be handed back.

### Making the link clickable

Two things have to be true, and both have safe defaults:

- **An absolute URL.** `CLAUDE_WRAPPER_PUBLIC_BASE_URL` wins whenever it is set.
  Otherwise the wrapper derives the base from the request's own `Host` /
  `X-Forwarded-Host` / `X-Forwarded-Proto` headers, which is correct behind a
  typical reverse proxy and needs no configuration. Set
  `CLAUDE_WRAPPER_DERIVE_BASE_URL=off` to go back to the old non-clickable
  `→ file_id=…` text.
- **Auth the browser can satisfy.** A link click sends no `Authorization`
  header, so with `CLAUDE_WRAPPER_API_KEYS` set every download used to 401.
  Each link now carries `?exp=…&sig=…`: an HMAC over exactly that one file id
  and expiry. It grants reading that one blob — listing, metadata and delete
  still require a real API key, and no API key ever appears in chat text.
  The signing key defaults to one derived from `CLAUDE_WRAPPER_API_KEYS`
  (stable across workers and restarts); set `CLAUDE_WRAPPER_DOWNLOAD_SIGNING_KEY`
  explicitly if you don't want an API-key rotation to invalidate outstanding
  links. Links expire after `CLAUDE_WRAPPER_DOWNLOAD_URL_TTL` seconds
  (default 30 days; `0` = never).

### Getting Claude to write a file at all

`CLAUDE_WRAPPER_WORKSPACE_HINT` (on by default) tells Claude that its working
directory is delivered to the user, so it writes a requested document or dataset
to a file instead of pasting the whole thing into the reply. Without it Claude
has no way to know the file goes anywhere, and mostly won't create one. It is
forced off for JSON-mode requests, where the client wants the value in the reply
body. Override the text with `CLAUDE_WRAPPER_WORKSPACE_PROMPT`.

### If your client sends `tools` (Open WebUI native function calling)

A request that declares `tools` is served by the **tool bridge**, which calls the
Anthropic Messages API directly — no Claude Code CLI, so no workspace, no `Write`
tool, and **no generated files on those turns**. Open WebUI in *Native* function
calling mode sends its whole tool roster on every message, so this silently
disables file downloads for the entire chat.

The fix is client-side, and needs no wrapper configuration: in Open WebUI set
the model's **Function Calling** setting from **Native** to **Default**. Open
WebUI then runs its own tool-selection call and sends no `tools[]` on the
completion, so the request takes the CLI path — downloads work, and function
calling stays fully intact.

There is deliberately no wrapper-side switch for this. A single turn cannot be
served both ways: the CLI runs its own tool loop and cannot surface a
caller-declared tool. Routing a tools-carrying turn to the CLI anyway would
silently drop the client's tools, so a real agentic client (the Vercel AI SDK,
LangChain) offering a tool on its opening turn would get prose instead of a
`tool_call` and its loop would stall on the first hop.

## Conversation continuity

OpenAI's Chat Completions API is stateless — the client sends the full
history every request. The wrapper mirrors that: it hashes the
history-excluding-last-user-turn and maps that to a Claude Code session
id, which is resumed on follow-up turns.

To pin a session explicitly pass `session_id` in the request body:

```json
{
  "model": "claude-sonnet-4-6",
  "session_id": "my-assistant-for-alice",
  "messages": [ ... ]
}
```

The response body echoes `session_id` so clients can round-trip it.

## Per-conversation usage cap (usage checkpoint)

Every request spends your Anthropic session/subscription quota, and a single
long conversation — especially at `max`/`ultracode` effort — can eat a large
slice of it with no warning. The wrapper can cap that **per conversation** and
ask before spending more.

**The cap is ON by default at the Max 5× ($100) plan** (allowance 7,500,000,
checkpoint every 375,000 tokens). Override it **by plan**, **by an explicit
number**, or turn it **off**:

```bash
# (a) By subscription plan — pro | max 5x ($100) | max 20x ($200)  [default: max 5x]
CLAUDE_WRAPPER_SESSION_PLAN="max 5x"
CLAUDE_WRAPPER_PRO_SESSION_TOKENS=1500000        # Pro anchor; Max scales 5x/20x from it

# (b) Or set the allowance directly (this wins over the plan)
CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE=0         # one "session's worth" of tokens (0 = use plan)

# Disable entirely:
CLAUDE_WRAPPER_SESSION_PLAN=off

CLAUDE_WRAPPER_SESSION_BLOCK_PERCENT=5            # block = 5% of the allowance
CLAUDE_WRAPPER_BUDGET_CONTINUE_KEYWORD=continue,proceed,keep going,go on,yes
```

> ⚠️ **The per-plan token figures are estimates.** Anthropic does not publish a
> token number for the Pro/Max *session* windows, and the wrapper can't query it
> (the subscription quota is exposed nowhere in the API — see below). What's
> defined is the *relationship*: Max is "5×" ($100) and "20×" ($200) of Pro. So
> the plan setting anchors on `CLAUDE_WRAPPER_PRO_SESSION_TOKENS` and scales from
> there. The default anchor (1,500,000) is **calibrated from real usage** — a
> heavy ~2h Claude Code session measured ~1.54M billable tokens (input +
> cache-creation + output, excluding near-free cache reads), which the operator
> reported as 21% of a Max-5× window → ~7.5M per window → ~1.5M Pro anchor. Tune
> it to your own usage; the cap is a safety checkpoint, not an exact mirror of
> Anthropic's accounting. On startup the wrapper logs the resolved `plan` /
> `allowance` / `block` so you can confirm.

| Plan setting | Multiplier | Allowance (default anchor 1.5M) | Block @ 5% |
|---|---|---|---|
| `pro` / `pro $20` | 1× | 1,500,000 | 75,000 |
| `max 5x` / `max $100` **(default)** | 5× | 7,500,000 | 375,000 |
| `max 20x` / `max $200` | 20× | 30,000,000 | 1,500,000 |
| `off` / `none` | — | disabled | — |

How it works:

- The wrapper tracks tokens (input + output) spent by each conversation.
- A **block** is `allowance × percent` (e.g. 1,000,000 × 5% = 50,000 tokens).
  Each conversation starts with one block of headroom.
- Once a conversation has spent its current block, the **next** request doesn't
  call Claude — it returns a checkpoint message:
  > ⏸️ **Usage checkpoint.** This conversation has used **52,000 tokens**,
  > reaching its **50,000-token** budget block (5% of the configured session
  > allowance). Reply **continue** to allow another block, or start a new chat
  > to reset.
- Replying with a continue keyword grants one more block and proceeds. Starting
  a **new chat** (new session) begins with a fresh budget.

### Checking usage: `stats` / `context`

Send **`stats`** or **`context`** (or `/stats`, `/context`) as the whole chat
message and the wrapper answers **instantly, without spawning Claude** — even
while the conversation is paused at a checkpoint, and at zero token cost:

> 📊 **Usage stats**
> - **Spent (this conversation):** 412,300 tokens across 7 requests (5.5% of the session allowance)
> - **Remaining before the next checkpoint:** 337,700 of 750,000 tokens (2 × 375,000-token blocks)
> - **Session allowance:** 7,500,000 tokens (max_5x plan), 5% per block
> - **Session key:** `conv-1a2b3c…`

Only a message that *is* the command triggers it — a prompt that merely
mentions "stats" goes to Claude as usual. The same numbers are exposed
programmatically at `GET /v1/usage/{session_id}` (the session key that chat
responses return in their `session_id` field):

```bash
curl -fsS -H "Authorization: Bearer $KEY" http://localhost:8000/v1/usage/conv-1a2b3c | jq
```

Because the check happens before Claude is spawned, a paused conversation costs
nothing until you confirm. The cap ships **enabled** at the Max 5× plan; set
`CLAUDE_WRAPPER_SESSION_PLAN=off` to disable it.

## Concurrency

- FastAPI + uvicorn serve requests async. Different sessions execute
  fully in parallel.
- Requests addressing the **same** `session_id` (or deriving the same
  hash) are serialized by an in-process `asyncio.Lock`, because Claude
  Code's session JSONL file is not safe to write from two processes
  simultaneously.
- Scale horizontally by running multiple containers behind a sticky
  load balancer keyed on `session_id`.

## Auth

There are two independent auth layers:

**1. Claude Code → Anthropic.** The container needs to authenticate
itself to Anthropic. In order of preference:

- **Interactive OAuth (recommended).** Run `docker compose run --rm -it
  claude-wrapper login` once. This executes `claude login` inside the
  container and writes the resulting credentials to
  `/home/claude/.claude/`, which is backed by the `claude-home` named
  volume so they survive `docker compose down/up`, image rebuilds, and
  machine reboots.
- **Long-lived OAuth token.** `docker compose run --rm -it
  claude-wrapper setup-token` runs `claude setup-token`, which prints a
  URL + code, accepts the resulting token, and stores it in the same
  volume. Good for headless / CI setups where you never want to
  interactively log in again.
- **Env vars.** Set `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` in
  `.env`. Takes precedence over nothing; skip if you've done the
  interactive login.
- **Share the host account's login.** If the host already has a Claude
  Code login you'd rather reuse, point `CLAUDE_HOST_CREDENTIALS` at its
  `~/.claude/.credentials.json` and add the optional overlay:

  ```bash
  docker compose -f docker-compose.yml -f docker-compose.host-credentials.yml up -d
  ```

  This bind-mounts that single file read-only. Because it's one file
  rather than a directory, an in-container token refresh can't replace
  the inode — refreshes have to happen host-side, and an expired token
  means logging in again *on the host*. The file is mode 600, so
  `CLAUDE_UID` must match its owner. Prefer the container's own login
  for anything long-running or headless.

  Note this is a deliberately narrow mount. Bind-mounting the host's
  whole `~/.claude` instead co-mingles its live daemon, session locks,
  and 700-mode sessions dir, which breaks `claude --resume`: the first
  turn of a chat succeeds and every follow-up fails with
  `error_during_execution`.

Entrypoint subcommands available via `docker compose run --rm -it
claude-wrapper <cmd>`:

| Command | Purpose |
| --- | --- |
| `serve` (default) | Start the uvicorn API server |
| `login` / `init` | Interactive Claude Code OAuth login |
| `setup-token` | Mint a long-lived OAuth token |
| `shell` | Drop into bash inside the container |
| `claude …` | Run any other `claude` CLI command |

**2. Client → wrapper.** If `CLAUDE_WRAPPER_API_KEYS` is set
(comma-separated), every request to the wrapper must include
`Authorization: Bearer <one-of-those-keys>`. When unset, the server is
unauthenticated — bind it to loopback or a private network only.

## Data & persistence

The compose file mounts two named volumes:

- `claude-data` → `/data` (uploaded + generated files, session registry,
  per-session workspaces).
- `claude-home` → `/home/claude/.claude` (Claude Code's own state,
  including the OAuth credentials from `login` / `setup-token`).

…plus one host bind mount:

- `${CLAUDE_INBOX_DIR:-./inbox}` → `/data/inbox`, read-only. See
  [Host drop folder](#host-drop-folder-datainbox).

The container runs unprivileged as `CLAUDE_UID:CLAUDE_GID` with
`cap_drop: ALL` and `no-new-privileges`, so everything it touches on the
host must be readable by that uid.

## Supported models

The `/v1/models` endpoint lists the Claude models the wrapper accepts. The
list is **built once at startup by scanning the installed Claude Code binary**
for the model ids it ships with (the current Opus / Sonnet / Haiku families and
the `fable` / `mythos` codename families, including the `[1m]` long-context
variants), so it tracks whatever Claude Code version is installed instead of a
hardcoded set. Set
`CLAUDE_WRAPPER_MODEL_DISCOVERY=off` to serve a static built-in list instead;
discovery also falls back to that list automatically if the binary can't be
read. Pass `"model": "auto"` to use `CLAUDE_WRAPPER_DEFAULT_MODEL`.

Each effort-capable model is advertised with one variant per effort level it
accepts (the *family rule*): Opus 4.5+ and the `fable` / `mythos` codename
families expose `(low)`/`(medium)`/`(high)`/`(xhigh)`/`(max)` plus `(ultracode)`;
Sonnet 4.6+ exposes through `(xhigh)`; Haiku and older models expose none.
Selecting a variant like `claude-opus-4-8 (xhigh)` sets the per-request effort.

The `(ultracode)` variant is special: it requests xhigh effort **plus** Claude
Code's dynamic-workflow (multi-agent) orchestration. Because ultracode is gated
on dynamic workflows being enabled — and that setting defaults off in a headless
container — the wrapper turns it on in the same overlay
(`--settings '{"enableWorkflows": true, "ultracode": true}'`). An org-policy
`disableWorkflows` or an account-level launch gate can still override this; those
are account-side and cannot be set by the wrapper.

## Limitations

- A single turn can serve the client's tools or run Claude Code, never both.
  Requests carrying `tools` go to the tool bridge (Messages API, `tool_calls`
  returned, no CLI and therefore no generated files); requests without them run
  the agentic CLI path, where Claude manages its own tools internally
  (Read/Write/Bash/etc.) and only the final assistant text is surfaced. See
  "Generated files" for the Open WebUI setting that avoids the choice entirely.
- Fine-tuning endpoints return 501: Claude models aren't user-tunable
  through this path. Use `/v1/assistants` to save per-customer
  instructions instead.
- Delegated endpoints (audio, images) rely on Claude installing tools
  the first time they're invoked. Cold-call latency is high; subsequent
  calls reuse the cached install.
- The realtime WebSocket is text-only — audio goes through
  `/v1/audio/*` instead of the realtime socket.
