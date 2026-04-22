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
| `CLAUDE_WRAPPER_API_KEYS` | Comma-separated bearer tokens clients must send. Leave blank on a trusted network. |
| `CLAUDE_WRAPPER_PORT` | Host + container port. Defaults to `8000`. |
| `CLAUDE_WRAPPER_DEFAULT_MODEL` | Used when a request sets `"model": "auto"`. |
| `ANTHROPIC_API_KEY` | Skip the interactive login — use API key auth instead. |
| `CLAUDE_CODE_OAUTH_TOKEN` | Skip the interactive login — use a pre-minted OAuth token. |

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
| `POST` | `/v1/responses` | OpenAI Responses API (maps to chat) |
| `POST` | `/v1/embeddings` | Dense vectors via fastembed / sentence-transformers |
| `POST` | `/v1/moderations` | Content classification via Claude |

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

When Claude writes into `outputs/` in the session workspace, the wrapper
detects the new files and:

1. Registers each one with `/v1/files` (`purpose=assistant_output`).
2. Appends a "Generated files" block to the assistant message listing
   each file's `file_id`, mime, and size.
3. If the request body sets `"inline_generated_files": true`, the raw
   bytes are included inline as base64 under `message.attachments[*].content_base64`.

A typical prompt:

> "Transcode the attached `.wav` to MP3 and write it to
> `outputs/result.mp3`."

…produces a response with a `file_id` the client can download with
`GET /v1/files/{id}/content`.

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

## Supported models

The `/v1/models` endpoint lists the Claude models the wrapper accepts:
current Opus / Sonnet / Haiku family. Pass `"model": "auto"` to use
`CLAUDE_WRAPPER_DEFAULT_MODEL`.

## Limitations

- No tool-calling over the OpenAI function/tool surface — Claude Code
  manages its own tools internally (Read/Write/Bash/etc.) and surfaces
  only the final assistant text.
- Fine-tuning endpoints return 501: Claude models aren't user-tunable
  through this path. Use `/v1/assistants` to save per-customer
  instructions instead.
- Delegated endpoints (audio, images) rely on Claude installing tools
  the first time they're invoked. Cold-call latency is high; subsequent
  calls reuse the cached install.
- The realtime WebSocket is text-only — audio goes through
  `/v1/audio/*` instead of the realtime socket.
