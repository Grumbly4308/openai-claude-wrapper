# claude-wrapper

OpenAI-compatible HTTP API in front of [Claude Code](https://docs.claude.com/en/docs/claude-code)
(default) or the [OpenAI Codex CLI](https://github.com/openai/codex), selected
per deployment, packaged as a container.

- Drop-in replacement for `https://api.openai.com` in any OpenAI client.
- Handles text, images, audio, video, PDFs and arbitrary binary files through
  `chat/completions` multimodal content and a `/v1/files` API.
- Serves many concurrent clients in parallel; requests that target the
  same conversation are serialized automatically to keep Claude Code's
  session log consistent.
- Runs as a multi-container sandbox where the agent has no route to the
  internet except a domain allowlist — the supported deployment. (A
  single-container layout exists but is **sunset**: kept for local
  development and rollback only — see
  [Single-container layout (sunset)](#single-container-layout-sunset).)

---

## Contents

- [Requirements](#requirements)
- [Choosing the wrapped agent](#choosing-the-wrapped-agent)
- [Quick start (Docker Compose)](#quick-start-docker-compose)
- [Quick start (Podman)](#quick-start-podman)
- [Quick start (Codex)](#quick-start-codex)
- [Sandboxed deployment](#sandboxed-deployment-network-isolated-agent)
- [Single-container layout (sunset)](#single-container-layout-sunset)
- [Configuration reference](#configuration-reference)
- [Endpoints](#endpoints)
- [Chat features](#chat-features)
- [Files in and out](#files-in-and-out)
- [Conversation continuity](#conversation-continuity)
- [Per-conversation usage cap](#per-conversation-usage-cap-usage-checkpoint)
- [Models and reasoning effort](#models-and-reasoning-effort)
- [Auth](#auth)
  - [Codex → OpenAI](#codex--openai)
- [Data and persistence](#data-and-persistence)
- [Concurrency](#concurrency)
- [Running the tests](#running-the-tests)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)
- [Limitations and known gaps](#limitations-and-known-gaps)

---

## Requirements

- A container runtime with Compose v2 semantics. Either:
  - **Docker Engine** with the `docker compose` plugin, or
  - **Podman** (rootless is fine) — see [Quick start (Podman)](#quick-start-podman)
    for the two ways to drive it and the caveats of each.
- An Anthropic account that can log into Claude Code, **or** an
  `ANTHROPIC_API_KEY` / `CLAUDE_CODE_OAUTH_TOKEN`.
- For a Codex deployment instead: an OpenAI account on a ChatGPT plan that can
  log into the Codex CLI, **or** an `OPENAI_API_KEY` — see
  [Choosing the wrapped agent](#choosing-the-wrapped-agent).
- Disk: budget **~5 GB** to be comfortable. The image itself lands around
  2–3 GB (ffmpeg, imagemagick, librsvg, the Claude Code CLI, and `fastembed`,
  which pulls `onnxruntime`). On top of that, the default embedding model
  (~100 MB) downloads on first use, `faster-whisper` is installed lazily on
  the first transcription, and `CLAUDE_WRAPPER_MAX_UPLOAD_BYTES` defaults to
  2 GiB **per upload**.

The compose files use `${VAR:-default}` interpolation throughout and do not
declare a `version:` key, so any Compose v2-compatible frontend works.

---

## Choosing the wrapped agent

The wrapper drives one agent per deployment: Claude Code (the default) or the
OpenAI Codex CLI. Selection is deploy-time, not per-request — you pick a
stack, and only the selected agent's container runs and only its models are
advertised on `/v1/models`:

- **`docker-compose.yml`** — the Claude stack, and today's behavior unchanged.
  It pins `CLAUDE_WRAPPER_AGENT: "claude"` on its app services.
- **`docker-compose.codex.yml`** — the Codex stack: same topology, same
  security guarantees, with `codex-agent` and `codex-refresher` in place of
  their Claude counterparts and `CLAUDE_WRAPPER_AGENT: "codex"` pinned on its
  app services. Everything machine-global lives in its own `codex-*`
  namespace — compose project, container names, image tag, volumes, and the
  published port (`CODEX_WRAPPER_PORT`, default `8001`) — so both stacks run
  side by side on one machine. See [Quick start (Codex)](#quick-start-codex).

The env var is pinned as a literal in each compose file on purpose — a stale
`.env` must not be able to half-select agents across containers. As a knob it
matters only in the sunset
[single-container layout](#single-container-layout-sunset), where it alone
selects the agent. Validation fails closed: any value other than `claude` or
`codex` refuses to boot with an error naming the variable, instead of silently
serving the wrong model list.

At startup the wrapper also verifies the topology it was handed: it asks the
agent shim's `/healthz` which agent that container runs. A mismatch — say a
codex wrapper pointed at a claude agent by a stale `CLAUDE_WRAPPER_AGENT_URL`
— refuses to boot, naming both values; an agent container that is merely not
up yet only logs a warning, because `depends_on` does not guarantee readiness
and the wrapper must not crash-loop while the agent boots.

Why two files rather than compose `profiles:`? Profiles would require
`COMPOSE_PROFILES` in every existing `.env` just to keep the default stack
starting (a hard failure on `git pull && docker compose up -d`),
podman-compose's support for them is unreliable, and this README already uses
the word "profiles" for
[per-model capability profiles](#per-model-capability-profiles). File choice
is also the selection convention the repo already has (`docker-compose.yml`
vs `docker-compose.single.yml`).

---

## Quick start (Docker Compose)

This brings up the **sandboxed stack — the default and supported way the
wrapper ships**: the FastAPI server as the only published port, the agent in a
network-isolated container whose sole egress is a squid domain allowlist, and
a credential-refresher sidecar. Topology details:
[Sandboxed deployment](#sandboxed-deployment-network-isolated-agent). The
retired single-container layout survives for local development as
[`docker-compose.single.yml`](#single-container-layout-sunset).

### 1. Configure

```bash
git clone <this-repo>
cd claude-wrapper
cp .env.example .env

# Set the container identity from the account that will RUN the containers.
# This is not auto-detected, and the shipped default (1000) is wrong for you
# if `id -u` says anything else.
sed -i "s/^CLAUDE_UID=.*/CLAUDE_UID=$(id -u)/; s/^CLAUDE_GID=.*/CLAUDE_GID=$(id -g)/" .env
grep CLAUDE_.ID .env
```

> **Get this right before the first build.** `CLAUDE_UID` is used in two places
> in every compose file: as a **build arg** (which creates the in-image `claude`
> user and chowns `/data` to it) and as `user:` (the uid the process actually
> runs as). If those disagree, `/data` is owned by one uid and written by
> another, and the server crash-loops at import with
> `PermissionError: [Errno 13] Permission denied: '/data/assistants'`. Because
> it is a build arg, changing it later needs `--build`, and if volumes already
> exist it needs the re-own procedure in
> [Changing CLAUDE_UID after first run](#changing-claude_uid-after-first-run).

Then open `.env` and set what else you need. The full list is in
[Configuration reference](#configuration-reference); these are the ones that
matter on day one:

| Variable | Purpose | Default |
| --- | --- | --- |
| `CLAUDE_UID` / `CLAUDE_GID` | Uid/gid the container runs as. Must match the host account owning any bind-mounted path. | `1000` |
| `CLAUDE_INBOX_DIR` | Host drop folder, bind-mounted read-only at `/data/inbox`. | `./inbox` |
| `CLAUDE_WRAPPER_API_KEYS` | Comma-separated bearer tokens clients must send. Blank = unauthenticated. | blank |
| `CLAUDE_WRAPPER_PORT` | Host + container port. | `8000` |
| `CLAUDE_WRAPPER_DEFAULT_MODEL` | Used when a request sets `"model": "auto"` or omits the model. | `claude-opus-4-8` |
| `ANTHROPIC_API_KEY` | Skip the interactive login — use API key auth instead. | blank |
| `CLAUDE_CODE_OAUTH_TOKEN` | Skip the interactive login — use a pre-minted OAuth token. | blank |

The uid also has to match the host account owning the inbox (and the credentials
file, if you use the overlay in [Auth](#auth)) — otherwise the container reads
those paths as a stranger and mode-600 files are simply unreadable.

**Copy `.env.example` to `.env` even if you change nothing in it.** Several
compose defaults are deliberately different from the code defaults, and the
compose file interpolates an *empty string* where `.env` would supply a value.
The one that bites: `CLAUDE_WRAPPER_CLARIFY_DISALLOWED_TOOLS` defaults to
`AskUserQuestion` in code but interpolates to empty from compose, which silently
turns off the AskUserQuestion suppression described under
[Clarifying questions](#clarifying-questions-interactive). See
[Defaults that differ](#defaults-that-differ-between-code-and-compose).

### 2. Build the image

```bash
docker compose build
```

Builds `localhost/claude-wrapper:latest` from `node:22-bookworm-slim`. Layer caching means
only the layer holding `src/` rebuilds after you edit code, but note the build
is **not reproducible**: the base image is a mutable tag (not digest-pinned) and
`Dockerfile:28` installs `@anthropic-ai/claude-code@latest`, so two builds a week
apart can ship different CLI versions — and therefore a different `/v1/models`
list.

Verify:

```bash
docker images localhost/claude-wrapper:latest
```

### 3. Initialize Claude Code credentials (one time)

Stores the credentials in the shared `claude-home` volume, where they survive
restarts, rebuilds and `docker compose down`. The interactive OAuth callback
cannot complete from inside the isolated agent container, so the bootstrap
runs in the refresher — it has ordinary networking and the writable mount
(details: [First-time login](#first-time-login)):

```bash
# Interactive — type /login at the prompt, complete the flow, /exit:
docker compose run --rm -it claude-refresher claude

# OR headless / CI — prints a URL + code, accepts a long-lived token:
docker compose run --rm -it claude-refresher setup-token
```

Skip this entirely if you set `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN`
in `.env`.

### 4. Run the server

```bash
docker compose up -d
```

The container binds `0.0.0.0:${CLAUDE_WRAPPER_PORT:-8000}` and is reachable from
loopback, LAN, or any peer container. Confirm:

```bash
curl -fsS http://localhost:8000/healthz     # {"status":"ok"}
docker compose ps        # claude-wrapper, claude-agent, claude-squid,
                         # claude-refresher — all Up
docker compose logs -f claude-wrapper
```

Then verify the sandbox is actually fencing egress — the two-command check in
[Sandboxed deployment](#sandboxed-deployment-network-isolated-agent).

The startup log reports the resolved model list, whether runs execute locally or
remotely, and the state of the knowledge base, clarification and download-link
features. Read it once — every one of those misconfigurations otherwise fails
quietly. See [Troubleshooting](#troubleshooting).

### 5. First request

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

If `CLAUDE_WRAPPER_API_KEYS` is set, pass one of those tokens as the OpenAI
`api_key` / `Authorization: Bearer …` header instead of `sk-anything`.

### Lifecycle cheatsheet

```bash
docker compose down                          # stop (keeps volumes + credentials)
docker compose down -v                       # stop and wipe everything
docker compose build && docker compose up -d # rebuild after a code change
docker compose exec claude-wrapper bash      # shell inside the running container
docker compose run --rm -it claude-wrapper claude --help   # any claude CLI command
docker compose exec claude-wrapper ls /data/workspace      # what Claude wrote
```

---

## Quick start (Podman)

Podman runs this stack unmodified. Nothing in the project calls `podman`
directly — the isolation comes from the compose topology, so whichever runtime
you have is the one it uses. Rootless podman is a good fit here: the containers
already run unprivileged (`USER claude` in the image, plus `user:`,
`cap_drop: ALL` and `no-new-privileges:true` in compose), and rootless adds a
user namespace underneath that.

**Drive it with Compose v2 against the rootless podman socket.** That is the one
path this README documents. Compose v2 is a client that speaks the Docker API
over `DOCKER_HOST`, and podman's socket serves that API — so **no Docker daemon,
no `docker` CLI and no `docker` group are involved**, and the containers still
run under rootless podman. Everything in the
[Docker quick start](#quick-start-docker-compose) then works verbatim; only the
spelling changes, from `docker compose` to `docker-compose`.

### Setup (one time)

```bash
# 1. Enable podman's Docker-compatible API socket (rootless, user-level)
systemctl --user enable --now podman.socket
loginctl enable-linger "$USER"          # keep containers alive after logout
systemctl --user enable --now podman-restart.service   # start containers at boot

# 2. Install the standalone Compose v2 binary (a single static file)
sudo curl -fsSL -o /usr/local/bin/docker-compose \
  "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)"
sudo chmod +x /usr/local/bin/docker-compose
docker-compose version                  # expect v2.x

# 3. Point it at podman, persistently
echo "export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/podman/podman.sock" >> ~/.bashrc
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/podman/podman.sock
docker-compose ps                       # smoke test: connects, lists nothing
```

Then follow [1. Configure](#1-configure) onward, substituting `docker-compose`
for `docker compose`. Steps 1–5 are otherwise identical. The
[codex stack](#quick-start-codex) works the same way — add
`-f docker-compose.codex.yml` to each command, exactly as under Docker.

### Surviving reboots

Three things must all be true, and each fails silently on its own:

1. **The compose files use `restart: always` — keep it that way.**
   `podman-restart.service` is one `podman start --all --filter
   restart-policy=always` at boot; `unless-stopped` does not match the filter,
   so with a daemonless runtime nothing ever starts it again. (This repo used
   `unless-stopped` until 2026-08-11, which is exactly the "enabled the service
   but the stack stays down" symptom.)
2. **Linger, for the account that owns the containers.** Without it that
   user's systemd manager — and every user-level unit, podman-restart
   included — starts at *login*, not at boot.
3. **The units enabled in that same account's manager.** `systemctl --user`
   acts on whoever runs it; enabling the service as your login user does
   nothing for a stack owned by a service account.

For a dedicated service account (say `claude`, uid 1001), run all of it as
root once:

```bash
loginctl enable-linger claude
sudo -u claude XDG_RUNTIME_DIR=/run/user/1001 \
    systemctl --user enable --now podman.socket podman-restart.service

# verify all three legs:
loginctl show-user claude -p Linger                 # Linger=yes
sudo -u claude XDG_RUNTIME_DIR=/run/user/1001 \
    systemctl --user is-enabled podman-restart.service   # enabled
podman inspect claude-wrapper --format '{{.HostConfig.RestartPolicy.Name}}'  # always
```

The real test is a reboot: `podman ps` afterwards (as the owning user) must
list the stack. If it is empty, walk the three legs above in order.

### Confirm you are still rootless

```bash
podman info --format '{{.Host.Security.Rootless}}'   # true
systemctl --user status podman.socket                # a *user* unit
ps -o user=,pid= -p "$(podman inspect -f '{{.State.Pid}}' claude-wrapper)"
```

**Do not run the compose command under `sudo`,** do not enable `podman.socket`
as a system unit, and do not point `DOCKER_HOST` at `/run/podman/podman.sock`.
Any of the three moves the containers to root-owned podman. Because `DOCKER_HOST`
lives in your shell profile, `sudo docker-compose …` silently drops it and falls
back to the root socket — so a command that "only works with sudo" is a signal
something is misconfigured, not a reason to escalate.

### Rootless podman caveats

- **tmpfs mounts are root-owned 0755, not 1777.** Docker's default lets an
  unprivileged service write its own tmpfs; podman's does not. Squid runs as
  `proxy` and puts its pid file on the tmpfs at `/var/cache/squid`, so under
  podman it fails at startup with
  `FATAL: failed to open /var/cache/squid/squid.pid: (13) Permission denied`.
  `docker-compose.yml` pins `mode=1777` on both squid tmpfs mounts to
  make the two runtimes agree. Pin the mode on any tmpfs you add for the same
  reason — and note how this one presents, because it is the worst kind of
  failure: `restart: unless-stopped` turns the crash into an invisible loop, so
  `ps` shows the container "up" and only the restart count gives it away.

  ```bash
  podman inspect claude-squid --format '{{.State.Status}} restarts={{.RestartCount}}'
  ```

  A four-digit restart count means squid has never once started, and the agent
  container has had no egress the entire time — which surfaces as CLI runs that
  hang or exit 1, nowhere near the actual cause.
- **Your host uid maps to container uid 0, not to itself.** Named volumes live
  inside the user namespace and are unaffected, but a **bind mount** owned by
  you appears root-owned inside the container, so mode-600 files in the inbox
  are unreadable by the `claude` user. The fix is `userns_mode: keep-id` on the
  app services, which makes host and container uid the same identity. Decide
  this **before** you populate volumes — switching later remaps ownership under
  data you have already written.
- **SELinux hosts (Fedora/RHEL).** The bind-mounted `sandbox/squid.conf`,
  `sandbox/allowlist.txt` and the inbox need a `:z` suffix or the container is
  denied access. It is deliberately left out of the checked-in compose files
  because relabeling touches files in your checkout — add it locally if your host
  enforces SELinux.

### If you are stuck with podman-compose

It works, but it is not what this README is written against, and it diverges in
two ways that have both caused real failures here:

- **`run` rejects `-i`/`-t`.** `podman-compose run` allocates a TTY itself and
  errors with `unrecognized arguments: -it`. Drop the flags, or bypass compose
  for that step: `podman exec -it claude-wrapper /app/entrypoint.sh login`
  (codex stack: `podman exec -it codex-refresher /app/entrypoint.sh
  codex-login`).
- **`${VAR:-default}` may not expand.** The symptom is a service whose
  environment holds the literal string, e.g. uvicorn dying with
  `Invalid value for '--port': '${CLAUDE_WRAPPER_AGENT_PORT:-8791}'`. Check with
  `podman exec claude-agent printenv CLAUDE_WRAPPER_AGENT_PORT`. Fix by
  upgrading (`pipx install podman-compose`) or by setting every value explicitly
  in `.env` rather than relying on the inline defaults.

It also creates a **pod** per project, which Compose v2 does not. If you switch,
tear the old world down first or `up` will fail with `container name … already
in use` and leave a half-started stack:

```bash
podman rm -f claude-wrapper claude-agent claude-squid 2>/dev/null
podman pod ls && podman pod rm -f <pod-name>
```

(On the codex stack the names are `codex-wrapper codex-agent codex-refresher
codex-squid` — no name is shared with the claude stack, so the two can run
side by side.)

---

## Quick start (Codex)

This brings up the **same sandboxed stack driving the OpenAI Codex CLI**
instead of Claude Code: `docker-compose.codex.yml` swaps in a `codex-agent`
and a `codex-refresher` while keeping the topology — and its security
guarantees — identical. The steps mirror the
[Docker quick start](#quick-start-docker-compose); only the compose file and
the credential flow differ. Background:
[Choosing the wrapped agent](#choosing-the-wrapped-agent).

### 1. Configure

Follow [1. Configure](#1-configure) from the Docker quick start unchanged —
same `.env`, same `CLAUDE_UID`/`CLAUDE_GID` warning. One codex-specific knob:
the stack publishes on `CODEX_WRAPPER_PORT` (default `8001`), not
`CLAUDE_WRAPPER_PORT` — separate variables so both stacks can share the
`.env` and the machine. Then open
`sandbox/allowlist.txt` and uncomment `api.openai.com` in the OpenAI Codex
block. It ships commented out — the default Claude deployment should not
carry OpenAI egress — and it is split by auth mode: if you will use a
ChatGPT-plan login rather than an API key, uncomment `chatgpt.com` and
`auth.openai.com` too; an API-key-only deployment leaves them commented.

### 2. Build the image

```bash
docker compose -f docker-compose.codex.yml build
```

Same Dockerfile, its own `localhost/codex-wrapper:latest` tag — the codex
compose file sets the `INSTALL_CODEX=1` build arg, which adds
`@openai/codex@latest` to the image. Like the Claude CLI it is unpinned, so
two builds a week apart can ship different codex versions. The tag is
separate from the claude stack's `localhost/claude-wrapper:latest` on
purpose: the two builds differ (a Claude-built image contains no codex
binary at all), and a shared tag would let each stack's build silently
clobber the other's image.

### 3. Initialize Codex credentials (one time)

Stores the login in the shared `codex-home` volume, where it survives
restarts, rebuilds and `docker compose down`. The interactive flow cannot
complete from inside the isolated agent container, so the bootstrap runs in
the refresher — it has ordinary networking and the writable mount (details:
[First-time login (Codex)](#first-time-login-codex)):

```bash
# Device-code flow — prints a URL + code to enter in any browser:
docker compose -f docker-compose.codex.yml run --rm -it codex-refresher codex-login
```

Skip this entirely if you set `OPENAI_API_KEY` in `.env`. A third option is
persisting an API key to the volume with `codex login --with-api-key`, which
then requires uncommenting the wrapper's read-only `codex-home` mount for
function-calling (tools) requests to read it — see
[Codex → OpenAI](#codex--openai).

If you'd rather use a raw container, get the volume name from the running
agent rather than guessing it — a `-v` naming a volume that does not exist
**creates an empty one**:

```bash
podman inspect codex-agent \
    --format '{{range .Mounts}}{{.Name}} -> {{.Destination}}{{"\n"}}{{end}}'

podman exec codex-agent stat -c '%y' /home/claude/.codex/auth.json
```

That last line is the check that matters: the mtime of `auth.json` must be
from just now. If it has not moved, the credential did not land, whatever the
CLI printed.

### 4. Run the server

```bash
docker compose -f docker-compose.codex.yml up -d
```

Confirm:

```bash
curl -fsS http://localhost:8001/healthz     # {"status":"ok"}
docker compose -f docker-compose.codex.yml ps
                         # codex-wrapper, codex-agent, codex-refresher,
                         # codex-squid — all Up
docker compose -f docker-compose.codex.yml logs -f codex-wrapper
```

Then verify the sandbox is fencing egress, exactly as in the
[sandboxed deployment](#sandboxed-deployment-network-isolated-agent) check —
substituting `codex-agent` for `claude-agent` and `api.openai.com` for the
allowed host.

---

## Sandboxed deployment (network-isolated agent)

The default stack (`docker-compose.yml`) splits the wrapper into containers so
the FastAPI server is the **only** externally reachable service and the agent —
where model-driven tool use actually executes — has no route to the internet
except through a domain allowlist:

```
[clients] ──> claude-wrapper (FastAPI — the only published port)
                  │  backend network (internal: no external route)
                  ├──> claude-agent  (Claude Code CLI behind src/agent_shim.py)
                  │        │  HTTP(S)_PROXY egress only
                  └───────>└──> squid ──> sandbox/allowlist.txt hosts
```

`docker-compose.codex.yml` is the same picture with `codex-agent` in place of
`claude-agent` — the Codex CLI behind the same `src/agent_shim.py`, the same
squid, the same single published port.

```bash
docker compose up -d --build
```

**Do the one-time login before you rely on it, and read
[First-time login](#first-time-login) first.** It targets `claude-agent`, not
`claude-wrapper` — that is where the CLI runs and where its credentials live —
and the interactive OAuth callback cannot complete from inside the isolated
agent, so the bootstrap needs a throwaway container with ordinary networking.

Verify the allowlist end-to-end once it is up:

```bash
# allowed host — completes
docker compose exec claude-agent \
    curl -sS -o /dev/null -w '%{http_code}\n' https://api.anthropic.com/
# unlisted host — 403 from squid, in milliseconds
docker compose exec claude-agent \
    curl -sS -o /dev/null -w '%{http_code}\n' https://example.com/
```

### How it works

- When `CLAUDE_WRAPPER_AGENT_URL` is set, the wrapper stops spawning `claude`
  locally and sends each run to the **agent shim** (`src/agent_shim.py`), a
  minimal service exposing `GET /healthz` and `POST /run` that spawns the CLI and
  streams its stream-json stdout back verbatim. The shim never takes the binary
  path from the caller, confines `cwd` to the shared workspace volume, and can
  require a bearer token (`CLAUDE_WRAPPER_AGENT_TOKEN`). Codex runs behind the
  **same shim contract**: same `/healthz` + `/run` surface, the binary still
  comes from the shim container's own environment (never from the caller), and
  a caller's env overlay cannot steer binary or loader resolution either.
- The per-session workspace is a volume mounted **at the same path in both
  containers**, so uploads materialized by the API and files generated by the CLI
  need no copying. The file store, session registry and usage ledgers stay
  API-only; the CLI's credentials and session logs stay agent-only (the API mounts
  `claude-home` read-only, for the tool bridge).
- Squid is a **CONNECT-only** proxy — no TLS interception (`ssl_bump` appears
  nowhere), no caching, no buffering of the token stream. It runs unprivileged
  (`user: proxy`, `cap_drop: ALL`, pid file in `/var/cache/squid`) and enforces
  `sandbox/allowlist.txt`: one host per line, a leading dot matching all
  subdomains, exactly squid's `dstdomain` semantics. Denied hosts fail in
  milliseconds, so the CLI's probes to unlisted hosts don't read as latency.
  Plain HTTP on port 80 to an allowlisted host is permitted too, not only CONNECT.
- The agent's proxy env (`HTTP_PROXY`/`HTTPS_PROXY`) is inherited by the CLI and
  every Bash subshell it runs — curl, pip and git all flow through the allowlist
  with no code changes.
- `CLAUDE_CODE_PROXY_RESOLVES_HOSTS` is **off by default, on purpose**. It was
  set for the internal network's dead DNS, but its shim also swallows
  resolution of the proxy's own hostname for the CLI's OAuth client — with it
  on, logins fail with `Invalid IP address: undefined` and token refresh never
  fires (measured A/B; CREDENTIALS-FIX.md Round 4). Current CLIs hand target
  hostnames to Squid in the CONNECT line anyway, so runs work without it. If
  an older CLI dies before any CONNECT, the `.env` knob turns it back on — at
  the cost of in-agent OAuth. The knob is Claude-only: codex's HTTPS and
  WebSocket transports both honor `HTTPS_PROXY` natively, so there is no codex
  equivalent to reach for.

### The shipped allowlist

Thirteen hosts are active out of the box — the OpenAI Codex block that sits
below them ships fully commented:

`api.anthropic.com`, `claude.ai`, `claude.com`, `code.claude.com`,
`platform.claude.com`, `downloads.claude.ai`, `bridge.claudeusercontent.com`,
`mcp-proxy.anthropic.com`, `.github.com`, `.gitlab.com`, `.npmjs.org`,
`pypi.org`, `files.pythonhosted.org`.

The eight Claude/Anthropic endpoints beyond `api.anthropic.com` support the
interactive OAuth login and CLI features; `sandbox/allowlist.txt`'s own comments
recommend deleting them on an API-key-only deployment. To change the list, edit
the file and restart the squid container — the `./sandbox allow` helper referenced
in that file's comments **does not exist in this repository**.

The commented **OpenAI Codex block** (uncommenting it is step 1 of
[Quick start (Codex)](#quick-start-codex)) is split by auth mode, in the same
every-line-is-an-exit spirit as the Claude guidance: `api.openai.com` is
always needed under codex — the model API plus the tools passthrough — while
`chatgpt.com` and `auth.openai.com` serve the ChatGPT-plan login only, so an
API-key deployment leaves them commented. Squid is CONNECT-only, so the
WebSocket transport codex tries first (`wss://api.openai.com`) rides the same
CONNECT:443 entry as HTTPS.

### Operational notes

Every API feature works identically to the single-container layout — chat and
Responses (streaming included), generated-file download links, sessions and
`--resume`, the token budget, JSON mode, the tool bridge, and the delegated
endpoints — with these caveats:

- **Knowledge base:** the `curl` calls to `$OPENWEBUI_BASE_URL` run in the agent
  container. If OpenWebUI is on your internal network, add its host to
  `SANDBOX_EXTRA_NO_PROXY` (direct, needs a route) or to the allowlist (via squid).
- **Audio/embeddings first use:** `faster-whisper` and `sentence-transformers`
  download weights from `huggingface.co` — add it (and `cdn-lfs.huggingface.co`)
  to the allowlist, or pre-install the models. pip itself is already covered.
- **Function calling (`tools`)** is served by the API container's direct
  Messages-API call, routed through the same proxy, so `api.anthropic.com` must
  stay on the allowlist. Under codex the passthrough calls `api.openai.com`
  instead, which must stay allowlisted for the same reason.
- **`http(s)` `image_url` fetches** leave the API container and are governed by
  the allowlist too — which turns the SSRF surface noted under
  [Multimodal input](#multimodal-input) into a policy decision.
- **The host-credentials overlay does not apply here.**
  `docker-compose.host-credentials.yml` targets the `claude-wrapper` service, but
  the CLI runs in `claude-agent` under this topology, so the overlay is inert.
- The agent's isolation is enforced by network topology (hard); the API
  container's egress discipline is proxy-env only (soft), because its inbound port
  needs a non-internal network. Firewall the host if you want both hard.
- The squid image (`docker.io/ubuntu/squid:latest`), the Claude Code CLI and
  the codex CLI (when the image is built with `INSTALL_CODEX=1`) are all
  unpinned. Pin them yourself if you need reproducibility.

---

## Single-container layout (sunset)

The original one-container deployment survives as `docker-compose.single.yml`
for local development and as a rollback path. It is **no longer the supported
way to ship**: no network isolation for tool use, no egress allowlist, and new
deployment features land in the default sandboxed stack first.

```bash
docker compose -f docker-compose.single.yml build
docker compose -f docker-compose.single.yml run --rm -it claude-wrapper login
docker compose -f docker-compose.single.yml up -d
```

The [host-credentials overlay](docker-compose.host-credentials.yml) applies to
this layout only:

```bash
docker compose -f docker-compose.single.yml -f docker-compose.host-credentials.yml up -d
```

`CLAUDE_WRAPPER_AGENT` works in this layout too — here the env var alone
selects the agent, since there is no per-agent compose file. Codex support is
built for the sandboxed stack; running it single-container additionally
requires `CLAUDE_WRAPPER_SESSION_PLAN=off` (the token cap's plan presets are
Anthropic-shaped — see
[Defaults that differ](#defaults-that-differ-between-code-and-compose)) and an
image built with `--build-arg INSTALL_CODEX=1`. Either way the layout remains
sunset.

## Configuration reference

Everything is environment-driven. `.env.example` is the annotated master copy;
this table is the code-level truth. Booleans are false only for
`0`/`false`/`no`/`off`/`disabled` (case-insensitive); anything else is true.

### Identity, paths, ports

| Variable | Purpose | Default |
| --- | --- | --- |
| `CLAUDE_UID` / `CLAUDE_GID` | Build arg + runtime uid/gid. | `1000` |
| `CLAUDE_WRAPPER_PORT` | uvicorn port, published host port, healthcheck. | `8000` |
| `CODEX_WRAPPER_PORT` | Same, for the codex stack (`docker-compose.codex.yml` feeds it into `CLAUDE_WRAPPER_PORT`). A separate variable so one `.env` can run both stacks side by side without a host-port collision. | `8001` |
| `CLAUDE_WRAPPER_HOST` | uvicorn bind address. | `0.0.0.0` |
| `CLAUDE_WRAPPER_WORKERS` | `uvicorn --workers`. **Leave at 1** — see [Concurrency](#concurrency). | `1` |
| `CLAUDE_INBOX_DIR` | Host drop folder → `/data/inbox` (read-only). | `./inbox` |
| `CLAUDE_WRAPPER_DATA` | Root data dir. | `/data` |
| `CLAUDE_WRAPPER_WORKSPACE` | Per-conversation CLI workspaces. | `$DATA/workspace` |
| `CLAUDE_WRAPPER_FILES` | Blob store. | `$DATA/files` |
| `CLAUDE_WRAPPER_SESSIONS` | Session registry (usage ledgers in `usage/` beneath it). | `$DATA/sessions` |
| `CLAUDE_CONFIG_DIR` | Where the CLI keeps all its state. | `/home/claude/.claude` |
| `CLAUDE_WRAPPER_CLAUDE_BIN` | Path or name of the `claude` executable. | `claude` |

The four path variables are baked into the image; `entrypoint.sh` dereferences
them with no fallback, so a stripped environment aborts at start.

### Auth

| Variable | Purpose | Default |
| --- | --- | --- |
| `CLAUDE_WRAPPER_API_KEYS` | Comma-separated bearer tokens. Non-empty turns auth on for every `/v1/*` route. | blank (auth off) |
| `ANTHROPIC_API_KEY` | Anthropic API key for the CLI and the tool bridge. | blank |
| `CLAUDE_CODE_OAUTH_TOKEN` | Pre-minted OAuth token. | blank |
| `CLAUDE_HOST_CREDENTIALS` | Host `~/.claude/.credentials.json` for the overlay. Compose errors if the overlay is used and this is unset. | none |
| `CLAUDE_WRAPPER_CREDENTIALS_FILE` | Where the tool bridge reads the CLI's OAuth token. | `~/.claude/.credentials.json` |
| `CLAUDE_WRAPPER_AGENT` | Which agent the wrapper drives: `claude` \| `codex`. Any other value refuses to boot. | `claude` |
| `OPENAI_API_KEY` | OpenAI API key for the codex CLI and the tools passthrough (codex only). | blank |
| `CLAUDE_WRAPPER_CODEX_BIN` | Path or name of the `codex` executable. | `codex` |
| `CLAUDE_WRAPPER_CODEX_CREDENTIALS_FILE` | Where the wrapper reads `codex login`'s auth state. | `$CODEX_HOME/auth.json` |
| `CLAUDE_WRAPPER_OPENAI_BASE_URL` | OpenAI API base for the tools passthrough (codex only). | `https://api.openai.com` |
| `CLAUDE_WRAPPER_CODEX_MODELS` | Comma-separated override of the advertised codex model list. | built-in list |

Note `OPENAI_API_KEY` and the `CODEX_REFRESH_*` refresher knobs (see
`.env.example`) have no `CLAUDE_WRAPPER_` prefix.

### Model and effort

| Variable | Purpose | Default |
| --- | --- | --- |
| `CLAUDE_WRAPPER_DEFAULT_MODEL` | Model for `"model": "auto"` or an absent model. Always added to the advertised list. | `claude-opus-4-8` |
| `CODEX_WRAPPER_DEFAULT_MODEL` | Same, for the codex stack (`docker-compose.codex.yml` feeds it into `CLAUDE_WRAPPER_DEFAULT_MODEL`). A separate variable so the shared `.env`'s Claude id doesn't become the codex default — which would advertise a Claude model on `/v1/models` and fail every `auto` request. Empty = `gpt-6-astra`. | blank |
| `CLAUDE_WRAPPER_EFFORT` | Server-default reasoning effort. Empty means the `--effort` flag is not passed at all. | code: empty · compose: `medium` |
| `CLAUDE_WRAPPER_MODEL_DISCOVERY` | `auto` scans the installed CLI binary; `off` serves the static fallback list. | `auto` |
| `CLAUDE_WRAPPER_ANTHROPIC_BASE_URL` | Messages API base for the tool bridge. | `https://api.anthropic.com` |
| `CLAUDE_WRAPPER_TOOLS_MAX_TOKENS` | `max_tokens` when a tool-bridge client omits it. | `8192` |

### Requests, uploads, timeouts

| Variable | Purpose | Default |
| --- | --- | --- |
| `CLAUDE_WRAPPER_MAX_UPLOAD_BYTES` | Upload ceiling; over it returns 413. | `2147483648` (2 GiB) |
| `CLAUDE_WRAPPER_REQUEST_TIMEOUT` | Per-request / CLI-read timeout, seconds. | `1800` |
| `CLAUDE_WRAPPER_PDF_INLINE_MAX_CHARS` | `>0` inlines extracted PDF text into the prompt; `0` hands Claude the file path instead. | `0` (off) |

These three parse with a bare `int()` — a malformed value raises at import and
the process never starts. Every other numeric variable falls back to its default.

### Generated-file download links

| Variable | Purpose | Default |
| --- | --- | --- |
| `CLAUDE_WRAPPER_PUBLIC_BASE_URL` | Absolute base for download links. Wins over derivation. | blank |
| `CLAUDE_WRAPPER_DERIVE_BASE_URL` | Derive the base from `Host` / `X-Forwarded-*`. | `on` |
| `CLAUDE_WRAPPER_DOWNLOAD_SIGNING_KEY` | HMAC key for capability links. If blank and API keys exist, derived from them via scrypt. | derived |
| `CLAUDE_WRAPPER_DOWNLOAD_URL_TTL` | Link lifetime in seconds; `0` = never expires (still signed). | `2592000` (30 days) |
| `CLAUDE_WRAPPER_WORKSPACE_HINT` | Tell Claude its working directory is delivered to the user. | code: **off** · compose: `on` |
| `CLAUDE_WRAPPER_WORKSPACE_PROMPT` | Override the injected hint text. | built-in |

### Streaming and reasoning

| Variable | Purpose | Default |
| --- | --- | --- |
| `CLAUDE_WRAPPER_REASONING_CHANNEL` | Where reasoning/progress frames go: `details`, `reasoning_content`, `think_tags`, `none`. | `details` |
| `CLAUDE_WRAPPER_SSE_SHOW_ACTIVITY` | Surface tool/subagent activity as reasoning frames. | `true` |
| `CLAUDE_WRAPPER_SSE_HEARTBEAT` | Seconds between keep-alive SSE comments. | `15` |
| `CLAUDE_WRAPPER_SSE_PREAMBLE_BYTES` | One-time comment padding to flush buffering proxies; `0` disables. | `2048` |
| `CLAUDE_WRAPPER_SSE_PROGRESS_SECONDS` | Silence before a visible "still working" tick; `0` disables. | `25` |
| `CLAUDE_WRAPPER_STREAM_PARTIAL` | Pass `--include-partial-messages` for token-by-token deltas. | `true` |

`details` is the default because Open WebUI's OpenAI provider renders **neither**
the `reasoning_content` field nor inline `<think>` tags — it does render a
`<details type="reasoning">` block embedded in the content.

### Clarification protocol

| Variable | Purpose | Default |
| --- | --- | --- |
| `CLAUDE_WRAPPER_CLARIFY` | Inject the ask-then-stop protocol on chat/responses. | `on` |
| `CLAUDE_WRAPPER_CLARIFY_PROMPT` | Override the injected instruction. | built-in |
| `CLAUDE_WRAPPER_CLARIFY_DISALLOWED_TOOLS` | Comma-separated tools passed to `--disallowedTools`. | code: `AskUserQuestion` · compose: **empty** |

### Usage cap

| Variable | Purpose | Default |
| --- | --- | --- |
| `CLAUDE_WRAPPER_SESSION_PLAN` | `pro` \| `max 5x` \| `max 20x` \| `off`. | `max 5x` |
| `CLAUDE_WRAPPER_PRO_SESSION_TOKENS` | Pro-tier anchor; Max scales ×5 / ×20 from it. | `1500000` |
| `CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE` | Explicit allowance; `>0` overrides the plan. | `0` |
| `CLAUDE_WRAPPER_SESSION_BLOCK_PERCENT` | Block size as a percentage of the allowance; `<=0` disables. | `5` |
| `CLAUDE_WRAPPER_BUDGET_CONTINUE_KEYWORD` | Keywords that grant another block. | `continue,proceed,keep going,go on,yes` |

Plan matching is substring-based: anything containing `pro` is Pro, `20x`/`200`
is Max 20×, `5x`/`100`/`max` is Max 5×, and anything else (`off`, `none`,
`disabled`) disables the cap.

### OpenWebUI knowledge base

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENWEBUI_BASE_URL` | Master switch for the KB prompt addendum. | blank (off) |
| `OPENWEBUI_API_KEY` | Bearer for OpenWebUI's retrieval API. | blank |
| `OPENWEBUI_DEFAULT_COLLECTION` | Default knowledge-base **id** (not display name). | blank |

Note these three have no `CLAUDE_WRAPPER_` prefix.

### Sandbox topology

| Variable | Purpose | Default |
| --- | --- | --- |
| `CLAUDE_WRAPPER_AGENT_URL` | Set → runs execute in the agent container instead of a local subprocess. | blank (local) |
| `CLAUDE_WRAPPER_AGENT_TOKEN` | Shared bearer the shim requires. | blank (no check) |
| `CLAUDE_WRAPPER_AGENT_PORT` | Port the shim binds inside the agent container. | `8791` |
| `SANDBOX_EXTRA_NO_PROXY` | Extra hosts appended to the agent's `NO_PROXY`. | blank |

If you change `CLAUDE_WRAPPER_AGENT_PORT`, change the port inside
`CLAUDE_WRAPPER_AGENT_URL` to match — the number is spelled out in both places.

### Defaults that differ between code and compose

Four variables resolve differently depending on whether you run under compose
with a populated `.env` or import the app directly (the last only in the codex
compose file):

| Variable | Code default | Compose default | Consequence |
| --- | --- | --- | --- |
| `CLAUDE_WRAPPER_CLARIFY_DISALLOWED_TOOLS` | `AskUserQuestion` | empty string | Compose passes an empty value, which **overrides** the code default and stops `--disallowedTools` from being passed at all. The dead `AskUserQuestion` tool is then reachable again. |
| `CLAUDE_WRAPPER_EFFORT` | empty (no flag) | `medium` | A compose deployment pins effort at medium; a bare-Python run lets the CLI pick. |
| `CLAUDE_WRAPPER_WORKSPACE_HINT` | `false` | `on` | Off in code deliberately, because the hint changes the *shape* of the reply and `/v1/completions`, assistants runs and the batches worker share the same path. |
| `CLAUDE_WRAPPER_SESSION_PLAN` | `max 5x` | `off` (`docker-compose.codex.yml`, via `CODEX_WRAPPER_SESSION_PLAN`) | The [usage cap](#per-conversation-usage-cap-usage-checkpoint)'s plan presets are Anthropic-shaped; no ChatGPT-plan calibration exists, so the codex stack ships the cap off — on its own variable, so the shared `.env`'s `max 5x` cannot silently re-enable it. Set `CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE` for a custom cap. |

Removed and ignored: `CLAUDE_WRAPPER_JSON_SNIFF`. It is documented as removed in
`.env.example` and read nowhere in the code.

---

## Endpoints

The wrapper covers the commonly used OpenAI surface, not all of it — see
[Limitations and known gaps](#limitations-and-known-gaps) for the routes OpenAI
has that this does not. Endpoints that aren't naturally served by an LLM (audio,
images, embeddings) are implemented by having Claude Code do the work inside a
per-request workspace; see [Delegation design](#delegation-design).

Every `/v1/*` route requires `Authorization: Bearer <key>` when
`CLAUDE_WRAPPER_API_KEYS` is set, **except** the two noted below.

### Text

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | Streaming + non-streaming multimodal chat |
| `POST` | `/v1/completions` | Legacy text-prompt completion |
| `POST` | `/v1/responses` | Responses API — streaming + multi-turn chaining |
| `POST` | `/v1/embeddings` | Dense vectors |
| `POST` | `/v1/moderations` | Content classification via Claude |

`/v1/completions` accepts multiple prompts in one request (each becomes a
separate run) and returns 400 if you combine `stream` with multiple prompts.
When streaming, it emits **chat-shaped** `chat.completion.chunk` frames rather
than `text_completion` chunks — clients tolerate this in practice.

`/v1/embeddings` tries four backends in order: fastembed →
sentence-transformers → Claude-generated vectors → a deterministic hash
embedding, so a response is always produced even with no embedding library
present. `dimensions` truncation and `encoding_format: base64` are supported.

`/v1/responses` is the modern "ask and response" primitive. It accepts a string
or structured `input` (plus optional `instructions`) and returns a `response`
object whose `output_text` flattens the assistant message.

- **Streaming** (`"stream": true`) emits the typed Responses event protocol —
  `response.created` → `response.output_text.delta` … → `response.completed`.
  There is no `[DONE]` sentinel; the stream ends on the terminal
  `response.completed` / `response.failed` event.
- **Multi-turn chaining:** pass a prior response's `id` back as
  `previous_response_id` to continue the same Claude session.

### Audio

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text via faster-whisper (pip-installed on first call) + ffmpeg |
| `POST` | `/v1/audio/translations` | Speech → English text, same pipeline |
| `POST` | `/v1/audio/speech` | Text-to-speech via espeak-ng, transcoded by ffmpeg |

Transcriptions support the `json`, `verbose_json`, `text`, `srt` and `vtt`
response formats.

### Images

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/images/generations` | Claude authors an SVG → PNG via rsvg-convert (imagemagick fallback) |
| `POST` | `/v1/images/edits` | Apply a prompt + optional mask to an image |
| `POST` | `/v1/images/variations` | N imagemagick-driven variations |

`/v1/images/generations` returns `b64_json` when asked; otherwise it returns a
**relative** `url` of `/v1/files/{id}/content` plus a non-standard `file_id`
field.

### Files

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/files` | Upload (multipart `file`, `purpose`; 413 over the size limit) |
| `GET` | `/v1/files` | List records, optional `?purpose=` filter |
| `GET` | `/v1/files/{id}` | Metadata |
| `DELETE` | `/v1/files/{id}` | Delete |
| `GET` | `/v1/files/{id}/content` | Stream the blob |

`GET /v1/files/{id}/content` is the one route with **dual auth**: a normal API
key *or* a valid unexpired `?exp=…&sig=…` capability link. Listing, metadata and
delete always require a real key.

### Batches

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/batches` | Create from an uploaded JSONL `input_file_id` |
| `GET` | `/v1/batches` | List |
| `GET` | `/v1/batches/{id}` | Retrieve |
| `POST` | `/v1/batches/{id}/cancel` | Mark `cancelling` |

Only `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
`/v1/responses` and `/v1/moderations` are permitted as batch endpoints; anything
else is a 400. Execution starts immediately in-process via `asyncio.create_task`
— this is **not a durable queue**, and in-flight batches are lost on restart.

### Fine-tuning (stubs)

| Method | Path | Behavior |
| --- | --- | --- |
| `POST` | `/v1/fine_tuning/jobs` | **501** — Claude isn't user-tunable; the body points you at `/v1/assistants` |
| `GET` | `/v1/fine_tuning/jobs` | **200** with an empty list |
| `GET` | `/v1/fine_tuning/jobs/{id}` | **404** |
| `POST` | `/v1/fine_tuning/jobs/{id}/cancel` | **501** |
| `GET` | `/v1/fine_tuning/jobs/{id}/events` | **200** with an empty list |

### Assistants, threads, runs

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/assistants` | Create a saved (model, instructions) config |
| `GET` | `/v1/assistants` | List |
| `GET` | `/v1/assistants/{id}` | Retrieve |
| `POST` | `/v1/assistants/{id}` | Modify (OpenAI's POST-to-update idiom) |
| `DELETE` | `/v1/assistants/{id}` | Delete |
| `POST` | `/v1/threads` | Create, optionally seeded with `messages` |
| `GET` | `/v1/threads/{id}` | Retrieve |
| `DELETE` | `/v1/threads/{id}` | Delete thread + messages |
| `POST` | `/v1/threads/{id}/messages` | Append |
| `GET` | `/v1/threads/{id}/messages` | List |
| `POST` | `/v1/threads/{id}/runs` | Create **and synchronously execute** a run |
| `GET` | `/v1/threads/{id}/runs` | List runs |
| `GET` | `/v1/threads/{id}/runs/{run_id}` | Retrieve a run |

`POST /v1/threads/{id}/runs` blocks until the run finishes and returns an
already-terminal run object, so OpenAI's poll-until-`completed` loop is a no-op
here. Runs always use the thread id as the session key.

### Vector stores

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/vector_stores` | Create, optionally ingesting `file_ids` |
| `GET` | `/v1/vector_stores` | List |
| `GET` | `/v1/vector_stores/{id}` | Retrieve |
| `DELETE` | `/v1/vector_stores/{id}` | Delete |
| `POST` | `/v1/vector_stores/{id}/files` | Index a file (chunk → embed → append) |
| `GET` | `/v1/vector_stores/{id}/files` | List indexed files |
| `DELETE` | `/v1/vector_stores/{id}/files/{vsfile_id}` | Detach a file entry |
| `POST` | `/v1/vector_stores/{id}/search` | Cosine similarity search |

Detaching a file removes its entry but **not its vectors** from the matrix, so
detached content can still surface in search results.

### Realtime

| Method | Path | Purpose |
| --- | --- | --- |
| `WS` | `/v1/realtime` | Text-only realtime protocol |
| `GET` | `/v1/realtime/sessions` | Discovery helper for OpenAI SDKs — **no auth** |

The WebSocket handles `session.update`, `response.create` and
`input_text.append`, and streams `response.output_text.delta` …
`response.completed`. It checks the `Authorization` header itself and closes with
code `4401` on failure. It calls the runner **directly** rather than going
through `/v1/chat/completions`, so none of the chat-endpoint behavior applies —
no tool bridge, no usage-ledger gate, no JSON mode, no generated-file trailer —
and it always uses `CLAUDE_WRAPPER_DEFAULT_MODEL`, ignoring any model the client
sends.

### Models, usage, health

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/models`, `/v1/models/{id}` | Advertise supported Claude models |
| `GET` | `/v1/usage/{session_id}` | Per-conversation token spend and remaining allowance |
| `GET` | `/healthz` | Liveness probe — **no auth** |

FastAPI's `/docs`, `/redoc` and `/openapi.json` are also served, and are
**unauthenticated**. Block them at your reverse proxy if that matters to you.

### Non-standard request and response fields

`ChatCompletionRequest` allows extra fields; these are the meaningful ones:

| Field | Effect |
| --- | --- |
| `session_id` | Pin the conversation explicitly instead of deriving it |
| `inline_generated_files` | Return generated bytes inline as base64 (non-streaming only) |
| `clarify` | `false` opts this one request out of the clarification protocol |

Responses echo `session_id`, and carry an `effort` object
(`{"applied", "source", "requested"}`) so clients can confirm which reasoning
effort actually took effect. While streaming, `effort` rides the first chunk.

### Delegation design

Non-text endpoints delegate to Claude Code through a per-request workspace:

1. The caller writes input bytes into `workspace/uploads/`.
2. It invokes Claude with a structured prompt naming the exact output file(s) to
   produce under `workspace/outputs/`.
3. Claude uses Bash / Read / Write to do the work — pip-install faster-whisper,
   invoke ffmpeg or espeak-ng, render an SVG with rsvg-convert.
4. The caller collects every file under `outputs/` and packages them in the
   endpoint's OpenAI-shaped response.

Each delegated request gets a fresh workspace keyed by a random id, so
concurrent requests never collide. Heavy dependencies are installed lazily on
first use and cached for later calls, which is why a cold call to
`/v1/audio/transcriptions` is slow and later ones are not.

---

## Chat features

### Structured output (`response_format`)

`/v1/chat/completions` and `/v1/responses` honor OpenAI's `response_format`. With
`{"type": "json_object"}` or `{"type": "json_schema", "json_schema": {…}}` the
wrapper appends a raw-JSON-only instruction to the prompt (including the schema
for `json_schema`) and reduces the reply to the bare JSON value before it leaves
the wrapper — markdown fences and surrounding prose are stripped, and the
file-reference trailer is suppressed. Streamed JSON-mode requests buffer the
answer (fences can span chunk deltas) and deliver the cleaned JSON as a single
content chunk; reasoning/progress frames are suppressed so the concatenated
content is pure JSON.

This is what clients built on the Vercel AI SDK's `generateObject` expect.
Requests without `response_format` (or with `{"type": "text"}`) are completely
unaffected. If the model produces no parseable JSON at all, the reply passes
through unchanged rather than masking what it said.

One interaction worth knowing: the instant replies described under
[usage cap](#per-conversation-usage-cap-usage-checkpoint) are wrapper-authored
prose, which would corrupt a structured-output client. In JSON mode they return
**HTTP 502** instead of the prose message.

### Streaming feedback on long runs

At high/max/ultracode effort Claude may think or run tools for many minutes
before the first answer token. To keep streams alive and visibly *working* —
rather than looking stalled, or being severed by a buffering proxy before any
headers reach the client — the wrapper:

- sends a one-time comment preamble to flush the response head past buffering
  proxies immediately;
- emits keep-alive comments, plus periodic **visible** "⏳ Still working…"
  ticks during silence;
- surfaces **tool/subagent activity** (`🔧 Bash: …`, `🔧 Read: …`).

Progress rides the reasoning channel, never the answer content. Tune it with the
`CLAUDE_WRAPPER_SSE_*` and `CLAUDE_WRAPPER_REASONING_CHANNEL` variables; set
`CLAUDE_WRAPPER_SSE_PROGRESS_SECONDS=0` and
`CLAUDE_WRAPPER_SSE_SHOW_ACTIVITY=false` to quiet it, or
`CLAUDE_WRAPPER_REASONING_CHANNEL=none` to suppress reasoning frames entirely.

### Clarifying questions (interactive)

Headless Claude Code has no interactive question UI: its `AskUserQuestion` card
is auto-dismissed, so Claude proceeds on assumptions and you never get to answer.
With the clarification protocol on (default), the wrapper:

- injects a system prompt (`--append-system-prompt`) teaching Claude that, when a
  decision genuinely changes the result, it should ask its questions as plain
  numbered text **and stop** — making the questions the whole turn — so you
  answer in your next message and the session resumes automatically;
- disables the dead interactive tool via `--disallowedTools AskUserQuestion`, so
  questions always arrive as answerable text.

Only chat and responses opt in; delegated task endpoints never pause. Disable
globally with `CLAUDE_WRAPPER_CLARIFY=off`, override the instruction with
`CLAUDE_WRAPPER_CLARIFY_PROMPT`, or opt a single request out with
`"clarify": false`.

Note the compose caveat in [Defaults that differ](#defaults-that-differ-between-code-and-compose):
the tool-suppression half of this only takes effect if
`CLAUDE_WRAPPER_CLARIFY_DISALLOWED_TOOLS` is actually set in your `.env`.

Under codex, both this protocol and the workspace hint travel as a leading
prompt block instead — `codex exec` has no `--append-system-prompt` — with the
same enable conditions. The `--disallowedTools AskUserQuestion` half is
Claude-only and moot for codex, which has no interactive question tool to
suppress.

### If your client sends `tools` (Open WebUI native function calling)

A request that declares `tools` is served by the **tool bridge**, which calls the
Anthropic Messages API directly — no Claude Code CLI, so no workspace, no `Write`
tool, and **no generated files on those turns**. Open WebUI in *Native* function
calling mode sends its whole tool roster on every message, which silently
disables file downloads for the entire chat.

The fix is client-side: in Open WebUI set the model's **Function Calling**
setting from **Native** to **Default**. Open WebUI then runs its own
tool-selection call and sends no `tools[]` on the completion, so the request
takes the CLI path — downloads work, and function calling stays intact.

There is deliberately no wrapper-side switch. A single turn cannot be served both
ways: the CLI runs its own tool loop and cannot surface a caller-declared tool.
Routing a tools-carrying turn to the CLI anyway would silently drop the client's
tools, so a real agentic client offering a tool on its opening turn would get
prose instead of a `tool_call` and its loop would stall.

The routing rule is agent-independent — tools or the CLI, never both in one
turn. Under codex the bridge is a near-pure proxy to OpenAI's own
`/v1/chat/completions`: native `response_format`, client-declared OpenAI
built-in tools and `tool_choice` pass through verbatim, and legacy `functions`
clients keep the legacy wire shape end-to-end. The wrapper-owned `memory` /
`time_calc` tools do not exist on this path, and tool-path streams carry no
`session_id` / `effort` extension fields — upstream chunks are forwarded
as-is.

### OpenWebUI knowledge base

Set `OPENWEBUI_BASE_URL` (plus `OPENWEBUI_API_KEY`, and optionally
`OPENWEBUI_DEFAULT_COLLECTION`) and the wrapper appends a **knowledge-base
addendum** to the system prompt on every CLI-path turn. This is not a retrieval
integration inside the wrapper — it teaches Claude to query OpenWebUI's RAG API
itself with its own Bash/curl tools:

1. **Discover:** `GET $OPENWEBUI_BASE_URL/api/v1/knowledge/` to list knowledge
   bases and their ids.
2. **Query:** `POST $OPENWEBUI_BASE_URL/api/v1/retrieval/query/collection` with
   `{"collection_names": ["<id>"], "query": "…", "k": 5}`.

The addendum is explicit that `collection_names` takes the knowledge base's
**id, not its display name** — the single most common cause of empty results.
It also instructs Claude to search before answering rather than answering from
memory, and to cite the chunks it used.

Secrets are never interpolated into the prompt: only the `$OPENWEBUI_*` variable
*names* appear in the injected text, and the values reach Claude through the
subprocess environment. The addendum applies to chat, completions, responses,
assistants runs and the batches worker; it does not apply to the delegated
endpoints or the tool-bridge path. Startup logs report whether it is enabled and
warn if the base URL is set without a key (every call would 401).

---

## Files in and out

### Multimodal input

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
- `file-<id>` — a reference to something uploaded to `/v1/files`.

`file_data` parts accept both a data URL and raw base64 with the type in a
sibling `mime_type` field; an explicit `mime_type` wins over the data URL's own.

> **SSRF note.** The `http(s)` form makes the *server* fetch whatever URL the
> client names — including hosts on your internal network the client couldn't
> reach directly, reachable by anyone holding an API key. Fine when the
> container's egress is restricted or its callers are trusted; if neither holds,
> front the wrapper with an egress allowlist (see
> [Sandboxed deployment](#sandboxed-deployment-network-isolated-agent)) or strip
> `image_url` parts at your gateway.

Uploaded and inlined binaries are written into the per-session workspace before
Claude Code is invoked, so Claude can open them with its `Read` tool — images,
PDFs, audio and video (use ffmpeg inside the container for the last).

### PDFs

By default a PDF is handed to Claude as a **file path** and Claude reads the
pages it needs. Set `CLAUDE_WRAPPER_PDF_INLINE_MAX_CHARS` above zero to inline
extracted text into the prompt instead, wrapped in `<<<PDF-START …>>>` /
`<<<PDF-END …>>>` markers with per-page `--- page N ---` separators. Extraction
stops once it is over budget, so a 900-page book is not fully materialized just
to be truncated, and the header line tells Claude it was truncated and that the
full file is still readable at the given path. A per-page extraction error
becomes an inline note rather than failing the request.

`tools/split_pdf.py` slices a PDF by page range or outline chapter host-side if
you would rather send only the relevant pages.

### Host drop folder (`/data/inbox`)

Some clients (Open WebUI in particular) extract a document to text before it ever
reaches the wrapper, so Claude sees the client's transcription rather than the
file. The inbox sidesteps that: whatever you put in `CLAUDE_INBOX_DIR` on the
host appears read-only inside the container at `/data/inbox`, and Claude reads
the real bytes.

```bash
cp report.pdf ./inbox/          # CLAUDE_INBOX_DIR on the host
```

Then name the path in chat: *"Read /data/inbox/report.pdf and summarise section 3."*

Two requirements, both easy to get wrong:

- **The host path must already exist.** A missing bind source is created as a
  root-owned directory, which the container then can't read.
- **It must be readable by `CLAUDE_UID`.**

The mount is read-only, so Claude can't modify or delete anything you drop there.
Under the sandbox topology the inbox mounts on the **agent** container, since
that is where the CLI runs.

### Uploading a file

```bash
curl -X POST http://localhost:8000/v1/files \
    -F 'file=@clip.mp4' \
    -F 'purpose=user_data'
```

Then reference the returned `id`:

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

### Generated files (Claude writes binaries back)

When Claude writes a file into the session workspace, the wrapper:

1. Registers it with `/v1/files` (`purpose=assistant_output`).
2. Appends a "Generated files" block to the assistant message listing each file
   as a markdown download link (mime, size, `file_id`).
3. If the request sets `"inline_generated_files": true`, includes the raw bytes
   as base64 under `message.attachments[*].content_base64` (non-streaming chat
   only — there is no field to carry them while streaming).

Attachment objects also carry a non-standard `url` key so SDK clients don't have
to parse markdown.

Files are excluded from delivery if they live under `uploads/` (the user's own
attachments) or under any dot-prefixed path — so `.scratch/` is a free channel
for intermediate work.

#### Making the link clickable

Two things have to be true, and both have safe defaults:

- **An absolute URL.** `CLAUDE_WRAPPER_PUBLIC_BASE_URL` wins whenever set.
  Otherwise the wrapper derives the base from the request's own `Host` /
  `X-Forwarded-Host` / `X-Forwarded-Proto` headers, which is correct behind a
  typical reverse proxy and needs no configuration. Forwarded headers are trusted
  deliberately: uvicorn's `--proxy-headers` only trusts `forwarded_allow_ips`,
  which a containerized reverse proxy is not, so without this a TLS-terminated
  deployment emits `http://` links that browsers block as mixed content. A forged
  `Host` only poisons links in the forger's own reply, because the download
  signature does not cover the host. Set `CLAUDE_WRAPPER_DERIVE_BASE_URL=off` for
  non-clickable `→ file_id=…` text instead.
- **Auth the browser can satisfy.** A link click sends no `Authorization` header,
  so with `CLAUDE_WRAPPER_API_KEYS` set every download would 401. Each link
  carries `?exp=…&sig=…`: an HMAC over exactly that one file id and expiry, with
  the expiry inside the MAC so a holder can neither extend it nor forge a
  never-expires link. It grants reading that one blob — listing, metadata and
  delete still require a real API key, and no API key appears in chat text.

  The signing key defaults to one derived from `CLAUDE_WRAPPER_API_KEYS` via
  **scrypt**, deliberately expensive because a published link is a known MAC pair
  over that key and a cheap hash would make any leaked link an offline oracle for
  your API keys. Set `CLAUDE_WRAPPER_DOWNLOAD_SIGNING_KEY` explicitly if you want
  the link secret independent of your API keys and stable across key rotation.
  Links expire after `CLAUDE_WRAPPER_DOWNLOAD_URL_TTL` seconds (default 30 days;
  `0` = never).

  Signature verification is gated on `CLAUDE_WRAPPER_API_KEYS` being set. A
  signing key with no API keys stamps signatures nobody checks — the boot log
  warns about exactly this.

#### Getting Claude to write a file at all

`CLAUDE_WRAPPER_WORKSPACE_HINT` tells Claude that its working directory is
delivered to the user, so it writes a requested document or dataset to a file
instead of pasting the whole thing into the reply. Without it Claude has no way
to know the file goes anywhere, and mostly won't create one.

**It is off in code and on in the shipped compose files.** Off is the right
default for programmatic callers: it changes the shape of the reply, and
`/v1/completions`, assistants runs and the batches worker share the same path, so
a script doing `completions.create("write me a python script…")` and reading
`choices[0].text` would silently start getting "I wrote it to script.py" plus a
link. Turn it on for chat-UI deployments. It is forced off for JSON-mode
requests, where the client wants the value in the reply body. Override the text
with `CLAUDE_WRAPPER_WORKSPACE_PROMPT`.

---

## Conversation continuity

OpenAI's Chat Completions API is stateless — the client sends the full history
every request. The wrapper mirrors that by hashing the conversation's **anchor**
(its system/developer messages plus the **first** user turn) and mapping that
hash to a Claude Code session id, which is resumed on follow-up turns. Hashing
only the anchor, rather than the whole transcript, is what makes every turn of
the same conversation land on the same session key so `--resume` works
indefinitely.

To pin a session explicitly, pass `session_id` in the request body:

```json
{
  "model": "claude-sonnet-4-6",
  "session_id": "my-assistant-for-alice",
  "messages": [ ... ]
}
```

The response body echoes `session_id` so clients can round-trip it.

### Self-healing resume

If Claude Code's transcript for a session disappears while the wrapper's
`session_key → uuid` mapping survives, `claude --resume <uuid>` fails identically
on every retry and the conversation is bricked. The failure string is not
"session not found" — it is `error_during_execution`, often with exit code 0 —
so a naive error match never fires.

The wrapper therefore drops the mapping whenever a resumed turn ends with an
error and produced **no assistant text at all**. The next turn mints a fresh
session id and replays the full transcript, costing one extra full-history turn
but recovering the conversation. A resume that streamed a real answer before a
late error is deliberately left intact.

---

## Per-conversation usage cap (usage checkpoint)

Every request spends your Anthropic session/subscription quota, and a single long
conversation — especially at `max`/`ultracode` effort — can eat a large slice of
it with no warning. The wrapper caps that **per conversation** and asks before
spending more.

**The cap is ON by default at the Max 5× ($100) plan** (allowance 7,500,000,
checkpoint every 375,000 tokens). Override it by plan, by an explicit number, or
turn it off:

```bash
# (a) By subscription plan — pro | max 5x ($100) | max 20x ($200)  [default: max 5x]
CLAUDE_WRAPPER_SESSION_PLAN="max 5x"
CLAUDE_WRAPPER_PRO_SESSION_TOKENS=1500000        # Pro anchor; Max scales 5x/20x from it

# (b) Or set the allowance directly (this wins over the plan)
CLAUDE_WRAPPER_SESSION_TOKEN_ALLOWANCE=0         # 0 = use the plan

# Disable entirely:
CLAUDE_WRAPPER_SESSION_PLAN=off

CLAUDE_WRAPPER_SESSION_BLOCK_PERCENT=5
CLAUDE_WRAPPER_BUDGET_CONTINUE_KEYWORD=continue,proceed,keep going,go on,yes
```

> ⚠️ **The per-plan token figures are estimates.** Anthropic does not publish a
> token number for the Pro/Max *session* windows, and the wrapper can't query it.
> What's defined is the *relationship*: Max is "5×" ($100) and "20×" ($200) of
> Pro. So the plan setting anchors on `CLAUDE_WRAPPER_PRO_SESSION_TOKENS` and
> scales from there. The default anchor (1,500,000) is calibrated from real usage
> — a heavy ~2h Claude Code session measured ~1.54M billable tokens, reported as
> 21% of a Max-5× window → ~7.5M per window → ~1.5M Pro anchor. Tune it to your
> own usage; this is a safety checkpoint, not a mirror of Anthropic's accounting.
> On startup the wrapper logs the resolved plan / allowance / block.

| Plan setting | Multiplier | Allowance (default anchor 1.5M) | Block @ 5% |
|---|---|---|---|
| `pro` / `pro $20` | 1× | 1,500,000 | 75,000 |
| `max 5x` / `max $100` **(default)** | 5× | 7,500,000 | 375,000 |
| `max 20x` / `max $200` | 20× | 30,000,000 | 1,500,000 |
| `off` / `none` | — | disabled | — |

How it works:

- The wrapper tracks input + output tokens spent by each conversation, in a JSON
  ledger under `sessions/usage/`.
- A **block** is `allowance × percent`. Each conversation starts with one block
  of headroom.
- Once a conversation has spent its current block, the **next** request doesn't
  call Claude — it returns a checkpoint message:
  > ⏸️ **Usage checkpoint.** This conversation has used **52,000 tokens**,
  > reaching its **50,000-token** budget block (5% of the configured session
  > allowance). Reply **continue** to allow another block, or start a new chat
  > to reset.
- Replying with a continue keyword grants one more block and proceeds. The
  keyword matches as the whole message *or* as a leading/trailing word, so
  "yes, continue" and "continue please" both work. A **new chat** starts fresh.

Note the block size is recomputed from current config on every read rather than
stored, so changing `CLAUDE_WRAPPER_SESSION_BLOCK_PERCENT` reinterprets existing
ledgers.

### Checking usage: `stats` / `context`

Send **`stats`** or **`context`** (or `/stats`, `/context`) as the whole chat
message and the wrapper answers instantly, **without spawning Claude** — even
while the conversation is paused at a checkpoint, and at zero token cost:

> 📊 **Usage stats**
> - **Spent (this conversation):** 412,300 tokens across 7 requests (5.5% of the session allowance)
> - **Remaining before the next checkpoint:** 337,700 of 750,000 tokens (2 × 375,000-token blocks)
> - **Session allowance:** 7,500,000 tokens (max_5x plan), 5% per block
> - **Session key:** `conv-1a2b3c…`

Only a message that *is* the command triggers it (surrounding punctuation and a
leading `/` are tolerated) — a prompt that merely mentions "stats" goes to Claude
as usual. The same numbers are available at `GET /v1/usage/{session_id}`:

```bash
curl -fsS -H "Authorization: Bearer $KEY" http://localhost:8000/v1/usage/conv-1a2b3c | jq
```

Because the check happens before Claude is spawned, a paused conversation costs
nothing until you confirm.

---

## Models and reasoning effort

`/v1/models` lists the Claude models the wrapper accepts. The list is **built
once at startup by scanning the installed Claude Code binary** for the model ids
it ships with, so it tracks whatever CLI version is installed instead of a
hardcoded set. A maintained denylist drops models Anthropic has deprecated
(deprecation isn't encoded in the binary). Set
`CLAUDE_WRAPPER_MODEL_DISCOVERY=off` to serve a static built-in list; discovery
also falls back to that list automatically if the binary can't be read. Pass
`"model": "auto"` to use `CLAUDE_WRAPPER_DEFAULT_MODEL`.

Under `CLAUDE_WRAPPER_AGENT=codex` there is no binary scan —
`CLAUDE_WRAPPER_MODEL_DISCOVERY` is claude-only. `/v1/models` serves a static
codex list, overridable with `CLAUDE_WRAPPER_CODEX_MODELS`, and every entry
carries `owned_by: openai`. `"model": "auto"` resolves to `gpt-6-astra`
unless `CLAUDE_WRAPPER_DEFAULT_MODEL` says otherwise.

Each effort-capable model is advertised with one variant per effort level it
accepts (the *family rule*):

| Family | Advertised efforts |
| --- | --- |
| Opus 4.5+ and the `fable` / `mythos` codename families | `(low)` `(medium)` `(high)` `(xhigh)` `(max)` `(ultracode)` |
| Sonnet 4.6+ | `(low)` `(medium)` `(high)` `(xhigh)` |
| Haiku and older models | none |

Selecting `claude-opus-4-8 (xhigh)` sets the per-request effort. A
`claude-opus-4-8:xhigh` shorthand also parses. A bare model id uses the server
default (`CLAUDE_WRAPPER_EFFORT`), and an explicitly empty effort means no
`--effort` flag is passed at all, letting the CLI choose. The `[1m]`
long-context suffix passes through the effort machinery untouched.

Responses report what actually happened in the `effort` field —
`{"applied": "xhigh", "source": "request", "requested": "xhigh"}` — where
`source` is one of `request`, `server-default`, `model-incapable` or
`effort-unsupported`. That is how you confirm a variant took effect rather than
being silently dropped for a model that doesn't support it.

Codex models advertise a different variant set: `(minimal)` `(low)` `(medium)`
`(high)` `(xhigh)` on every id — effort is a request parameter there, not
model-gated the way `claude --effort` is — and never `(max)` or `(ultracode)`,
which resolve as `effort-unsupported`. An explicit `:none` suffix is honored
(it reaches the CLI instead of silently falling back to codex's default)
though never advertised. The server default (`CLAUDE_WRAPPER_EFFORT`) applies
to CLI turns but **not** to the tools passthrough, which maps only the
explicit per-request suffix.

The `(ultracode)` variant is special: it requests xhigh effort **plus** Claude
Code's dynamic-workflow (multi-agent) orchestration. Because ultracode is gated
on dynamic workflows being enabled — and that setting defaults off in a headless
container — the wrapper turns it on in the same overlay
(`--settings '{"enableWorkflows": true, "ultracode": true}'`) and passes no
`--effort` flag. An org-policy `disableWorkflows` or an account-level launch gate
can still override this; those are account-side and cannot be set by the wrapper.

---

## Per-model capability profiles

The wrapper decides per model what is exposed and enforced, from one profile
registry (`src/capabilities.py`). Every `/v1/models` entry carries its
resolved profile as a `capabilities` list — the machine-readable source of
truth for UIs and sync tooling. **Absent any configuration this whole feature
is a no-op**: the built-in default reproduces the wrapper's classic behavior,
with one deliberate exception (terminal, below).

Capabilities: `vision`, `file_upload`, `web_search`, `code_interpreter`,
`terminal`, `memory`, `citations`, `image_generation`, `sub_agents`,
`client_tools`, `time_calc`.

| Variable | Meaning |
| --- | --- |
| `CLAUDE_WRAPPER_MODEL_PROFILES` | Path to the profile JSON (see `deploy/model-profiles.example.json`). Invalid config fails startup loudly. |
| `CLAUDE_WRAPPER_MODEL_PROFILE_OVERRIDES` | Inline JSON, same schema, applied after the file. |
| `CLAUDE_WRAPPER_EXPOSE_TERMINAL` | **Default off.** Hard gate for `terminal`: until set, the capability is masked out of every profile — a profile file alone can never hand a chat UI a shell — and chat runs carry `--disallowedTools Bash`. Set it to restore the classic everything-on behavior. |
| `CLAUDE_WRAPPER_BRIDGE_WEB_SEARCH` | Default off. Lets the tool bridge inject Anthropic's server-side web search for `web_search`-profiled models (new billing surface, hence its own opt-in). |
| `CLAUDE_WRAPPER_BRIDGE_MAX_TOOL_ROUNDS` | Hybrid-loop round cap per turn (default 8). |
| `CLAUDE_WRAPPER_IMAGE_BACKEND_URL` / `_KEY` / `_MODEL` | Optional OpenAI-compatible image backend; configured, `/v1/images/generations` proxies there instead of the SVG delegation path. |

**Where the profile file lives:** in the sandbox stack (the supported
deployment) it is `sandbox/profiles.json` — bind-mounted read-only into the
wrapper at `/etc/claude-wrapper/profiles.json`, which is the compose default
for `CLAUDE_WRAPPER_MODEL_PROFILES`. It ships as a no-op `{}`; edit it in the
checkout and restart the wrapper, the same workflow as `sandbox/allowlist.txt`.
(Sunset single-container layout: drop the file in the inbox and point the env
var at `/data/inbox/profiles.json`.)

Profile schema — `default` replaces the built-in set; `models` entries apply
in order (literal match first, then glob), each either replacing
(`capabilities`) or delta-ing (`add`/`remove`); effort and `[1m]` variants
inherit their base model:

```json
{
  "models": [
    {"match": "claude-haiku-*", "remove": ["sub_agents", "web_search"]},
    {"match": "claude-opus-5", "add": ["memory", "time_calc"]}
  ]
}
```

Enforcement is layered per path:

- **CLI (chat) runs** translate the profile into `--disallowedTools`
  (`terminal`→Bash, `web_search`→WebSearch/WebFetch, `sub_agents`→Task).
  In the sandbox topology that argv crosses the agent shim to the agent
  container unchanged, so gating is identical there. Internal delegation
  runs (audio/images/embeddings do their work through Bash) are never gated.
- **Tool bridge**: a model without `client_tools` answers a tools request
  with a 400 naming the denied tools. Profiled server-side tools (web
  search, code execution) are injected after the client's. Wrapper-owned
  tools — `memory` (per-conversation markdown files under
  `<data>/memory/`, the same file-path model Claude Code itself uses) and
  `time_calc` — execute inside the bridge's hybrid loop and are invisible
  to the client. They activate only on tools-carrying requests; tool-less
  chats take the CLI path, which has its own built-ins.

Under codex, profiles still gate the tool bridge and `/v1/models`
advertisement for codex ids, but the CLI-side translations above are
Claude-tool names and do not apply. In particular
`CLAUDE_WRAPPER_EXPOSE_TERMINAL` cannot disable codex's command execution:
the shell is intrinsic to that agent, and the container sandbox is the
enforcement boundary (see
[Limitations](#limitations-and-known-gaps)). The wrapper-owned `memory` /
`time_calc` capabilities are inert under codex — the OpenAI passthrough has
no hybrid loop to execute them in.

**OpenWebUI**: OpenWebUI doesn't map pulled model metadata into its
capability toggles, so `deploy/openwebui_capability_sync.py` — run **on the
OpenWebUI host** (cron or startup hook) — pulls the wrapper's `/v1/models`
and writes the toggles through OpenWebUI's local admin API. The wrapper never
contacts OpenWebUI.

## Auth

There are two independent auth layers.

### First-time login

**`claude login` is not an authentication command.** CLI 2.1.226 has no `login`
subcommand — the word is parsed as a prompt, so the CLI opens a chat session,
writes `history.jsonl` and `sessions/`, and never touches `.credentials.json`.
The only interactive auth path is the TUI's `/login` slash command. (The
`login` entrypoint subcommand detects this and drops you into the TUI.)

For the **single-container** layout:

```bash
docker compose run --rm -it claude-wrapper login   # opens the TUI; type /login
```

For the **sandboxed** layout, the interactive flows need a browser, which the
agent container will never have — `/login` and `setup-token` both stall at
"Opening browser to sign in…" there. (If you instead see
`OAuth error: Invalid IP address: undefined`, that's a different problem:
`CLAUDE_CODE_PROXY_RESOLVES_HOSTS` is set — it's off by default for exactly
this reason; see CREDENTIALS-FIX.md Round 4.)

The reliable approach is to do the one-time login in a container with ordinary
networking, writing into the same volume the agent reads. This is a legitimate
bootstrap, not a workaround: the sandbox exists to constrain model-driven tool
use, not the operator's initial setup. The `claude-refresher` service is
already in exactly that position — ordinary networking, the volume mounted
writable, env credentials pinned empty — so borrow it and compose resolves the
volume for you:

```bash
docker compose run --rm -it claude-refresher claude
# type /login at the prompt, complete the flow, then /exit
```

After that, the running `claude-refresher` keeps the login renewed
indefinitely — see [Keeping the credential alive](#keeping-the-credential-alive).
You log in once.

If you'd rather use a raw container, get the volume name from the running
agent rather than guessing it — a `-v` naming a volume that does not exist
**creates an empty one**, so the login succeeds into a volume nothing reads:

```bash
podman inspect claude-agent \
    --format '{{range .Mounts}}{{.Name}} -> {{.Destination}}{{"\n"}}{{end}}'

podman run --rm -it -v <that-name>:/home/claude/.claude localhost/claude-wrapper:latest claude
# type /login at the prompt, complete the flow, then /exit

podman exec claude-agent stat -c '%y' /home/claude/.claude/.credentials.json
```

That last line is the check that matters: the mtime must be from just now. If it
has not moved, the credential did not land, whatever the CLI printed.

`setup-token` is **not** an alternative route into the volume. It prints a token
and nothing else — verified with `claude-home` mounted writable, where it left
`.credentials.json` untouched. Its output is for `CLAUDE_CODE_OAUTH_TOKEN`, and
setting that suppresses file refresh entirely (see
[Keeping the credential alive](#keeping-the-credential-alive)). On an
API-key-only deployment you can delete the eight `claude.ai` / `claude.com`
entries from `sandbox/allowlist.txt`, which that file's own comments recommend.

### First-time login (Codex)

The same bootstrap logic applies to the codex stack: the interactive flows
cannot complete from inside the isolated `codex-agent`, and doing the one-time
login from a container with ordinary networking, writing into the volume the
agent reads, is a legitimate bootstrap, not a workaround — the sandbox exists
to constrain model-driven tool use, not the operator's initial setup. The
`codex-refresher` service is in exactly that position:

```bash
docker compose -f docker-compose.codex.yml run --rm -it codex-refresher codex-login
# follow the device-code flow in your browser
```

`codex-login` uses the device-auth flow deliberately: codex's default browser
flow spins up a callback server on `localhost:1455`, which is unreachable in a
`run` container unless you publish the port
(`run --rm -it -p 1455:1455 codex-refresher codex login`). Device auth needs
no callback.

The check that matters afterwards is the same one as for Claude — the mtime
of the credential file must be from just now:

```bash
podman exec codex-agent stat -c '%y' /home/claude/.codex/auth.json
```

After that, the running `codex-refresher` keeps the login renewed — see
[Keeping the credential alive](#keeping-the-credential-alive). Skip all of
this if you set `OPENAI_API_KEY` in `.env`.

### 1. Claude Code → Anthropic

The two mechanisms are genuinely different, and they cannot both be active —
an environment credential wins and then nothing renews the file underneath it.

- **Interactive login (renews itself).** Run the TUI and use `/login`. Writes
  `.credentials.json` to `/home/claude/.claude/`, backed by the `claude-home`
  volume so it survives `down`/`up`, rebuilds and reboots. The access token
  lasts hours and the CLI renews it from a refresh token — good for ~30 days —
  every time it runs with working egress. This is the only credential that
  sustains itself, and the only one that lives in the volume. Renewal is a
  POST to `platform.claude.com` (allowlisted), and the CLI's refresh client is
  measured to work through the CONNECT proxy — but renewal only happens when
  the CLI *runs*, so an idle deployment still drifts past expiry with nothing
  to renew it. That is why the sandbox compose file runs a `claude-refresher`
  service: same CLI, same volume, renewing the token whenever it nears expiry
  whether or not anyone is sending turns. With that service up, log in once
  and the login sustains itself. (`tools/credential_refresh_test.sh` checks
  in-agent renewal against your own stack in about a minute.)
- **Long-lived token (static).** `claude setup-token` prints a token valid for
  about a year. It is **not** written to the volume — the output is meant for
  `CLAUDE_CODE_OAUTH_TOKEN`. Nothing renews it and nothing can read its
  expiry, so keep it as the break-glass credential rather than the primary,
  and record its mint date as `CLAUDE_CODE_OAUTH_TOKEN_MINTED=YYYY-MM-DD` in
  `.env` — that date is the boot report's only way to warn you in the last
  month of the token's ~1 year instead of letting the deployment die silently.
  Note that setting the token stops the CLI (and the refresher) renewing any
  login in the volume, which then decays; the boot report warns when it
  detects this.
- **Env vars.** Set `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` in `.env`.
  The wrapper does not arbitrate between these and a persisted login — it only
  reports what it found. Precedence is decided by the Claude Code CLI (and,
  for the tool bridge, `ANTHROPIC_API_KEY` is checked before
  `CLAUDE_CODE_OAUTH_TOKEN`, which is checked before the on-disk credentials).
- **Share the host account's login.** Point `CLAUDE_HOST_CREDENTIALS` at the
  host's `~/.claude/.credentials.json` and add the overlay:

  ```bash
  docker compose -f docker-compose.single.yml -f docker-compose.host-credentials.yml up -d
  ```

  This bind-mounts that single file read-only. Because it's one file rather than
  a directory, an in-container token refresh can't replace the inode — refreshes
  have to happen host-side, and an expired token means logging in again *on the
  host*. The file is mode 600, so `CLAUDE_UID` must match its owner. Prefer the
  container's own login for anything long-running or headless.

  This is a deliberately narrow mount. Bind-mounting the host's whole `~/.claude`
  instead co-mingles its live daemon, session locks and 700-mode sessions dir,
  which breaks `claude --resume`: the first turn of a chat succeeds and every
  follow-up fails with `error_during_execution`.

  The overlay targets `claude-wrapper` and is therefore **inert under the sandbox
  topology**, where the CLI runs in `claude-agent`.

### Codex → OpenAI

The codex counterpart of the layer above, with one asymmetry that decides what
you deploy. Precedence:

- **`OPENAI_API_KEY` (environment)** wins whenever set. As with the Claude env
  vars, this is the single most surprising behaviour: an env key makes codex
  ignore the volume login entirely, and then nothing exercises — or refreshes —
  the file credential underneath it. The boot report warns when it detects the
  shadowing.
- **`auth.json` in the `codex-home` volume** otherwise — either a ChatGPT-plan
  login (from `codex-login`, self-renewing on use) or a persisted API key
  (`codex login --with-api-key`). The wrapper container's read-only mount of
  this file **ships commented out on purpose**: a ChatGPT-plan `auth.json`
  holds an OAuth refresh token the tools passthrough deliberately never uses,
  and mounting it into the one internet-facing container would expose it for
  zero benefit. Uncomment the mount only if you persist an API key there.

**The plan-token asymmetry: a ChatGPT-plan login drives CLI turns only.
Function-calling (`tools`) requests need an API key — without one they return
a 502 (`no_upstream_credential`) naming the two working options.** Unlike
Claude, whose plan login is bridge-usable, ChatGPT-plan tokens authenticate
against the Codex backend, not the OpenAI Platform API — sending one to
`/v1/chat/completions` is both non-functional and a plan-terms violation, so
the wrapper never tries.

When OpenAI rejects the wrapper's upstream credential (401/403), clients see
a fixed 502 message; the upstream body — which can echo fragments of the
presented key — goes to the server log only.

### Keeping the credential alive

An access token always expires; what you control is whether it gets renewed
before it does. The CLI rewrites `.credentials.json` when it runs — but only
when it runs, and only if it can reach the network. So a deployment that sits
idle, or whose egress breaks, drifts past expiry with nothing in its own logs,
and then answers every turn with:

```
Failed to authenticate. API Error: 401 OAuth access token has expired.
```

which the wrapper surfaces as `claude failed: claude exited 1:` — empty stderr,
no mention of credentials. Four things keep that from happening:

1. **The `claude-refresher` service** (sandbox topology — on by default, no
   setup). Renewal happens only when the CLI runs, so an idle deployment
   expires on schedule no matter how healthy its network path is; this
   sidecar renews from ordinary networking into the shared volume regardless
   of traffic and independent of the proxy: it watches `.credentials.json`
   and spends one minimal CLI turn whenever the access token drops below ~4h.
   It logs both expiries on every pass —
   `podman logs claude-refresher` is the place to see renewal actually happen,
   and to learn whether `refreshTokenExpiresAt` rolls forward (if it does, one
   login lasts forever; if not, expect a re-login every ~30 days and the boot
   report will say so a week out). It only helps when the file credential is
   in force: an env token in `.env` makes the CLI ignore the volume entirely.
2. **Mint a long-lived token** with `setup-token` rather than using the desktop
   `login` flow. It does not depend on the refresh cycle at all. Record
   `CLAUDE_CODE_OAUTH_TOKEN_MINTED` alongside it so its own ~1-year death gets
   a warning instead of a silent outage.
3. **Exercise the CLI on a timer** if you are on a short-lived login in the
   **single-container** layout, where the CLI has ordinary egress. A daily
   throwaway invocation is enough to keep the refresh current. (Under the
   sandbox topology this is exactly what the refresher service does, from a
   container that can actually reach the OAuth endpoint — a timer aimed at
   `claude-agent` keeps nothing alive there.)

   ```bash
   cat > ~/.config/systemd/user/claude-token-refresh.service <<'EOF'
   [Unit]
   Description=Keep the Claude Code OAuth token fresh

   [Service]
   Type=oneshot
   ExecStart=/usr/bin/podman exec claude-wrapper claude -p hi --output-format json
   EOF

   cat > ~/.config/systemd/user/claude-token-refresh.timer <<'EOF'
   [Unit]
   Description=Daily Claude OAuth refresh

   [Timer]
   OnCalendar=daily
   Persistent=true

   [Install]
   WantedBy=timers.target
   EOF

   systemctl --user daemon-reload
   systemctl --user enable --now claude-token-refresh.timer
   ```

   `loginctl enable-linger "$USER"` must be on or the timer stops when you log
   out — the same prerequisite the podman socket has.
4. **Read the boot log.** Every role reports the credential it will actually
   authenticate with, at a severity that matches how much trouble you are in:

   ```
   INFO  Claude credentials: Claude Code login (/home/claude/.claude/.credentials.json), valid for 89d
   WARN  Claude credentials: ... valid for 7h. This is a short-lived login that only stays valid while …
   ERROR Claude credentials: ... EXPIRED 3h ago. Every turn will fail with a 401 until this is replaced …
   ```

   Note that `/healthz` will keep returning 200 through all of this — it reports
   that uvicorn is alive, nothing more. It is not a credential check, and it was
   not one when squid was down either.

The codex story is the same shape with different machinery: codex refreshes
its ChatGPT tokens automatically **on use**, so an idle deployment is the
failure mode, and the `codex-refresher` service covers it — it watches
`auth.json` and spends one minimal turn whenever `last_refresh` goes stale.
That refresh turn runs under codex's own read-only sandbox — and against a
private `CODEX_HOME` seeded with `auth.json` alone, because the volume's
`config.toml` is agent-writable and codex executes config directives
(`mcp_servers` commands spawn as plain subprocesses): loading it in a
container with ordinary egress would hand a prompt-injected agent a way
around Squid entirely. The refresher runs no agent code and executes no
model-driven tool use; only the credential crosses back. `codex-login` takes
the same precaution. Treat the
refresh-lifetime figures as estimates — OpenAI does not publish them, and the
refresher logs the token age on every pass so you can watch the real
behaviour.

Back the credential up once it is minted; restoring a file beats re-running an
interactive flow through the sandbox:

```bash
podman exec claude-agent cat /home/claude/.claude/.credentials.json > ~/claude-oauth-backup.json
chmod 600 ~/claude-oauth-backup.json
```

### Entrypoint subcommands

Available via `docker compose run --rm -it claude-wrapper <cmd>` — except the
`codex-*`/`codex` rows, which exist only in codex-stack images (the claude
build ships no codex binary): run those as
`docker compose -f docker-compose.codex.yml run --rm -it codex-refresher <cmd>`.

| Command | Purpose |
| --- | --- |
| `serve` (also `start`, `run`, or no argument) | Start the uvicorn API server |
| `agent` / `shim` | Start the agent shim — used by the sandbox topology |
| `refresher` / `refresh` | Keep the volume's Claude login renewed — the claude-refresher service's role |
| `login` / `init` | Interactive Claude Code OAuth login |
| `setup-token` / `token` | Mint a long-lived OAuth token (printed, not saved) |
| `shell` / `bash` | Drop into bash inside the container |
| `claude …` | Run any other `claude` CLI command |
| `codex-login` | Codex device-code login — the codex stack's bootstrap |
| `codex-refresher` | Keep a ChatGPT-plan `auth.json` fresh — the codex-refresher service's role |
| `codex …` | Run any other `codex` CLI command |
| anything else | Executed verbatim |

### 2. Client → wrapper

If `CLAUDE_WRAPPER_API_KEYS` is set (comma-separated), every request must include
`Authorization: Bearer <one-of-those-keys>`; a bare key without the `Bearer`
prefix is also accepted. When unset, the server is unauthenticated — bind it to
loopback or a private network only.

Three routes deviate: `/healthz` and `GET /v1/realtime/sessions` never require
auth, and `GET /v1/files/{id}/content` accepts a signed capability link as an
alternative to a key. FastAPI's `/docs`, `/redoc` and `/openapi.json` are
unauthenticated as well.

---

## Data and persistence

`docker-compose.single.yml` mounts two named volumes:

- `claude-data` → `/data` — uploaded and generated files, the session registry,
  usage ledgers, batch records, per-session workspaces.
- `claude-home` → `/home/claude/.claude` — Claude Code's own state, including the
  OAuth credentials. The image sets `CLAUDE_CONFIG_DIR` to this path so the CLI's
  config file lands here too; by default it would write `~/.claude.json` at HOME
  root, outside the volume, where a `run --rm` login would discard it on exit.

…plus one host bind mount: `${CLAUDE_INBOX_DIR:-./inbox}` → `/data/inbox`,
read-only.

`docker-compose.yml` differs: **three** volumes (a separate
`claude-workspace` shared between the API and agent containers at the same path),
`claude-home` mounted **read-only** on the API container and writable on the
agent, and the inbox mounted on the agent rather than the API.

`docker-compose.codex.yml` keeps volumes of its own — `codex-data`,
`codex-workspace`, and `codex-home` → `/home/claude/.codex` (`auth.json`,
`config.toml` and the sqlite thread store — writable on `codex-agent` and
`codex-refresher`, and mounted read-only on the API container **only when
opted in**: the API-key file mode; the mount ships commented out — see
[Codex → OpenAI](#codex--openai)). Nothing is shared with the claude stack:
the two run concurrently, and a shared file store or registry would race.

Session-registry entries are still tagged with the agent that wrote them, so
a volume that does serve both agents over time — the
[single-container layout](#single-container-layout-sunset), where
`CLAUDE_WRAPPER_AGENT` alone flips the agent — starts fresh sessions instead
of cross-resuming: the other agent's entries are ignored, not deleted, and
switching back finds them again.

The containers run unprivileged as `CLAUDE_UID:CLAUDE_GID` with `cap_drop: ALL`
and `no-new-privileges`, so everything they touch on the host must be readable by
that uid.

The image contains only `src/`, `requirements.txt` and `entrypoint.sh` — `tests/`,
`tools/`, `deploy/` and `sandbox/` are excluded, so you cannot run the test suite
with `docker compose exec`.

---

## Concurrency

- FastAPI + uvicorn serve requests async. Different sessions execute fully in
  parallel.
- Requests addressing the **same** `session_id` (or deriving the same anchor
  hash) are serialized by an in-process `asyncio.Lock`, because Claude Code's
  session JSONL file is not safe to write from two processes simultaneously.
  Budget accounting has its own separate lock, because it runs before the session
  lock is taken.
- **Keep `CLAUDE_WRAPPER_WORKERS=1`.** That lock is per-process, so more than one
  uvicorn worker reintroduces exactly the concurrent-write corruption it exists
  to prevent.
- Scale horizontally by running multiple containers behind a sticky load balancer
  keyed on `session_id`.

---

## Running the tests

The suite is 23 files covering endpoints, JSON mode, the budget, downloads,
effort, the tool bridge, the KB addendum, PDF handling, the sandbox shim, resume
self-healing, model discovery — and, since the codex integration: the agent
selector, the codex runner, the OpenAI bridge and the entrypoint roles. It
stubs both CLIs, so **no Docker and no Anthropic or OpenAI credentials are
required**.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest        # pytest is NOT in requirements.txt
CLAUDE_WRAPPER_DATA=/tmp/cw-test python -m pytest tests -q
# 332 passed  (the exact count moves with every added test; treat ±a few as fine)
```

CI (`.github/workflows/ci.yml`) runs the same command on Python 3.11, preceded by
an import check (`python -c "import src.main"`) that catches dangling imports in
under a second — three separate import-level breakages reached `main` before that
guard existed. CI does not lint, type-check, build the image, or validate the
compose files.

Nine of the test files also run standalone (`python tests/test_endpoints.py`)
and print a `RESULT pass=N fail=M` line; the rest are pytest-only and use
fixtures. Prefer the pytest command — the standalone path in `test_endpoints.py`
executes 27 of the 31 tests defined in that file, and its `check()` helper counts
failures without asserting, so a failure there does not fail the file under
pytest either. Treat `test_endpoints.py` as a smoke test rather than a gate.

---

## Troubleshooting

**Read the startup log first.** The wrapper reports the resolved model list,
whether runs execute locally or in a remote agent container, knowledge-base
state, clarification state, and a download-link report covering the link base and
where it came from, whether signing is active, the TTL, and the workspace hint.
Each of the corresponding misconfigurations otherwise fails silently.

| Symptom | Likely cause |
| --- | --- |
| `PermissionError: … '/data/assistants'` at boot | `CLAUDE_UID` disagrees with the uid baked into the image — see [below](#changing-claude_uid-after-first-run) |
| `OAuth error: Invalid IP address: undefined` | `CLAUDE_CODE_PROXY_RESOLVES_HOSTS` is set (off by default since 2026-08-11) — unset it and recreate; see CREDENTIALS-FIX.md Round 4 |
| `claude login` opens a chat session instead of authenticating | There is no `login` subcommand; the word is parsed as a prompt. Use the TUI's `/login` |
| `claude failed: claude exited 1:` with empty stderr | Usually an expired credential. The boot log names it; `claude -p hi` inside the agent prints the real 401 — see [Keeping the credential alive](#keeping-the-credential-alive) |
| squid `FATAL: failed to open /var/cache/squid/squid.pid` | tmpfs mode under rootless podman; check `RestartCount` — see [Rootless podman caveats](#rootless-podman-caveats) |
| CLI turns hang or exit 1, no `TCP_DENIED` in the squid log | Squid is not actually running. `restart: unless-stopped` hides a crash loop; check `RestartCount`, not `Status` |
| `container name … already in use` on `up` | Leftovers from a previous run, or from podman-compose's pod — see [If you are stuck with podman-compose](#if-you-are-stuck-with-podman-compose) |
| uvicorn: `Invalid value for '--port'` | The compose frontend didn't expand `${VAR:-default}` — same section |
| Download links 401 in the browser | `CLAUDE_WRAPPER_API_KEYS` set but no signing key resolved, or the link expired |
| Links are plain `→ file_id=…` text | No `CLAUDE_WRAPPER_PUBLIC_BASE_URL` and `CLAUDE_WRAPPER_DERIVE_BASE_URL=off` |
| Claude never writes files | `CLAUDE_WRAPPER_WORKSPACE_HINT` is off (the code default) |
| Downloads stop working mid-chat | Client sent `tools`; see [the tool-bridge note](#if-your-client-sends-tools-open-webui-native-function-calling) |
| KB queries always 401 | `OPENWEBUI_BASE_URL` set without `OPENWEBUI_API_KEY` |
| KB queries return nothing | A collection *name* was used where the *id* is required |
| Claude answers on assumptions instead of asking | `CLAUDE_WRAPPER_CLARIFY_DISALLOWED_TOOLS` interpolated empty from compose |
| Every sandbox run dies instantly | The target host isn't on the allowlist; on older CLIs possibly DNS (see the `CLAUDE_CODE_PROXY_RESOLVES_HOSTS` note — do not set it casually, it disables in-agent OAuth) |
| Inbox files unreadable under rootless podman | Host uid maps to container 0; needs `userns_mode: keep-id` |
| Old chats fail fast, new chats work | A dead `--resume` target; the wrapper self-heals on the next turn |
| Tools requests 502 `no_upstream_credential` under codex | ChatGPT-plan-only deployment — plan tokens can't call the Platform API. Set `OPENAI_API_KEY`, or opt into the `auth.json` mount — see [Codex → OpenAI](#codex--openai) |
| Boot refusal naming two agents | The wrapper and agent container disagree on `CLAUDE_WRAPPER_AGENT` — stale `.env`, or `CLAUDE_WRAPPER_AGENT_URL` pointing at the other stack |
| `codex login` stalls in the agent container | Interactive flows can't complete behind the sandbox; bootstrap via `codex-refresher codex-login` (device auth) — see [First-time login (Codex)](#first-time-login-codex) |
| Every codex turn times out with silent stdout | Egress blocked — codex retries forever. Check the OpenAI allowlist block is uncommented |
| A "resumed with model …" warning in a codex turn | Harmless notice that the resumed thread previously used a different model |

### Changing CLAUDE_UID after first run

`CLAUDE_UID` is both a build arg and a runtime uid, so changing it touches three
things: the image, the running container, and the ownership of any volume that
was already populated under the old value. Doing only the first two leaves you
with `PermissionError: [Errno 13] Permission denied: '/data/assistants'` —
`/data` is owned by the old uid, and `assistants`, `batches` and `vector_stores`
are created at import time directly under it.

Diagnose by comparing the two:

```bash
docker inspect claude-wrapper --format '{{.Config.User}}'          # runtime uid
docker run --rm -u 0 -v <project>_claude-data:/data \
    localhost/claude-wrapper:latest ls -lna /data                            # owner uid
```

If they differ, fix all three in one pass:

```bash
docker-compose down                       # never -v: that destroys your login

sed -i "s/^CLAUDE_UID=.*/CLAUDE_UID=$(id -u)/; s/^CLAUDE_GID=.*/CLAUDE_GID=$(id -g)/" .env
docker-compose build                      # rebuild: the uid is a build arg

docker volume ls | grep claude            # confirm the project prefix
for v in claude-data claude-workspace claude-home; do
  docker run --rm -u 0 -v "<project>_$v:/vol" \
      localhost/claude-wrapper:latest chown -R "$(id -u):$(id -g)" /vol
done

docker-compose up -d
docker exec claude-wrapper id             # matches your host uid
```

The chown step is what preserves an existing login rather than making you repeat
it. Include `claude-workspace` even though the error names only `/data`: the
agent shim creates each session directory under the workspace root, so it would
fail the same way on the first real request instead of at boot.

---

## Repository layout

| Path | What it is |
| --- | --- |
| `src/` | The application. `main.py` (routes + SSE), `agent_runner.py` (agent-neutral runner core + sessions), `claude_runner.py` (Claude CLI dialect), `codex_runner.py` (codex exec dialect), `converters.py` (prompt building), `tool_bridge.py` (Messages API path), `openai_bridge.py` (OpenAI passthrough for codex tools requests), `agent_shim.py` (sandbox agent service), plus per-area routers. |
| `docker-compose.codex.yml` | The codex variant of the default sandboxed stack — see [Quick start (Codex)](#quick-start-codex). |
| `tests/` | 23 test files; see [Running the tests](#running-the-tests). |
| `sandbox/` | `squid.conf` and `allowlist.txt`, bind-mounted into the squid container. Config only — there is no `sandbox` executable despite what the comments in those files suggest. |
| `tools/` | Host-side helper scripts, not part of the service and not in the image. `split_pdf.py` slices a PDF by page range or outline chapter; `chat_with_pdfs.py` uploads PDFs as binary `file_id`s and posts a chat completion, bypassing Open WebUI's text extraction. |
| `deploy/` | **Historical.** An archived incident runbook for a mis-mounted `claude-home` volume. Both fixes shipped; `fix-claude-home-mount.sh` fails its own preflight by design and must not be run. |

---

## Limitations and known gaps

- **Tools or the CLI, never both in one turn.** Requests carrying `tools` go to
  the tool bridge (Messages API, `tool_calls` returned, no CLI and therefore no
  generated files); requests without them run the agentic CLI path, where Claude
  manages its own tools internally and only the final assistant text is surfaced.
- **OpenAI surface coverage is partial.** There is no list-threads, no
  thread/message/vector-store *modify*, no message retrieve/delete, no run
  cancel, and no `submit_tool_outputs`. Fine-tuning is stubbed (501 on writes,
  empty lists on reads).
- **Batches are not durable.** Execution is an in-process asyncio task; in-flight
  batches are lost on restart.
- **Vector-store deletes are soft.** Detaching a file leaves its vectors in the
  matrix, so its content can still appear in search results.
- **The realtime WebSocket is text-only**, bypasses the chat pipeline entirely,
  and ignores the client's model. Audio goes through `/v1/audio/*`.
- **Delegated endpoints have a slow first call** while Claude installs the tools
  it needs; later calls reuse the cached install.
- **`/docs`, `/redoc`, `/openapi.json` and `GET /v1/realtime/sessions` are
  unauthenticated** even when API keys are configured.
- **Builds are not reproducible.** The base image tag, the Claude Code CLI, the
  squid image — and the codex CLI, when built with `INSTALL_CODEX=1` — are all
  unpinned.
- **Codex has no per-tool CLI gating.** `--disallowedTools` is Claude-only;
  under codex, command execution is intrinsic to the agent and the container
  sandbox is the enforcement boundary. `CLAUDE_WRAPPER_EXPOSE_TERMINAL` cannot
  disable it — and `/v1/models` is honest about that: under codex the
  CLI-shaped capabilities (`terminal`, `web_search`, `sub_agents`) always
  advertise as present, even if a profile removes them, because the removal
  would not be enforced. Profiles still genuinely gate `client_tools` (the
  bridge enforces it).
- **No wrapper-owned tools on the codex bridge.** `memory` and `time_calc`
  live in the Anthropic hybrid loop; the OpenAI passthrough never injects them.
- **Codex-tuned model ids cannot take `tools` requests.** OpenAI serves the
  `*-codex` ids only via the Responses API, and the passthrough speaks
  `/v1/chat/completions` — such requests are rejected with a 400 naming the
  fix (use a non-codex id, or drop `tools` for the CLI path) rather than
  forwarded into an opaque upstream 404. The gate applies only against the
  default `api.openai.com`; a custom `CLAUDE_WRAPPER_OPENAI_BASE_URL` backend
  is assumed to know its own model surface.
- **The token budget is uncalibrated for ChatGPT plans.** The plan presets are
  Anthropic-shaped, so the codex stack ships the cap off.
- **Codex streaming is item-granular.** `codex exec --json` emits whole text
  blocks, so the CLI path has no token-by-token deltas under codex.
- **Delegated endpoints are exercised primarily under Claude.**
- **Agent identity is only verified at wrapper boot.** An agent container
  recreated later with the wrong agent fails per-turn rather than refusing at
  startup.
- **Single-worker only** — see [Concurrency](#concurrency).
