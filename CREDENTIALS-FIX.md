# Credential handling: what broke and how to fix it

Three separate problems got tangled into one confusing experience. Untangling
them first, because the fix for each is different.

## What actually changed

**Problem 1 — interactive login stopped working.** Commit `2a5440f` moved the
CLI out of `claude-wrapper` and into `claude-agent`, which sits alone on an
`internal: true` network with `HTTP(S)_PROXY` pointed at Squid and a new
`CLAUDE_CODE_PROXY_RESOLVES_HOSTS: "1"` (`docker-compose.sandbox.yml:212`).
Login worked before that commit because the CLI ran in a container with
ordinary networking.

Note what is *not* the cause. The docs blame the read-only `claude-home` mount
(`README.md:1237`) and the OAuth callback (`README.md:1240`). Both are wrong:
the volume is mounted **writable** on the agent (`docker-compose.sandbox.yml:222`),
and `setup-token` — which has no callback at all — fails identically. Whatever
breaks it, breaks both flows before either reaches an OAuth server.

**Problem 2 — credentials stopped refreshing.** This one is fully proven from
the files. Putting the token in `.env` causes compose to inject
`CLAUDE_CODE_OAUTH_TOKEN` into both containers (`docker-compose.sandbox.yml:94`
and `:192`). `src/agent_shim.py:128` copies the whole environment into every CLI
subprocess, and the CLI prefers an env token over the on-disk login — so
`.credentials.json` is never read and never rewritten. That is exactly why the
file timestamps do not move. **The 30-day refresh token in the volume is
quietly running down with nothing renewing it.**

**Problem 3 — the "1 year vs 30 days" confusion.** Not a bug. See below.

**Problem 5 — `login` was never a login.** Confirmed empirically on CLI 2.1.226:
`claude --help` lists `setup-token` and **no `login` subcommand**. `claude login`
is parsed as a prompt, so it starts a chat session — it writes `.claude.json`,
`history.jsonl` and `sessions/`, prints a `--resume` hint, and never touches
`.credentials.json`. `entrypoint.sh:76-79` probes with `claude login --help`,
which exits 0 regardless, so the probe always passes and the entrypoint's
`login` subcommand silently runs a chat session. Every
`docker compose run --rm -it claude-agent login` in the docs does this. The only
real interactive path is the TUI's `/login` slash command.

**Problem 4, now fixed — Squid crash-loop.** The rootless-podman tmpfs bug
(commit `aa10040`) meant the agent had no egress for hours, so nothing could
refresh even in principle. Resolved, but it masked problems 1 and 2 throughout.

## Token lifetimes, reconciled

| Artefact | Where it lives | Lifetime | What renews it |
| --- | --- | --- | --- |
| Access token | `.credentials.json` → `expiresAt` | ~8 hours | CLI, from the refresh token, when it runs |
| Refresh token | `.credentials.json` → `refreshTokenExpiresAt` | ~30 days (yours: Sep 9) | Unknown — see Open questions |
| Setup token | printed to stdout; now in `.env` | ~1 year | Nothing; static until it expires |

There is no contradiction. The 8h and 30d numbers describe the **old
interactive login** sitting in the volume. The 1-year token is a **different
artefact** that was printed to your terminal and never written to that file.
Nothing in this repo writes `.credentials.json` — the only writes are test
fixtures (`tests/test_credentials.py:43`).

**The minted token never reaches the volume, and cannot.** Tested directly:
`claude setup-token` run in a container with `claude-home` mounted **writable**
left `.credentials.json` untouched. It prints the token and says
`Use this token by setting: export CLAUDE_CODE_OAUTH_TOKEN=<token>`. There is no
file-persistence path for it. (An earlier theory that a missing `-v` was to
blame is therefore wrong — the mount makes no difference.)

## The fix

The goals interact in a way that forces one real tradeoff:

> **Any mechanism that loads the backup token into the environment re-creates
> problem 2.** Compose `secrets:`, an `env_file`, a `*_FILE` convention — all of
> them end with `CLAUDE_CODE_OAUTH_TOKEN` set, which is precisely what stops the
> login from refreshing. There is no way to have both credentials active at
> once.

So the backup must stay **off-stack**: a 600-mode file outside the repo, or a
password manager. It is break-glass — if the login dies, you paste it into
`.env` deliberately, knowing it suspends refresh until you remove it again.

Everything else is achievable:

1. Bootstrap the login into the volume from a plain container that mounts
   `claude-home` but is **not** on the internal network.
2. Remove `CLAUDE_CODE_OAUTH_TOKEN` from `.env` so the file credential is the
   one in force.
3. Confirm refresh actually happens by watching `expiresAt` move.

Step 3 is the one that might fail, and if it does, see the experiment in Open
questions before concluding anything.

## Checklist

### 1. Restore login-based credentials

- [ ] Back up the current volume credential before touching anything:
      `podman exec claude-agent cat /home/claude/.claude/.credentials.json > ~/creds-$(date +%F).json && chmod 600 ~/creds-$(date +%F).json`
- [ ] Find the volume's real name: `podman volume ls --format '{{.Name}}' | grep claude-home`
- [ ] Bootstrap the login with the volume mounted. **Use the TUI slash command —
      `login` is not a subcommand and silently starts a chat session instead:**
      `podman run --rm -it -v <volume>:/home/claude/.claude claude-wrapper:latest claude`
      then type `/login` at the prompt, complete the flow, `/exit`
- [ ] Confirm it landed and is fresh:
      `podman exec claude-agent python3 -c "import json,time; d=json.load(open('/home/claude/.claude/.credentials.json'))['claudeAiOauth']; print(time.ctime(d['expiresAt']/1000), time.ctime(d['refreshTokenExpiresAt']/1000))"`

### 2. Get the backup token out of `.env`

- [ ] Store the 1-year token off-stack: `umask 077; printf '%s\n' '<token>' > ~/.claude-backup-token` (or a password manager)
- [ ] Blank it in `.env`: `sed -i 's|^CLAUDE_CODE_OAUTH_TOKEN=.*|CLAUDE_CODE_OAUTH_TOKEN=|' .env`
- [ ] Recreate so the env change takes effect: `docker-compose -f docker-compose.sandbox.yml up -d --force-recreate`
- [ ] Verify the env var is genuinely gone from the agent:
      `podman exec claude-agent sh -c 'echo "[${CLAUDE_CODE_OAUTH_TOKEN}]"'` → expect `[]`
- [ ] Do **not** wire the backup in via compose `secrets:` or `env_file` — that reintroduces problem 2

### 3. Prove the refresh cycle works

- [ ] Record the current `expiresAt`
- [ ] Send a real CLI turn (not a `tools` request — those bypass the CLI):
      `curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}]}'`
- [ ] Re-read `expiresAt`. **Moved forward → the cycle works and you are done.**
- [ ] Also check whether `refreshTokenExpiresAt` moved — that determines whether
      you ever need to log in again (see Open questions)
- [ ] If `expiresAt` did not move, run the experiment in Open questions

### 4. Documentation corrections

- [ ] `README.md:1237` — the read-only mount is not why login fails; the volume is writable on the agent
- [ ] `README.md:1240` — `setup-token` fails inside the sandbox exactly as `login` does; it is not a workaround for it
- [ ] `README.md:1268`, `.env.example:50`, `entrypoint.sh:84` — `setup-token` prints a token for the environment; it does not reliably populate the credentials volume
- [ ] `README.md:1489` — troubleshooting row should read "any OAuth flow", not "interactive login"
- [ ] `README.md:1249` — the throwaway-container recipe must show `-v <volume>:/home/claude/.claude`; without it the credential is discarded
- [ ] Document that an env token suppresses file refresh — the single most
      surprising behaviour in this system, currently written down nowhere

### 5. Code fixes

- [ ] `src/config.py:605` — `read_credential_status()` returns on the env token
      *before* opening the file, so it cannot warn that a volume credential is
      decaying underneath it. Report both.
- [ ] `src/config.py` — the file branch reads only `expiresAt`; the number that
      matters for "do I need to log in again" is `refreshTokenExpiresAt`
- [ ] `entrypoint.sh:76-79` — **`cmd_login` is broken.** `claude login --help`
      exits 0 even though no `login` subcommand exists, so the probe always
      passes and `exec claude login` starts a chat session. Replace the probe
      with a real capability check (`claude --help | grep -qw login`), or drop
      the subcommand and document the TUI `/login` flow instead.
- [ ] `entrypoint.sh:7` — `has_saved_login()` is a pure existence test, so an
      expired credential counts as "logged in" and the warning never fires
- [ ] `docker-compose.sandbox.yml:204` — `NO_PROXY` has a trailing empty element
      when `SANDBOX_EXTRA_NO_PROXY` is unset. Hygiene only; not a cause of anything.

## Outcome (2026-08-11)

Steps 1 and 2 are done and verified on the live stack:

- Volume login established via the TUI `/login` in a container with `claude-home`
  mounted and ordinary networking. Fresh: access 8.0h, refresh 29.1d.
- `CLAUDE_CODE_OAUTH_TOKEN` blanked in `.env`; the agent's env is empty; the
  1-year token moved off-stack to `~/.claude-backup-token` (mode 600).
- A real chat turn succeeded authenticating **from the volume credential**, which
  proves the file login works end to end inside the sandbox.
- Refresh did not fire — expected, with ~8h still on the access token.

**The one open item is the refresh event itself.** Run `~/credcheck.sh` after
the access token has aged past ~8h of use:

- `mtime` moves and `access` resets toward 8h → refresh works in-sandbox; this
  whole problem is closed and no long-lived token is needed.
- `access` goes negative and turns start failing with 401 → refresh is blocked
  inside the sandbox. Fall back to the backup token, and the
  `CLAUDE_CODE_PROXY_RESOLVES_HOSTS` hypothesis below becomes the thing to test.

## Open questions

**Does the refresh cycle work at all inside the sandbox?** This is the one that
decides whether goal 1 is achievable. A sub-agent proposed that
`CLAUDE_CODE_PROXY_RESOLVES_HOSTS=1` is itself the cause of
`Invalid IP address: undefined` — that the flag installs a DNS shim which breaks
the HTTP client used for OAuth *and* for token refresh, failing before any
CONNECT. I am reporting this as a **hypothesis, not a finding**: it came from an
agent that investigated by decompiling the CLI, which is outside what I am
willing to rely on, and the flag is documented as mandatory for normal runs
(`README.md:362`).

Test it legitimately, without changing your running stack:

```bash
podman run --rm -it \
  -e HTTPS_PROXY=http://squid:3128 -e HTTP_PROXY=http://squid:3128 \
  --network <the sandbox backend network> \
  claude-wrapper:latest claude setup-token          # expect the same error

# same again, with the flag explicitly disabled
podman run --rm -it \
  -e HTTPS_PROXY=http://squid:3128 -e HTTP_PROXY=http://squid:3128 \
  -e CLAUDE_CODE_PROXY_RESOLVES_HOSTS= \
  --network <the sandbox backend network> \
  claude-wrapper:latest claude setup-token
```

If the second reaches an OAuth server and the first does not, the hypothesis
holds — and the consequence is significant: the flag that makes normal runs work
is the same one that prevents refresh, so login credentials can never
self-sustain in this topology and the long-lived token is the correct answer
after all.

**Resolved:** `setup-token` does not persist to the credentials volume even when
one is mounted writable — tested directly. The token is environment-only, so the
"backup must stay off-stack" conclusion above holds.

**Resolved:** there is no `login` subcommand on CLI 2.1.226; only the TUI's
`/login` performs authentication. This also means the
`OAuth error: Invalid IP address: undefined` seen in the sandboxed agent came
from the TUI's own auth prompt, not from the `login` subcommand — the two
failures we treated as one were partly the entrypoint bug.

**Which host does a refresh contact?** Not determinable from this repo; no
OAuth endpoint appears anywhere in `src/` or `sandbox/`. `console.anthropic.com`
is absent from the allowlist and is a plausible gap, but that is a suspicion.
Squid's access log during a refresh is the way to settle it.

**Does `refreshTokenExpiresAt` roll forward on each refresh?** Anthropic-side.
If it does, a used login is indefinite. If it is a hard 30-day wall, you re-login
monthly regardless — which would also make the backup token the primary.
