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
| Access token | `.credentials.json` → `expiresAt` | ~8 hours | CLI, from the refresh token, when it runs — **but not inside the sandbox; measured, see Outcome** |
| Refresh token | `.credentials.json` → `refreshTokenExpiresAt` | ~30 days (yours: Sep 9) | Nothing, since the renewal it would drive never fires here |
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

- [x] Run `tools/credential_refresh_test.sh`, which forces the question in one
      go rather than waiting ~8h for a natural expiry
- [x] **Result: it does not work here.** See Outcome. Steps 1 and 2 therefore do
      not give you a self-sustaining deployment, and the long-lived token is the
      primary credential under the sandbox topology, not the break-glass one.

### 4. Documentation corrections

All done in `7e57523` / `4f0bfbc`; line numbers below are from before those
commits.

- [x] `README.md:1237` — the read-only mount is not why login fails; the volume is writable on the agent
- [x] `README.md:1240` — `setup-token` fails inside the sandbox exactly as `login` does; it is not a workaround for it
- [x] `README.md:1268`, `.env.example:50`, `entrypoint.sh:84` — `setup-token` prints a token for the environment; it does not reliably populate the credentials volume
- [x] `README.md:1489` — troubleshooting row should read "any OAuth flow", not "interactive login"
- [x] `README.md:1249` — the throwaway-container recipe must show `-v <volume>:/home/claude/.claude`; without it the credential is discarded
- [x] Document that an env token suppresses file refresh — the single most
      surprising behaviour in this system, currently written down nowhere

### 5. Code fixes

- [x] `src/config.py:605` — `read_credential_status()` returns on the env token
      *before* opening the file, so it cannot warn that a volume credential is
      decaying underneath it. Report both. — done, `log_credential_status()`
      now warns about the shadowed login
- [x] `src/config.py` — the file branch reads only `expiresAt`; the number that
      matters for "do I need to log in again" is `refreshTokenExpiresAt`
- [x] `entrypoint.sh:76-79` — **`cmd_login` is broken.** `claude login --help`
      exits 0 even though no `login` subcommand exists, so the probe always
      passes and `exec claude login` starts a chat session. — done, it now
      capability-checks and otherwise opens the TUI with the `/login` hint
- [x] `entrypoint.sh:7` — `has_saved_login()` is a pure existence test, so an
      expired credential counts as "logged in" and the warning never fires.
      — done in round 2: it now reads the refresh window and treats a dead
      file as no login.
- [ ] `docker-compose.sandbox.yml:204` — `NO_PROXY` has a trailing empty element
      when `SANDBOX_EXTRA_NO_PROXY` is unset. Hygiene only; not a cause of anything.
      Still open.

## Outcome (2026-08-11)

Steps 1 and 2 are done and verified on the live stack:

- Volume login established via the TUI `/login` in a container with `claude-home`
  mounted and ordinary networking. Fresh: access 8.0h, refresh 29.1d.
- `CLAUDE_CODE_OAUTH_TOKEN` blanked in `.env`; the agent's env is empty; the
  1-year token moved off-stack to `~/.claude-backup-token` (mode 600).
- A real chat turn succeeded authenticating **from the volume credential**, which
  proves the file login works end to end inside the sandbox.
- Refresh did not fire — expected, with ~8h still on the access token.

**The refresh event has now been measured, and it does not happen.** The volume
login was left to age; by 03:42 its access token had lapsed on its own — the
first sign, since a working refresh would have renewed it during ordinary use.
`tools/credential_refresh_test.sh` then forced the question with a valid refresh
token 29.1 days from expiry:

- the turn failed with `claude failed: claude exited 1:` — the empty-stderr 401
  signature;
- `.credentials.json` was **not rewritten**: after the turn its mtime was still
  the test's own edit, and `expiresAt` still the value the test wrote.

The second point is the finding. A refresh either rewrites that file or did not
occur, and nothing rewrote it. Two limits on how far that stretches: the CLI's
exit code is opaque, so this shows the CLI did not renew rather than naming
which hop refused; and the credential was already expired when the test began,
so the turn was doomed either way — what the test adds is that a healthy refresh
token sat there unused. The control is in the record above: the same request
path succeeded from this same volume credential while its access token was
fresh, so the difference between working and failing is expiry alone.

**Consequence: a login cannot sustain itself under the sandbox topology** — on
its own. The conclusion this section first drew (long-lived token as primary,
step 2 in reverse) held for a few hours; "The fix, round 2" below supersedes
it by renewing the login from outside the sandbox, which restores the login as
primary and the token as break-glass. The remaining question about *why*
in-agent renewal is blocked is still the `CLAUDE_CODE_PROXY_RESOLVES_HOSTS`
experiment below.

## The fix, round 2 (2026-08-11): log in once, run forever

The measurement above proved renewal fails *from the agent's network
position* — but the bootstrap `/login` proved the same CLI renews the same
volume fine *from ordinary networking*. Those two facts compose: refresh does
not have to happen where the agent is, only into the volume the agent reads.

**`claude-refresher`** is that composition as a service: same image, same
`claude-home` volume mounted writable, sitting on the egress network with env
credentials pinned empty, watching `expiresAt` and spending one minimal CLI
turn whenever the access token drops below ~4h. The static token goes back to
break-glass — now with `CLAUDE_CODE_OAUTH_TOKEN_MINTED` so its own ~1-year
death gets a boot warning instead of a silent outage. `console.anthropic.com`
is also on the allowlist now, so in-agent renewal gets a fair retest, but
nothing depends on it working.

### Rollout checklist (live stack)

- [ ] Take the env token back out of `.env` (keep it in
      `~/.claude-backup-token`) — while it is set, every CLI ignores the file
      and the refresher refuses to start
- [ ] Add `CLAUDE_CODE_OAUTH_TOKEN_MINTED=<the day you minted it>` while
      you're in there, for the day you break the glass
- [ ] `docker-compose -f docker-compose.sandbox.yml up -d --build --force-recreate`
      (`--build`: the entrypoint gained the refresher role)
- [ ] Watch the first renewal: `podman logs -f claude-refresher`. The current
      volume credential — access expired, refresh good to Sep 9 — is the
      perfect test case: the first pass should revive it with **no re-login**
- [ ] `~/credcheck.sh` → `in force : volume credential`, access climbing back
      toward 8h
- [ ] One real turn through `localhost:8000/v1/chat/completions`
- [ ] Optional: `tools/credential_refresh_test.sh` to see whether the
      allowlist entry made in-agent refresh work too — interesting, not
      required
- [ ] Over the next weeks, check the refresher's log for whether
      `refreshTokenExpiresAt` rolls forward. If it does, the login is
      indefinite and this file is finished. If it is a hard 30-day wall, the
      refresher's log and the boot report both warn a week out, and a monthly
      `/login` via `run --rm -it claude-refresher claude` is the whole ritual

## Open questions

**Resolved — no, the refresh cycle does not work inside the sandbox.** Measured
with `tools/credential_refresh_test.sh`; see Outcome. Goal 1 is therefore not
achievable as stated, and what remains open is the cause. A sub-agent proposed
that
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
holds, and it would explain the measured failure: the flag that makes normal runs
work is the same one that prevents refresh. Note that the conclusion no longer
rests on it — the long-lived token is already the right answer on the evidence
above. This now only identifies the cause, and with it whether the topology could
be changed to make a login viable.

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
