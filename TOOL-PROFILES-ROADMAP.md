# Per-model tool & capability profiles — implementation roadmap

## Why

OpenWebUI tracks a capability set per model (Vision, File Upload, Web Search,
Code Interpreter, Terminal, Memory, …) plus a builtin-tool menu (Calendar,
Notes, Knowledge Base, Chat History, Automations, …). Today the wrapper has no
notion of any of this: every model behaves identically, the capability toggles
in OpenWebUI are set by hand, and nothing stops a client from declaring tools
at a model that shouldn't have them. The two sides drift.

Goal: make the wrapper the **single source of truth** for what each advertised
model can do — declared once in a profile registry, enforced on both execution
paths, exposed through `/v1/models`, and synced into OpenWebUI automatically.

## Current state (what the code does today)

- **CLI path** (`claude_runner.py`): Claude Code runs its own agentic loop
  with its built-in tools (bash, file ops, web search, subagents). The only
  tool restriction is the global `clarify_disallowed_tools` flag.
- **Tool-bridge path** (`tool_bridge.py`): a request carrying `tools` bypasses
  the CLI and hits the Messages API directly. The *client* owns the loop and
  executes tools. No per-model filtering, no server-side tools injected.
- **Model surface** (`model_discovery.py`, `config.py`): model ids are scanned
  from the CLI binary, effort variants appended, and served by `/v1/models`
  with no capability metadata.

## Capability → mechanism map

How each OpenWebUI capability is backed. "native" = an API feature the wrapper
already relays or can relay; nothing to execute.

| Capability | Backed by | Path |
|---|---|---|
| Vision | native image content blocks; gate by model family | both |
| File Upload / File Context | existing `/v1/files` + multimodal handling | both |
| Usage | native `usage` passthrough (already works) | both |
| Citations | `citations: {enabled}` on document blocks / web search results | bridge |
| Status Updates | existing reasoning/activity stream frames | CLI |
| Web Search | server-side `web_search_20260209` (older families: `_20250305`); CLI's own WebSearch tool on the CLI path | both |
| Code Interpreter | server-side `code_execution` tool (Anthropic sandbox) | bridge |
| Terminal | Claude Code's bash tool; profile grant **and** the `CLAUDE_WRAPPER_EXPOSE_TERMINAL` env gate both required; enforced via `--disallowedTools` on chat runs only | CLI |
| Memory | `memory_20250818` client-side tool, wrapper-owned storage per conversation | bridge |
| Image Generation | external backend behind `routes_images.py`, surfaced as a custom tool | bridge |
| Sub-agents | Claude Code's Task tool, gated per profile | CLI |
| Builtin domain tools (Calendar, Notes, KB, Chat History, Channels, Notifications, Tasks, Automations, Time & Calc) | OpenWebUI-owned tools; the wrapper's job is to relay tool calls faithfully and gate which models may receive them | bridge |

## Design

**Profile registry.** A JSON/YAML file (path via
`CLAUDE_WRAPPER_MODEL_PROFILES`) mapping model-id patterns to a capability
set. Resolution order: built-in default → profile file → inline env
overrides. The built-in default reproduces today's behavior exactly (CLI
built-in tools on, client tools allowed, vision/files on; capabilities that
need new wrapper machinery off), so an absent profile file changes nothing.
Effort and `[1m]` variants inherit their base model's profile.

**Enforcement points.**

1. `/v1/models` — attach the resolved capability set as extra metadata fields
   on each model entry (the OpenAI schema tolerates extra fields).
2. CLI path — translate the profile into `--disallowedTools` (terminal off →
   disallow Bash; web search off → disallow WebSearch/WebFetch; sub-agents
   off → disallow Task).
3. Tool-bridge path — reject or strip client-declared tools not permitted by
   the profile; inject enabled server-side tools (web search variant chosen
   per family, code execution); keep tool ordering deterministic so prompt
   caching survives.
4. OpenWebUI sync — a puller script living on the OpenWebUI host: it reads
   the wrapper's `/v1/models` (capabilities included) and writes the toggles
   into OpenWebUI's model registry via the local admin API. The wrapper side
   of this is nothing but the enriched `/v1/models`.

**Loop ownership (phase 4 decision).** Today the bridge is strictly
client-looped. Wrapper-owned tools (Memory, Image Generation) require a hybrid
loop: the wrapper executes *its* tools inline and only returns *client* tools
to OpenWebUI. This is the riskiest change — it gets its own phase and can ship
last without blocking anything earlier.

## Phases

**Phase 0 — Groundwork.** Capability vocabulary as an enum; profile schema +
loader with validation; behavior-preserving built-in default; resolution
order; unit tests.
Exit: `resolve_profile(model_id)` returns a stable capability set for every
advertised model, covered by tests.

**Phase 1 — Advertise.** `/v1/models` carries capability metadata; a
reference puller (`deploy/openwebui_capability_sync.py`, to be run on the
OpenWebUI host) reads it and writes the toggles via OpenWebUI's local admin
API. Exit: fresh OpenWebUI instance shows correct per-model toggles after
one puller run against the wrapper.

**Phase 2 — Enforce, CLI path.** Per-model `--disallowedTools` derived from
the profile, applied to **chat runs only** — delegation runs (audio, images,
embeddings working through Bash internally) are never gated, or the wrapper
breaks itself. `clarify_disallowed_tools` folded into the same mechanism.
Exit: a terminal-off model cannot run bash in chat; a sub-agents-off model
cannot spawn Task agents; with `CLAUDE_WRAPPER_EXPOSE_TERMINAL` set the
default argv is byte-for-byte today's (without it, chat runs carry
`--disallowedTools Bash` by design); delegation argv unchanged in all cases.

**Phase 3 — Enforce, tool bridge.** Client tool filtering per profile with a
clear error for denied tools; server-side web search + code execution
injection for enabled models (correct version per family); citations enabled
where profiled. Exit: `test_tool_bridge.py` extended to cover allow, deny,
and injection cases.

**Phase 4 — Wrapper-owned tools.** Hybrid loop in the bridge; Memory backed
by per-conversation storage under the existing data dir; Time & Calc as a
cheap first wrapper-owned tool to prove the loop. Exit: memory persists
across turns without OpenWebUI executing anything.

**Phase 5 — Image generation.** Pick a backend (env-configured URL), wire
`routes_images.py` to it, expose it as a wrapper-owned tool for profiled
models. Exit: image request round-trips through a profiled model.

**Phase 6 — Docs & hardening.** README configuration reference for profiles;
example profile file; end-to-end validation against a live OpenWebUI;
migration note for existing deployments (default profile = today's behavior).

## Decisions (settled)

- **Profile file format: JSON.** No new dependency.
- **Deny behavior in the bridge: hard 400** naming the denied tool. Silent
  stripping hides misconfiguration.
- **Memory storage: file-path model** — a per-conversation directory of
  markdown files under the data dir. This is what Claude Code itself uses
  organically (memory directory + `MEMORY.md` index) and it matches the
  `memory_20250818` tool's path-based command set exactly.
- **Capability exposure: pull, never push.** Capabilities ride as extra
  fields on `/v1/models`, visible the moment anything pulls the model list.
  The wrapper never contacts OpenWebUI. Because OpenWebUI doesn't map pulled
  metadata into its own capability toggles (those live in its internal model
  records, writable via its local admin API), a small sync script bridges the
  gap — but it **resides on the OpenWebUI host**, pulls the wrapper's
  `/v1/models`, and writes the toggles locally. This repo ships it as a
  reference artifact only (`deploy/`); it is not part of the wrapper process.
  If OpenWebUI later honors pulled metadata, the script is deleted and
  nothing else changes.
- **Terminal is opt-in, hard-gated by env.** Exposing a shell to a chat UI
  must be a deliberate operator action, not a profile-file side effect:
  `CLAUDE_WRAPPER_EXPOSE_TERMINAL` (default off) masks the `terminal`
  capability out of every resolved profile — including explicit profile
  grants — until set. With the toggle on, profiles decide per model as
  usual. This gates the UI-facing capability only: the wrapper's internal
  delegation runs (audio/images/embeddings doing their work through Claude
  Code's Bash) are not chat-path enforcement and keep working regardless.

## Non-goals

- No Agent SDK adoption — the CLI path already is the Claude Code harness.
- No mid-conversation tool set changes (cache-hostile); the toolset is fixed
  when a conversation starts.
- No reimplementation of OpenWebUI's builtin domain tools — those stay
  client-owned.
