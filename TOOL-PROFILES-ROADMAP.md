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
| Terminal | Claude Code's bash tool, gated per profile via `--disallowedTools` | CLI |
| Memory | `memory_20250818` client-side tool, wrapper-owned storage per conversation | bridge |
| Image Generation | external backend behind `routes_images.py`, surfaced as a custom tool | bridge |
| Sub-agents | Claude Code's Task tool, gated per profile | CLI |
| Builtin domain tools (Calendar, Notes, KB, Chat History, Channels, Notifications, Tasks, Automations, Time & Calc) | OpenWebUI-owned tools; the wrapper's job is to relay tool calls faithfully and gate which models may receive them | bridge |

## Design

**Profile registry.** A JSON/YAML file (path via
`CLAUDE_WRAPPER_MODEL_PROFILES`) mapping model-id patterns to a capability
set. Resolution order: family-derived defaults (e.g. all ≥4.x models get
vision) → profile file → env overrides. One `default` profile; per-model
entries override it. Effort variants inherit their base model's profile.

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
4. OpenWebUI sync — a script that reads the wrapper's model list + profiles
   and pushes the capability toggles into OpenWebUI's model registry via its
   admin API, so the UI never drifts from the wrapper.

**Loop ownership (phase 4 decision).** Today the bridge is strictly
client-looped. Wrapper-owned tools (Memory, Image Generation) require a hybrid
loop: the wrapper executes *its* tools inline and only returns *client* tools
to OpenWebUI. This is the riskiest change — it gets its own phase and can ship
last without blocking anything earlier.

## Phases

**Phase 0 — Groundwork.** Capability vocabulary as an enum; profile schema +
loader with validation; family-derived defaults; resolution order; unit tests.
Exit: `resolve_profile(model_id)` returns a stable capability set for every
advertised model, covered by tests.

**Phase 1 — Advertise.** `/v1/models` carries capability metadata; new
`tools/sync_openwebui_capabilities.py` pushes toggles into OpenWebUI. Exit:
fresh OpenWebUI instance shows correct per-model toggles after one sync run.

**Phase 2 — Enforce, CLI path.** Per-model `--disallowedTools` derived from
the profile; `clarify_disallowed_tools` folded into the same mechanism. Exit:
a terminal-off model cannot run bash; a sub-agents-off model cannot spawn
Task agents.

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

## Open decisions

- Profile file format: JSON (no new deps) vs YAML (needs a dep — house rule
  says avoid; leaning JSON).
- Deny behavior in the bridge: hard 400 vs silently stripping denied tools.
  Leaning 400 — silent stripping hides misconfiguration.
- Memory storage shape: flat per-conversation JSON vs the memory tool's
  file-path model. Decide in phase 4.
- OpenWebUI sync auth: admin API key via env; confirm the endpoint shape
  against the deployed OpenWebUI version before phase 1 lands.

## Non-goals

- No Agent SDK adoption — the CLI path already is the Claude Code harness.
- No mid-conversation tool set changes (cache-hostile); the toolset is fixed
  when a conversation starts.
- No reimplementation of OpenWebUI's builtin domain tools — those stay
  client-owned.
