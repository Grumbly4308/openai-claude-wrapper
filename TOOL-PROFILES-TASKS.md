# Per-model tool profiles — task checklist

Working checklist for [TOOL-PROFILES-ROADMAP.md](TOOL-PROFILES-ROADMAP.md).
Check items off as they land; keep one PR per phase where practical.

## Phase 0 — Groundwork

- [x] Define the capability enum (vision, file_upload, web_search,
      code_interpreter, terminal, memory, citations, image_generation,
      sub_agents, client_tools) in `src/capabilities.py`
- [x] Define the profile schema: `default` entry + ordered per-model-pattern
      entries (replace or add/remove deltas)
- [x] Implement the profile loader (`CLAUDE_WRAPPER_MODEL_PROFILES` JSON file
      + `CLAUDE_WRAPPER_MODEL_PROFILE_OVERRIDES` inline JSON), with
      validation errors that name the offending entry
- [x] Implement the behavior-preserving built-in default (vision, file_upload,
      web_search, terminal, sub_agents, client_tools on — i.e. exactly what
      every model does today; code_interpreter/memory/citations/image_generation
      off until their phases land)
- [x] Implement `resolve_profile(model_id)` with resolution order
      built-in default → file → env overrides; effort and `[1m]` variants
      inherit the base model
- [x] Unit tests: `tests/test_profiles.py` covering defaults, overrides,
      pattern matching, effort-variant inheritance, invalid config
- [x] Hard-gate the terminal capability behind `CLAUDE_WRAPPER_EXPOSE_TERMINAL`
      (default off): profile grants are masked until the operator opts in, so
      a profile file alone can never expose a shell to the UI

## Phase 1 — Advertise

- [ ] Attach resolved capabilities as extra fields on `/v1/models` entries
- [ ] Extend `tests/test_endpoints.py` for the enriched model list
- [ ] Write `deploy/openwebui_capability_sync.py` — a **puller that resides
      on the OpenWebUI host**: reads the wrapper's `/v1/models` (capabilities
      included) and writes the toggles via OpenWebUI's local admin API. The
      wrapper never contacts OpenWebUI. (Needed because OpenWebUI doesn't map
      pulled metadata into its capability toggles; delete if it ever does)
- [ ] Confirm the OpenWebUI model-update endpoint shape against the deployed
      version; pin the minimum supported OpenWebUI version in the script
- [ ] Document puller usage (env vars: wrapper URL, OpenWebUI URL + admin
      key; run via cron or OpenWebUI startup hook)

## Phase 2 — Enforce: CLI path

- [ ] Map capabilities → Claude Code tool names (terminal→Bash,
      web_search→WebSearch/WebFetch, sub_agents→Task)
- [ ] Build per-request `--disallowedTools` from the model's profile in
      `claude_runner.py` — **chat runs only**; delegation runs (audio,
      images, embeddings — they do their work through Bash) are never gated
- [ ] Fold `clarify_disallowed_tools` into the same mechanism (keep the env
      var working; deprecation note in README)
- [ ] Tests: terminal-off model gets Bash disallowed in chat; delegation argv
      never gated; with `CLAUDE_WRAPPER_EXPOSE_TERMINAL` set the default
      profile's argv is byte-for-byte today's

## Phase 3 — Enforce: tool bridge

- [ ] Filter client-declared tools against the profile; return a 400 naming
      the denied tool (decided: hard-fail over silent strip)
- [ ] Inject server-side web search when profiled: `web_search_20260209` for
      capable families, `web_search_20250305` otherwise
- [ ] Inject server-side code execution when profiled
- [ ] Enable citations on document blocks when profiled
- [ ] Keep injected + client tools deterministically ordered (cache safety)
- [ ] Extend `tests/test_tool_bridge.py`: allow, deny, injection, version
      selection per family, ordering stability

## Phase 4 — Wrapper-owned tools (hybrid loop)

- [x] Decide memory storage shape — file-path model (what Claude Code uses
      organically; matches `memory_20250818` semantics)
- [ ] Implement the hybrid loop in the bridge: wrapper-owned tools execute
      inline, client tools still return to the caller
- [ ] Guard against infinite loops (max iterations per turn)
- [ ] Implement Time & Calc as the first wrapper-owned tool (proves the loop)
- [ ] Implement Memory (`memory_20250818`) with per-conversation storage
      under the existing data dir
- [ ] Tests: hybrid turn with one wrapper tool + one client tool; memory
      persists across turns; iteration cap trips cleanly

## Phase 5 — Image generation

- [ ] Choose/configure the external backend (env: backend URL + key)
- [ ] Wire `routes_images.py` to the backend
- [ ] Expose image generation as a wrapper-owned tool for profiled models
- [ ] Tests with the backend faked

## Phase 6 — Docs & hardening

- [ ] README: profiles configuration reference + example profile file,
      including `CLAUDE_WRAPPER_EXPOSE_TERMINAL` and why it defaults off
- [ ] Ship `deploy/` example profile matching current default behavior
- [ ] End-to-end pass against a live OpenWebUI (toggles, tool calls, denials)
- [ ] Migration note: absent profile file == today's behavior
- [ ] Run full test suite + `qa` if available; fix fallout
