# Per-model tool profiles — task checklist

Working checklist for [TOOL-PROFILES-ROADMAP.md](TOOL-PROFILES-ROADMAP.md).
Check items off as they land; keep one PR per phase where practical.

## Phase 0 — Groundwork

- [ ] Define the capability enum (vision, file_upload, web_search,
      code_interpreter, terminal, memory, citations, image_generation,
      sub_agents, client_tools) in `src/capabilities.py`
- [ ] Define the profile schema: `default` entry + per-model-pattern overrides
- [ ] Implement the profile loader (`CLAUDE_WRAPPER_MODEL_PROFILES`, JSON),
      with validation errors that name the offending entry
- [ ] Implement family-derived defaults (all discovered ≥4.x models: vision,
      file_upload, usage on; everything else off)
- [ ] Implement `resolve_profile(model_id)` with resolution order
      defaults → file → env, effort variants inheriting the base model
- [ ] Unit tests: `tests/test_profiles.py` covering defaults, overrides,
      pattern matching, effort-variant inheritance, invalid config

## Phase 1 — Advertise

- [ ] Attach resolved capabilities as extra fields on `/v1/models` entries
- [ ] Extend `tests/test_endpoints.py` for the enriched model list
- [ ] Write `tools/sync_openwebui_capabilities.py` (reads wrapper models +
      profiles, pushes capability toggles via OpenWebUI admin API)
- [ ] Confirm the OpenWebUI model-update endpoint shape against the deployed
      version; pin the minimum supported OpenWebUI version in the script
- [ ] Document sync usage (env vars: OpenWebUI URL + admin key)

## Phase 2 — Enforce: CLI path

- [ ] Map capabilities → Claude Code tool names (terminal→Bash,
      web_search→WebSearch/WebFetch, sub_agents→Task)
- [ ] Build per-request `--disallowedTools` from the model's profile in
      `claude_runner.py`
- [ ] Fold `clarify_disallowed_tools` into the same mechanism (keep the env
      var working; deprecation note in README)
- [ ] Tests: terminal-off model gets Bash disallowed; default profile
      preserves today's behavior byte-for-byte on the argv

## Phase 3 — Enforce: tool bridge

- [ ] Filter client-declared tools against the profile; return a 400 naming
      the denied tool (decision: hard-fail over silent strip — revisit if it
      proves noisy)
- [ ] Inject server-side web search when profiled: `web_search_20260209` for
      capable families, `web_search_20250305` otherwise
- [ ] Inject server-side code execution when profiled
- [ ] Enable citations on document blocks when profiled
- [ ] Keep injected + client tools deterministically ordered (cache safety)
- [ ] Extend `tests/test_tool_bridge.py`: allow, deny, injection, version
      selection per family, ordering stability

## Phase 4 — Wrapper-owned tools (hybrid loop)

- [ ] Decide memory storage shape (flat JSON vs memory-tool file model)
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

- [ ] README: profiles configuration reference + example profile file
- [ ] Ship `deploy/` example profile matching current default behavior
- [ ] End-to-end pass against a live OpenWebUI (toggles, tool calls, denials)
- [ ] Migration note: absent profile file == today's behavior
- [ ] Run full test suite + `qa` if available; fix fallout
