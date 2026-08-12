#!/usr/bin/env python3
"""Pull per-model capabilities from a claude-wrapper into OpenWebUI's toggles.

Runs on the OPENWEBUI host (cron, systemd timer, or startup hook) — this is a
puller, the wrapper never contacts OpenWebUI. It reads the wrapper's
/v1/models, whose entries carry a `capabilities` list (the wrapper's resolved
per-model profile, see src/capabilities.py in the wrapper repo), and writes
the matching capability toggles into OpenWebUI's model records via its local
admin API. Stdlib only — no pip installs on the OpenWebUI host.

Environment:
  WRAPPER_BASE_URL     wrapper origin, e.g. http://wrapper:8000   (required)
  WRAPPER_API_KEY      bearer token if the wrapper enforces auth  (optional)
  OPENWEBUI_BASE_URL   default http://localhost:8080
  OPENWEBUI_API_KEY    an ADMIN account's API key                 (required)
  SYNC_DRY_RUN         set to 1 to print planned changes without writing

Endpoint assumptions (verified against OpenWebUI 0.6.x — re-check on
upgrade; if OpenWebUI ever maps pulled model metadata into its own toggles,
delete this script):
  GET  {openwebui}/api/v1/models/                → existing model records
  POST {openwebui}/api/v1/models/model/update?id=… → update a record
  POST {openwebui}/api/v1/models/create          → create a record

Capability mapping (wrapper → OpenWebUI meta.capabilities):
  vision→vision, file_upload→file_upload+file_context, web_search→web_search,
  code_interpreter→code_interpreter, terminal→terminal, memory→memory,
  citations→citations, image_generation→image_generation,
  client_tools→builtin_tools. usage and status_updates are always on —
  the wrapper reports usage and streams activity for every model.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

CAPABILITY_MAP = {
    "vision": ("vision",),
    "file_upload": ("file_upload", "file_context"),
    "web_search": ("web_search",),
    "code_interpreter": ("code_interpreter",),
    "terminal": ("terminal",),
    "memory": ("memory",),
    "citations": ("citations",),
    "image_generation": ("image_generation",),
    "client_tools": ("builtin_tools",),
    # sub_agents has no OpenWebUI toggle; enforced wrapper-side only.
}
ALWAYS_ON = ("usage", "status_updates")
ALL_TOGGLES = sorted({t for ts in CAPABILITY_MAP.values() for t in ts} | set(ALWAYS_ON))


def _request(method: str, url: str, token: str | None, body: dict | None = None) -> object:
    req = urllib.request.Request(url, method=method)
    req.add_header("Accept", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        sys.exit(f"{method} {url} -> HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        sys.exit(f"{method} {url} -> {e.reason}")


def _toggles_for(wrapper_caps: list[str]) -> dict[str, bool]:
    toggles = {t: False for t in ALL_TOGGLES}
    for cap in wrapper_caps:
        for t in CAPABILITY_MAP.get(cap, ()):
            toggles[t] = True
    for t in ALWAYS_ON:
        toggles[t] = True
    return toggles


def main() -> int:
    wrapper = (os.environ.get("WRAPPER_BASE_URL") or "").rstrip("/")
    if not wrapper:
        sys.exit("WRAPPER_BASE_URL is required")
    openwebui = (os.environ.get("OPENWEBUI_BASE_URL") or "http://localhost:8080").rstrip("/")
    owui_key = os.environ.get("OPENWEBUI_API_KEY") or ""
    dry_run = (os.environ.get("SYNC_DRY_RUN") or "").strip() in ("1", "true", "yes")
    if not owui_key and not dry_run:
        sys.exit("OPENWEBUI_API_KEY is required (an admin account's API key)")

    models = _request("GET", f"{wrapper}/v1/models", os.environ.get("WRAPPER_API_KEY"))
    entries = models.get("data") if isinstance(models, dict) else None
    if not entries:
        sys.exit(f"{wrapper}/v1/models returned no models")

    existing = _request("GET", f"{openwebui}/api/v1/models/", owui_key)
    by_id = {m.get("id"): m for m in existing} if isinstance(existing, list) else {}

    changed = created = unchanged = 0
    for entry in entries:
        model_id = entry.get("id")
        caps = entry.get("capabilities")
        if not model_id or not isinstance(caps, list):
            print(f"skip {model_id!r}: no capabilities field (wrapper too old?)")
            continue
        toggles = _toggles_for(caps)
        record = by_id.get(model_id)
        current = ((record or {}).get("meta") or {}).get("capabilities") or {}
        if record and all(current.get(k) == v for k, v in toggles.items()):
            unchanged += 1
            continue
        if dry_run:
            print(f"would sync {model_id}: {toggles}")
            changed += 1
            continue
        if record:
            record.setdefault("meta", {})["capabilities"] = toggles
            _request(
                "POST", f"{openwebui}/api/v1/models/model/update?id={model_id}", owui_key, record
            )
            changed += 1
        else:
            _request(
                "POST",
                f"{openwebui}/api/v1/models/create",
                owui_key,
                {
                    "id": model_id,
                    "name": model_id,
                    "meta": {"capabilities": toggles},
                    "params": {},
                },
            )
            created += 1
        print(f"synced {model_id}: {sorted(k for k, v in toggles.items() if v)}")

    print(f"done: {changed} updated, {created} created, {unchanged} already in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
