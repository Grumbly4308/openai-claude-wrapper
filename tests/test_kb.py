"""Unit tests for the OpenWebUI knowledge-base addendum.

The addendum is injected into Claude's system prompt only when
OPENWEBUI_BASE_URL is set, and teaches Claude to (1) discover knowledge-base
ids via GET /api/v1/knowledge/ and (2) query them by id via
POST /api/v1/retrieval/query/collection. These tests pin that contract so the
self-query path can't silently regress.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
from pathlib import Path

# config.SETTINGS is built at import time and mkdirs its data dir — point it at
# a tempdir so importing src.* never touches /data.
_TMP = tempfile.mkdtemp(prefix="claude-wrapper-kb-test-")
os.environ["CLAUDE_WRAPPER_DATA"] = _TMP
os.environ["CLAUDE_WRAPPER_MODEL_DISCOVERY"] = "off"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.converters as conv  # noqa: E402

_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, note: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"PASS  {name}")
    else:
        _FAIL += 1
        print(f"FAIL  {name} {note}")
    assert cond, f"{name} {note}"


def _preparer() -> conv.MessagePreparer:
    # _kb_addendum reads only SETTINGS, so the file store / workspace are unused.
    return conv.MessagePreparer(file_store=None, workspace_root=Path(_TMP))


def _with_settings(**overrides):
    return dataclasses.replace(conv.SETTINGS, **overrides)


def test_disabled_when_no_base_url() -> None:
    conv.SETTINGS = _with_settings(openwebui_base_url="")
    check("kb.disabled.empty", _preparer()._kb_addendum() == "")


def test_enabled_addendum_has_discover_and_query() -> None:
    conv.SETTINGS = _with_settings(
        openwebui_base_url="http://owui:8080",
        openwebui_api_key="sk-x",
        openwebui_default_collection="kb-abc123",
    )
    out = _preparer()._kb_addendum()
    check("kb.discover.endpoint", "/api/v1/knowledge/" in out, note=out)
    check("kb.query.endpoint", "/api/v1/retrieval/query/collection" in out)
    check("kb.body.collection_names", "collection_names" in out)
    # The single most important correctness note: query by id, not display name.
    check("kb.id_not_name", "NOT its display name" in out)
    check("kb.auth.header", "Authorization: Bearer $OPENWEBUI_API_KEY" in out)
    check("kb.default.hint", "kb-abc123" in out)


def test_no_auth_header_without_key() -> None:
    conv.SETTINGS = _with_settings(
        openwebui_base_url="http://owui:8080",
        openwebui_api_key="",
        openwebui_default_collection="",
    )
    out = _preparer()._kb_addendum()
    check("kb.noauth.absent", "Authorization: Bearer" not in out, note=out)
    check("kb.noauth.still_enabled", "/api/v1/knowledge/" in out)
    check("kb.nodefault.no_hint", "default to" not in out)


def main() -> int:
    tests = [
        test_disabled_when_no_base_url,
        test_enabled_addendum_has_discover_and_query,
        test_no_auth_header_without_key,
    ]
    for t in tests:
        try:
            t()
        except AssertionError:
            pass
        except Exception as e:
            check(t.__name__, False, note=f"exception: {e!r}")
    print(f"\nRESULT pass={_PASS} fail={_FAIL}")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
