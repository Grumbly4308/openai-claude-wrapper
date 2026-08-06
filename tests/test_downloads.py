"""Browser-clickable download links: capability tokens, origin, route auth.

Deliberately NOT in test_endpoints.py, whose check() has no assert and swallows
every failure under pytest. Settings cannot come from os.environ either: src.config
is imported once per pytest process by whichever module collects first
(alphabetically test_budget.py), so a preamble here would be ignored under a full
run and only work when this file is run alone. Everything is varied with
dataclasses.replace against the *consuming* module's SETTINGS binding — each
consumer did `from .config import SETTINGS` and holds its own reference — via
monkeypatch, so nothing leaks into later modules.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import re
import sys
import tempfile
import time
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="claude-wrapper-downloads-")
os.environ.setdefault("CLAUDE_WRAPPER_DATA", _TMP)
os.environ.setdefault("CLAUDE_WRAPPER_MODEL_DISCOVERY", "off")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from src import deps as src_deps
from src import download_tokens, request_origin
from src import main as src_main
from src.deps import FILE_STORE
from src.file_store import FileStore
from src.main import app

client = TestClient(app)

KEY = "test-signing-key"


# ---------- download_tokens (pure) ----------


def test_mint_verify_round_trip():
    q = download_tokens.mint("file-" + "a" * 32, KEY, 3600)
    exp, sig = _split(q)
    assert download_tokens.verify("file-" + "a" * 32, exp, sig, KEY) is True


def test_signature_is_scoped_to_one_file():
    a, b = "file-" + "a" * 32, "file-" + "b" * 32
    exp, sig = _split(download_tokens.mint(a, KEY, 3600))
    assert download_tokens.verify(b, exp, sig, KEY) is False


def test_tampered_expiry_is_rejected():
    """The expiry is inside the MAC, so it cannot be extended by the holder."""
    fid = "file-" + "c" * 32
    exp, sig = _split(download_tokens.mint(fid, KEY, 3600))
    assert download_tokens.verify(fid, str(int(exp) + 86400), sig, KEY) is False


def test_tampered_or_malformed_inputs_are_rejected():
    fid = "file-" + "d" * 32
    exp, sig = _split(download_tokens.mint(fid, KEY, 3600))
    assert download_tokens.verify(fid, exp, sig + "x", KEY) is False
    assert download_tokens.verify(fid, "not-a-number", sig, KEY) is False
    assert download_tokens.verify(fid, None, sig, KEY) is False
    assert download_tokens.verify(fid, exp, None, KEY) is False
    assert download_tokens.verify(fid, exp, sig, "other-key") is False


def test_expired_signature_is_rejected():
    fid = "file-" + "e" * 32
    past = int(time.time()) - 60
    assert download_tokens.verify(fid, str(past), download_tokens.signature(fid, past, KEY), KEY) is False


def test_zero_ttl_never_expires_but_is_still_signed():
    fid = "file-" + "f" * 32
    exp, sig = _split(download_tokens.mint(fid, KEY, 0))
    assert exp == "0"
    assert download_tokens.verify(fid, exp, sig, KEY) is True
    # ...and exp=0 cannot be forged onto a deployment that signs a real TTL.
    assert download_tokens.verify(fid, "0", download_tokens.signature(fid, 999, KEY), KEY) is False


def test_empty_key_mints_nothing_and_verifies_nothing():
    fid = "file-" + "a" * 32
    assert download_tokens.mint(fid, "", 3600) == ""
    assert download_tokens.verify(fid, "0", "anything", "") is False


def _split(query: str) -> tuple[str, str]:
    parts = dict(p.split("=", 1) for p in query.split("&"))
    return parts["exp"], parts["sig"]


# ---------- _file_download_url ----------


def _main_settings(monkeypatch, **overrides):
    monkeypatch.setattr(
        src_main, "SETTINGS", dataclasses.replace(src_main.SETTINGS, **overrides)
    )


def test_public_base_url_wins_over_derived_origin(monkeypatch):
    _main_settings(monkeypatch, public_base_url="https://configured", download_signing_key="")
    monkeypatch.setattr(src_main, "current_origin", lambda: "https://derived")
    assert src_main._file_download_url("file-1") == "https://configured/v1/files/file-1/content"


def test_derived_origin_used_when_public_base_url_is_empty(monkeypatch):
    _main_settings(monkeypatch, public_base_url="", download_signing_key="")
    monkeypatch.setattr(src_main, "current_origin", lambda: "https://derived")
    assert src_main._file_download_url("file-1") == "https://derived/v1/files/file-1/content"


def test_no_base_and_no_origin_returns_none(monkeypatch):
    """Back-compat: the caller then degrades to the plain-text trailer."""
    _main_settings(monkeypatch, public_base_url="")
    monkeypatch.setattr(src_main, "current_origin", lambda: "")
    assert src_main._file_download_url("file-1") is None


def test_query_present_only_when_a_signing_key_is_configured(monkeypatch):
    _main_settings(monkeypatch, public_base_url="https://x", download_signing_key="")
    assert "?" not in src_main._file_download_url("file-1")

    _main_settings(
        monkeypatch,
        public_base_url="https://x",
        download_signing_key=KEY,
        download_url_ttl_seconds=3600,
    )
    url = src_main._file_download_url("file-1")
    assert re.fullmatch(r"https://x/v1/files/file-1/content\?exp=\d+&sig=[A-Za-z0-9_-]+", url)


# ---------- _append_file_references (both branches) ----------


ATTACHMENT = {"id": "file-9", "filename": "report.csv", "mime_type": "text/csv", "bytes": 12}


def test_trailer_renders_a_markdown_link_when_a_url_is_available(monkeypatch):
    _main_settings(monkeypatch, public_base_url="https://x", download_signing_key="")
    out = src_main._append_file_references("done", [ATTACHMENT])
    assert out == (
        "done\n\nGenerated files:\n"
        "- [report.csv](https://x/v1/files/file-9/content) (text/csv, 12 bytes, file_id=`file-9`)"
    )


def test_trailer_degrades_to_plain_text_without_a_url(monkeypatch):
    _main_settings(monkeypatch, public_base_url="")
    monkeypatch.setattr(src_main, "current_origin", lambda: "")
    out = src_main._append_file_references("done", [ATTACHMENT])
    assert out == "done\n\nGenerated files:\n- report.csv (text/csv, 12 bytes) → file_id=file-9"


# ---------- _resolve_workspace_hint ----------
#
# The hint is what makes Claude write a deliverable to a file instead of pasting
# it inline, so it is the switch the whole download feature hangs off. It had no
# coverage at all: replacing the body with a constant True or a constant False
# left the suite green.


class _Req:
    def __init__(self, response_format=None):
        self.response_format = response_format


class _Fmt:
    def __init__(self, type_):
        self.type = type_
        self.json_schema = None


def test_workspace_hint_on_for_an_ordinary_turn():
    assert src_main._resolve_workspace_hint(_Req()) is True
    assert src_main._resolve_workspace_hint(_Req(_Fmt("text"))) is True


def test_workspace_hint_off_in_json_mode():
    """A structured-output client wants the value in the reply body. Nudging
    Claude to put the deliverable in a file instead would starve it."""
    assert src_main._resolve_workspace_hint(_Req(_Fmt("json_object"))) is False
    assert src_main._resolve_workspace_hint(_Req(_Fmt("json_schema"))) is False


# ---------- origin derivation through a real request ----------


def _origin_seen_by(headers=None, root_path="") -> str:
    scope_headers = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return request_origin._origin_from_scope(
        {"type": "http", "scheme": "http", "headers": scope_headers, "root_path": root_path}
    )


def test_origin_from_scope():
    assert _origin_seen_by({"host": "wrapper:8000"}) == "http://wrapper:8000"
    # X-Forwarded-* wins: uvicorn only trusts proxy headers from
    # forwarded_allow_ips, so scope["scheme"] is http behind a TLS terminator.
    assert (
        _origin_seen_by({"host": "internal", "x-forwarded-host": "chat.example", "x-forwarded-proto": "https"})
        == "https://chat.example"
    )
    assert _origin_seen_by({"host": "wrapper", "x-forwarded-proto": "https, http"}) == "https://wrapper"
    assert _origin_seen_by({"host": "wrapper"}, root_path="/api") == "http://wrapper/api"
    assert _origin_seen_by({}) == ""


def test_derive_base_url_off_leaves_the_origin_empty(monkeypatch):
    monkeypatch.setattr(
        request_origin,
        "SETTINGS",
        dataclasses.replace(request_origin.SETTINGS, derive_base_url=False),
    )
    captured: list[str] = []

    @app.get("/_test_origin_off")
    async def _probe():
        captured.append(request_origin.current_origin())
        return {}

    try:
        client.get("/_test_origin_off", headers={"Host": "wrapper"})
        assert captured == [""]
    finally:
        app.router.routes = [r for r in app.router.routes if getattr(r, "path", "") != "/_test_origin_off"]


def test_origin_is_live_inside_the_request(monkeypatch):
    captured: list[str] = []

    @app.get("/_test_origin_on")
    async def _probe():
        captured.append(request_origin.current_origin())
        return {}

    try:
        client.get(
            "/_test_origin_on",
            headers={"X-Forwarded-Proto": "https", "X-Forwarded-Host": "chat.example"},
        )
        assert captured == ["https://chat.example"]
    finally:
        app.router.routes = [r for r in app.router.routes if getattr(r, "path", "") != "/_test_origin_on"]
    # The ContextVar is scoped to the request: at module scope it reads empty,
    # which is what keeps a caller outside any request on the plain-text trailer
    # rather than minting a link to whatever host asked last.
    assert request_origin.current_origin() == ""


def test_a_task_spawned_in_a_request_inherits_that_origin():
    """Documents real behavior that the module comment used to deny.

    routes_batches.py launches its worker with asyncio.create_task inside the
    POST handler. create_task snapshots the current context, so the worker keeps
    the submitting request's origin for the batch's whole life -- including after
    the middleware's finally-block reset. Batch outputs therefore DO get links,
    addressed to the origin that submitted the batch. That is defensible (it is
    the only origin that caller ever had), but it is the opposite of what the
    code claimed, so pin it rather than leave it to be rediscovered.
    """

    async def scenario():
        seen: list[str] = []

        async def worker():
            seen.append(request_origin.current_origin())

        token = request_origin._ORIGIN.set("https://chat.example")
        task = asyncio.create_task(worker())      # as routes_batches.py does
        request_origin._ORIGIN.reset(token)       # as the middleware's finally does
        await task
        return seen

    assert asyncio.run(scenario()) == ["https://chat.example"]
    assert request_origin.current_origin() == ""


# ---------- the download route ----------


@pytest.fixture
def stored_file():
    return asyncio.run(
        FILE_STORE.save_bytes(b"a,b\n1,2\n", filename="report.csv", purpose="assistant_output")
    )


def _require_auth(monkeypatch, signing_key=KEY):
    """Turn auth on for src.deps only — that is where auth_dependency reads it."""
    monkeypatch.setattr(
        src_deps,
        "SETTINGS",
        dataclasses.replace(
            src_deps.SETTINGS,
            api_keys=frozenset({"secret-key"}),
            require_auth=True,
            download_signing_key=signing_key,
        ),
    )


def _get(file_id, query="", headers=None):
    return client.get(f"/v1/files/{file_id}/content{query}", headers=headers or {})


def test_valid_signature_downloads_without_a_header(monkeypatch, stored_file):
    _require_auth(monkeypatch)
    r = _get(stored_file.id, "?" + download_tokens.mint(stored_file.id, KEY, 3600))
    assert r.status_code == 200
    assert r.content == b"a,b\n1,2\n"


def test_missing_and_bad_signatures_are_rejected(monkeypatch, stored_file):
    _require_auth(monkeypatch)
    assert _get(stored_file.id).status_code == 401
    bad = _get(stored_file.id, "?exp=99999999999&sig=nonsense")
    assert bad.status_code == 401
    assert bad.json()["detail"] == "invalid or expired download link"


def test_non_ascii_signature_is_a_401_not_a_500(monkeypatch, stored_file):
    # hmac.compare_digest raises TypeError on a non-ASCII str. This is the one
    # route reachable without a credential, so a mangled link has to fail closed
    # with a 401 rather than a traceback per request. Needs an unexpired exp to
    # get past the earlier short-circuits and actually reach the compare.
    _require_auth(monkeypatch)
    bad = _get(stored_file.id, "?exp=99999999999&sig=%C3%A9")
    assert bad.status_code == 401
    assert bad.json()["detail"] == "invalid or expired download link"


def test_expired_signature_is_rejected_by_the_route(monkeypatch, stored_file):
    _require_auth(monkeypatch)
    past = int(time.time()) - 60
    sig = download_tokens.signature(stored_file.id, past, KEY)
    assert _get(stored_file.id, f"?exp={past}&sig={sig}").status_code == 401


def test_a_signature_for_another_file_is_rejected(monkeypatch, stored_file):
    _require_auth(monkeypatch)
    other = "file-" + "0" * 32
    q = download_tokens.mint(other, KEY, 3600)
    assert _get(stored_file.id, "?" + q).status_code == 401


def test_api_key_header_alone_still_works(monkeypatch, stored_file):
    """Pins the SDK-client contract (and test_endpoints.py's files.content check)."""
    _require_auth(monkeypatch)
    r = _get(stored_file.id, headers={"Authorization": "Bearer secret-key"})
    assert r.status_code == 200
    # A bogus signature alongside a valid key must not turn a 200 into a 401.
    r2 = _get(stored_file.id, "?exp=1&sig=nope", headers={"Authorization": "Bearer secret-key"})
    assert r2.status_code == 200


def test_wrong_api_key_is_still_rejected(monkeypatch, stored_file):
    _require_auth(monkeypatch)
    assert _get(stored_file.id, headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_auth_off_needs_neither_header_nor_signature(stored_file):
    # require_auth is False for the whole test process (every module pops
    # CLAUDE_WRAPPER_API_KEYS), i.e. today's behavior, unchanged.
    assert _get(stored_file.id).status_code == 200


def test_malformed_file_ids_are_rejected_by_the_store(monkeypatch, stored_file):
    assert asyncio.run(FILE_STORE.get("../../etc/passwd")) is None
    assert asyncio.run(FILE_STORE.get("file-XYZ")) is None
    assert asyncio.run(FILE_STORE.delete("../../etc")) is False
    assert _get("file-XYZ").status_code == 404
    assert _get("..%2F..%2Fetc%2Fpasswd").status_code in (404, 400)


def test_the_id_guard_stops_delete_from_escaping_the_store(tmp_path):
    """The guard is load-bearing, not decorative.

    delete() does shutil.rmtree(root / file_id) on an unchecked join, so a
    traversing id recursively deletes an arbitrary directory. The assertions
    above cannot show this: their inputs resolve to paths that do not exist, so
    they return None/False via the .exists() check whether or not the guard is
    present, and pass identically with it removed. This one puts a real
    directory at the other end of the traversal, so it fails if the guard goes.
    """
    root = tmp_path / "store"
    root.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("do not delete me")

    store = FileStore(root)
    assert asyncio.run(store.delete("../victim")) is False
    assert victim.exists(), "traversing id escaped the store root and deleted a sibling"
    assert (victim / "keep.txt").read_text() == "do not delete me"


def test_minted_url_authenticates_against_the_live_route(monkeypatch, stored_file):
    """Round trip: the exact URL the trailer emits actually downloads the file."""
    _require_auth(monkeypatch)
    _main_settings(
        monkeypatch,
        public_base_url="http://testserver",
        download_signing_key=KEY,
        download_url_ttl_seconds=3600,
    )
    url = src_main._file_download_url(stored_file.id)
    r = client.get(url.removeprefix("http://testserver"))
    assert r.status_code == 200
    assert r.content == b"a,b\n1,2\n"
