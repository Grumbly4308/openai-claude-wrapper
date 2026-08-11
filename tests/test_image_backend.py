"""External image backend proxy on /v1/images/generations (phase 5).

With CLAUDE_WRAPPER_IMAGE_BACKEND_URL configured, generations proxy to any
OpenAI-compatible backend instead of the SVG delegation path; unset, the
delegation path is untouched (covered by test_endpoints.py).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="claude-wrapper-test-imgbackend-")
os.environ.setdefault("CLAUDE_WRAPPER_DATA", _TMP)
os.environ.setdefault("CLAUDE_WRAPPER_MODEL_DISCOVERY", "off")
os.environ.pop("CLAUDE_WRAPPER_API_KEYS", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import pytest
from fastapi.testclient import TestClient

from src import routes_images
from src.main import app

client = TestClient(app)


@pytest.fixture
def backend(monkeypatch):
    """Point the images route at a mocked OpenAI-compatible backend."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"created": 1, "data": [{"url": "https://img.example/out.png"}]},
        )

    monkeypatch.setattr(routes_images, "_IMAGE_BACKEND_URL", "http://img.test")
    monkeypatch.setattr(routes_images, "_IMAGE_BACKEND_KEY", "img-key")
    monkeypatch.setattr(routes_images, "_IMAGE_BACKEND_MODEL", "sdxl")
    monkeypatch.setattr(
        routes_images,
        "_backend_client",
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    yield captured
    routes_images._backend_client = None


def test_generations_proxy_to_backend(backend):
    r = client.post(
        "/v1/images/generations",
        json={"prompt": "a lighthouse at dusk", "n": 2, "size": "512x512"},
    )
    assert r.status_code == 200
    assert r.json()["data"][0]["url"] == "https://img.example/out.png"
    assert backend["url"] == "http://img.test/v1/images/generations"
    assert backend["headers"]["authorization"] == "Bearer img-key"
    # Model override wins; prompt/n/size forwarded.
    body = backend["body"]
    assert body["model"] == "sdxl"
    assert body["prompt"] == "a lighthouse at dusk"
    assert body["n"] == 2 and body["size"] == "512x512"


def test_backend_error_maps_to_502(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    monkeypatch.setattr(routes_images, "_IMAGE_BACKEND_URL", "http://img.test")
    monkeypatch.setattr(
        routes_images,
        "_backend_client",
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    r = client.post("/v1/images/generations", json={"prompt": "x"})
    routes_images._backend_client = None
    assert r.status_code == 502
    assert "image backend error 500" in r.json()["detail"]
