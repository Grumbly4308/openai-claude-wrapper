"""End-to-end endpoint smoke tests.

Stubs ``ClaudeRunner`` so no real Claude Code invocation happens: each
delegated endpoint writes the expected output files into its
per-request workspace directly. Exercises every OpenAI-shaped route to
catch shape regressions before packaging.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
from pathlib import Path


# ---- environment setup before importing anything from src ----
_TMP = tempfile.mkdtemp(prefix="claude-wrapper-test-")
os.environ["CLAUDE_WRAPPER_DATA"] = _TMP
os.environ["CLAUDE_WRAPPER_DEFAULT_MODEL"] = "claude-sonnet-4-6"
os.environ.pop("CLAUDE_WRAPPER_API_KEYS", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- stub ClaudeRunner before importing the FastAPI app ----
from src.claude_runner import ClaudeResult, ClaudeRunner, StreamEvent

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


async def _stub_run_collect(self, prompt, session_key, model=None, env_extra=None, extra_args=None):
    cwd = self.workspace_root / session_key
    (cwd / "outputs").mkdir(parents=True, exist_ok=True)
    text = "ok"

    if session_key.startswith("moderate-"):
        text = json.dumps(
            {
                "flagged": False,
                "categories": {"hate": False, "violence": False, "sexual": False},
                "category_scores": {"hate": 0.01, "violence": 0.0, "sexual": 0.0},
            }
        )
    elif session_key.startswith("stt-"):
        (cwd / "outputs" / "transcript.json").write_text('{"text": "hello world"}')
        (cwd / "outputs" / "transcript.txt").write_text("hello world")
    elif session_key.startswith("tts-"):
        (cwd / "outputs" / "speech.mp3").write_bytes(b"ID3\x00stubaudio")
    elif session_key.startswith("imggen-"):
        (cwd / "outputs" / "gen-1.png").write_bytes(TINY_PNG)
    elif session_key.startswith("imgedit-"):
        (cwd / "outputs" / "edit-1.png").write_bytes(TINY_PNG)
    elif session_key.startswith("imgvar-"):
        (cwd / "outputs" / "var-1.png").write_bytes(TINY_PNG)
    elif session_key.startswith("embed-"):
        texts = json.loads((cwd / "uploads" / "texts.json").read_text())
        (cwd / "outputs" / "embeddings.json").write_text(json.dumps([[0.1] * 384 for _ in texts]))
    else:
        text = "hello from stub claude"

    return ClaudeResult(
        session_uuid="stub-uuid",
        final_text=text,
        stop_reason="stop",
        input_tokens=1,
        output_tokens=1,
        events=[StreamEvent(kind="system", raw={"new_outputs": []})],
    )


async def _stub_run_stream(self, prompt, session_key, model=None, env_extra=None, extra_args=None):
    yield StreamEvent(kind="text", text="hello ")
    yield StreamEvent(kind="text", text="stream")
    yield StreamEvent(
        kind="final",
        text="hello stream",
        raw={"stop_reason": "stop", "new_outputs": [], "session_uuid": "stub"},
    )


ClaudeRunner.run_collect = _stub_run_collect
ClaudeRunner.run_stream = _stub_run_stream

# ---- now import the app ----
from fastapi.testclient import TestClient  # noqa: E402

from src.main import app  # noqa: E402


client = TestClient(app)


# ---------- helpers ----------

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


# ---------- health / models ----------


def test_health() -> None:
    r = client.get("/healthz")
    check("health", r.status_code == 200 and r.json() == {"status": "ok"})


def test_models() -> None:
    r = client.get("/v1/models")
    check("models.list", r.status_code == 200 and isinstance(r.json().get("data"), list))
    r = client.get("/v1/models/claude-sonnet-4-6")
    check("models.retrieve", r.status_code == 200 and r.json()["id"] == "claude-sonnet-4-6")
    r = client.get("/v1/models/does-not-exist")
    check("models.retrieve.404", r.status_code == 404)


# ---------- chat / completions / responses ----------


def test_chat_completion() -> None:
    r = client.post(
        "/v1/chat/completions",
        json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]},
    )
    check("chat.completions", r.status_code == 200 and r.json()["choices"][0]["message"]["content"])


def test_chat_completion_stream() -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as r:
        chunks = [ln for ln in r.iter_lines() if ln]
    has_done = any("[DONE]" in c for c in chunks)
    check("chat.completions.stream", has_done and len(chunks) >= 2)


def test_legacy_completions() -> None:
    r = client.post(
        "/v1/completions",
        json={"model": "claude-sonnet-4-6", "prompt": "hello"},
    )
    check(
        "completions",
        r.status_code == 200 and r.json()["choices"][0]["text"],
        note=r.text[:200],
    )


def test_responses_api() -> None:
    r = client.post(
        "/v1/responses",
        json={"model": "claude-sonnet-4-6", "input": "hi"},
    )
    check(
        "responses",
        r.status_code == 200 and r.json()["output_text"],
        note=r.text[:200],
    )


# ---------- embeddings ----------


def test_embeddings() -> None:
    r = client.post("/v1/embeddings", json={"input": ["foo", "bar"], "model": "text-embedding-3-small"})
    body = r.json()
    ok = (
        r.status_code == 200
        and len(body["data"]) == 2
        and isinstance(body["data"][0]["embedding"], list)
        and len(body["data"][0]["embedding"]) > 0
    )
    check("embeddings", ok, note=r.text[:200])


def test_embeddings_base64() -> None:
    r = client.post(
        "/v1/embeddings",
        json={"input": "foo", "encoding_format": "base64"},
    )
    body = r.json()
    check(
        "embeddings.base64",
        r.status_code == 200 and isinstance(body["data"][0]["embedding"], str),
    )


# ---------- moderations ----------


def test_moderations() -> None:
    r = client.post("/v1/moderations", json={"input": "hello"})
    body = r.json()
    ok = r.status_code == 200 and "flagged" in body["results"][0]
    check("moderations", ok, note=r.text[:200])


# ---------- audio ----------


def test_transcriptions() -> None:
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("audio.wav", b"RIFF....WAVE", "audio/wav")},
        data={"model": "whisper-1"},
    )
    check("audio.transcriptions", r.status_code == 200 and "text" in r.json(), note=r.text[:200])


def test_translations() -> None:
    r = client.post(
        "/v1/audio/translations",
        files={"file": ("audio.wav", b"RIFF....WAVE", "audio/wav")},
        data={"model": "whisper-1", "response_format": "text"},
    )
    check("audio.translations", r.status_code == 200, note=r.text[:200])


def test_speech() -> None:
    r = client.post(
        "/v1/audio/speech",
        json={"model": "claude-tts", "input": "hello", "voice": "en", "response_format": "mp3"},
    )
    check(
        "audio.speech",
        r.status_code == 200 and r.headers.get("content-type", "").startswith("audio/"),
        note=r.text[:200] if r.status_code != 200 else "",
    )


# ---------- images ----------


def test_image_gen() -> None:
    r = client.post(
        "/v1/images/generations",
        json={"prompt": "a red circle", "n": 1, "size": "256x256", "response_format": "b64_json"},
    )
    body = r.json()
    ok = r.status_code == 200 and body["data"] and "b64_json" in body["data"][0]
    check("images.generations", ok, note=r.text[:200])


def test_image_edit() -> None:
    r = client.post(
        "/v1/images/edits",
        data={"prompt": "invert colors", "n": 1, "size": "256x256"},
        files={"image": ("x.png", TINY_PNG, "image/png")},
    )
    check("images.edits", r.status_code == 200 and r.json()["data"], note=r.text[:200])


def test_image_var() -> None:
    r = client.post(
        "/v1/images/variations",
        data={"n": 1, "size": "256x256"},
        files={"image": ("x.png", TINY_PNG, "image/png")},
    )
    check("images.variations", r.status_code == 200 and r.json()["data"], note=r.text[:200])


# ---------- files ----------


def test_files_cycle() -> None:
    r = client.post("/v1/files", files={"file": ("a.txt", b"hello", "text/plain")}, data={"purpose": "user_data"})
    fid = r.json().get("id")
    check("files.upload", r.status_code == 200 and fid)

    r = client.get(f"/v1/files/{fid}")
    check("files.retrieve", r.status_code == 200 and r.json()["id"] == fid)

    r = client.get(f"/v1/files/{fid}/content")
    check("files.content", r.status_code == 200 and b"hello" in r.content)

    r = client.get("/v1/files")
    check("files.list", r.status_code == 200 and any(f["id"] == fid for f in r.json()["data"]))

    r = client.delete(f"/v1/files/{fid}")
    check("files.delete", r.status_code == 200 and r.json()["deleted"])


# ---------- batches ----------


def test_batches() -> None:
    """Batches run as asyncio.create_task — exercise via a persistent
    async client so the background task has a live loop."""
    import asyncio

    import httpx

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://batchtest") as ac:
            batch_lines = "\n".join(
                json.dumps(
                    {
                        "custom_id": f"r{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "claude-sonnet-4-6",
                            "messages": [{"role": "user", "content": f"item {i}"}],
                        },
                    }
                )
                for i in range(2)
            )
            r = await ac.post(
                "/v1/files",
                files={"file": ("batch.jsonl", batch_lines.encode(), "application/jsonl")},
                data={"purpose": "batch"},
            )
            fid = r.json()["id"]

            r = await ac.post(
                "/v1/batches",
                json={
                    "input_file_id": fid,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
            )
            bid = r.json().get("id")
            check("batches.create", r.status_code == 200 and bool(bid), note=r.text[:200])

            status: dict = {}
            for _ in range(50):
                await asyncio.sleep(0.1)
                status = (await ac.get(f"/v1/batches/{bid}")).json()
                if status.get("status") in ("completed", "failed"):
                    break
            check(
                "batches.complete",
                status.get("status") == "completed" and status["request_counts"]["completed"] == 2,
                note=str(status),
            )
            r = await ac.get("/v1/batches")
            check("batches.list", r.status_code == 200)

    asyncio.run(_run())


# ---------- assistants + threads + runs ----------


def test_assistants_flow() -> None:
    r = client.post(
        "/v1/assistants",
        json={"model": "claude-sonnet-4-6", "name": "tester", "instructions": "Be concise."},
    )
    aid = r.json().get("id")
    check("assistants.create", r.status_code == 200 and aid, note=r.text[:200])

    check("assistants.list", client.get("/v1/assistants").status_code == 200)
    check("assistants.retrieve", client.get(f"/v1/assistants/{aid}").status_code == 200)

    r = client.post("/v1/threads", json={})
    tid = r.json().get("id")
    check("threads.create", r.status_code == 200 and tid)

    r = client.post(f"/v1/threads/{tid}/messages", json={"role": "user", "content": "hi"})
    check("threads.messages.create", r.status_code == 200 and r.json()["role"] == "user")

    check("threads.messages.list", client.get(f"/v1/threads/{tid}/messages").status_code == 200)

    r = client.post(f"/v1/threads/{tid}/runs", json={"assistant_id": aid})
    rid = r.json().get("id")
    check("threads.runs.create", r.status_code == 200 and r.json()["status"] == "completed", note=r.text[:200])
    check("threads.runs.retrieve", client.get(f"/v1/threads/{tid}/runs/{rid}").status_code == 200)

    check("assistants.delete", client.delete(f"/v1/assistants/{aid}").json()["deleted"])
    check("threads.delete", client.delete(f"/v1/threads/{tid}").json()["deleted"])


# ---------- vector stores ----------


def test_vector_stores() -> None:
    r = client.post("/v1/files", files={"file": ("doc.txt", b"cats are animals. dogs bark loudly.", "text/plain")}, data={"purpose": "user_data"})
    fid = r.json()["id"]

    r = client.post("/v1/vector_stores", json={"name": "test", "file_ids": [fid]})
    sid = r.json().get("id")
    check("vector_stores.create", r.status_code == 200 and sid, note=r.text[:200])

    check("vector_stores.list", client.get("/v1/vector_stores").status_code == 200)
    check("vector_stores.retrieve", client.get(f"/v1/vector_stores/{sid}").status_code == 200)
    check("vector_stores.files.list", client.get(f"/v1/vector_stores/{sid}/files").status_code == 200)

    r = client.post(f"/v1/vector_stores/{sid}/search", json={"query": "what barks?", "max_num_results": 2})
    body = r.json()
    check("vector_stores.search", r.status_code == 200 and "data" in body, note=r.text[:200])

    check("vector_stores.delete", client.delete(f"/v1/vector_stores/{sid}").json()["deleted"])


# ---------- fine-tuning ----------


def test_fine_tuning() -> None:
    r = client.post("/v1/fine_tuning/jobs", json={"training_file": "file-x", "model": "claude"})
    check("fine_tuning.create.501", r.status_code == 501)
    check("fine_tuning.list", client.get("/v1/fine_tuning/jobs").status_code == 200)


# ---------- realtime ----------


def test_realtime_discovery() -> None:
    r = client.get("/v1/realtime/sessions")
    check("realtime.discovery", r.status_code == 200 and r.json()["object"] == "realtime.session")


def test_realtime_websocket() -> None:
    try:
        with client.websocket_connect("/v1/realtime") as ws:
            created = json.loads(ws.receive_text())
            assert created["type"] == "session.created"
            ws.send_text(json.dumps({"type": "response.create", "input": "hi"}))
            events = []
            for _ in range(10):
                try:
                    events.append(json.loads(ws.receive_text()))
                except Exception:
                    break
                if events[-1].get("type") == "response.completed":
                    break
        ok = any(e.get("type") == "response.output_text.delta" for e in events) and events[-1].get("type") == "response.completed"
        check("realtime.websocket", ok, note=str(events[:3]))
    except Exception as e:
        check("realtime.websocket", False, note=repr(e))


# ---------- run ----------


def main() -> int:
    tests = [
        test_health,
        test_models,
        test_chat_completion,
        test_chat_completion_stream,
        test_legacy_completions,
        test_responses_api,
        test_embeddings,
        test_embeddings_base64,
        test_moderations,
        test_transcriptions,
        test_translations,
        test_speech,
        test_image_gen,
        test_image_edit,
        test_image_var,
        test_files_cycle,
        test_batches,
        test_assistants_flow,
        test_vector_stores,
        test_fine_tuning,
        test_realtime_discovery,
        test_realtime_websocket,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            check(t.__name__, False, note=f"exception: {e!r}")
    print(f"\nRESULT pass={_PASS} fail={_FAIL}")
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
