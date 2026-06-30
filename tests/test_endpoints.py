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
os.environ["CLAUDE_WRAPPER_MODEL_DISCOVERY"] = "off"
os.environ.pop("CLAUDE_WRAPPER_API_KEYS", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- stub ClaudeRunner before importing the FastAPI app ----
from src.claude_runner import ClaudeResult, ClaudeRunner, StreamEvent

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


async def _stub_run_collect(self, prompt, session_key, model=None, env_extra=None, extra_args=None, effort=None, **_kwargs):
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


async def _stub_run_stream(self, prompt, session_key, model=None, env_extra=None, extra_args=None, effort=None, **_kwargs):
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


def test_chat_completion_effort_surfaced() -> None:
    # Non-stream: an explicit Opus variant is echoed back as applied/request.
    r = client.post(
        "/v1/chat/completions",
        json={"model": "claude-opus-4-8 (high)", "messages": [{"role": "user", "content": "hi"}]},
    )
    eff = r.json().get("effort")
    check(
        "chat.effort.sync",
        bool(eff) and eff["applied"] == "high" and eff["source"] == "request",
        note=str(eff),
    )

    # Bare model falls back to the server default (source != request).
    r = client.post(
        "/v1/chat/completions",
        json={"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}]},
    )
    eff = r.json().get("effort")
    check("chat.effort.default_source", bool(eff) and eff["source"] == "server-default", note=str(eff))

    # Streaming: the first chunk carries the resolved effort.
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "claude-opus-4-8 (ultracode)",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as r:
        chunks = [ln for ln in r.iter_lines() if ln]
    effort_chunk = next((c for c in chunks if c.startswith("data: ") and '"effort"' in c), None)
    eff = json.loads(effort_chunk[6:]).get("effort") if effort_chunk else None
    check(
        "chat.effort.stream",
        bool(eff) and eff["applied"] == "ultracode" and eff["source"] == "request",
        note=str(eff),
    )


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


def test_chat_completion_stream_heartbeat() -> None:
    """A slow runner with no early text must still emit keep-alive comments,
    and thinking must arrive on the reasoning channel, not in answer content."""
    import asyncio as _asyncio

    import src.main as _main

    async def _slow_run_stream(self, prompt, session_key, model=None, env_extra=None, extra_args=None, effort=None, **_kwargs):
        await _asyncio.sleep(0.15)  # silent "thinking" gap — should trigger heartbeats
        yield StreamEvent(kind="thinking", text="pondering")
        yield StreamEvent(kind="text", text="answer")
        yield StreamEvent(
            kind="final",
            text="answer",
            raw={"stop_reason": "stop", "new_outputs": [], "session_uuid": "stub"},
        )

    orig_run_stream = ClaudeRunner.run_stream
    orig_hb = _main._STREAM_HEARTBEAT_SECONDS
    orig_channel = _main._REASONING_CHANNEL
    ClaudeRunner.run_stream = _slow_run_stream
    _main._STREAM_HEARTBEAT_SECONDS = 0.05
    _main._REASONING_CHANNEL = "reasoning_content"  # exercise the reasoning_content channel
    try:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as r:
            raw = r.read().decode()
    finally:
        ClaudeRunner.run_stream = orig_run_stream
        _main._STREAM_HEARTBEAT_SECONDS = orig_hb
        _main._REASONING_CHANNEL = orig_channel

    has_heartbeat = ": keep-alive" in raw
    has_reasoning = '"reasoning_content":"pondering"' in raw
    has_answer = '"content":"answer"' in raw
    has_done = "[DONE]" in raw
    # Thinking must NOT leak into answer content.
    no_thinking_in_content = '"content":"pondering"' not in raw
    check(
        "chat.completions.stream.heartbeat",
        has_heartbeat and has_reasoning and has_answer and has_done and no_thinking_in_content,
        note=f"hb={has_heartbeat} reason={has_reasoning} ans={has_answer} done={has_done} clean={no_thinking_in_content}",
    )


def test_chat_completion_stream_progress_and_activity() -> None:
    """During a long no-answer-text phase the feed must show life: tool/subagent
    activity surfaced as reasoning_content, a periodic visible 'still working'
    tick, and the preamble that flushes buffering proxies — all without leaking
    progress noise into the answer content."""
    import asyncio as _asyncio

    import src.main as _main

    async def _busy_run_stream(self, prompt, session_key, model=None, env_extra=None, extra_args=None, effort=None, **_kwargs):
        yield StreamEvent(kind="tool_use", tool_name="Bash", tool_input={"command": "pytest -q"})
        await _asyncio.sleep(0.18)  # silent stretch -> should trigger a progress tick
        yield StreamEvent(kind="text", text="final answer")
        yield StreamEvent(
            kind="final",
            text="final answer",
            raw={"stop_reason": "stop", "new_outputs": [], "session_uuid": "stub"},
        )

    orig_stream = ClaudeRunner.run_stream
    orig_hb = _main._STREAM_HEARTBEAT_SECONDS
    orig_prog = _main._STREAM_PROGRESS_SECONDS
    orig_channel = _main._REASONING_CHANNEL
    ClaudeRunner.run_stream = _busy_run_stream
    _main._STREAM_HEARTBEAT_SECONDS = 0.04
    _main._STREAM_PROGRESS_SECONDS = 0.08
    _main._REASONING_CHANNEL = "reasoning_content"  # exercise the reasoning_content channel
    try:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "claude-opus-4-8 (max)",
                "messages": [{"role": "user", "content": "hard problem"}],
                "stream": True,
            },
        ) as r:
            raw = r.read().decode()
    finally:
        ClaudeRunner.run_stream = orig_stream
        _main._STREAM_HEARTBEAT_SECONDS = orig_hb
        _main._STREAM_PROGRESS_SECONDS = orig_prog
        _main._REASONING_CHANNEL = orig_channel

    has_preamble = raw.startswith(":  ") and "          " in raw[:2100]
    has_tool = '"reasoning_content":"\\ud83d\\udd27 Bash: pytest -q' in raw or "🔧 Bash: pytest -q" in raw
    has_tick = "Still working" in raw
    has_answer = '"content":"final answer"' in raw
    has_done = "[DONE]" in raw
    # Progress noise rides reasoning_content, never the answer content.
    no_progress_in_content = '"content":"⏳' not in raw and '"content":"🔧' not in raw
    check(
        "chat.completions.stream.progress",
        has_preamble and has_tool and has_tick and has_answer and has_done and no_progress_in_content,
        note=f"preamble={has_preamble} tool={has_tool} tick={has_tick} ans={has_answer} done={has_done} clean={no_progress_in_content}",
    )


def test_chat_completion_stream_think_tags() -> None:
    """With CLAUDE_WRAPPER_REASONING_CHANNEL=think_tags, reasoning rides the
    *content* channel wrapped in a single <think>…</think> block (for Open WebUI
    builds that don't render reasoning_content), closed before the first answer
    token. No reasoning_content frames are emitted in this mode."""
    import asyncio as _asyncio

    import src.main as _main

    async def _run(self, prompt, session_key, model=None, env_extra=None, extra_args=None, effort=None, **_kwargs):
        yield StreamEvent(kind="tool_use", tool_name="Bash", tool_input={"command": "ls"})
        yield StreamEvent(kind="thinking", text="pondering")
        yield StreamEvent(kind="text", text="answer")
        yield StreamEvent(
            kind="final",
            text="answer",
            raw={"stop_reason": "stop", "new_outputs": [], "session_uuid": "stub"},
        )
        await _asyncio.sleep(0)

    orig_run_stream = ClaudeRunner.run_stream
    orig_channel = _main._REASONING_CHANNEL
    ClaudeRunner.run_stream = _run
    _main._REASONING_CHANNEL = "think_tags"
    try:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as r:
            raw = r.read().decode()
    finally:
        ClaudeRunner.run_stream = orig_run_stream
        _main._REASONING_CHANNEL = orig_channel

    opened = '"content":"<think>' in raw
    has_thought = "pondering" in raw
    closed = "</think>" in raw
    has_answer = '"content":"answer"' in raw
    # No reasoning_content channel is used in think_tags mode.
    no_reasoning_channel = "reasoning_content" not in raw
    # Block is balanced and closed before the answer token.
    ordered = (
        opened
        and closed
        and has_answer
        and raw.index("<think>") < raw.index("</think>") < raw.index('"content":"answer"')
    )
    has_done = "[DONE]" in raw
    check(
        "chat.completions.stream.think_tags",
        opened and has_thought and closed and has_answer and no_reasoning_channel and ordered and has_done,
        note=f"open={opened} thought={has_thought} close={closed} ans={has_answer} "
        f"no_rc={no_reasoning_channel} ordered={ordered} done={has_done}",
    )


def test_partial_stream_normalization() -> None:
    """With --include-partial-messages: incremental text/thinking deltas are
    surfaced from stream_event, and the consolidated assistant text/thinking
    blocks are suppressed (so they don't double-emit what the deltas already
    streamed) while tool_use is still taken from the consolidated block."""
    from src.claude_runner import _normalize_stream_event

    think_delta = {
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "hmm"},
        },
    }
    text_delta = {
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "hi"},
        },
    }
    consolidated = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
            ]
        },
    }

    td = _normalize_stream_event(think_delta, partial=True)
    xd = _normalize_stream_event(text_delta, partial=True)
    cons_kinds = [e.kind for e in _normalize_stream_event(consolidated, partial=True)]
    whole_kinds = [e.kind for e in _normalize_stream_event(consolidated, partial=False)]

    delta_thinking = len(td) == 1 and td[0].kind == "thinking" and td[0].text == "hmm"
    delta_text = len(xd) == 1 and xd[0].kind == "text" and xd[0].text == "hi"
    # Partial mode: consolidated text/thinking suppressed, tool_use kept.
    suppressed = "text" not in cons_kinds and "thinking" not in cons_kinds
    tool_kept = "tool_use" in cons_kinds
    # Whole-block mode (no --include-partial-messages): all three still emit.
    whole_ok = {"text", "thinking", "tool_use"} <= set(whole_kinds)

    check(
        "claude_runner.partial_stream_normalization",
        delta_thinking and delta_text and suppressed and tool_kept and whole_ok,
        note=f"think={delta_thinking} text={delta_text} suppressed={suppressed} "
        f"tool={tool_kept} whole={whole_ok}",
    )


def test_chat_completion_stream_terminates_on_post_loop_error() -> None:
    """If post-stream bookkeeping raises after bytes are on the wire, the SSE
    stream must STILL terminate with a clean error frame + [DONE] — never a
    truncated chunked body (which aiohttp clients like Open WebUI surface as
    'TransferEncodingError: Not enough data to satisfy transfer length header')."""
    import src.main as _main

    async def _boom(*a, **k):
        raise OSError("simulated post-loop failure")

    orig = _main._register_generated_files
    _main._register_generated_files = _boom
    try:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as r:
            raw = r.read().decode()
    finally:
        _main._register_generated_files = orig

    frames = [f for f in raw.strip().split("\n\n") if f]
    ends_with_done = frames[-1].strip() == "data: [DONE]"
    has_error_frame = any('"type": "upstream_error"' in f for f in frames)
    # The streamed answer text must still have been delivered before the failure.
    has_answer = '"content":"hello stream"' in raw or '"content":"stream"' in raw
    check(
        "chat.completions.stream.post_loop_error",
        ends_with_done and has_error_frame and has_answer,
        note=f"done={ends_with_done} error_frame={has_error_frame} answer={has_answer}",
    )


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
    body = r.json()
    # Shape: an object response with a message item AND a flattened output_text,
    # and an id that round-trips for chaining (resp_ prefix).
    ok = (
        r.status_code == 200
        and body.get("object") == "response"
        and body.get("output_text")
        and body["output"][0]["content"][0]["type"] == "output_text"
        and str(body.get("id", "")).startswith("resp_")
    )
    check("responses", ok, note=r.text[:200])


def test_responses_api_chaining() -> None:
    """The returned id, passed back as previous_response_id, must reattach to the
    SAME session — otherwise multi-turn Responses threads silently fork."""
    r1 = client.post("/v1/responses", json={"model": "claude-sonnet-4-6", "input": "first turn"})
    first_id = r1.json()["id"]

    r2 = client.post(
        "/v1/responses",
        json={
            "model": "claude-sonnet-4-6",
            "input": "second turn",
            "previous_response_id": first_id,
        },
    )
    # Same session => same derived response id. A fresh random id would mean the
    # chain broke and a new Claude session was started.
    same_session = r2.status_code == 200 and r2.json()["id"] == first_id
    check("responses.chaining", same_session, note=f"{first_id} vs {r2.json().get('id')}")


def test_responses_api_stream() -> None:
    """Streaming must emit the Responses event protocol (typed events), not
    chat.completion chunks, and must NOT use the chat-style [DONE] sentinel."""
    with client.stream(
        "POST",
        "/v1/responses",
        json={"model": "claude-sonnet-4-6", "input": "hi", "stream": True},
    ) as r:
        raw = r.read().decode()

    has_created = '"type":"response.created"' in raw or '"type": "response.created"' in raw
    has_delta = "response.output_text.delta" in raw
    has_completed = "response.completed" in raw
    # The streamed deltas must reconstruct the answer.
    has_text = '"delta":"hello ' in raw or '"delta": "hello ' in raw
    # Responses protocol terminates on response.completed — no chat-style [DONE],
    # and definitely no chat.completion.chunk objects leaking through.
    no_done = "[DONE]" not in raw
    no_chat_chunk = "chat.completion.chunk" not in raw
    check(
        "responses.stream",
        has_created and has_delta and has_completed and has_text and no_done and no_chat_chunk,
        note=f"created={has_created} delta={has_delta} done_ev={has_completed} "
        f"text={has_text} no_done={no_done} no_chat={no_chat_chunk}",
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
        test_chat_completion_effort_surfaced,
        test_chat_completion_stream,
        test_chat_completion_stream_heartbeat,
        test_chat_completion_stream_progress_and_activity,
        test_legacy_completions,
        test_responses_api,
        test_responses_api_chaining,
        test_responses_api_stream,
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
