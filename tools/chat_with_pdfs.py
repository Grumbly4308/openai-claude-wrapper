#!/usr/bin/env python3
"""Send a chat completion to the wrapper with PDFs attached as binary file_ids.

Bypasses Open-WebUI's text-extraction pipeline so the wrapper's PDF inliner
(see src/converters.py:_render_file_block) gets the actual binary and can
hand the full extracted text to Claude.

Usage:
    chat_with_pdfs.py --pdf book1.pdf --pdf book2.pdf \
        --prompt "summarize chapter 1 of each"

    chat_with_pdfs.py --pdf book.pdf --prompt-file notes.md \
        --base-url http://localhost:8000 --api-key $WRAPPER_KEY \
        --model 'claude-opus-4-7[1m]'

Env defaults:
    WRAPPER_BASE_URL   default base URL (else http://localhost:8000)
    WRAPPER_API_KEY    bearer token (omit if wrapper has auth disabled)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx


def upload_pdf(client: httpx.Client, base: str, headers: dict, path: Path) -> str:
    with path.open("rb") as f:
        resp = client.post(
            f"{base}/v1/files",
            headers=headers,
            files={"file": (path.name, f, "application/pdf")},
            data={"purpose": "user_data"},
            timeout=600,
        )
    resp.raise_for_status()
    return resp.json()["id"]


def chat(
    client: httpx.Client,
    base: str,
    headers: dict,
    model: str,
    prompt: str,
    file_ids: list[str],
    stream: bool,
) -> None:
    content: list[dict] = []
    for fid in file_ids:
        content.append({"type": "file", "file": {"file_id": fid}})
    content.append({"type": "text", "text": prompt})

    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }

    if not stream:
        resp = client.post(
            f"{base}/v1/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json=body,
            timeout=1800,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        print(text)
        return

    with client.stream(
        "POST",
        f"{base}/v1/chat/completions",
        headers={**headers, "Content-Type": "application/json"},
        json=body,
        timeout=1800,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload.strip() == "[DONE]":
                print()
                return
            try:
                evt = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in evt.get("choices", []):
                delta = choice.get("delta") or {}
                chunk = delta.get("content")
                if chunk:
                    print(chunk, end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Chat with the wrapper with PDFs attached as binary.")
    p.add_argument("--pdf", action="append", type=Path, required=True, help="PDF to attach (repeatable)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="User prompt text")
    g.add_argument("--prompt-file", type=Path, help="Read user prompt from a file")
    p.add_argument(
        "--base-url",
        default=os.environ.get("WRAPPER_BASE_URL", "http://localhost:8000"),
    )
    p.add_argument("--api-key", default=os.environ.get("WRAPPER_API_KEY", ""))
    p.add_argument("--model", default="claude-opus-4-7[1m]")
    p.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = p.parse_args(argv)

    for pdf in args.pdf:
        if not pdf.exists():
            print(f"error: {pdf} does not exist", file=sys.stderr)
            return 2

    prompt = args.prompt or args.prompt_file.read_text(encoding="utf-8")

    headers: dict[str, str] = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    base = args.base_url.rstrip("/")

    with httpx.Client() as client:
        file_ids: list[str] = []
        for pdf in args.pdf:
            print(f"uploading {pdf.name}...", file=sys.stderr)
            fid = upload_pdf(client, base, headers, pdf)
            print(f"  file_id={fid}", file=sys.stderr)
            file_ids.append(fid)

        print(f"--- streaming response from {args.model} ---", file=sys.stderr)
        chat(client, base, headers, args.model, prompt, file_ids, stream=not args.no_stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
