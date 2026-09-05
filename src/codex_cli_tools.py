"""Client-declared function calling served by the codex CLI, with no API key.

The third tool path, and the only one a ChatGPT-plan deployment can use:

  tool_bridge.py     claude + Anthropic Messages API  (plan login works there)
  openai_bridge.py   codex  + OpenAI Platform API     (needs OPENAI_API_KEY)
  THIS MODULE        codex  + the codex CLI itself    (needs no key at all)

The asymmetry that forces this to exist: a Claude Code plan login can call the
Messages API, so tool_bridge serves function calling for free. A ChatGPT-plan
login authenticates only against the Codex backend — it cannot call
api.openai.com — so under codex the bridge demands a separate Platform key that
plan-only operators do not have (README, "Codex → OpenAI").

Since the CLI itself has no protocol for returning a caller's tool calls, the
tool contract is carried IN the prompt: the client's tool definitions are
rendered as a protocol block, codex answers with one JSON envelope, and this
module turns that envelope back into OpenAI-shaped ``tool_calls``. The client
then executes the tool and sends results back as ``role: "tool"`` messages,
which converters.py already renders into the transcript — so the loop closes
without any new session machinery.

Honest limits (README, "Limitations"): prompt-declared tool calling is weaker
than a native function-calling API. A model can ignore the envelope, so
malformed answers are repaired once and otherwise degrade to a plain text
answer rather than an error. Public surface mirrors tool_bridge/openai_bridge
(``complete`` / ``stream``) so main.py selects it like any other bridge.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from .json_mode import extract_raw_json
from .models import ChatCompletionRequest
from .tool_bridge import BridgeResult

log = logging.getLogger("claude_wrapper.codex_cli_tools")

# The envelope codex is asked to emit. One shape for both outcomes so there is
# a single thing to parse and a single thing to repair.
_PROTOCOL = """\
## Tool-calling protocol (MANDATORY)

The caller has declared the tools below. You cannot execute them yourself —
the caller executes them and sends you the results.

Reply with EXACTLY ONE JSON object and nothing else. No prose, no code fence.

To call one or more tools:
  {"tool_calls": [{"name": "<tool name>", "arguments": {<arguments object>}}]}

To answer directly, when no tool is needed or the results already suffice:
  {"content": "<your answer>"}

Rules:
- "arguments" is a JSON OBJECT matching that tool's parameter schema, never a
  string, and never a guess at a required value you were not given.
- Call a tool only if it is in the list below.
- Earlier tool results appear in the transcript as `Tool:` turns. Use them.
- Never wrap the object in a code fence or add commentary around it.

### Available tools

"""


def build_tool_protocol(req: ChatCompletionRequest) -> str:
    """The protocol block prepended to a tools request's prompt."""
    lines = [_PROTOCOL]
    for t in req.tools or []:
        if t.type != "function":
            # Non-function tool types are an OpenAI server-side concept; the
            # CLI has nothing to map them onto. Skipped rather than faked.
            continue
        fn = t.function
        schema = fn.parameters if fn.parameters is not None else {"type": "object"}
        lines.append(f"- **{fn.name}**: {fn.description or '(no description)'}")
        lines.append(f"  parameters: {json.dumps(schema, ensure_ascii=False)}")
    choice = req.tool_choice
    if isinstance(choice, dict):
        forced = (choice.get("function") or {}).get("name")
        if forced:
            lines.append(f"\nYou MUST call the tool `{forced}` on this turn.")
    elif choice == "none":
        lines.append("\nDo NOT call any tool on this turn; answer directly.")
    elif choice == "required":
        lines.append("\nYou MUST call at least one tool on this turn.")
    return "\n".join(lines) + "\n"


def _coerce_arguments(raw: Any) -> str:
    """Tool-call arguments as the JSON STRING the OpenAI wire shape requires.

    Models emit an object (as instructed), a JSON string (over-literal
    reading), or something else entirely; all three become a valid JSON
    string so clients never see a malformed `arguments`.
    """
    if isinstance(raw, str):
        try:
            json.loads(raw)
            return raw  # already a JSON string
        except json.JSONDecodeError:
            return json.dumps({"input": raw}, ensure_ascii=False)
    if raw is None:
        return "{}"
    try:
        return json.dumps(raw, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def parse_envelope(text: str, declared: frozenset[str]) -> tuple[Optional[str], Optional[list]]:
    """(content, tool_calls) from codex's reply.

    Returns tool_calls only for tools the client actually declared — a
    hallucinated tool name would otherwise reach the client as a call it
    cannot execute and has no way to refuse. Anything unparseable degrades to
    (text, None): a plain answer beats a 502 the caller cannot act on.
    """
    raw = extract_raw_json(text)
    if raw is None:
        return text, None
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        return text, None
    if not isinstance(doc, dict):
        return text, None

    calls = doc.get("tool_calls")
    if isinstance(calls, list) and calls:
        out: list[dict[str, Any]] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            # Accept the OpenAI-native nesting too ({"function": {...}}), which
            # models trained on that shape reach for despite the instructions.
            fn = call.get("function") if isinstance(call.get("function"), dict) else call
            name = str(fn.get("name") or "").strip()
            if not name or name not in declared:
                log.warning("codex proposed undeclared tool %r; dropping", name)
                continue
            out.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": _coerce_arguments(
                            fn.get("arguments", fn.get("parameters"))
                        ),
                    },
                }
            )
        if out:
            return None, out
        # Every proposed call was undeclared — fall through to text so the
        # caller gets something rather than an empty turn.

    content = doc.get("content")
    if isinstance(content, str):
        return content, None
    return text, None


async def _run(req: ChatCompletionRequest, run_model: str, session_key: str, runner):
    """One codex turn carrying the protocol block, returned as a BridgeResult."""
    from .deps import PREPARER

    # The configured singleton, so uploads/attachments materialize exactly as
    # they do on the chat path — a tools request may carry them too.
    prompt, _paths = await PREPARER.prepare_messages(req.messages, session_key)
    result = await runner.run_collect(
        prompt=build_tool_protocol(req) + "\n" + prompt,
        session_key=session_key,
        model=run_model,
        # The protocol IS the turn's instruction set; the clarify prompt would
        # compete with it for the reply shape.
        clarify=False,
    )
    if result.error:
        # Surfaced the same way the CLI chat path surfaces a failed turn.
        from .tool_bridge import openai_error

        raise openai_error(
            502,
            f"codex failed: {result.error}",
            err_type="api_error",
            code="upstream_error",
        )
    declared = frozenset(
        t.function.name for t in (req.tools or []) if t.type == "function"
    )
    content, tool_calls = parse_envelope(result.final_text or "", declared)
    return BridgeResult(
        content=content,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


async def complete(
    req: ChatCompletionRequest, run_model: str, session_key: str = ""
) -> BridgeResult:
    """Non-streaming tools request served by the CLI."""
    from .deps import RUNNER

    return await _run(req, run_model, session_key, RUNNER)


async def stream(
    req: ChatCompletionRequest,
    run_model: str,
    model_label: str,
    session_key: str,
    effort_info: dict[str, Any],
    on_usage: Optional[Callable[[int, int], Awaitable[None]]] = None,
) -> AsyncIterator[bytes]:
    """Streaming tools request.

    The turn is run to completion and then emitted as chunks: codex's --json
    stream is item-granular (no text deltas), and a tool call is only usable
    once whole, so there is nothing to stream incrementally. The frame
    sequence is still the one clients expect — role chunk, payload, finish,
    [DONE] — so a strict SSE client cannot tell the difference.
    """
    from .deps import RUNNER

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def _frame(delta: dict[str, Any], finish: Optional[str] = None) -> bytes:
        payload = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_label,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"

    result = await _run(req, run_model, session_key, RUNNER)
    if on_usage is not None:
        await on_usage(result.input_tokens, result.output_tokens)

    yield _frame({"role": "assistant"})
    if result.tool_calls:
        # Indexed exactly as the OpenAI streaming shape requires; arguments
        # arrive whole in one delta, which is legal (clients concatenate).
        yield _frame(
            {
                "tool_calls": [
                    {**call, "index": i} for i, call in enumerate(result.tool_calls)
                ]
            }
        )
    elif result.content:
        yield _frame({"content": result.content})
    yield _frame({}, finish=result.finish_reason)
    yield b"data: [DONE]\n\n"
