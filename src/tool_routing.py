"""Which side owns the agent loop for a tools-carrying chat request.

The tool bridge (tool_bridge.py) serves requests where the CLIENT runs the
loop: one Messages API call, tool_calls handed back. That is the right answer
only when the client is contractually owed a call. A chat UI in native
function-calling mode sends its whole tool roster on every message, whether or
not the turn has anything to do with them -- and the bridge has no CLI, so no
session workspace and no generated files. Those turns can be served by the
agentic path instead, at the cost of the client's tools being dropped for that
turn (the CLI cannot surface them). That tradeoff is the operator's to make,
via CLAUDE_WRAPPER_TOOLS_MODE.

Deliberately imports nothing from the package: these are pure predicates over a
request object, unit-testable without the app or the settings singleton.
"""

from __future__ import annotations

from typing import Any

# Accepted values of CLAUDE_WRAPPER_TOOLS_MODE. "bridge" is today's behavior
# (every tools-carrying request goes to the bridge); "agentic" runs the CLI for
# the turns where no tool call is owed.
TOOLS_MODES = ("bridge", "agentic")


def tool_call_is_owed(req: Any) -> bool:
    """True when returning prose instead of a tool_call would break the client.

    Either the client forced a call, or the transcript ENDS mid-loop. Only the
    tail matters: settled tool history further up is finished business, and
    treating it as owed would strand a conversation on the bridge forever once
    any tool had ever run -- which is exactly the case this change exists to fix.
    """
    tc = req.tool_choice
    if isinstance(tc, dict):
        return True  # a named function was forced
    if isinstance(tc, str) and tc.strip().lower() == "required":
        return True
    last = req.messages[-1] if req.messages else None
    return bool(last is not None and (last.role == "tool" or last.tool_calls))


def use_tool_bridge(req: Any, mode: str) -> bool:
    """Routing decision for run_chat_completion. `mode` is SETTINGS.tools_mode.

    No tools => never the bridge. Owed a call => always the bridge, in every
    mode. Anything else is ambiguous, and the operator's mode decides; an
    unrecognized mode behaves as "bridge" so a typo cannot silently change
    routing.
    """
    if not req.tools:
        return False
    if tool_call_is_owed(req):
        return True
    return mode != "agentic"
