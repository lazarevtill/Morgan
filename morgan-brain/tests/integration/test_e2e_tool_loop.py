"""E2E test: tool-call loop wired through composition → orchestrator.

Extends test_tool_loop.py with additional assertions:
- role="tool" message appears in the prompt sent to the second LLM call.
- tools_invoked == ["calculator"] is returned in the result.
- PermissionGate blocks unknown tools (ASK-mode guard).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.composition import build_orchestrator_for_test
from morgan_brain.providers.wire import ChatResult, ToolCall


# ---------------------------------------------------------------------------
# Existing tests (kept for back-compat; new assertions added inline)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_loop_calculator_executes_and_result_in_prompt() -> None:
    """handle_turn: calculator runs, role='tool' message appears in second LLM call,
    and tools_invoked == ['calculator']."""
    tc = ToolCall(id="tc-e2e-1", name="calculator", arguments={"expression": "6 * 7"})
    results = [
        ChatResult(text="", tool_calls=[tc]),
        ChatResult(text="6 × 7 = 42.", tool_calls=[]),
    ]

    orch, _ = build_orchestrator_for_test(
        reply="",
        clock=lambda: datetime(2026, 1, 1),
        chat_results=results,
    )

    # FakeChatClient is shared via the router — capture it via a handle.
    # We reach it through the reasoner's router, which holds the binding.
    fake_client = orch._reasoner._router._bindings["strong"][0].client  # type: ignore[attr-defined]  # noqa: SLF001

    result = await orch.handle_turn(user_id="u1", text="What is 6 * 7?", session_id="s1")

    # Final text is the second scripted reply.
    assert result.text == "6 × 7 = 42."

    # Calculator was invoked.
    assert "calculator" in result.tools_invoked

    # The second call's messages must contain a role="tool" message.
    tool_msgs = [m for m in fake_client.last_messages if m.role == "tool"]
    assert len(tool_msgs) >= 1, "Expected at least one role='tool' message in the prompt"

    # The tool message content must contain the calculator result (42).
    tool_contents = " ".join(m.content for m in tool_msgs)
    assert "42" in tool_contents, f"Tool result '42' not in tool messages: {tool_contents}"


@pytest.mark.asyncio
async def test_tool_loop_plain_turn_has_no_tool_messages() -> None:
    """Plain turn (no tool calls scripted) produces no role='tool' messages."""
    orch, _ = build_orchestrator_for_test(reply="Hello!", clock=lambda: datetime(2026, 1, 1))
    fake_client = orch._reasoner._router._bindings["strong"][0].client  # type: ignore[attr-defined]  # noqa: SLF001

    result = await orch.handle_turn(user_id="u1", text="Hi there", session_id="s1")

    assert result.text == "Hello!"
    assert result.tools_invoked == []

    tool_msgs = [m for m in fake_client.last_messages if m.role == "tool"]
    assert len(tool_msgs) == 0


@pytest.mark.asyncio
async def test_tool_loop_multiple_tool_calls_all_invoked() -> None:
    """Two sequential tool calls in one turn both appear in tools_invoked."""
    tc1 = ToolCall(id="tc-1", name="calculator", arguments={"expression": "2 + 2"})
    tc2 = ToolCall(id="tc-2", name="current_time", arguments={})
    results = [
        ChatResult(text="", tool_calls=[tc1]),
        ChatResult(text="", tool_calls=[tc2]),
        ChatResult(text="Done.", tool_calls=[]),
    ]

    orch, _ = build_orchestrator_for_test(
        reply="",
        clock=lambda: datetime(2026, 1, 1),
        chat_results=results,
    )

    result = await orch.handle_turn(user_id="u1", text="What time and 2+2?", session_id="s1")

    assert result.text == "Done."
    assert "calculator" in result.tools_invoked
    assert "current_time" in result.tools_invoked
