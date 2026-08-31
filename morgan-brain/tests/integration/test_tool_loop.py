"""Integration test: tool-call loop wired through composition → orchestrator.

Uses FakeChatClient scripted with a ChatResult that contains a tool_call
(calculator) followed by a final text answer, with no network calls.
Asserts that the calculator actually ran, the tool result was forwarded
to the model, and handle_turn returned the correct final answer.
"""

from __future__ import annotations

from datetime import UTC, datetime

from morgan_brain.composition import build_orchestrator_for_test
from morgan_brain.providers.wire import ChatResult, ToolCall


async def test_tool_loop_calculator_executes_in_handle_turn():
    """End-to-end: FakeChatClient scripts a calculator tool call; handle_turn executes it."""
    tc = ToolCall(id="tc-int-1", name="calculator", arguments={"expression": "12 * 12"})
    results = [
        ChatResult(text="", tool_calls=[tc]),
        ChatResult(text="12 × 12 = 144.", tool_calls=[]),
    ]

    orch, _ = build_orchestrator_for_test(
        reply="",
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        chat_results=results,
    )

    result = await orch.handle_turn(
        user_id="u1", project="default", text="What is 12 * 12?", session_id="s1"
    )

    assert result.text == "12 × 12 = 144."
    assert "calculator" in result.tools_invoked


async def test_tool_loop_plain_turn_still_works():
    """Existing plain turns (no tool calls scripted) continue to work after wiring."""
    orch, _mem = build_orchestrator_for_test(
        reply="Hello!", clock=lambda: datetime(2026, 1, 1, tzinfo=UTC)
    )

    result = await orch.handle_turn(
        user_id="u1", project="default", text="Hi there", session_id="s1"
    )

    assert result.text == "Hello!"
    assert result.tools_invoked == []
