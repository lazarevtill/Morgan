"""Unit tests for turn_id + SignalStore wiring (commit 1).

Asserts:
- handle_turn returns a ReasoningResult (basic smoke).
- RESPONSE_GENERATED payload includes a ``turn_id`` key.
- After a turn the SignalStore has a base signal for that turn_id.
- ChatResponse model exposes a turn_id field.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from morgan_brain.composition import build_orchestrator_for_test_with_signals

CLOCK = lambda: datetime(2026, 1, 1)  # noqa: E731


@pytest.mark.asyncio
async def test_handle_turn_returns_result() -> None:
    orch, _, _ = build_orchestrator_for_test_with_signals(reply="hello", clock=CLOCK)
    result = await orch.handle_turn(user_id="u1", text="hi", session_id="s1")
    assert result.text == "hello"


@pytest.mark.asyncio
async def test_response_generated_carries_turn_id() -> None:
    """RESPONSE_GENERATED event payload must include a non-empty turn_id."""
    from morgan_brain.interfaces.events import EventType

    orch, _, bus = build_orchestrator_for_test_with_signals(reply="hello", clock=CLOCK)

    events_payloads: list[dict[str, Any]] = []

    async def _capture(event: Any) -> None:
        events_payloads.append(event.payload)

    bus.subscribe(EventType.RESPONSE_GENERATED, _capture)

    await orch.handle_turn(user_id="u1", text="hello", session_id="s1")

    assert len(events_payloads) >= 1
    payload = events_payloads[0]
    assert "turn_id" in payload
    assert isinstance(payload["turn_id"], str)
    assert len(payload["turn_id"]) > 0


@pytest.mark.asyncio
async def test_signal_recorded_after_handle_turn() -> None:
    """After a turn, the SignalStore has exactly one base signal for that turn."""
    orch, signal_store, _ = build_orchestrator_for_test_with_signals(reply="pong", clock=CLOCK)

    await orch.handle_turn(user_id="u1", text="ping", session_id="sess1")

    signals = await signal_store.for_user("u1")
    assert len(signals) == 1
    sig = signals[0]
    assert sig.query == "ping"
    assert sig.original_reply == "pong"
    assert sig.session_id == "sess1"
    assert len(sig.turn_id) > 0


def test_turn_id_in_chat_response() -> None:
    """ChatResponse must expose a turn_id field."""
    from morgan_brain.apps.brain_api.app import ChatResponse

    resp = ChatResponse(response="hi", model_used="m", turn_id="abc123")
    assert resp.turn_id == "abc123"
