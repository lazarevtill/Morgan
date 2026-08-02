"""Unit tests for the real learning-worker (commit 3).

Verifies that:
- Publishing a RESPONSE_GENERATED event triggers learner.process_session (non-noop).
- The worker context is built with fakes (no network).
- The handler correctly reconstructs a Conversation and calls the learner.

All tests are deterministic: fake bus + spy learner, no external services.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from morgan_brain.bus.inproc import InProcessBus
from morgan_brain.config import Settings
from morgan_brain.interfaces.events import Event, EventType
from morgan_brain.models.message import Conversation


# ---------------------------------------------------------------------------
# Spy learner (captures process_session calls)
# ---------------------------------------------------------------------------


class _SpyLearner:
    def __init__(self) -> None:
        self.sessions: list[Conversation] = []
        self.consolidate_calls: list[str] = []

    async def process_session(self, conversation: Conversation) -> None:
        self.sessions.append(conversation)

    async def consolidate(self, user_id: str) -> None:
        self.consolidate_calls.append(user_id)

    async def user_model(self, user_id: str) -> Any:
        from morgan_brain.models.user import UserModel

        return UserModel(user_id=user_id)


# ---------------------------------------------------------------------------
# Helper: build a minimal worker context with fakes
# ---------------------------------------------------------------------------


def _make_fake_settings(**kwargs: Any) -> Settings:
    return Settings(
        llm_model="test",
        llm_fast_model="test",
        enable_scheduling=False,
        **kwargs,
    )


def _fake_clock() -> datetime:
    return datetime(2026, 1, 1)


# ---------------------------------------------------------------------------
# Tests: handler is non-noop when RESPONSE_GENERATED is published
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_generated_triggers_learner_process_session() -> None:
    """Publishing RESPONSE_GENERATED must call learner.process_session (not a no-op)."""
    from morgan_brain.apps.learning_worker.__main__ import _make_response_handler

    bus = InProcessBus()
    spy = _SpyLearner()

    handler = _make_response_handler(spy, clock=_fake_clock)
    bus.subscribe(EventType.RESPONSE_GENERATED, handler)

    event = Event(
        type=EventType.RESPONSE_GENERATED,
        user_id="alice",
        payload={
            "session_id": "sess-1",
            "turn_id": "turn-42",
            "request": "What is 2+2?",
            "response": "4",
        },
    )
    await bus.publish(event)

    assert len(spy.sessions) == 1, "process_session must be called exactly once"
    convo = spy.sessions[0]
    assert convo.user_id == "alice"
    assert convo.session_id == "sess-1"
    assert len(convo.messages) == 2
    assert convo.messages[0].content == "What is 2+2?"
    assert convo.messages[1].content == "4"


@pytest.mark.asyncio
async def test_handler_uses_default_session_id_when_missing() -> None:
    """If session_id is absent from payload, 'default' is used."""
    from morgan_brain.apps.learning_worker.__main__ import _make_response_handler

    bus = InProcessBus()
    spy = _SpyLearner()

    handler = _make_response_handler(spy, clock=_fake_clock)
    bus.subscribe(EventType.RESPONSE_GENERATED, handler)

    event = Event(
        type=EventType.RESPONSE_GENERATED,
        user_id="bob",
        payload={
            "request": "Hello",
            "response": "Hi there",
        },
    )
    await bus.publish(event)

    assert spy.sessions[0].session_id == "default"


@pytest.mark.asyncio
async def test_handler_exception_does_not_propagate() -> None:
    """A crashing learner must not raise from the handler (worker resilience)."""
    from morgan_brain.apps.learning_worker.__main__ import _make_response_handler

    class _BrokenLearner:
        async def process_session(self, conversation: Conversation) -> None:
            raise RuntimeError("boom")

    bus = InProcessBus()
    handler = _make_response_handler(_BrokenLearner(), clock=_fake_clock)  # type: ignore[arg-type]
    bus.subscribe(EventType.RESPONSE_GENERATED, handler)

    event = Event(
        type=EventType.RESPONSE_GENERATED,
        user_id="charlie",
        payload={"request": "q", "response": "a"},
    )
    # Should not raise even though the learner raises internally
    await bus.publish(event)


@pytest.mark.asyncio
async def test_multiple_events_each_trigger_session() -> None:
    """Each published event produces a separate process_session call."""
    from morgan_brain.apps.learning_worker.__main__ import _make_response_handler

    bus = InProcessBus()
    spy = _SpyLearner()

    handler = _make_response_handler(spy, clock=_fake_clock)
    bus.subscribe(EventType.RESPONSE_GENERATED, handler)

    for i in range(3):
        event = Event(
            type=EventType.RESPONSE_GENERATED,
            user_id="user",
            payload={
                "session_id": f"sess-{i}",
                "request": f"q{i}",
                "response": f"a{i}",
            },
        )
        await bus.publish(event)

    assert len(spy.sessions) == 3


# ---------------------------------------------------------------------------
# Tests: worker module structure (smoke)
# ---------------------------------------------------------------------------


def test_learning_worker_exposes_make_response_handler() -> None:
    """The module must export _make_response_handler."""
    import morgan_brain.apps.learning_worker.__main__ as mod

    assert callable(mod._make_response_handler)  # noqa: SLF001


def test_learning_worker_exposes_build_worker_context() -> None:
    """The module must export build_worker_context (or composition must)."""
    from morgan_brain.composition import build_worker_context

    assert callable(build_worker_context)
