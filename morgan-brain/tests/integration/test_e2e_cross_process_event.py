"""E2E test: cross-process event path simulation.

Simulates the two-process (brain-api publisher / learning-worker subscriber) path
using an in-memory bus double shared between an orchestrator (publisher) and the
worker's real ``_make_response_handler`` (subscriber).

When the orchestrator publishes RESPONSE_GENERATED, the worker handler must call
``learner.process_session`` — proven via a spy. No real Redis.

This tests that the boundary between brain-api and the learning-worker is correctly
wired: the same bus, the same event shape, the same handler factory.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from morgan_brain.apps.learning_worker.__main__ import _make_response_handler
from morgan_brain.bus.inproc import InProcessBus
from morgan_brain.composition import _assemble
from morgan_brain.config import Settings
from morgan_brain.interfaces.events import Event, EventType
from morgan_brain.models.message import Conversation
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter

CLOCK = lambda: datetime(2026, 1, 1)  # noqa: E731


# ---------------------------------------------------------------------------
# Spy learner
# ---------------------------------------------------------------------------


class _SpyLearner:
    """Captures process_session calls without running any real logic."""

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
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_process_handler_called_on_response_generated() -> None:
    """Publishing RESPONSE_GENERATED on a shared bus triggers the worker handler.

    The shared bus is the in-memory substitute for Redis Streams. The orchestrator
    publishes the event; the worker's _make_response_handler consumes it.
    """
    shared_bus = InProcessBus()
    spy = _SpyLearner()

    # Wire the worker handler onto the shared bus (simulates learning-worker startup).
    worker_handler = _make_response_handler(spy, clock=CLOCK)
    shared_bus.subscribe(EventType.RESPONSE_GENERATED, worker_handler)

    # Build orchestrator using the shared bus (simulates brain-api).
    fake_client = FakeChatClient(reply="worker should see this")
    reg = CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    router = RoleRouter(
        reg=reg,
        bindings={"strong": [Binding("fake", "test-model", fake_client)]},
    )
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")

    orch, *_ = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
        bus=shared_bus,
    )

    # Simulate brain-api serving a request.
    result = await orch.handle_turn(
        user_id="alice",
        text="What is the capital of France?",
        session_id="sess-cross",
    )

    assert result.text == "worker should see this"

    # The worker handler must have been called exactly once.
    assert len(spy.sessions) == 1
    convo = spy.sessions[0]
    assert convo.user_id == "alice"
    assert convo.session_id == "sess-cross"
    assert len(convo.messages) == 2
    assert convo.messages[0].content == "What is the capital of France?"
    assert convo.messages[1].content == "worker should see this"


@pytest.mark.asyncio
async def test_cross_process_multiple_turns_all_processed() -> None:
    """Each turn published via the orchestrator reaches the worker handler."""
    shared_bus = InProcessBus()
    spy = _SpyLearner()

    worker_handler = _make_response_handler(spy, clock=CLOCK)
    shared_bus.subscribe(EventType.RESPONSE_GENERATED, worker_handler)

    fake_client = FakeChatClient(replies=["r1", "r2", "r3"])
    reg = CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    router = RoleRouter(
        reg=reg,
        bindings={"strong": [Binding("fake", "test-model", fake_client)]},
    )
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")

    orch, *_ = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
        bus=shared_bus,
    )

    for i in range(3):
        await orch.handle_turn(
            user_id="bob",
            text=f"turn {i}",
            session_id="sess-multi",
        )

    # Worker handler should have been called 3 times (once per RESPONSE_GENERATED).
    assert len(spy.sessions) == 3


@pytest.mark.asyncio
async def test_cross_process_worker_handler_spy_via_direct_publish() -> None:
    """Direct publish of RESPONSE_GENERATED event triggers learner.process_session (spy).

    This decouples from the orchestrator and tests the handler contract directly:
    the exact event shape brain-api publishes is what the worker handler expects.
    """
    bus = InProcessBus()
    spy = _SpyLearner()

    handler = _make_response_handler(spy, clock=CLOCK)
    bus.subscribe(EventType.RESPONSE_GENERATED, handler)

    event = Event(
        type=EventType.RESPONSE_GENERATED,
        user_id="charlie",
        payload={
            "session_id": "sess-direct",
            "turn_id": "turn-xyz",
            "request": "Direct question",
            "response": "Direct answer",
        },
    )
    await bus.publish(event)

    assert len(spy.sessions) == 1
    convo = spy.sessions[0]
    assert convo.user_id == "charlie"
    assert convo.session_id == "sess-direct"
    assert convo.messages[0].content == "Direct question"
    assert convo.messages[1].content == "Direct answer"
