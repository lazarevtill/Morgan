"""Stage 2 (GAP-2): history + base signal persist regardless of the event-bus backend.

Local persistence (session history + the base interaction signal) now lives in the
Orchestrator's step 7 — written in-process and synchronously — so the documented 2-process
Redis topology no longer silently degrades every turn to turn 1. This test drives a turn over
a **non-InProcessBus** (exactly the condition under which the old InProcessBus-gated subscriber
never fired) and asserts both are written, while consolidation is still announced on the bus
for the worker to handle off-path.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.composition import _assemble
from morgan_brain.config import Settings
from morgan_brain.interfaces.events import Event, EventType
from morgan_brain.learning.history import SessionHistoryStore, session_key
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter

CLOCK = lambda: datetime(2026, 1, 1)  # noqa: E731


class _NonInProcBus:
    """Stand-in for the Redis bus: NOT an InProcessBus, so ``_register_turn_storage`` is
    skipped (as in production, where the worker — not brain-api — subscribes). Captures
    published events; runs no subscribers in-process."""

    def __init__(self) -> None:
        self.published: list[Event] = []

    def subscribe(self, event_type: EventType, handler: object) -> None:
        pass  # no in-process subscribers, mirroring the Redis topology

    async def publish(self, event: Event) -> None:
        self.published.append(event)


def _router() -> RoleRouter:
    reg = CapabilityRegistry.from_seed(
        {
            "fake/test-model": {
                "supports_tools": True,
                "json_mode": "json_schema",
                "context_window": 32768,
            }
        }
    )
    return RoleRouter(
        reg=reg, bindings={"strong": [Binding("fake", "test-model", FakeChatClient(reply="ok"))]}
    )


@pytest.mark.asyncio
async def test_history_and_signal_persist_under_non_inproc_bus() -> None:
    history = SessionHistoryStore()
    bus = _NonInProcBus()
    orch, _mm, signal_store, _rec, _ex, _sk, _ln = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=_router(),
        settings=Settings(llm_model="test-model", llm_fast_model="test-model"),
        clock=CLOCK,
        temporal_path=":memory:",
        history_store=history,
        bus=bus,  # type: ignore[arg-type]
    )

    _, turn_id = await orch.handle_turn_with_id(
        user_id="u1", text="My name is Sam", session_id="s1"
    )

    # History written in-process despite the non-inproc bus (the GAP-2 break is closed).
    msgs = history.recent(session_key("u1", "s1"))
    assert [m.content for m in msgs] == ["My name is Sam", "ok"]

    # Base signal recorded too — so feedback/learning attaches to a real row, not a stub.
    sigs = await signal_store.for_user("u1")
    assert any(s.turn_id == turn_id and s.original_reply == "ok" for s in sigs)

    # Consolidation is still announced on the bus (the worker handles it off-path under Redis).
    assert any(e.type is EventType.RESPONSE_GENERATED for e in bus.published)
