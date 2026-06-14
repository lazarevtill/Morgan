"""Integration test: session history threaded into turn2's ReasoningRequest.

Two consecutive turns in the same session:
- Turn 1: user says "My name is Alice", assistant replies "Got it."
- Turn 2: user says "What is my name?"

After turn 2, the FakeChatClient's ``last_messages`` must contain a message
with Alice in the content — proving turn 1's history reached the prompt.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.composition import _assemble
from morgan_brain.config import Settings
from morgan_brain.learning.history import SessionHistoryStore as _HSS
from morgan_brain.learning.history import session_key
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter

CLOCK = lambda: datetime(2026, 1, 1)  # noqa: E731


def _make_router(fake_client: FakeChatClient) -> RoleRouter:
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
        reg=reg,
        bindings={"strong": [Binding("fake", "test-model", fake_client)]},
    )


@pytest.mark.asyncio
async def test_turn2_history_contains_turn1_exchange() -> None:
    """Turn2's prompt must include turn1's user message (Alice) in the history."""
    fake_client = FakeChatClient(replies=["Got it.", "You are Alice."])
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    history_store = _HSS()
    router = _make_router(fake_client)

    orch, _, _, _, _, _, _ = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
        history_store=history_store,
    )

    session_id = "session-abc"
    user_id = "u1"
    hkey = session_key(user_id, session_id)  # history is keyed per-user

    # Turn 1
    history1 = history_store.recent(hkey)
    result1 = await orch.handle_turn(
        user_id=user_id, text="My name is Alice", session_id=session_id, history=history1
    )
    assert result1.text == "Got it."

    # After turn 1, the cold-path subscriber should have appended to history_store.
    # Give the in-process bus time to fire (it's synchronous in InProcessBus).
    history_after_turn1 = history_store.recent(hkey)
    assert len(history_after_turn1) == 2  # user + assistant

    # Turn 2 — fetch history first (as the API handler does)
    history2 = history_store.recent(hkey)
    result2 = await orch.handle_turn(
        user_id=user_id, text="What is my name?", session_id=session_id, history=history2
    )
    assert result2.text == "You are Alice."

    # The LLM received turn1's messages in the context
    messages_sent = fake_client.last_messages
    full_context = " ".join(m.content for m in messages_sent)
    assert "Alice" in full_context, (
        f"Turn 1 history not in turn 2 prompt. Messages: {messages_sent}"
    )
