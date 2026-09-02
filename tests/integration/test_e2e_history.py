"""E2E test: two-turn session history threading.

Two consecutive turns in the same session using the full composition stack.
Asserts that turn 2's prompt (FakeChatClient.last_messages) contains turn 1's
user message and assistant reply.

This is a more comprehensive complement to test_history_threading.py, exercising
the full build_orchestrator_for_test builder path.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from morgan_brain.composition import _assemble
from morgan_brain.config import Settings
from morgan_brain.learning.history import SessionHistoryStore, session_key
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter

CLOCK = lambda: datetime(2026, 1, 1, tzinfo=UTC)  # noqa: E731


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
async def test_turn2_prompt_contains_turn1_user_and_assistant_messages() -> None:
    """Turn 2's LLM prompt must contain both the user message and reply from turn 1."""
    fake_client = FakeChatClient(replies=["I live in Berlin.", "You said you live in Berlin."])
    history_store = SessionHistoryStore()
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    router = _make_router(fake_client)

    orch, *_ = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
        history_store=history_store,
    )

    session_id = "sess-history-e2e"
    user_id = "u-history"
    hkey = session_key(user_id, session_id)

    # Turn 1.
    h1 = history_store.recent(hkey, project="default")
    result1 = await orch.handle_turn(
        user_id=user_id,
        project="default",
        text="I live in Berlin.",
        session_id=session_id,
        history=h1,
    )
    assert result1.text == "I live in Berlin."

    # history_store must now have 2 entries (user + assistant).
    after_turn1 = history_store.recent(hkey, project="default")
    assert len(after_turn1) == 2

    # Turn 2.
    h2 = history_store.recent(hkey, project="default")
    result2 = await orch.handle_turn(
        user_id=user_id,
        project="default",
        text="Where do I live?",
        session_id=session_id,
        history=h2,
    )
    assert result2.text == "You said you live in Berlin."

    # The prompt sent for turn 2 must include turn 1's exchange.
    messages = fake_client.last_messages
    full_context = " ".join(m.content for m in messages)
    assert "Berlin" in full_context, f"Turn 1 'Berlin' not found in turn 2 messages: {messages}"


@pytest.mark.asyncio
async def test_history_is_session_scoped_not_cross_session() -> None:
    """Session-scoped history messages (role=user/assistant) from session A must NOT
    appear in session B's prompt as history entries.

    NOTE: User-scoped episodic memory (vector recall) is intentionally shared across
    sessions for the same user — that is correct behavior. This test only verifies that
    the explicit session history (injected via history_store.recent()) is session-scoped:
    session B's prompt must not include role=user/role=assistant messages from session A.
    """
    fake_client = FakeChatClient(replies=["Alice noted.", "Your name is unknown here."])
    history_store = SessionHistoryStore()
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    router = _make_router(fake_client)

    orch, *_ = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
        history_store=history_store,
    )

    user_id = "u-scope"

    # Session A: turn 1.
    h_a = history_store.recent(session_key(user_id, "session-a"), project="default")
    await orch.handle_turn(
        user_id=user_id,
        project="default",
        text="My name is Alice.",
        session_id="session-a",
        history=h_a,
    )

    # Session B uses EMPTY history (different session).
    h_b = history_store.recent(session_key(user_id, "session-b"), project="default")
    assert len(h_b) == 0, "Session B should start with empty history"

    await orch.handle_turn(
        user_id=user_id,
        project="default",
        text="What is my name?",
        session_id="session-b",
        history=h_b,
    )

    # last_messages for session B must NOT include the session-A history messages
    # (role=user "My name is Alice." and role=assistant "Alice noted.").
    # Session B history should only contribute zero history messages (none injected).
    messages = fake_client.last_messages
    history_msgs = [m for m in messages if m.role in ("user", "assistant")]
    # The only user message should be the current turn (session B's question).
    user_msgs = [m for m in history_msgs if m.role == "user"]
    # Session A's "My name is Alice." must not appear as an injected history message.
    assert not any(m.content == "My name is Alice." for m in user_msgs), (
        f"Session A's user message leaked into session B as history: {messages}"
    )
    # Session A's "Alice noted." must not appear as an injected assistant history message.
    assistant_msgs = [m for m in history_msgs if m.role == "assistant"]
    assert not any(m.content == "Alice noted." for m in assistant_msgs), (
        f"Session A's assistant message leaked into session B as history: {messages}"
    )


@pytest.mark.asyncio
async def test_history_grows_across_turns() -> None:
    """After N turns the history store has 2*N entries for the session."""
    fake_client = FakeChatClient(
        replies=["r1", "r2", "r3"],
    )
    history_store = SessionHistoryStore()
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    router = _make_router(fake_client)

    orch, *_ = _assemble(
        embedder=FakeEmbedder(dim=16),
        router=router,
        settings=settings,
        clock=CLOCK,
        temporal_path=":memory:",
        history_store=history_store,
    )

    session_id = "sess-grow"
    user_id = "u-grow"
    hkey = session_key(user_id, session_id)

    for i in range(3):
        h = history_store.recent(hkey, project="default")
        await orch.handle_turn(
            user_id=user_id, project="default", text=f"msg {i}", session_id=session_id, history=h
        )

    entries = history_store.recent(hkey, project="default", limit=100)
    assert len(entries) == 6, f"Expected 6 history entries, got {len(entries)}"
