"""Fresh-session admission is atomic with turn persistence, never a chat-default change."""

import asyncio
from datetime import UTC, datetime

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.history import SessionHistoryStore
from morgan_brain.models import Memory, Message, Role

NOW = datetime(2026, 10, 1, tzinfo=UTC)


class HookEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__(dim=4)
        self.hook = None

    async def embed(self, text):
        await asyncio.sleep(0)
        if self.hook:
            hook, self.hook = self.hook, None
            hook()
        return await super().embed(text)


def setup():
    conn = open_db(":memory:")
    embedder = HookEmbedder()
    gate = MemoryGate(build_memory_module(conn, embedder=embedder, dim=4, clock=lambda: NOW))
    history = SessionHistoryStore(conn, clock=lambda: NOW)
    return conn, gate, history, embedder


async def save(gate, history, **kwargs):
    memories = [
        Memory(id=identity, user_id="owner", project="personal", content=text)
        for identity, text in (("new-user", "Continue"), ("new-reply", "Draft"))
    ]
    entries = [
        (
            "owner:fresh",
            "personal",
            Message(user_id="owner", project="personal", role=role, content=memory.content),
        )
        for role, memory in zip((Role.USER, Role.ASSISTANT), memories, strict=True)
    ]
    await gate.store_turn(
        memories,
        history=history,
        history_entries=entries,
        expected_generation=gate.capture_erasure_generation(),
        **kwargs,
    )


@pytest.mark.parametrize("during_embedding", [False, True])
async def test_occupied_session_refuses_without_partial_new_events(during_embedding):
    conn, gate, history, embedder = setup()
    try:

        def insert():
            history.append(
                "owner:fresh",
                Message(user_id="owner", role=Role.USER, content="Concurrent turn"),
                project="personal",
            )

        if during_embedding:
            embedder.hook = insert
        else:
            insert()
        with pytest.raises(ValueError, match="occupied"):
            await save(gate, history, fresh_session=True)
        assert await gate.get("new-user", user_id="owner") is None
        assert await gate.get("new-reply", user_id="owner") is None
        assert [
            m.content for m in history.recent("owner:fresh", project="personal", user_id="owner")
        ] == ["Concurrent turn"]
    finally:
        conn.close()


async def test_fresh_check_is_owner_and_project_scoped():
    conn, gate, history, _ = setup()
    try:
        history.append(
            "owner:fresh",
            Message(user_id="foreign", role=Role.USER, content="Foreign owner"),
            project="personal",
        )
        history.append(
            "owner:fresh",
            Message(user_id="owner", role=Role.USER, project="orchid", content="Foreign project"),
            project="orchid",
        )
        await save(gate, history, fresh_session=True)
        assert [
            m.content for m in history.recent("owner:fresh", project="personal", user_id="owner")
        ] == ["Continue", "Draft"]
        assert await gate.get("new-user", user_id="owner") is not None
    finally:
        conn.close()


async def test_existing_default_turn_appends_to_occupied_history_unchanged():
    conn, gate, history, _ = setup()
    try:
        history.append("owner:fresh", Message(user_id="owner", role=Role.USER, content="Previous"))
        await save(gate, history)
        assert [
            m.content for m in history.recent("owner:fresh", project="personal", user_id="owner")
        ] == ["Previous", "Continue", "Draft"]
    finally:
        conn.close()
