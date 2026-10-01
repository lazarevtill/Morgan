"""Synthetic chat preparation never partially persists or survives a committed forget."""

from datetime import UTC, datetime

import pytest

from morgan_brain.app.chat import Chat
from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.erasure import StoreInterruptedByForget
from morgan_brain.memory.store.history import SessionHistoryStore, session_key
from morgan_brain.models import Memory, Message, OriginKind, Role
from morgan_brain.providers.wire import ChatResult

NOW = datetime(2026, 10, 1, tzinfo=UTC)


class ControlledEmbedder(FakeEmbedder):
    def __init__(self, conn, *, fail_at=None, hook=None):
        super().__init__(dim=4)
        self.conn = conn
        self.calls = 0
        self.fail_at = fail_at
        self.hook = hook

    async def embed(self, text):
        self.calls += 1
        assert not self.conn.in_transaction, "Embedding awaited while transaction held"
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic embedding failure")
        if self.hook is not None:
            await self.hook(self.calls)
        return await super().embed(text)


class ReplyClient:
    def __init__(self, conn, hook=None):
        self.conn = conn
        self.hook = hook
        self.calls = 0

    async def agenerate(self, messages, *, model):
        self.calls += 1
        assert not self.conn.in_transaction, "Generation awaited while transaction held"
        if self.hook is not None:
            await self.hook()
        return ChatResult(text="Synthetic response", model=model)


def stack(conn, embedder=None, client_hook=None):
    embedder = embedder or ControlledEmbedder(conn)
    module = build_memory_module(conn, embedder=embedder, dim=4, clock=lambda: NOW)
    gate = MemoryGate(module)
    history = SessionHistoryStore(conn, clock=lambda: NOW)
    chat = Chat(
        gate=gate,
        history=history,
        client=ReplyClient(conn, client_hook),
        model="fake",
        clock=lambda: NOW,
    )
    return module, gate, history, chat


def turn():
    memories = [
        Memory(id=identity, user_id="owner", content=text, origin_kind=OriginKind.ASK)
        for identity, text in [("input", "Synthetic input"), ("output", "Synthetic output")]
    ]
    entries = [
        ("owner:session", "personal", Message(user_id="owner", role=role, content=memory.content))
        for memory, role in zip(memories, (Role.USER, Role.ASSISTANT), strict=True)
    ]
    return memories, entries


def assert_empty(conn):
    for query in (
        "SELECT COUNT(*) FROM memories",
        "SELECT COUNT(*) FROM vec_meta",
        "SELECT COUNT(*) FROM vec_items",
        "SELECT COUNT(*) FROM fts_memories",
        "SELECT COUNT(*) FROM memory_entities",
        "SELECT COUNT(*) FROM session_history",
        "SELECT COUNT(*) FROM projects",
    ):
        assert conn.execute(query).fetchone()[0] == 0


@pytest.mark.parametrize("fail_at", [2, 3])
async def test_first_or_second_memory_embedding_failure_leaves_no_turn(fail_at):
    conn = open_db(":memory:")
    try:
        _, _, _, chat = stack(conn, ControlledEmbedder(conn, fail_at=fail_at))
        before = conn.serialize()
        with pytest.raises(RuntimeError, match="synthetic embedding failure"):
            await chat.ask(user_id="owner", project="personal", text="Synthetic input")
        assert conn.serialize() == before
        assert_empty(conn)
    finally:
        conn.close()


@pytest.mark.parametrize("stage", ["recall", "model", "first", "second"])
async def test_forget_during_any_upstream_await_cancels_turn_and_fresh_retry_works(tmp_path, stage):
    path = str(tmp_path / "synthetic.db")
    conn, eraser_conn = open_db(path), open_db(path)
    _, eraser, _, _ = stack(eraser_conn)
    trigger = {"recall": 1, "first": 2, "second": 3}.get(stage)
    forgotten = False

    async def forget_once():
        nonlocal forgotten
        if not forgotten:
            forgotten = True
            await eraser.forget(user_id="owner", project="personal")

    async def embedding_hook(call):
        if call == trigger:
            await forget_once()

    try:
        _, _, _, chat = stack(
            conn,
            ControlledEmbedder(conn, hook=embedding_hook),
            forget_once if stage == "model" else None,
        )
        with pytest.raises(StoreInterruptedByForget, match="retry explicitly"):
            await chat.ask(user_id="owner", project="personal", text="Synthetic input")
        assert forgotten
        assert_empty(conn)
        assert conn.execute("SELECT generation FROM erasure_state").fetchone()[0] == 1
        assert (
            await chat.ask(user_id="owner", project="personal", text="Fresh explicit input")
            == "Synthetic response"
        )
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM session_history").fetchone()[0] == 2
    finally:
        conn.close()
        eraser_conn.close()


@pytest.mark.parametrize("site", ["event", "vector", "fts", "entity", "history"])
async def test_second_event_or_history_failure_rolls_back_whole_turn(monkeypatch, site):
    conn = open_db(":memory:")
    module, gate, history, _ = stack(conn)
    target, method = {
        "event": (module._episodics, "put"),
        "vector": (module._vectors, "upsert"),
        "fts": (module._fts, "add"),
        "entity": (module._entities, "add"),
        "history": (history, "append"),
    }[site]
    original = getattr(target, method)
    calls = 0

    def fail_sync(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic write failure")
        return original(*args, **kwargs)

    async def fail_async(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic write failure")
        return await original(*args, **kwargs)

    monkeypatch.setattr(target, method, fail_async if site == "vector" else fail_sync)
    memories, entries = turn()
    try:
        before = conn.serialize()
        with pytest.raises(RuntimeError, match="synthetic write failure"):
            await gate.store_turn(
                memories,
                history=history,
                history_entries=entries,
                expected_generation=gate.capture_erasure_generation(),
            )
        assert conn.serialize() == before
        assert_empty(conn)
        assert all(memory.embedding is None and memory.recorded_at is None for memory in memories)
    finally:
        conn.close()


@pytest.mark.parametrize(
    "mismatch", ["connection", "owner", "context", "key", "content", "roles", "duplicate", "count"]
)
async def test_turn_shape_refused_before_embedding(mismatch):
    conn, other = open_db(":memory:"), open_db(":memory:")
    embedder = ControlledEmbedder(conn)
    _, gate, history, _ = stack(conn, embedder)
    memories, entries = turn()
    if mismatch == "connection":
        history = stack(other)[2]
    elif mismatch == "owner":
        memories[1].user_id = "foreign"
    elif mismatch == "context":
        memories[1].project = "foreign"
    elif mismatch == "key":
        entries[1] = ("foreign:session", *entries[1][1:])
    elif mismatch == "content":
        entries[1][2].content = "Changed"
    elif mismatch == "roles":
        entries[1][2].role = Role.USER
    elif mismatch == "duplicate":
        memories[1].id = memories[0].id
    elif mismatch == "count":
        memories.pop()
    try:
        with pytest.raises(ValueError):
            await gate.store_turn(
                memories, history=history, history_entries=entries, expected_generation=0
            )
        assert embedder.calls == 0
        assert_empty(conn)
    finally:
        conn.close()
        other.close()


async def test_success_is_whole_attributed_turn_and_replay_cannot_duplicate_history():
    conn = open_db(":memory:")
    _, gate, history, chat = stack(conn)
    try:
        assert (
            await chat.ask(user_id="owner", project="personal", text="Synthetic input")
            == "Synthetic response"
        )
        records = conn.execute("SELECT source, author_id FROM memories ORDER BY rowid").fetchall()
        assert [row["source"] for row in records] == ["unknown", "agent_inferred"]
        assert records[1]["author_id"] == "model:fake"
        assert conn.execute("SELECT COUNT(*) FROM vec_meta").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM session_history").fetchone()[0] == 2
        memories, entries = turn()
        await gate.store_turn(
            memories, history=history, history_entries=entries, expected_generation=0
        )
        before = conn.serialize()
        with pytest.raises(ValueError, match="turn replay"):
            await gate.store_turn(
                memories, history=history, history_entries=entries, expected_generation=0
            )
        assert conn.serialize() == before
    finally:
        conn.close()


async def test_batch_snapshots_all_caller_events_and_history_before_first_await():
    conn = open_db(":memory:")
    memories, entries = turn()

    async def mutate_caller(call):
        if call == 1:
            memories[1].content = "Caller mutation"
            entries[1][2].content = "Caller history mutation"

    _, gate, history, _ = stack(conn, ControlledEmbedder(conn, hook=mutate_caller))
    try:
        await gate.store_turn(
            memories, history=history, history_entries=entries, expected_generation=0
        )
        assert (await gate.get("output", user_id="owner")).content == "Synthetic output"
        assert (
            conn.execute("SELECT content FROM session_history ORDER BY id DESC LIMIT 1").fetchone()[
                0
            ]
            == "Synthetic output"
        )
    finally:
        conn.close()


async def test_outer_transaction_refuses_before_embedding_and_preserves_owner_transaction():
    conn = open_db(":memory:")
    embedder = ControlledEmbedder(conn)
    _, gate, history, _ = stack(conn, embedder)
    memories, entries = turn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="no active transaction"):
            await gate.store_turn(
                memories, history=history, history_entries=entries, expected_generation=0
            )
        assert conn.in_transaction and embedder.calls == 0
        conn.rollback()
        assert_empty(conn)
    finally:
        conn.close()


async def test_single_store_outer_transaction_refuses_before_embedding_without_changes():
    conn = open_db(":memory:")
    embedder = ControlledEmbedder(conn)
    _, gate, _, _ = stack(conn, embedder)
    try:
        await gate.store(Memory(id="existing", user_id="owner", content="Existing source"))
        calls = embedder.calls
        before = conn.serialize()
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="no active transaction"):
            await gate.store(Memory(id="new", user_id="owner", content="Would embed under lock"))
        assert embedder.calls == calls and conn.in_transaction
        conn.rollback()
        assert conn.serialize() == before
        assert (
            await gate.store(Memory(id="existing", user_id="owner", content="Existing source"))
            == "existing"
        )
        assert embedder.calls == calls
    finally:
        conn.close()


async def test_ask_outer_transaction_refuses_before_recall_or_generation():
    conn = open_db(":memory:")
    embedder = ControlledEmbedder(conn)
    _, gate, _, chat = stack(conn, embedder)
    try:
        before = conn.serialize()
        with gate.write_transaction():
            with pytest.raises(ValueError, match="no active transaction"):
                await chat.ask(user_id="owner", project="personal", text="Should not generate")
            assert conn.in_transaction
            assert embedder.calls == 0
            assert chat._client.calls == 0
        assert conn.serialize() == before
    finally:
        conn.close()


async def test_input_effective_time_precedes_model_and_reply_time_follows_it():
    conn = open_db(":memory:")
    initial = datetime(2026, 10, 1, tzinfo=UTC)
    completed = datetime(2026, 10, 2, tzinfo=UTC)
    instant = initial

    def clock():
        return instant

    async def complete_model():
        nonlocal instant
        instant = completed

    module = build_memory_module(conn, embedder=ControlledEmbedder(conn), dim=4, clock=clock)
    history = SessionHistoryStore(conn, clock=clock)
    chat = Chat(
        gate=MemoryGate(module),
        history=history,
        client=ReplyClient(conn, complete_model),
        model="fake",
        clock=clock,
    )
    try:
        await chat.ask(user_id="owner", project="personal", text="Synthetic input")
        events = conn.execute(
            "SELECT created_at, recorded_at FROM memories ORDER BY rowid"
        ).fetchall()
        assert [datetime.fromisoformat(row["created_at"]) for row in events] == [initial, completed]
        assert all(datetime.fromisoformat(row["recorded_at"]) == completed for row in events)
        assert all(
            datetime.fromisoformat(row[0]) == completed
            for row in conn.execute("SELECT created_at FROM session_history")
        )
    finally:
        conn.close()


async def test_colliding_session_keys_do_not_leak_foreign_history_into_native_prompt():
    conn = open_db(":memory:")
    _, gate, history, _ = stack(conn)

    class RecordingClient(ReplyClient):
        async def agenerate(self, messages, *, model):
            self.messages = messages
            return await super().agenerate(messages, model=model)

    client = RecordingClient(conn)
    chat = Chat(gate=gate, history=history, client=client, model="fake", clock=lambda: NOW)
    own_key = session_key("foo", "bar:baz")
    assert own_key == session_key("foo:bar", "baz")
    try:
        history.append(
            own_key,
            Message(user_id="foo", role=Role.USER, content="Synthetic own earlier turn"),
            project="personal",
        )
        for i in range(12):
            history.append(
                own_key,
                Message(
                    user_id="foo:bar", role=Role.USER, content=f"Synthetic foreign private turn {i}"
                ),
                project="personal",
            )
        await chat.ask(
            user_id="foo", session_id="bar:baz", project="personal", text="Synthetic current input"
        )
        assert not any("foreign private" in message.content for message in client.messages)
        assert any(message.content == "Synthetic own earlier turn" for message in client.messages)
        rows = conn.execute(
            "SELECT user_id FROM session_history ORDER BY id DESC LIMIT 2"
        ).fetchall()
        assert [row[0] for row in rows] == ["foo", "foo"]
    finally:
        conn.close()
