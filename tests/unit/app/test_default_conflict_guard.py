"""Default conflict handling uses real synthetic Gate metadata, never a model verdict."""

from datetime import UTC, datetime, timedelta

import pytest

from morgan_brain.app.chat import Chat, build_messages
from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.erasure import StoreInterruptedByForget
from morgan_brain.memory.store.history import SessionHistoryStore, session_key
from morgan_brain.models import Memory, MemoryQuery, MemorySource
from morgan_brain.providers.wire import ChatResult

NOW = datetime(2026, 10, 1, tzinfo=UTC)


class Client:
    def __init__(self):
        self.requests = []

    async def agenerate(self, messages, *, model):
        self.requests.append((messages, model))
        return ChatResult(text="Synthetic generated answer", model=model)


class Embedder(FakeEmbedder):
    def __init__(self):
        super().__init__(dim=4)
        self.calls = 0
        self.fail_at = None
        self.hook = None

    async def embed(self, text):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic embedding failure")
        if self.hook:
            await self.hook(self.calls)
        return await super().embed(text)


def stack():
    conn = open_db(":memory:")
    embedder = Embedder()
    gate = MemoryGate(build_memory_module(conn, embedder=embedder, dim=4, clock=lambda: NOW))
    history = SessionHistoryStore(conn, clock=lambda: NOW)
    client = Client()
    chat = Chat(gate=gate, history=history, client=client, model="fake", clock=lambda: NOW)
    return conn, embedder, gate, history, client, chat


async def fork(gate, source="user_stated"):
    root = Memory(
        id="root",
        user_id="owner",
        project="personal",
        content="I chose a glaze",
        source=source,
        author_id="person",
        created_at=NOW - timedelta(days=3),
    )
    await gate.store(root)
    for identity, text, days in [
        ("amber", "I chose amber glaze", 2),
        ("cobalt", "I chose cobalt glaze", 1),
    ]:
        await gate.store(
            root.model_copy(
                update={
                    "id": identity,
                    "content": text,
                    "created_at": NOW - timedelta(days=days),
                    "revises_event_ids": ["root"],
                }
            )
        )


@pytest.mark.parametrize(
    "question,language",
    [
        ("Which glaze did I choose?", "en"),
        ("Какую глазурь я выбрал?", "ru"),
        ("What is two plus two?", "en"),
    ],
)
@pytest.mark.parametrize("source", ["user_stated", "agent_inferred"])
async def test_conflict_clarification_skips_model_and_preserves_true_author(
    question, language, source
):
    conn, _, gate, history, client, chat = stack()
    try:
        await fork(gate, source)
        recalled = await gate.recall(
            MemoryQuery(user_id="owner", project="personal", text=question)
        )
        assert any(record.revision_state == "conflicted" for record in recalled.memories)
        reply = await chat.ask(
            user_id="owner",
            project="personal",
            text=question,
            session_id="s",
            source=MemorySource.USER_STATED,
            author_id="owner",
        )
        assert client.requests == []
        assert "recall" in reply and "evidence" in reply and "remember" in reply
        assert len(reply.encode("utf-8")) <= 512
        assert "amber" not in reply and "cobalt" not in reply
        assert ("неразреш" in reply) if language == "ru" else ("unresolved" in reply)
        rows = history.recent(session_key("owner", "s"), user_id="owner", project="personal")
        assert [row.content for row in rows] == [question, reply]
        result = conn.execute(
            "SELECT id FROM memories WHERE origin_kind='ask' ORDER BY rowid"
        ).fetchall()
        assert len(result) == 2
        stored = [await gate.get(row["id"], user_id="owner") for row in result]
        assert stored[0].source is MemorySource.USER_STATED
        assert stored[1].source is MemorySource.AGENT_INFERRED
        assert stored[1].author_id == "morgan:conflict-guard"
    finally:
        conn.close()


async def test_join_resolves_guard_and_nonconflict_wire_is_unchanged():
    conn, _, gate, history, client, chat = stack()
    try:
        await fork(gate)
        await chat.ask(user_id="owner", project="personal", text="Which glaze?", session_id="s")
        assert client.requests == []
        await gate.store(
            Memory(
                id="join",
                user_id="owner",
                project="personal",
                content="I chose jade glaze",
                source="user_stated",
                author_id="person",
                created_at=NOW,
                revises_event_ids=["amber", "cobalt"],
            )
        )
        question = "Which glaze?"
        recalled = await gate.recall(
            MemoryQuery(user_id="owner", project="personal", text=question)
        )
        assert all(record.revision_state != "conflicted" for record in recalled.memories)
        prior = history.recent(session_key("owner", "s"), project="personal", user_id="owner")
        expected = build_messages(memories=recalled.memories, history=prior, text=question)
        reply = await chat.ask(user_id="owner", project="personal", text=question, session_id="s")
        assert reply == "Synthetic generated answer"
        assert client.requests == [(expected, "fake")]
        rows = conn.execute("SELECT id FROM memories WHERE content=?", (reply,)).fetchall()
        assert (await gate.get(rows[0]["id"], user_id="owner")).author_id == "model:fake"
    finally:
        conn.close()


@pytest.mark.parametrize("failure", ["embedding", "forget"])
async def test_guard_keeps_atomic_failure_and_erasure_checks(failure):
    conn, embedder, gate, history, client, chat = stack()
    try:
        await fork(gate)
        if failure == "embedding":
            embedder.fail_at = 6
        else:

            async def erase(calls):
                if calls == 5:
                    await gate.forget(user_id="owner", project="personal")

            embedder.hook = erase
        with pytest.raises(RuntimeError if failure == "embedding" else StoreInterruptedByForget):
            await chat.ask(user_id="owner", project="personal", text="Which glaze?", session_id="s")
        assert client.requests == []
        assert history.recent(session_key("owner", "s"), project="personal", user_id="owner") == []
        assert (
            conn.execute("SELECT count(*) FROM memories WHERE origin_kind='ask'").fetchone()[0] == 0
        )
    finally:
        conn.close()


async def test_guard_does_not_upgrade_unknown_caller_input():
    conn, _, gate, _, client, chat = stack()
    try:
        await fork(gate)
        await chat.ask(user_id="owner", project="personal", text="Which glaze?", session_id="s")
        assert client.requests == []
        rows = conn.execute(
            "SELECT id FROM memories WHERE origin_kind='ask' ORDER BY rowid"
        ).fetchall()
        stored = [await gate.get(row["id"], user_id="owner") for row in rows]
        assert stored[0].source is MemorySource.UNKNOWN
        assert stored[0].author_id == ""
        assert stored[1].source is MemorySource.AGENT_INFERRED
        assert stored[1].author_id == "morgan:conflict-guard"
    finally:
        conn.close()


@pytest.mark.parametrize("change", ["fork", "join"])
async def test_revision_change_between_recall_and_commit_refuses_atomically(change):
    from morgan_brain.memory.errors import EvidenceChanged

    conn, _, gate, history, _, chat = stack()
    try:
        if change == "join":
            await fork(gate)
            parents = ["amber", "cobalt"]
        else:
            await gate.store(
                Memory(
                    id="root",
                    user_id="owner",
                    project="personal",
                    content="I chose a glaze",
                    source="user_stated",
                    author_id="person",
                    created_at=NOW - timedelta(days=3),
                )
            )
            await gate.store(
                Memory(
                    id="amber",
                    user_id="owner",
                    project="personal",
                    content="I chose amber",
                    source="user_stated",
                    author_id="person",
                    created_at=NOW - timedelta(days=2),
                    revises_event_ids=["root"],
                )
            )
            parents = ["root"]
        original = gate.recall

        async def recall_then_change(query):
            result = await original(query)
            await gate.store(
                Memory(
                    id="concurrent",
                    user_id="owner",
                    project="personal",
                    content="Concurrent correction",
                    source="user_stated",
                    author_id="person",
                    created_at=NOW,
                    revises_event_ids=parents,
                )
            )
            return result

        gate.recall = recall_then_change
        with pytest.raises(EvidenceChanged):
            await chat.ask(user_id="owner", project="personal", text="Which glaze?", session_id="s")
        assert history.recent(session_key("owner", "s"), project="personal", user_id="owner") == []
        assert (
            conn.execute("SELECT count(*) FROM memories WHERE origin_kind='ask'").fetchone()[0] == 0
        )
    finally:
        conn.close()


async def test_program_notice_has_per_call_result_provenance():
    from morgan_brain.app.chat import TurnRequest

    conn, _, gate, _, client, chat = stack()
    try:
        await fork(gate)
        result = await chat.ask_with_provenance(
            TurnRequest(user_id="owner", project="personal", text="Which glaze?")
        )
        assert result.model_used is None
        assert result.response_author_id == "morgan:conflict-guard"
        assert client.requests == []
        assert "eligible_leaf_count" in result.answer
        assert "revision_truncated" in result.answer
        assert "8" in result.answer
    finally:
        conn.close()


async def test_large_fork_can_be_resolved_in_bounded_stages():
    conn, _, gate, _, client, chat = stack()
    try:
        await gate.store(
            Memory(
                id="root",
                user_id="owner",
                project="personal",
                content="Glaze origin",
                source="user_stated",
                author_id="person",
                created_at=NOW - timedelta(days=2),
            )
        )
        leaves = [f"leaf-{index}" for index in range(9)]
        for identity in leaves:
            await gate.store(
                Memory(
                    id=identity,
                    user_id="owner",
                    project="personal",
                    content="Glaze branch " + identity,
                    source="user_stated",
                    author_id="person",
                    created_at=NOW - timedelta(days=1),
                    revises_event_ids=["root"],
                )
            )
        await chat.ask(user_id="owner", project="personal", text="Which glaze?")
        assert client.requests == []
        for identity, parents in [("partial", leaves[:8]), ("complete", ["partial", leaves[8]])]:
            await gate.store(
                Memory(
                    id=identity,
                    user_id="owner",
                    project="personal",
                    content="Supported glaze correction",
                    source="user_stated",
                    author_id="person",
                    created_at=NOW,
                    revises_event_ids=parents,
                )
            )
            if identity == "partial":
                current = await gate.evidence(
                    user_id="owner", project="personal", evidence_ids=[identity]
                )
                assert current.records[0].revision_state == "conflicted"
        assert (
            await chat.ask(user_id="owner", project="personal", text="Which glaze?")
            == "Synthetic generated answer"
        )
        assert len(client.requests) == 1
    finally:
        conn.close()
