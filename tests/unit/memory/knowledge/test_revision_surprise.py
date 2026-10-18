"""Explicit revision delivery is independent of lexical novelty; no provider calls."""

from datetime import UTC, datetime

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.knowledge.consolidation import MemoryConsolidator
from morgan_brain.memory.knowledge.surprise import keep_surprising
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemorySource, TemporalFact
from tests.fakes import FakeChatClient

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def event(identity: str, text: str, parents: list[str] | None = None) -> Memory:
    return Memory(
        id=identity,
        user_id="u",
        project="p",
        content=text,
        source=MemorySource.USER_STATED,
        author_id="person:u",
        created_at=NOW,
        revises_event_ids=parents or [],
    )


def fact(text: str) -> TemporalFact:
    return TemporalFact(
        user_id="u",
        project="p",
        subject="user",
        predicate="permission",
        object=text,
        source=MemorySource.USER_STATED,
        created_at=NOW,
    )


@pytest.mark.parametrize(
    "text",
    [
        "user may not publish the project report",
        "пользователь не разрешает публиковать отчёт проекта",
    ],
)
def test_low_novelty_revision_is_retained(text: str) -> None:
    revision = event("revision", text, ["parent"])
    ordinary = event("ordinary", text)
    assert keep_surprising([ordinary, revision], [fact(text)]) == [revision]


@pytest.mark.parametrize("revision_count", [2, 35])
def test_revisions_take_priority_with_same_cap_and_stable_order(revision_count: int) -> None:
    revisions = [event(f"r{i}", "known", ["parent"]) for i in range(revision_count)]
    ordinary = [event(f"o{i}", f"novel word {i}") for i in range(40)]
    selected = keep_surprising(ordinary + revisions, [fact("known")])
    assert len(selected) == 30
    assert selected == revisions[:30] + ordinary[: max(0, 30 - revision_count)]
    assert keep_surprising(ordinary, []) == ordinary[:30]


async def test_actual_gate_delivers_active_revision_without_applying_noop() -> None:
    conn = open_db(":memory:")
    module = build_memory_module(conn, embedder=FakeEmbedder(dim=16), dim=16, clock=lambda: NOW)
    gate = MemoryGate(module)
    original = event("parent", "user may publish the project report")
    revision = event("revision", "user may not publish the project report", ["parent"])
    await gate.store(original)
    await gate.store(revision)
    current = fact(original.content)
    await gate.upsert_fact(current)
    client = FakeChatClient(
        replies=['{"ops":[{"op":"NOOP","subject":"user","predicate":"permission"}]}']
    )
    consolidator = MemoryConsolidator(gate=gate, client=client, model="fake", clock=lambda: NOW)
    assert await consolidator.consolidate("u", project="p") == []
    assert client.calls == 1
    prompt = client.last_messages[-1].content
    assert '"id": "revision"' in prompt
    assert '"id": "parent"' not in prompt
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1
    conn.close()
