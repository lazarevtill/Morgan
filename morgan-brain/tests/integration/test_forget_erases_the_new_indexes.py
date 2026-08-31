"""`forget()` must reach the semantic index and the persona graph.

Both are derived from the memories being erased -- an entity node, a co-occurrence edge
and a persona node all encode what the owner said. Leaving them behind means `morgan
forget` reports success while the derived shape of that project's content is still on
disk, and the persona graph is the most personal store in the system.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from morgan_brain.models.base import Entity
from morgan_brain.models.memory import Memory
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.retrieval.semantic_index import SemanticIndex
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.episodic import EpisodicStore
from morgan_brain.modules.memory.stores.sqlite_vector import SqliteVectorIndex
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.personalization.persona_graph import PersonaGraph

U = "u1"
P = "plata"
T0 = datetime(2026, 8, 1, tzinfo=UTC)


@pytest.fixture
def stack(tmp_path):
    conn = open_db(str(tmp_path / "morgan.db"))
    semantic = SemanticIndex(conn)
    persona = PersonaGraph(conn)
    module = MemoryModule(
        embedder=FakeEmbedder(dim=8),
        vectors=SqliteVectorIndex(conn, dim=8),
        temporal=SqliteTemporalStore(conn=conn),
        clock=lambda: T0,
        fts=FtsIndex(conn),
        entities=EntityIndex(conn),
        episodics=EpisodicStore(conn),
        semantic=semantic,
    )
    yield module, semantic, persona, conn
    conn.close()


async def _populate(module, semantic, persona, project: str) -> None:
    await module.store(
        Memory(
            user_id=U,
            project=project,
            content="Harbor blocked it",
            entities=[Entity(name="harbor"), Entity(name="gitlab")],
        )
    )
    semantic.ensure_schemas(user_id=U, project=project)
    semantic.assign(user_id=U, project=project, entity="harbor", schema_name="work")
    semantic.assign(user_id=U, project=project, entity="gitlab", schema_name="work")
    semantic.observe_cooccurrence(user_id=U, project=project, names=["harbor", "gitlab"])
    persona.observe(
        user_id=U,
        project=project,
        description="impatient",
        entity="harbor",
        valence=-0.5,
        session_id="s1",
        now=T0,
    )


async def test_forget_erases_the_semantic_index(stack):
    module, semantic, persona, conn = stack
    await _populate(module, semantic, persona, P)

    await module.forget(user_id=U, project=P)

    for table in (
        "mem_entity_nodes",
        "mem_entity_edges",
        "mem_schema_edges",
        "mem_schemas",
    ):
        n = conn.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE user_id = ? AND project = ?",  # noqa: S608
            (U, P),
        ).fetchone()["n"]
        assert n == 0, f"{table} still holds rows after forget()"


async def test_forget_erases_the_persona_graph(stack):
    module, semantic, persona, _conn = stack
    await _populate(module, semantic, persona, P)

    await module.forget(user_id=U, project=P)

    assert persona.all_nodes(user_id=U, project=P) == []


async def test_forget_reports_what_it_erased_from_them(stack):
    module, semantic, persona, _conn = stack
    await _populate(module, semantic, persona, P)

    report = await module.forget(user_id=U, project=P)

    assert report.persona_nodes == 1
    assert report.index_entries > 0


async def test_forget_leaves_another_project_alone(stack):
    module, semantic, persona, _conn = stack
    await _populate(module, semantic, persona, P)
    await _populate(module, semantic, persona, "other")

    await module.forget(user_id=U, project=P)

    assert persona.all_nodes(user_id=U, project="other")
    assert semantic.schema_of(user_id=U, project="other", entity="harbor") == "work"


async def test_forgetting_a_project_with_no_index_rows_is_not_an_error(stack):
    module, _semantic, _persona, _conn = stack
    report = await module.forget(user_id=U, project="never-used")
    assert report.persona_nodes == 0
    assert report.index_entries == 0


async def test_forget_erases_the_pattern_register(stack):
    """A correction class is distilled from this project's edits: derived content."""
    from morgan_brain.learning.patterns import PatternRegister

    module, semantic, persona, conn = stack
    patterns = PatternRegister(conn)
    await _populate(module, semantic, persona, P)
    patterns.record(user_id=U, project=P, title="replies are too long", now=T0)

    await module.forget(user_id=U, project=P)

    assert patterns.all_patterns(user_id=U, project=P) == []


async def test_forget_keeps_the_decision_receipts(stack):
    """Receipts record why the champion is what it is, and the champion is deliberately
    not erased. Deleting the reasoning while keeping the prompt it justified leaves the
    least explicable of the two states."""
    from morgan_brain.learning.receipts import ReceiptStore

    module, semantic, persona, conn = stack
    receipts = ReceiptStore(conn)
    await _populate(module, semantic, persona, P)
    receipts.record(
        prompt_name="system-prompt", verdict="promoted", candidate_body="be terse", now=T0
    )

    await module.forget(user_id=U, project=P)

    assert len(receipts.recent()) == 1
