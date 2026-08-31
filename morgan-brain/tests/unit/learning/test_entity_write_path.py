"""The entity ranking is one of the three signals `MemoryModule.recall` fuses, and it was
dead: no write path set `Memory.entities`, so `memory_entities` only ever held rows the
tests inserted by hand. These tests pin the write path itself, through the real
consolidation learner and the real gate -- not through a hand-built Memory.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from morgan_brain.learning.consolidation import MemoryConsolidator
from morgan_brain.learning.learner import ConsolidationLearner
from morgan_brain.models.message import Conversation, Message, Role
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.episodic import EpisodicStore
from morgan_brain.modules.memory.stores.sqlite_vector import SqliteVectorIndex
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter
from morgan_brain.security.memory_gate import MemoryGate


@pytest.fixture
def wiring(tmp_path):
    conn: sqlite3.Connection = open_db(str(tmp_path / "morgan.db"))
    entities = EntityIndex(conn)
    module = MemoryModule(
        embedder=FakeEmbedder(dim=4),
        vectors=SqliteVectorIndex(conn, dim=4),
        temporal=SqliteTemporalStore(conn=conn),
        clock=lambda: datetime(2026, 8, 31, tzinfo=UTC),
        fts=FtsIndex(conn),
        entities=entities,
        episodics=EpisodicStore(conn),
    )
    gate = MemoryGate(module)
    yield conn, entities, gate, _build_learner(gate)
    conn.close()


def _build_learner(gate: MemoryGate) -> ConsolidationLearner:
    """A real learner over a fake provider stack.

    ``process_session`` never reaches the consolidator, but building the real object
    rather than a stub keeps the test honest about what the write path actually
    depends on.
    """
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
        reg=reg, bindings={"strong": [Binding("fake", "test-model", FakeChatClient(replies=[]))]}
    )
    return ConsolidationLearner(
        consolidator=MemoryConsolidator(
            gate=gate,
            router=router,
            capability_registry=reg,
            clock=lambda: datetime(2026, 8, 31, tzinfo=UTC),
        ),
        gate=gate,
        clock=lambda: datetime(2026, 8, 31, tzinfo=UTC),
    )


def _conversation(text: str) -> Conversation:
    return Conversation(
        user_id="u1",
        project="acme",
        session_id="s1",
        messages=[Message(user_id="u1", role=Role.USER, content=text)],
    )


async def test_a_stored_turn_populates_the_entity_index(wiring):
    conn, _entities, _gate, learner = wiring

    await learner.process_session(_conversation("Harbor blocked the deploy for Alice"))

    rows = conn.execute("SELECT name FROM memory_entities WHERE user_id = 'u1'").fetchall()
    assert {r["name"] for r in rows} == {"harbor", "alice"}


async def test_the_entity_signal_actually_returns_the_memory(wiring):
    """Populating the table is only half of it -- the third fused ranking has to be
    non-empty for the same query the orchestrator would issue."""
    _conn, entities, _gate, learner = wiring
    await learner.process_session(_conversation("Harbor blocked the deploy"))

    ranking = entities.search({"harbor"}, user_id="u1", top_k=8, project="acme")
    assert ranking, "entity ranking is empty: the third recall signal is still dead"


async def test_cyrillic_turns_are_indexed_too(wiring):
    conn, _entities, _gate, learner = wiring

    await learner.process_session(_conversation("Ромашка отправила образец"))

    rows = conn.execute("SELECT name FROM memory_entities WHERE user_id = 'u1'").fetchall()
    assert "ромашка" in {r["name"] for r in rows}


async def test_entities_are_scoped_to_the_conversation_project(wiring):
    conn, _entities, _gate, learner = wiring
    await learner.process_session(_conversation("Harbor blocked the deploy"))

    rows = conn.execute("SELECT DISTINCT project FROM memory_entities").fetchall()
    assert [r["project"] for r in rows] == ["acme"]
