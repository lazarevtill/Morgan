"""The entity ranking is one of the three signals `MemoryModule.recall` fuses. It is only
non-empty if the write path extracts entities -- which is why extraction happens inside
``store`` for every caller, rather than in whichever caller remembers to do it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.db import open_db
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.models import Memory


@pytest.fixture
def wiring(tmp_path):
    conn = open_db(str(tmp_path / "morgan.db"))
    module = build_memory_module(
        conn, embedder=FakeEmbedder(dim=4), dim=4, clock=lambda: datetime(2026, 8, 31, tzinfo=UTC)
    )
    yield conn, module, MemoryGate(module)
    conn.close()


async def _remember(gate: MemoryGate, text: str) -> None:
    await gate.store(Memory(user_id="u1", project="acme", content=text))


async def test_a_stored_memory_populates_the_entity_index(wiring):
    conn, _module, gate = wiring
    await _remember(gate, "Harbor blocked the deploy for Alice")
    rows = conn.execute("SELECT name FROM memory_entities WHERE user_id = 'u1'").fetchall()
    assert {r["name"] for r in rows} == {"harbor", "alice"}


async def test_the_entity_signal_actually_returns_the_memory(wiring):
    """Populating the table is only half of it -- the third fused ranking has to be
    non-empty for the same query recall would issue."""
    _conn, module, gate = wiring
    await _remember(gate, "Harbor blocked the deploy")
    ranking = module._entities.search({"harbor"}, user_id="u1", top_k=8, project="acme")
    assert ranking, "entity ranking is empty: the third recall signal is dead"


async def test_cyrillic_memories_are_indexed_too(wiring):
    conn, _module, gate = wiring
    await _remember(gate, "Ромашка отправила образец")
    rows = conn.execute("SELECT name FROM memory_entities WHERE user_id = 'u1'").fetchall()
    assert "ромашка" in {r["name"] for r in rows}


async def test_entities_are_scoped_to_the_memory_project(wiring):
    conn, _module, gate = wiring
    await _remember(gate, "Harbor blocked the deploy")
    rows = conn.execute("SELECT DISTINCT project FROM memory_entities").fetchall()
    assert [r["project"] for r in rows] == ["acme"]


async def test_the_semantic_index_is_filed_in_the_same_write(wiring):
    """A memory that reaches the entity index but not the upper index is invisible to
    routing -- and a pool that excludes it would cut it from recall."""
    _conn, module, gate = wiring
    await _remember(gate, "Harbor blocked the deploy")
    assert module._semantic.schema_of(user_id="u1", project="acme", entity="harbor") == "work"
