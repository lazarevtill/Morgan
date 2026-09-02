"""The upper index, end to end through the real composition wiring.

The unit tests prove each piece. This proves they are connected: a memory stored through
the gate files its entities, and a later recall routes through them -- neither of which
happens if the index is built but never read, or read but never built.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.db import open_db
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.models import Memory, MemoryQuery

U = "u1"
P = "acme"
T0 = datetime(2026, 8, 31, tzinfo=UTC)


@pytest.fixture
def stack(tmp_path):
    conn = open_db(str(tmp_path / "morgan.db"))
    module = build_memory_module(conn, embedder=FakeEmbedder(dim=16), dim=16, clock=lambda: T0)
    yield MemoryGate(module), module._semantic
    conn.close()


async def _remember(gate: MemoryGate, text: str) -> None:
    await gate.store(Memory(user_id=U, project=P, content=text))


async def test_a_stored_memory_files_its_entities_into_the_index(stack):
    gate, semantic = stack
    await _remember(gate, "Harbor blocked the deploy again")
    assert semantic.schema_of(user_id=U, project=P, entity="harbor") == "work"


async def test_recall_routes_through_what_was_filed(stack):
    """Two unrelated memories, one query. Without routing the fused ranking would offer
    both; the pool leaves only the one the query points at."""
    gate, _semantic = stack
    await _remember(gate, "Harbor blocked the deploy again")
    await _remember(gate, "the Dentist appointment moved to the gym slot")

    hits = await gate.recall(MemoryQuery(user_id=U, project=P, text="Harbor", top_k=8))
    contents = [m.content for m in hits]
    assert "Harbor blocked the deploy again" in contents
    assert "the Dentist appointment moved to the gym slot" not in contents


async def test_cross_project_recall_is_never_routed(stack):
    """all_projects=True is the explicit escape hatch. The index is per project, so
    routing it would make the escape hatch stricter than the default."""
    gate, _semantic = stack
    await _remember(gate, "Harbor blocked the deploy again")
    await _remember(gate, "the Dentist appointment moved to the gym slot")

    hits = await gate.recall(
        MemoryQuery(user_id=U, project=P, all_projects=True, text="Harbor", top_k=8)
    )
    assert len(hits) == 2


async def test_an_unrouted_query_still_recalls_everything(stack):
    gate, _semantic = stack
    await _remember(gate, "Harbor blocked the deploy again")
    await _remember(gate, "the Dentist appointment moved to the gym slot")

    hits = await gate.recall(MemoryQuery(user_id=U, project=P, text="moved slot", top_k=8))
    assert any("Dentist" in m.content for m in hits)
