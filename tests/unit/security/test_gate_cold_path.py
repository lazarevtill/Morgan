"""MemoryGate must be the single chokepoint for the cold path too.

Before this test existed, `MemoryConsolidator` held a raw `SqliteTemporalStore` reference and
called `current_facts`/`close_fact`/`set_confidence` directly -- operations the gate did not
expose. A project filter enforced only at the gate did not bind consolidation, so the nightly
consolidator could read and supersede facts across project boundaries. These tests pin the gate
as the only path to those operations, with project scoping enforced end to end.
"""

from __future__ import annotations

import inspect

import pytest

from morgan_brain.models.memory import TemporalFact
from morgan_brain.security.memory_gate import MemoryGate


async def test_close_fact_is_exposed_on_the_gate(gate: MemoryGate) -> None:
    fid = await gate.upsert_fact(
        TemporalFact(user_id="u", project="p", subject="s", predicate="p", object="o")
    )
    await gate.close_fact(fid, user_id="u", project="p")
    assert await gate.current_facts(user_id="u", project="p") == []


async def test_current_facts_is_project_scoped(gate: MemoryGate) -> None:
    await gate.upsert_fact(
        TemporalFact(user_id="u", project="acme", subject="s", predicate="p", object="o")
    )
    assert await gate.current_facts(user_id="u", project="personal") == []


async def test_gate_rejects_empty_project(gate: MemoryGate) -> None:
    with pytest.raises(PermissionError):
        await gate.current_facts(user_id="u", project="")


async def test_set_confidence_is_exposed_on_the_gate(gate: MemoryGate) -> None:
    fid = await gate.upsert_fact(
        TemporalFact(user_id="u", project="p", subject="s", predicate="p", object="o")
    )
    await gate.set_confidence(fid, user_id="u", project="p", value=0.3)
    facts = await gate.current_facts(user_id="u", project="p")
    assert facts[0].confidence == 0.3


async def test_close_fact_cannot_reach_another_projects_fact(gate: MemoryGate) -> None:
    """A caller who knows a fact id must not be able to close a fact from a different project."""
    fid = await gate.upsert_fact(
        TemporalFact(user_id="u", project="acme", subject="s", predicate="p", object="o")
    )
    await gate.close_fact(fid, user_id="u", project="personal")
    assert await gate.current_facts(user_id="u", project="acme") != []


async def test_close_fact_cannot_reach_another_users_fact(gate: MemoryGate) -> None:
    fid = await gate.upsert_fact(
        TemporalFact(user_id="u1", project="p", subject="s", predicate="p", object="o")
    )
    await gate.close_fact(fid, user_id="u2", project="p")
    assert await gate.current_facts(user_id="u1", project="p") != []


def test_consolidator_does_not_hold_a_raw_store() -> None:
    """Regression: consolidation must go through the gate, not around it."""
    from morgan_brain.learning.consolidation import MemoryConsolidator

    params = inspect.signature(MemoryConsolidator.__init__).parameters
    assert "temporal" not in params
    assert "gate" in params
