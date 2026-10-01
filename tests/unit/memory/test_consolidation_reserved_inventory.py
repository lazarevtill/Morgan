"""Reserved structural heads never consume or invalidate the factual proposal inventory."""

from datetime import UTC, datetime

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.checkpoints import CHECKPOINT_PREDICATE
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.knowledge.basis import ConsolidationInputLimit, StaleConsolidationProposal
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.working_context import WORKING_CONTEXT_PREDICATE
from morgan_brain.models import TemporalFact

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def setup():
    conn = open_db(":memory:")
    gate = MemoryGate(
        build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4, clock=lambda: NOW)
    )
    return conn, gate


async def fact(gate, subject, predicate="preference", value="ceramic"):
    return await gate.upsert_fact(
        TemporalFact(
            user_id="owner",
            project="personal",
            subject=subject,
            predicate=predicate,
            object=value,
            valid_from=NOW,
        ),
        now=NOW,
    )


async def capture(gate):
    return await gate.capture_consolidation_basis(
        user_id="owner",
        project="personal",
        event_ids=[],
        generation=gate.capture_erasure_generation(),
    )


async def test_256_regular_plus_structural_heads_fit_exact_fact_limit():
    conn, gate = setup()
    try:
        regular = {await fact(gate, f"subject-{i}") for i in range(256)}
        await fact(gate, "working_checkpoint:gift", WORKING_CONTEXT_PREDICATE, "view")
        await fact(gate, "checkpoint:task", CHECKPOINT_PREDICATE, "checkpoint")
        prepared = await capture(gate)
        assert set(prepared.basis.fact_ids) == regular
        assert len(prepared.facts) == 256
    finally:
        conn.close()


async def test_structural_changes_do_not_stale_basis_but_regular_changes_do():
    conn, gate = setup()
    try:
        regular = await fact(gate, "person")
        await fact(gate, "working_checkpoint:gift", WORKING_CONTEXT_PREDICATE, "view")
        await fact(gate, "checkpoint:task", CHECKPOINT_PREDICATE, "checkpoint")
        prepared = await capture(gate)
        assert prepared.basis.fact_ids == (regular,)
        await fact(gate, "working_checkpoint:gift", WORKING_CONTEXT_PREDICATE, "updated view")
        await fact(gate, "checkpoint:task", CHECKPOINT_PREDICATE, "updated checkpoint")
        with gate.write_transaction():
            await gate.check_consolidation_basis(prepared.basis)
        await fact(gate, "person", value="steel")
        with (
            gate.write_transaction(),
            pytest.raises(StaleConsolidationProposal, match="fact basis"),
        ):
            await gate.check_consolidation_basis(prepared.basis)
    finally:
        conn.close()


async def test_257_actual_regular_facts_still_refuse():
    conn, gate = setup()
    try:
        for i in range(257):
            await fact(gate, f"subject-{i}")
        with pytest.raises(ConsolidationInputLimit, match="256"):
            await capture(gate)
    finally:
        conn.close()
