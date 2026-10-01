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
from morgan_brain.models import Memory, MemorySource, TemporalFact

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


@pytest.mark.parametrize("regular_changed", [False, True])
async def test_actual_non_noop_apply_with_structural_heads(regular_changed):
    from morgan_brain.memory.knowledge.consolidation import FactOp, FactOpBatch, MemoryConsolidator

    class NoGeneration:
        async def agenerate(self, *args, **kwargs):
            raise AssertionError("Applying a prepared operation must not call a model")

    conn, gate = setup()
    try:
        await gate.store(
            Memory(
                id="source",
                user_id="owner",
                content="Ceramic lasts.",
                source=MemorySource.USER_STATED,
                author_id="person:owner",
                created_at=NOW,
            )
        )
        await fact(gate, "existing")
        await fact(gate, "working_checkpoint:gift", WORKING_CONTEXT_PREDICATE, "view")
        await fact(gate, "checkpoint:task", CHECKPOINT_PREDICATE, "checkpoint")
        prepared = await gate.capture_consolidation_basis(
            user_id="owner",
            project="personal",
            event_ids=["source"],
            generation=gate.capture_erasure_generation(),
        )
        await fact(gate, "working_checkpoint:gift", WORKING_CONTEXT_PREDICATE, "updated view")
        await fact(gate, "checkpoint:task", CHECKPOINT_PREDICATE, "updated checkpoint")
        if regular_changed:
            await fact(gate, "existing", value="steel")
        op = FactOp(
            op="ADD",
            subject="new preference",
            predicate="material",
            object="ceramic",
            support_event_ids=["source"],
        )
        consolidator = MemoryConsolidator(
            gate=gate, client=NoGeneration(), model="fake", clock=lambda: NOW
        )
        if regular_changed:
            with pytest.raises(StaleConsolidationProposal, match="fact basis"):
                await consolidator.apply(
                    "owner", FactOpBatch(ops=[op]), project="personal", basis=prepared.basis
                )
        else:
            assert await consolidator.apply(
                "owner", FactOpBatch(ops=[op]), project="personal", basis=prepared.basis
            ) == [op]
        added = [
            f
            for f in await gate.current_facts(user_id="owner", project="personal")
            if f.subject == "new preference"
        ]
        assert len(added) == (0 if regular_changed else 1)
        if added:
            assert added[0].support_event_ids == ["source"]
    finally:
        conn.close()


@pytest.mark.parametrize("predicate", [WORKING_CONTEXT_PREDICATE, CHECKPOINT_PREDICATE])
async def test_legacy_predicate_fact_remains_recallable_consolidatable_and_not_a_context(predicate):
    from morgan_brain.memory.knowledge.consolidation import FactOp, FactOpBatch, MemoryConsolidator
    from morgan_brain.models import MemoryQuery

    class NoGeneration:
        async def agenerate(self, *args, **kwargs):
            raise AssertionError("No model calls")

    conn, gate = setup()
    try:
        legacy = await fact(gate, "legacy-person", predicate)
        recalled = await gate.recall(
            MemoryQuery(user_id="owner", project="personal", text="ceramic", top_k=10)
        )
        assert any(item.id == legacy for item in recalled.memories)
        listed = await gate.list_working_contexts(user_id="owner", project="personal")
        assert listed.items == []
        prepared = await capture(gate)
        assert prepared.basis.fact_ids == (legacy,)
        await fact(gate, "legacy-person", predicate, "steel")
        with gate.write_transaction(), pytest.raises(StaleConsolidationProposal):
            await gate.check_consolidation_basis(prepared.basis)
        await gate.store(
            Memory(
                id="legacy-source",
                user_id="owner",
                content="Titanium lasts.",
                source=MemorySource.USER_STATED,
                author_id="person:owner",
                created_at=NOW,
            )
        )
        prepared = await gate.capture_consolidation_basis(
            user_id="owner",
            project="personal",
            event_ids=["legacy-source"],
            generation=gate.capture_erasure_generation(),
        )
        op = FactOp(
            op="UPDATE",
            subject="legacy-person",
            predicate=predicate,
            object="titanium",
            support_event_ids=["legacy-source"],
        )
        worker = MemoryConsolidator(
            gate=gate, client=NoGeneration(), model="fake", clock=lambda: NOW
        )
        assert await worker.apply(
            "owner", FactOpBatch(ops=[op]), project="personal", basis=prepared.basis
        ) == [op]
        current = await gate.current_facts(user_id="owner", project="personal")
        assert len(current) == 1 and current[0].object == "titanium"
    finally:
        conn.close()
