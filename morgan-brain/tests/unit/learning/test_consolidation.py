"""Unit tests for MemoryConsolidator — Phase 2B, bi-temporal consolidation.

All tests are deterministic:
  - clock injected (no datetime.now())
  - FakeChatClient (no network)
  - SqliteTemporalStore(":memory:") (no disk)
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from morgan_brain.learning.consolidation import (
    FactOp,
    FactOpBatch,
    FactOpKind,
    MemoryConsolidator,
)
from morgan_brain.models.memory import Memory, MemoryKind, MemorySource, TemporalFact
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter
from morgan_brain.security.memory_gate import MemoryGate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2026, 6, 1, tzinfo=timezone.utc)
T_FAR = datetime(2027, 6, 1, tzinfo=timezone.utc)  # ~1 year later for decay tests


def _make_batch(**ops_kwargs: object) -> str:
    """Serialise a FactOpBatch to JSON for use as FakeChatClient reply."""
    batch = FactOpBatch(ops=[FactOp(**ops_kwargs)])  # type: ignore[arg-type]
    return batch.model_dump_json()


def _build_stack(
    fake_replies: list[str],
    clock: object,
) -> tuple[MemoryConsolidator, SqliteTemporalStore, MemoryGate]:
    temporal = SqliteTemporalStore(":memory:")
    embedder = FakeEmbedder(dim=16)
    vector_index = InMemoryVectorIndex()
    memory_module = MemoryModule(
        embedder=embedder,
        vectors=vector_index,
        temporal=temporal,
        clock=lambda: T0,  # type: ignore[arg-type]
    )
    gate = MemoryGate(memory_module)

    fake_client = FakeChatClient(replies=fake_replies)
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
        reg=reg,
        bindings={"strong": [Binding("fake", "test-model", fake_client)]},
    )
    consolidator = MemoryConsolidator(
        gate=gate,
        temporal=temporal,
        router=router,
        capability_registry=reg,
        clock=clock,  # type: ignore[arg-type]
        role="strong",
    )
    return consolidator, temporal, gate


# ---------------------------------------------------------------------------
# FactOpBatch model validation
# ---------------------------------------------------------------------------


def test_fact_op_kinds_are_all_string_enum() -> None:
    for kind in FactOpKind:
        assert isinstance(kind.value, str)


def test_fact_op_batch_round_trip() -> None:
    batch = FactOpBatch(
        ops=[
            FactOp(op=FactOpKind.ADD, subject="user", predicate="lives_in", object="Berlin"),
            FactOp(op=FactOpKind.NOOP, subject="user", predicate="works_at", object=""),
        ]
    )
    serialised = batch.model_dump_json()
    restored = FactOpBatch.model_validate_json(serialised)
    assert len(restored.ops) == 2
    assert restored.ops[0].op is FactOpKind.ADD


# ---------------------------------------------------------------------------
# propose()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_propose_calls_llm_and_returns_batch() -> None:
    reply = _make_batch(op="ADD", subject="user", predicate="lives_in", object="Berlin")
    consolidator, _, _ = _build_stack([reply], clock=lambda: T0)

    episodic = [
        Memory(
            user_id="u1",
            kind=MemoryKind.EPISODIC,
            content="I live in Berlin",
            source=MemorySource.USER_STATED,
        )
    ]
    batch = await consolidator.propose("u1", episodic, existing_facts=[])
    assert len(batch.ops) == 1
    assert batch.ops[0].op is FactOpKind.ADD
    assert batch.ops[0].object == "Berlin"


# ---------------------------------------------------------------------------
# apply() — ADD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_add_creates_current_fact() -> None:
    consolidator, temporal, gate = _build_stack([], clock=lambda: T0)

    batch = FactOpBatch(
        ops=[FactOp(op=FactOpKind.ADD, subject="user", predicate="lives_in", object="Berlin")]
    )
    applied = await consolidator.apply("u1", batch)

    assert len(applied) == 1
    current = await temporal.current_facts(user_id="u1")
    assert len(current) == 1
    assert current[0].object == "Berlin"
    assert current[0].valid_to is None


# ---------------------------------------------------------------------------
# apply() — UPDATE (contradiction)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_update_closes_old_fact_and_opens_new() -> None:
    consolidator, temporal, gate = _build_stack([], clock=lambda: T1)

    # Seed an existing fact: Berlin
    await gate.upsert_fact(
        TemporalFact(
            user_id="u1",
            subject="user",
            predicate="lives_in",
            object="Berlin",
            source=MemorySource.USER_STATED,
        )
    )

    batch = FactOpBatch(
        ops=[FactOp(op=FactOpKind.UPDATE, subject="user", predicate="lives_in", object="Munich")]
    )
    applied = await consolidator.apply("u1", batch)

    assert len(applied) == 1
    current = await temporal.current_facts(user_id="u1")
    assert len(current) == 1
    assert current[0].object == "Munich"

    history = await temporal.history(user_id="u1", subject="user", predicate="lives_in")
    assert len(history) == 2
    closed = next(f for f in history if f.object == "Berlin")
    assert closed.valid_to is not None  # interval was closed


# ---------------------------------------------------------------------------
# apply() — DELETE closes interval (no hard delete)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_delete_closes_interval_not_hard_delete() -> None:
    consolidator, temporal, gate = _build_stack([], clock=lambda: T1)

    await gate.upsert_fact(
        TemporalFact(
            user_id="u1",
            subject="user",
            predicate="prefers",
            object="dark_mode",
            source=MemorySource.USER_STATED,
        )
    )

    batch = FactOpBatch(
        ops=[FactOp(op=FactOpKind.DELETE, subject="user", predicate="prefers", object="dark_mode")]
    )
    applied = await consolidator.apply("u1", batch)

    assert len(applied) == 1
    # No longer current
    current = await temporal.current_facts(user_id="u1")
    assert len(current) == 0
    # But retained in history
    history = await temporal.history(user_id="u1", subject="user", predicate="prefers")
    assert len(history) == 1
    assert history[0].valid_to is not None  # interval closed, not deleted


# ---------------------------------------------------------------------------
# apply() — NOOP is skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_noop_is_skipped() -> None:
    consolidator, temporal, _ = _build_stack([], clock=lambda: T0)

    batch = FactOpBatch(
        ops=[FactOp(op=FactOpKind.NOOP, subject="user", predicate="lives_in", object="Berlin")]
    )
    applied = await consolidator.apply("u1", batch)

    assert applied == []
    current = await temporal.current_facts(user_id="u1")
    assert current == []


# ---------------------------------------------------------------------------
# apply() — dedup: ADD of already-current fact becomes NOOP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_add_dedup_skips_existing_fact() -> None:
    consolidator, temporal, gate = _build_stack([], clock=lambda: T0)

    await gate.upsert_fact(
        TemporalFact(
            user_id="u1",
            subject="user",
            predicate="lives_in",
            object="Berlin",
            source=MemorySource.USER_STATED,
        )
    )

    batch = FactOpBatch(
        ops=[FactOp(op=FactOpKind.ADD, subject="user", predicate="lives_in", object="Berlin")]
    )
    applied = await consolidator.apply("u1", batch)

    # Should be treated as NOOP — not applied
    assert applied == []
    # Still only one fact in current
    current = await temporal.current_facts(user_id="u1")
    assert len(current) == 1


# ---------------------------------------------------------------------------
# consolidate() — orchestrates propose → apply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consolidate_orchestrates_propose_and_apply() -> None:
    reply = _make_batch(
        op="ADD", subject="user", predicate="works_at", object="Acme",
        reason="user mentioned employer"
    )
    consolidator, temporal, gate = _build_stack([reply], clock=lambda: T0)

    # Seed an episodic memory so gate.recall returns something
    module = gate._store  # type: ignore[attr-defined]
    await module.store(
        Memory(
            user_id="u1",
            kind=MemoryKind.EPISODIC,
            content="I work at Acme",
            source=MemorySource.USER_STATED,
            created_at=T0,
        )
    )

    ops = await consolidator.consolidate("u1")

    assert len(ops) >= 1
    current = await temporal.current_facts(user_id="u1")
    facts = [f for f in current if f.predicate == "works_at"]
    assert len(facts) == 1
    assert facts[0].object == "Acme"


# ---------------------------------------------------------------------------
# decay_confidence()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decay_reduces_confidence_with_age() -> None:
    consolidator, temporal, gate = _build_stack([], clock=lambda: T0)

    await gate.upsert_fact(
        TemporalFact(
            user_id="u1",
            subject="user",
            predicate="lives_in",
            object="Berlin",
            source=MemorySource.AGENT_INFERRED,
            confidence=1.0,
        )
    )

    # Run decay far in the future (> 1 half-life of 30 days)
    await consolidator.decay_confidence("u1", half_life_days=30.0, now=T_FAR)

    current = await temporal.current_facts(user_id="u1")
    assert len(current) == 1
    # Confidence must have decreased from 1.0
    assert current[0].confidence < 1.0


@pytest.mark.asyncio
async def test_decay_flags_facts_below_threshold_as_stale() -> None:
    consolidator, temporal, gate = _build_stack([], clock=lambda: T0)

    await gate.upsert_fact(
        TemporalFact(
            user_id="u1",
            subject="user",
            predicate="lives_in",
            object="Berlin",
            source=MemorySource.AGENT_INFERRED,
            confidence=1.0,
        )
    )

    # Very far future → confidence will be below default threshold 0.2
    very_far = datetime(2030, 1, 1, tzinfo=timezone.utc)
    stale = await consolidator.decay_confidence(
        "u1", half_life_days=30.0, now=very_far, stale_threshold=0.9
    )

    assert len(stale) >= 1
    assert all(f.predicate == "lives_in" for f in stale)


@pytest.mark.asyncio
async def test_decay_deterministic_given_same_clock() -> None:
    """Same clock → same result twice."""
    consolidator, temporal, gate = _build_stack([], clock=lambda: T0)

    await gate.upsert_fact(
        TemporalFact(
            user_id="u1",
            subject="user",
            predicate="works_at",
            object="Corp",
            source=MemorySource.AGENT_INFERRED,
            confidence=0.9,
        )
    )

    await consolidator.decay_confidence("u1", half_life_days=30.0, now=T_FAR)
    conf_a = (await temporal.current_facts(user_id="u1"))[0].confidence

    # Reset confidence to 0.9 manually by upserting again
    # (TemporalFact upsert supersedes — we need a fresh store to test idempotency)
    consolidator2, temporal2, gate2 = _build_stack([], clock=lambda: T0)
    await gate2.upsert_fact(
        TemporalFact(
            user_id="u1",
            subject="user",
            predicate="works_at",
            object="Corp",
            source=MemorySource.AGENT_INFERRED,
            confidence=0.9,
        )
    )
    await consolidator2.decay_confidence("u1", half_life_days=30.0, now=T_FAR)
    conf_b = (await temporal2.current_facts(user_id="u1"))[0].confidence

    assert conf_a == pytest.approx(conf_b)
