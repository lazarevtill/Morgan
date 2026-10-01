"""Correction lifecycle uses real source stores and public scoped reads."""

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.revisions import RevisionError
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.episodic import EventIdentityConflict
from morgan_brain.models import Memory, MemoryQuery, MemorySource, MemoryStatus, TemporalFact

JAN = datetime(2026, 1, 1, tzinfo=UTC)
JUN = datetime(2026, 6, 1, tzinfo=UTC)
SEP = datetime(2026, 9, 1, tzinfo=UTC)


class CountingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__(dim=4)
        self.calls = 0

    async def embed(self, text):
        self.calls += 1
        return await super().embed(text)


def build(conn, *, at=SEP, embedder=None):
    return MemoryGate(
        build_memory_module(conn, embedder=embedder or FakeEmbedder(dim=4), dim=4, clock=lambda: at)
    )


def event(identity, *, at=JAN, parents=None, **fields):
    return Memory(
        id=identity,
        user_id="owner",
        content="synthetic " + identity,
        source=MemorySource.USER_STATED,
        author_id="person:owner",
        effective_at=at,
        revises_event_ids=parents or [],
        **fields,
    )


async def records(gate, identities, at=SEP):
    return (
        await gate.evidence(
            user_id="owner", project="personal", evidence_ids=identities, effective_at=at
        )
    ).records


async def test_linear_future_late_and_quarantine_preserve_raw_sources(tmp_path):
    conn = open_db(str(tmp_path / "linear.db"))
    gate = build(conn)
    try:
        await gate.store(event("A"))
        await gate.store(event("B", at=JUN, parents=["A"]))
        await gate.store(event("Q", at=SEP, parents=["B"], status=MemoryStatus.QUARANTINED))
        before = await records(gate, ["A", "B", "Q"], at=JAN)
        assert [m.revision_state for m in before] == ["active", "inactive", "inactive"]
        after = await records(gate, ["A", "B", "Q"])
        assert [m.revision_state for m in after] == ["inactive", "active", "inactive"]
        assert after[1].revision_root_id == "A" and after[1].revises_event_ids == ["A"]
        assert after[2].status is MemoryStatus.QUARANTINED
        assert after[0].content == "synthetic A"
        assert after[1].recorded_at == SEP and after[1].created_at == JUN
    finally:
        conn.close()


async def test_fork_multi_parent_resolution_and_canonical_replay_survive_restart(tmp_path):
    path = str(tmp_path / "fork.db")
    conn = open_db(path)
    embedder = CountingEmbedder()
    gate = build(conn, embedder=embedder)
    await gate.store(event("A"))
    await gate.store(event("B", at=JUN, parents=["A"]))
    await gate.store(event("C", at=JUN, parents=["A"]))
    branch = await records(gate, ["B", "C"], JUN)
    assert all(m.revision_state == "conflicted" for m in branch)
    assert all(m.eligible_leaf_ids == ["B", "C"] for m in branch)
    await gate.store(event("D", at=SEP, parents=["C", "B"]))
    baseline = conn.serialize()
    calls = embedder.calls
    assert await gate.store(event("D", at=SEP, parents=["B", "C"])) == "D"
    assert embedder.calls == calls and conn.serialize() == baseline
    with pytest.raises(EventIdentityConflict, match="revises_event_ids"):
        await gate.store(event("D", at=SEP, parents=["A"]))
    assert embedder.calls == calls and conn.serialize() == baseline
    conn.close()
    conn = open_db(path)
    try:
        gate = build(conn)
        final = await records(gate, ["A", "B", "C", "D"])
        assert [m.revision_state for m in final] == ["inactive", "inactive", "inactive", "active"]
        assert final[-1].eligible_leaf_ids == ["D"]
    finally:
        conn.close()


@pytest.mark.parametrize("changes", [{"user_id": "other"}, {"project": "work"}])
async def test_parent_unavailable_is_scoped_before_hydration_and_embedding(changes):
    conn = open_db(":memory:")
    embedder = CountingEmbedder()
    gate = build(conn, embedder=embedder)
    try:
        parent = event("A")
        parent = parent.model_copy(update=changes)
        await gate.store(parent)
        conn.execute("UPDATE memories SET entities='malformed foreign payload' WHERE id='A'")
        conn.commit()
        before, calls = conn.serialize(), embedder.calls
        with pytest.raises(RevisionError, match="revision_parent_unavailable"):
            await gate.store(event("B", at=JUN, parents=["A"]))
        assert conn.serialize() == before and embedder.calls == calls
    finally:
        conn.close()


async def test_source_actor_family_and_parent_quarantine_refusals_are_atomic():
    conn = open_db(":memory:")
    embedder = CountingEmbedder()
    gate = build(conn, embedder=embedder)
    try:
        await gate.store(event("A"))
        await gate.store(event("other-root"))
        await gate.store(event("Q", status=MemoryStatus.QUARANTINED))
        before, calls = conn.serialize(), embedder.calls
        candidates = [
            event("wrong-author", at=JUN, parents=["A"]).model_copy(update={"author_id": "other"}),
            event("cross-root", at=JUN, parents=["A", "other-root"]),
            event("quarantine-parent", at=JUN, parents=["Q"]),
            event("backdated", at=JAN, parents=["A"]).model_copy(
                update={"created_at": datetime(2025, 1, 1, tzinfo=UTC)}
            ),
        ]
        for candidate in candidates:
            with pytest.raises(RevisionError):
                await gate.store(candidate)
            assert conn.serialize() == before and embedder.calls == calls
        for parents in (["A"] * 2, ["A"] * 9, [None], "A"):
            with pytest.raises(ValidationError):
                event("bad", at=JUN, parents=parents)
    finally:
        conn.close()


async def test_declared_support_is_rechecked_under_lock_and_raw_fact_remains_available():
    conn = open_db(":memory:")
    gate = build(conn, at=JUN)
    try:
        await gate.store(event("A"))
        prepared = TemporalFact(
            id="fA",
            user_id="owner",
            subject="user",
            predicate="prefers",
            object="tea",
            source=MemorySource.AGENT_INFERRED,
            support_event_ids=["A"],
        )
        await gate.upsert_fact(prepared)
        await gate.store(event("B", at=JUN, parents=["A"]))
        before = conn.serialize()
        with pytest.raises(RevisionError, match="stale_revision_basis"):
            await gate.upsert_fact(prepared.model_copy(update={"id": "stale"}))
        assert conn.serialize() == before
        recalled = (await gate.recall(MemoryQuery(user_id="owner", text="prefers"))).memories
        assert not any(m.id == "fA" for m in recalled)
        exact = await records(gate, ["fA"])
        assert exact[0].support_state == "inactive_support"
        await gate.upsert_fact(
            TemporalFact(
                id="legacy",
                user_id="owner",
                subject="other",
                predicate="prefers",
                object="tea",
                source=MemorySource.AGENT_INFERRED,
            )
        )
        assert (await records(gate, ["legacy"]))[0].support_state == "unsupported"
    finally:
        conn.close()


class PausedEmbedder(CountingEmbedder):
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def embed(self, text):
        self.entered.set()
        await self.release.wait()
        return await super().embed(text)


async def test_two_real_connections_preserve_concurrent_siblings(tmp_path):
    path = str(tmp_path / "concurrent.db")
    left_conn, right_conn = open_db(path), open_db(path)
    paused = PausedEmbedder()
    left, right = build(left_conn), build(right_conn)
    try:
        await left.store(event("A"))
        left = build(left_conn, embedder=paused)
        pending = asyncio.create_task(left.store(event("B", at=JUN, parents=["A"])))
        await asyncio.wait_for(paused.entered.wait(), timeout=5)
        await right.store(event("C", at=JUN, parents=["A"]))
        paused.release.set()
        await asyncio.wait_for(pending, timeout=5)
        state = await records(left, ["B", "C"])
        assert all(m.revision_state == "conflicted" for m in state)
        assert all(m.eligible_leaf_ids == ["B", "C"] for m in state)
    finally:
        left_conn.close()
        right_conn.close()


async def test_migration_preserves_legacy_null_root_and_known_author_without_invention(tmp_path):
    path = str(tmp_path / "legacy-revisions.db")
    conn = open_db(path)
    gate = build(conn)
    await gate.store(event("A"))
    old = dict(conn.execute("SELECT * FROM memories WHERE id='A'").fetchone())
    conn.execute("DROP INDEX idx_memories_revision_family")
    conn.execute("ALTER TABLE memories DROP COLUMN revises_event_ids")
    conn.execute("ALTER TABLE memories DROP COLUMN revision_root_id")
    conn.execute("PRAGMA user_version=9")
    conn.commit()
    conn.close()
    conn = open_db(path)
    try:
        gate = build(conn)
        migrated = dict(conn.execute("SELECT * FROM memories WHERE id='A'").fetchone())
        for key in old.keys() - {"revision_root_id", "revises_event_ids"}:
            assert migrated[key] == old[key]
        assert migrated["revision_root_id"] is None and migrated["revises_event_ids"] == "[]"
        await gate.store(event("B", at=JUN, parents=["A"]))
        state = await records(gate, ["A", "B"])
        assert [record.revision_state for record in state] == ["inactive", "active"]
        assert state[-1].revision_root_id == "A"
    finally:
        conn.close()


async def test_conflict_references_remain_visible_when_retrieval_returns_one_branch():
    conn = open_db(":memory:")
    gate = build(conn)
    try:
        await gate.store(event("A"))
        await gate.store(event("B", at=JUN, parents=["A"]))
        await gate.store(event("C", at=JUN, parents=["A"]))
        result = await gate.recall(MemoryQuery(user_id="owner", text="synthetic B", top_k=1))
        assert len(result.memories) == 1
        assert result.memories[0].revision_state == "conflicted"
        assert result.memories[0].eligible_leaf_ids == ["B", "C"]
    finally:
        conn.close()


async def test_leaf_references_are_bounded_without_misclassifying_unlisted_leaves():
    conn = open_db(":memory:")
    gate = build(conn)
    try:
        await gate.store(event("A"))
        for index in range(34):
            await gate.store(event(f"B{index:02}", at=JUN, parents=["A"]))
        state = (await records(gate, ["B33"]))[0]
        assert state.revision_state == "conflicted" and state.eligible_leaf_count == 34
        assert len(state.eligible_leaf_ids) == 32 and state.revision_truncated
        assert "B33" not in state.eligible_leaf_ids
    finally:
        conn.close()


def test_ambiguous_time_alias_and_malformed_parent_type_are_refused():
    with pytest.raises(ValidationError, match="conflicting event time aliases"):
        Memory(user_id="owner", content="synthetic", created_at=JAN, effective_at=JUN)
    with pytest.raises(ValidationError, match="malformed parent ID"):
        Memory(user_id="owner", content="synthetic", revises_event_ids=[1])


async def test_microsecond_activation_and_offset_cutoff_are_exact():
    conn = open_db(":memory:")
    gate = build(conn)
    parent_time = datetime(2026, 1, 1, 0, 0, 0, 100, tzinfo=UTC)
    child_time = datetime(2026, 1, 1, 0, 0, 0, 200, tzinfo=UTC)
    try:
        await gate.store(event("A", at=parent_time))
        await gate.store(event("B", at=child_time, parents=["A"]))
        before = await records(gate, ["A", "B"], parent_time)
        assert [record.revision_state for record in before] == ["active", "inactive"]
        same_instant = datetime.fromisoformat("2026-01-01T03:00:00.000200+03:00")
        boundary = await records(gate, ["A", "B"], same_instant)
        assert [record.revision_state for record in boundary] == ["inactive", "active"]
    finally:
        conn.close()


async def test_evidence_is_one_snapshot_when_another_connection_adds_a_fork(tmp_path, monkeypatch):
    from morgan_brain.memory.revisions import RevisionResolver
    from morgan_brain.memory.store.episodic import EpisodicStore

    path = str(tmp_path / "snapshot.db")
    left_conn, right_conn = open_db(path), open_db(path)
    left, right = build(left_conn), build(right_conn)
    original_family = RevisionResolver.family
    added = False

    def family_then_add_fork(resolver, record):
        nonlocal added
        state = original_family(resolver, record)
        if not added:
            added = True
            # Source-only fixture write on an independent connection; exact evidence
            # does not depend on indexes. The new sibling is a valid prepared event.
            sibling = event("C", at=JUN, parents=["A"]).model_copy(
                update={"revision_root_id": "A", "recorded_at": SEP}
            )
            EpisodicStore(right_conn, initialize=False).put(sibling)
        return state

    try:
        await left.store(event("A"))
        await right.store(event("B", at=JUN, parents=["A"]))
        monkeypatch.setattr(RevisionResolver, "family", family_then_add_fork)
        first = await left.evidence(
            user_id="owner", project="personal", evidence_ids=["A", "B", "C"], effective_at=SEP
        )
        assert first.missing_ids == ["C"]
        assert first.records[-1].revision_state == "active"
        assert first.records[-1].eligible_leaf_ids == ["B"]
        after = await records(left, ["B", "C"])
        assert all(record.revision_state == "conflicted" for record in after)
    finally:
        left_conn.close()
        right_conn.close()


async def test_outward_current_facts_uses_injected_clock_and_filters_revised_support():
    conn = open_db(":memory:")
    gate = build(conn, at=JAN)
    try:
        await gate.store(event("A"))
        await gate.upsert_fact(
            TemporalFact(
                id="fA",
                user_id="owner",
                subject="user",
                predicate="drink",
                object="tea",
                source=MemorySource.AGENT_INFERRED,
                support_event_ids=["A"],
            )
        )
        await gate.upsert_fact(
            TemporalFact(
                id="future",
                user_id="owner",
                subject="other",
                predicate="drink",
                object="water",
                source=MemorySource.USER_STATED,
                valid_from=SEP,
            )
        )
        current = await gate.current_facts(user_id="owner")
        assert [fact.id for fact in current] == ["fA"]
        assert current[0].support_state == "current"
        await gate.store(event("B", at=JAN, parents=["A"]))
        assert await gate.current_facts(user_id="owner") == []
        raw = await records(gate, ["fA"])
        assert raw[0].support_state == "inactive_support"
        await gate.upsert_fact(
            TemporalFact(
                id="unsupported",
                user_id="owner",
                subject="legacy",
                predicate="drink",
                object="coffee",
                source=MemorySource.AGENT_INFERRED,
            )
        )
        current = await gate.current_facts(user_id="owner")
        assert [fact.id for fact in current] == ["unsupported"]
        assert current[0].support_state == "unsupported"
        later = build(conn, at=SEP)
        current = await later.current_facts(user_id="owner")
        assert {fact.id for fact in current} == {"future", "unsupported"}
        assert next(fact for fact in current if fact.id == "future").support_state == "intrinsic"
    finally:
        conn.close()
