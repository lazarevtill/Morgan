"""Durable identity, bounded scoped reads, lineage and protected user assertions."""

from datetime import UTC, datetime

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.migrations import code_version
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemoryKind, MemoryQuery, MemorySource, Scope, TemporalFact

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def build(conn):
    return MemoryGate(
        build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4, clock=lambda: NOW)
    )


def fact(identity="fact", **changes):
    return TemporalFact.model_validate(
        {
            "id": identity,
            "user_id": "owner",
            "subject": "user",
            "predicate": "drink",
            "object": "tea",
            "source": MemorySource.USER_STATED,
            **changes,
        }
    )


async def test_fact_default_is_unknown_and_persisted_legacy_source_is_preserved(tmp_path):
    path = str(tmp_path / "fact-attribution.db")
    conn = open_db(path)
    gate = build(conn)
    unknown = TemporalFact(
        id="unknown", user_id="owner", subject="agent", predicate="drink", object="tea"
    )
    assert unknown.source is MemorySource.UNKNOWN
    await gate.upsert_fact(unknown)
    await gate.upsert_fact(fact("legacy", source=MemorySource.USER_STATED))
    conn.close()
    conn = open_db(path)
    try:
        gate = build(conn)
        result = await gate.evidence(
            user_id="owner", project="personal", evidence_ids=["unknown", "legacy"]
        )
        assert [record.source for record in result.records] == [
            MemorySource.UNKNOWN,
            MemorySource.USER_STATED,
        ]
    finally:
        conn.close()


async def test_recall_and_progressive_evidence_preserve_identity_after_restart(
    tmp_path, monkeypatch
):
    path = str(tmp_path / "evidence.db")
    conn = open_db(path)
    gate = build(conn)
    await gate.store(
        Memory(id="root", user_id="owner", content="tea", source=MemorySource.USER_STATED)
    )
    await gate.upsert_fact(
        fact(
            source=MemorySource.AGENT_INFERRED,
            support_event_ids=["root"],
            author_id="consolidator",
            scope=Scope.SHARED,
            confidence=0.8,
        )
    )
    conn.close()
    conn = open_db(path)
    gate = build(conn)
    try:
        recalled = (await gate.recall(MemoryQuery(user_id="owner", text="drink"))).memories
        semantic = next(record for record in recalled if record.kind.value == "semantic")
        assert semantic.id == "fact" and semantic.author_id == "consolidator"
        assert semantic.scope is Scope.SHARED and semantic.confidence == 0.8
        assert semantic.recorded_at == NOW and semantic.valid_from == NOW
        assert semantic.support_event_ids == ["root"]

        async def unavailable_embed(text):
            raise AssertionError("exact evidence must not embed")

        monkeypatch.setattr(gate._store._embedder, "embed", unavailable_embed)
        evidence = await gate.evidence(
            user_id="owner", project="personal", evidence_ids=[semantic.id]
        )
        assert evidence.schema_version == "morgan.evidence.v1"
        assert evidence.records == [semantic] and evidence.missing_ids == []
        roots = await gate.evidence(
            user_id="owner", project="personal", evidence_ids=evidence.records[0].support_event_ids
        )
        assert [record.id for record in roots.records] == ["root"]
    finally:
        conn.close()


async def test_evidence_is_bounded_and_explicitly_scoped(monkeypatch):
    conn = open_db(":memory:")
    gate = build(conn)
    try:
        await gate.upsert_fact(fact())
        await gate.upsert_fact(fact("other", user_id="other"))
        await gate.upsert_fact(fact("work", project="work"))
        await gate.store(Memory(id="corrupt-other", user_id="other", content="Foreign content"))
        conn.execute("UPDATE memories SET entities='malformed' WHERE id='corrupt-other'")
        conn.commit()
        foreign = await gate.evidence(
            user_id="owner", project="personal", evidence_ids=["corrupt-other"]
        )
        assert foreign.records == [] and foreign.missing_ids == ["corrupt-other"]
        recalled = await gate.recall(MemoryQuery(user_id="owner", text="drink", all_projects=True))
        assert {record.id for record in recalled.memories} == {"fact", "work"}
        result = await gate.evidence(
            user_id="owner",
            project="personal",
            evidence_ids=["fact", "other", "work", "absent", "fact"],
        )
        assert result.requested_ids == ["fact", "other", "work", "absent"]
        assert [record.id for record in result.records] == ["fact"]
        assert result.records[0].source is MemorySource.USER_STATED
        assert result.records[0].support_event_ids == []
        assert result.missing_ids == ["other", "work", "absent"]
        boundary = await gate.evidence(
            user_id="owner", project="personal", evidence_ids=[f"missing-{i}" for i in range(32)]
        )
        assert len(boundary.missing_ids) == 32
        longest_id = "x" * 256
        longest = await gate.evidence(
            user_id="owner", project="personal", evidence_ids=[longest_id]
        )
        assert longest.missing_ids == [longest_id]

        async def fail_read(**kwargs):
            raise AssertionError("invalid request reached storage")

        monkeypatch.setattr(gate._store, "evidence", fail_read)
        for identities in ([], ["id"] * 33, [""], [" "], ["x" * 257]):
            with pytest.raises(ValueError):
                await gate.evidence(user_id="owner", project="personal", evidence_ids=identities)
    finally:
        conn.close()


async def test_support_scope_type_and_current_user_protection_are_atomic():
    conn = open_db(":memory:")
    gate = build(conn)
    try:
        await gate.store(
            Memory(id="foreign", user_id="other", content="coffee", source=MemorySource.USER_STATED)
        )
        await gate.store(
            Memory(id="own", user_id="owner", content="tea", source=MemorySource.USER_STATED)
        )
        await gate.upsert_fact(fact())
        before = conn.serialize()
        for roots in (["foreign"], ["fact"], ["missing"]):
            with pytest.raises(ValueError, match="support"):
                await gate.upsert_fact(
                    fact("invalid", source=MemorySource.AGENT_INFERRED, support_event_ids=roots)
                )
            assert conn.serialize() == before
        with pytest.raises(ValueError, match="user statement"):
            await gate.upsert_fact(
                fact("inferred", source=MemorySource.AGENT_INFERRED, support_event_ids=["own"])
            )
        assert conn.serialize() == before
        await gate.close_fact("fact", user_id="owner", project="personal", now=NOW)
        await gate.upsert_fact(
            fact("inferred", source=MemorySource.AGENT_INFERRED, support_event_ids=["own"])
        )
        assert [row.id for row in await gate.current_facts(user_id="owner")] == ["inferred"]
        historical = await gate.evidence(user_id="owner", project="personal", evidence_ids=["fact"])
        assert historical.records[0].valid_to == NOW
    finally:
        conn.close()


@pytest.mark.parametrize("incoming_start", [None, datetime(2026, 1, 1, tzinfo=UTC)])
@pytest.mark.parametrize("source", [MemorySource.UNKNOWN, MemorySource.AGENT_INFERRED])
async def test_cancelled_schedule_cannot_bypass_effective_user_statement_protection(
    incoming_start, source
):
    conn = open_db(":memory:")
    gate = build(conn)
    future = datetime(2027, 1, 1, tzinfo=UTC)
    try:
        await gate.upsert_fact(fact("present", source=MemorySource.USER_STATED))
        await gate.upsert_fact(fact("scheduled", object="water", valid_from=future))
        await gate.close_fact("scheduled", user_id="owner", project="personal", now=NOW)
        before = conn.serialize()
        with pytest.raises(ValueError, match="user statement"):
            await gate.upsert_fact(fact("inferred", source=source, valid_from=incoming_start))
        assert conn.serialize() == before
        effective = (await gate.recall(MemoryQuery(user_id="owner", text="drink"))).memories
        assert [record.id for record in effective] == ["present"]
        assert effective[0].valid_to == future
    finally:
        conn.close()


@pytest.mark.parametrize("source", [MemorySource.UNKNOWN, MemorySource.AGENT_INFERRED])
@pytest.mark.parametrize("incoming_end", [NOW, datetime(2026, 1, 1, tzinfo=UTC)])
async def test_empty_or_negative_inference_cannot_close_user_predecessor(source, incoming_end):
    conn = open_db(":memory:")
    gate = build(conn)
    try:
        await gate.upsert_fact(fact("present", source=MemorySource.USER_STATED))
        await gate.upsert_fact(fact("scheduled", valid_from=datetime(2027, 1, 1, tzinfo=UTC)))
        await gate.close_fact("scheduled", user_id="owner", project="personal", now=NOW)
        before = conn.serialize()
        with pytest.raises(ValueError, match="user statement"):
            await gate.upsert_fact(fact("invalid", source=source, valid_to=incoming_end))
        assert conn.serialize() == before
        effective = (await gate.recall(MemoryQuery(user_id="owner", text="drink"))).memories
        assert [record.id for record in effective] == ["present"]
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("incoming_start", "incoming_end"),
    [
        (datetime(2026, 1, 1, tzinfo=UTC), None),
        (datetime(2027, 5, 1, tzinfo=UTC), datetime(2027, 7, 1, tzinfo=UTC)),
    ],
)
async def test_cancelled_future_head_preserves_finite_future_user_window(
    incoming_start, incoming_end
):
    conn = open_db(":memory:")
    gate = build(conn)
    first_start = datetime(2027, 6, 1, tzinfo=UTC)
    next_start = datetime(2027, 9, 1, tzinfo=UTC)
    try:
        await gate.upsert_fact(fact("future-user", valid_from=first_start))
        await gate.upsert_fact(fact("next-user", object="water", valid_from=next_start))
        await gate.close_fact("next-user", user_id="owner", project="personal", now=NOW)
        before = conn.serialize()
        with pytest.raises(ValueError, match="user statement"):
            await gate.upsert_fact(
                fact(
                    "overlap",
                    source=MemorySource.AGENT_INFERRED,
                    valid_from=incoming_start,
                    valid_to=incoming_end,
                )
            )
        assert conn.serialize() == before
    finally:
        conn.close()


async def test_explicit_past_inference_cannot_overlap_ended_user_history():
    conn = open_db(":memory:")
    gate = build(conn)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 6, 1, tzinfo=UTC)
    try:
        await gate.upsert_fact(fact("history", valid_from=start))
        await gate.close_fact("history", user_id="owner", project="personal", now=end)
        before = conn.serialize()
        with pytest.raises(ValueError, match="user statement"):
            await gate.upsert_fact(
                fact(
                    "overlap",
                    source=MemorySource.AGENT_INFERRED,
                    valid_from=datetime(2026, 5, 1, tzinfo=UTC),
                    valid_to=end,
                )
            )
        assert conn.serialize() == before
        await gate.upsert_fact(fact("boundary", source=MemorySource.AGENT_INFERRED, valid_from=end))
    finally:
        conn.close()


@pytest.mark.parametrize("source", [MemorySource.UNKNOWN, MemorySource.AGENT_INFERRED])
async def test_ended_user_history_does_not_block_new_inferred_assertion(source):
    conn = open_db(":memory:")
    gate = build(conn)
    try:
        await gate.upsert_fact(fact("historical", source=MemorySource.USER_STATED))
        await gate.close_fact("historical", user_id="owner", project="personal", now=NOW)
        await gate.upsert_fact(fact("new", source=source))
        effective = (await gate.recall(MemoryQuery(user_id="owner", text="drink"))).memories
        assert [record.id for record in effective] == ["new"]
        historical = await gate.evidence(
            user_id="owner", project="personal", evidence_ids=["historical"]
        )
        assert historical.records[0].valid_to == NOW
    finally:
        conn.close()


async def test_version_eight_upgrade_preserves_legacy_facts_without_inventing_lineage():
    conn = open_db(":memory:")
    gate = build(conn)
    try:
        await gate.upsert_fact(fact())
        conn.execute("ALTER TABLE facts DROP COLUMN recorded_at")
        conn.execute("ALTER TABLE facts DROP COLUMN support_event_ids")
        conn.execute("PRAGMA user_version=8")
        conn.commit()
        reopened = build(conn)
        result = await reopened.evidence(user_id="owner", project="personal", evidence_ids=["fact"])
        assert result.records[0].recorded_at is None
        assert result.records[0].support_event_ids == []
        assert result.records[0].id == "fact"
        assert conn.execute("PRAGMA user_version").fetchone()[0] == code_version()
    finally:
        conn.close()


async def test_evidence_and_recall_preserve_literal_subject_and_object_identifiers():
    conn = open_db(":memory:")
    gate = build(conn)
    try:
        await gate.upsert_fact(
            fact(subject="alex_work@example.com", predicate="uses_account", object="account_foo_v2")
        )
        result = await gate.evidence(user_id="owner", project="personal", evidence_ids=["fact"])
        recalled = await gate.recall(MemoryQuery(user_id="owner", text="account"))
        expected = "alex_work@example.com uses account account_foo_v2"
        assert result.records[0].content == expected
        assert recalled.memories[0].content == expected
        persisted = (await gate.current_facts(user_id="owner"))[0]
        assert persisted.subject == "alex_work@example.com"
        assert persisted.object == "account_foo_v2"
    finally:
        conn.close()


@pytest.mark.parametrize(
    "root_fields",
    [
        {"source": "unknown"},
        {"source": "agent_inferred"},
        {"kind": "semantic"},
        {"status": "quarantined"},
        {"project": "work"},
    ],
)
async def test_support_roots_require_stored_source_events_in_exact_context(root_fields):
    conn = open_db(":memory:")
    gate = build(conn)
    try:
        root = Memory.model_validate(
            {
                "id": "root",
                "user_id": "owner",
                "content": "Synthetic source",
                "source": "user_stated",
                **root_fields,
            }
        )
        await gate.store(root)
        before = conn.serialize()
        with pytest.raises(ValueError, match="scoped source events"):
            await gate.upsert_fact(
                fact(source=MemorySource.AGENT_INFERRED, support_event_ids=["root"])
            )
        assert conn.serialize() == before
    finally:
        conn.close()


async def test_tool_observed_source_root_can_support_inference():
    conn = open_db(":memory:")
    gate = build(conn)
    try:
        await gate.store(
            Memory(
                id="tool",
                user_id="owner",
                content="Synthetic observation",
                source=MemorySource.TOOL_OBSERVED,
            )
        )
        await gate.upsert_fact(fact(source=MemorySource.AGENT_INFERRED, support_event_ids=["tool"]))
        result = await gate.evidence(user_id="owner", project="personal", evidence_ids=["fact"])
        assert result.records[0].support_event_ids == ["tool"]
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("foreign_field", "foreign_value"),
    [(None, None), ("user_id", "other"), ("project", "work")],
)
async def test_legacy_duplicate_identity_fails_closed_only_inside_requested_scope(
    foreign_field, foreign_value
):
    conn = open_db(":memory:")
    gate = build(conn)
    try:
        await gate.store(Memory(id="event", user_id="owner", content="Scoped event"))
        await gate.upsert_fact(fact())
        # Before the cross-table identity guard, legacy writers could mint the same ID.
        conn.execute("UPDATE facts SET id='event' WHERE id='fact'")
        if foreign_field == "user_id":
            conn.execute("UPDATE facts SET user_id=? WHERE id='event'", (foreign_value,))
        elif foreign_field == "project":
            conn.execute("UPDATE facts SET project=? WHERE id='event'", (foreign_value,))
        conn.commit()
        if foreign_field is None:
            with pytest.raises(ValueError, match="Ambiguous evidence identity"):
                await gate.evidence(user_id="owner", project="personal", evidence_ids=["event"])
        else:
            result = await gate.evidence(
                user_id="owner", project="personal", evidence_ids=["event"]
            )
            assert len(result.records) == 1
            assert result.records[0].content == "Scoped event"
            assert result.records[0].kind.value == "episodic"
            assert result.missing_ids == []
    finally:
        conn.close()


@pytest.mark.parametrize("event_first", [False, True])
@pytest.mark.parametrize(
    "other_scope", [("owner", "personal"), ("other", "personal"), ("owner", "work")]
)
async def test_cross_table_ids_are_ambiguous_only_in_exact_owner_and_context(
    event_first, other_scope
):
    conn = open_db(":memory:")
    gate = build(conn)
    event = Memory(id="shared", user_id="owner", project="personal", content="Synthetic event")
    assertion = fact("shared", user_id=other_scope[0], project=other_scope[1])
    try:
        first, second = (event, assertion) if event_first else (assertion, event)

        async def store(record):
            if isinstance(record, Memory):
                return await gate.store(record)
            return await gate.upsert_fact(record)

        await store(first)
        if other_scope == ("owner", "personal"):
            before = conn.serialize()
            with pytest.raises(ValueError, match="ID is already used"):
                await store(second)
            assert conn.serialize() == before
        else:
            await store(second)
            owner_result = await gate.evidence(
                user_id="owner", project="personal", evidence_ids=["shared"]
            )
            foreign_result = await gate.evidence(
                user_id=other_scope[0], project=other_scope[1], evidence_ids=["shared"]
            )
            assert owner_result.records[0].kind is MemoryKind.EPISODIC
            assert foreign_result.records[0].kind is MemoryKind.SEMANTIC
    finally:
        conn.close()
