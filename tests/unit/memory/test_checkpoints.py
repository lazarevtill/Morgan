"""Optional checkpoints retain source integrity and refuse stale writers."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.checkpoints import (
    AmbiguousCheckpoint,
    Checkpoint,
    CheckpointContext,
    CheckpointItem,
    ReportedProgress,
    StaleCheckpoint,
)
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.knowledge.basis import UnsupportedConsolidationOperation
from morgan_brain.memory.knowledge.consolidation import MemoryConsolidator
from morgan_brain.memory.knowledge.fact_ops import FactOp, FactOpBatch
from morgan_brain.memory.revisions import RevisionError
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemorySource, Scope, TemporalFact
from tests.fakes import FakeChatClient

JAN = datetime(2026, 1, 1, tzinfo=UTC)
JUN = datetime(2026, 6, 1, tzinfo=UTC)


def state(**fields):
    return Checkpoint(kind="goal", title="Read independently", objective="Practise daily", **fields)


def build(conn, clock=lambda: JAN):
    return MemoryGate(build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4, clock=clock))


async def source(gate, identity="A", **fields):
    return await gate.store(
        Memory(
            id=identity,
            user_id="owner",
            content="synthetic reading plan",
            source=MemorySource.USER_STATED,
            author_id="owner",
            effective_at=JAN,
            **fields,
        )
    )


async def test_create_update_stale_writer_and_scope(tmp_path):
    conn = open_db(str(tmp_path / "checkpoint.db"))
    other = open_db(str(tmp_path / "checkpoint.db"))
    try:
        gate, writer = build(conn), build(other)
        await source(gate)
        first = await gate.put_checkpoint(
            state(),
            checkpoint_id="reading",
            context=CheckpointContext(user_id="owner"),
            support_event_ids=["A"],
        )
        result = await gate.get_checkpoint("reading", user_id="owner")
        assert result.fact.id == first and result.eligibility == "current"
        assert result.fact.source is MemorySource.AGENT_INFERRED
        assert result.fact.project == "personal" and result.state.kind == "goal"
        assert await gate.get_checkpoint("reading", user_id="other") is None
        with pytest.raises(StaleCheckpoint):
            await writer.put_checkpoint(
                state(),
                checkpoint_id="reading",
                context=CheckpointContext(user_id="owner"),
                support_event_ids=["A"],
            )
        second = await writer.put_checkpoint(
            state(status="paused"),
            checkpoint_id="reading",
            context=CheckpointContext(user_id="owner"),
            support_event_ids=["A"],
            expected_fact_id=first,
        )
        before = conn.serialize()
        with pytest.raises(StaleCheckpoint):
            await gate.put_checkpoint(
                state(status="completed"),
                checkpoint_id="reading",
                context=CheckpointContext(user_id="owner"),
                support_event_ids=["A"],
                expected_fact_id=first,
            )
        assert conn.serialize() == before
        historical = await gate.evidence(
            user_id="owner", project="personal", evidence_ids=[first, second]
        )
        assert len(historical.records) == 2
    finally:
        other.close()
        conn.close()


async def test_revision_future_activation_rebuild_and_invalid_support(tmp_path):
    conn = open_db(str(tmp_path / "basis.db"))
    clock = [JAN]
    gate = build(conn, lambda: clock[0])
    try:
        await source(gate)
        first = await gate.put_checkpoint(
            state(),
            checkpoint_id="reading",
            context=CheckpointContext(user_id="owner"),
            support_event_ids=["A"],
        )
        await gate.store(
            Memory(
                id="B",
                user_id="owner",
                content="new plan",
                source=MemorySource.USER_STATED,
                author_id="owner",
                effective_at=JUN,
                revises_event_ids=["A"],
            )
        )
        assert (await gate.get_checkpoint("reading", user_id="owner")).eligibility == "current"
        clock[0] = JUN
        result = await gate.get_checkpoint("reading", user_id="owner")
        assert result.eligibility == "needs_rebuild" and result.state is None
        with pytest.raises(RevisionError):
            await gate.put_checkpoint(
                state(),
                checkpoint_id="reading",
                context=CheckpointContext(user_id="owner"),
                support_event_ids=["A"],
                expected_fact_id=first,
            )
        for support in (["missing"], ["agent"]):
            if support == ["agent"]:
                await gate.store(
                    Memory(
                        id="agent",
                        user_id="owner",
                        content="reported completion",
                        source=MemorySource.AGENT_INFERRED,
                    )
                )
            with pytest.raises(ValueError):
                await gate.put_checkpoint(
                    state(),
                    checkpoint_id="other",
                    context=CheckpointContext(user_id="owner"),
                    support_event_ids=support,
                )
        new = await gate.put_checkpoint(
            state(),
            checkpoint_id="reading",
            context=CheckpointContext(user_id="owner"),
            support_event_ids=["B"],
            expected_fact_id=first,
        )
        assert (await gate.get_checkpoint("reading", user_id="owner")).fact.id == new
    finally:
        conn.close()


@pytest.mark.parametrize(
    "fields",
    [
        {"version": "morgan.checkpoint.v2"},
        {"title": "x" * 121},
        {"objective": " "},
        {"applies_to": ["a"] * 9},
        {"permission": "book"},
        {"progress": [{"text": "done", "verification": "user_confirmed"}]},
    ],
)
def test_bounded_codec_refuses_unknown_fields_and_versions(fields):
    with pytest.raises(ValidationError):
        Checkpoint.model_validate(
            {"kind": "task", "title": "Task", "objective": "Resume", **fields}
        )


def test_item_lineage_and_nested_mutation_revalidated():
    item = CheckpointItem(text="Check constraint", evidence_ids=["A"])
    checkpoint = state(next_steps=[item])
    with pytest.raises(ValueError, match="item evidence"):
        checkpoint.encode([])
    assert "A" in checkpoint.encode(["A"])
    checkpoint.next_steps.extend([item] * 4)
    with pytest.raises(ValidationError):
        checkpoint.encode(["A"])


def test_total_encoding_bound_and_unverified_reference_separation():
    ids = ["x" * 253 + str(i).zfill(3) for i in range(16)]
    progress = ReportedProgress(text="Reported work", evidence_ids=ids)
    checkpoint = state(
        progress=[progress] * 4,
        next_steps=[CheckpointItem(text="Propose", evidence_ids=ids)] * 4,
        open_questions=[CheckpointItem(text="Ask", evidence_ids=ids)] * 4,
    )
    with pytest.raises(ValueError, match="byte encoding bound"):
        checkpoint.encode(ids)
    report = state(progress=[ReportedProgress(text="Reported completion", reference_ids=["AGENT"])])
    assert "AGENT" in report.encode([])
    assert report.progress[0].verification == "unverified_agent_report"


async def test_reopen_serialized_snapshot_and_forget(tmp_path):
    path = str(tmp_path / "original.db")
    conn = open_db(path)
    gate = build(conn)
    await source(gate)
    identity = await gate.put_checkpoint(
        state(),
        checkpoint_id="resume",
        context=CheckpointContext(user_id="owner"),
        support_event_ids=["A"],
    )
    conn.execute("VACUUM INTO ?", (str(tmp_path / "copy.db"),))
    conn.close()
    copy = open_db(str(tmp_path / "copy.db"))
    try:
        restored = build(copy)
        result = await restored.get_checkpoint("resume", user_id="owner")
        assert result.fact.id == identity and result.eligibility == "current"
        assert await restored.get_checkpoint("resume", user_id="owner", project="other") is None
        await restored.forget(user_id="owner", project="personal")
        assert await restored.get_checkpoint("resume", user_id="owner") is None
        with pytest.raises(ValueError, match="scoped source"):
            await restored.put_checkpoint(
                state(),
                checkpoint_id="resume",
                context=CheckpointContext(user_id="owner"),
                support_event_ids=["A"],
            )
    finally:
        copy.close()


async def test_unsupported_legacy_unknown_schema_and_protected_user_fact(tmp_path):
    conn = open_db(str(tmp_path / "legacy.db"))
    gate = build(conn)
    try:
        await gate.put_checkpoint(
            state(),
            checkpoint_id="empty",
            context=CheckpointContext(user_id="owner"),
            support_event_ids=[],
        )
        assert (await gate.get_checkpoint("empty", user_id="owner")).eligibility == "unsupported"
        raw = '{"version":"morgan.checkpoint.v9"}'
        identity = await gate.upsert_fact(
            TemporalFact(
                user_id="owner",
                subject="checkpoint:future",
                predicate="resumable_state_v1",
                object=raw,
            )
        )
        result = await gate.get_checkpoint("future", user_id="owner")
        assert result.eligibility == "unsupported_version" and result.state is None
        assert result.fact.object == raw
        user = await gate.upsert_fact(
            TemporalFact(
                user_id="owner",
                subject="checkpoint:protected",
                predicate="resumable_state_v1",
                object=state().encode([]),
                source=MemorySource.USER_STATED,
            )
        )
        with pytest.raises(ValueError, match="user statement"):
            await gate.put_checkpoint(
                state(),
                checkpoint_id="protected",
                context=CheckpointContext(user_id="owner"),
                support_event_ids=[],
                expected_fact_id=user,
            )
        assert identity in [
            r.id
            for r in (
                await gate.evidence(user_id="owner", project="personal", evidence_ids=[identity])
            ).records
        ]
    finally:
        conn.close()


@pytest.mark.parametrize(
    "raw",
    ["{", "[]", "[" * 2000 + "0" + "]" * 2000, "x" * 32769],
    ids=["malformed", "wrong-shape", "deep-nesting", "oversized"],
)
async def test_malformed_legacy_state_refuses_without_rewrite(tmp_path, raw):
    conn = open_db(str(tmp_path / "invalid.db"))
    gate = build(conn)
    try:
        await gate.upsert_fact(
            TemporalFact(
                user_id="owner",
                subject="checkpoint:invalid",
                predicate="resumable_state_v1",
                object=raw,
            )
        )
        result = await gate.get_checkpoint("invalid", user_id="owner")
        assert result.eligibility == "invalid_state" and result.state is None
        assert result.fact.object == raw
    finally:
        conn.close()


async def test_cancelled_schedule_retains_predecessor_cas(tmp_path):
    conn = open_db(str(tmp_path / "cancelled.db"))
    gate = build(conn)
    try:
        await source(gate)
        predecessor = await gate.put_checkpoint(
            state(),
            checkpoint_id="scheduled",
            context=CheckpointContext(user_id="owner"),
            support_event_ids=["A"],
        )
        scheduled = await gate.upsert_fact(
            TemporalFact(
                user_id="owner",
                subject="checkpoint:scheduled",
                predicate="resumable_state_v1",
                object=state(status="paused").encode(["A"]),
                source=MemorySource.AGENT_INFERRED,
                support_event_ids=["A"],
                valid_from=JUN,
            )
        )
        await gate.close_fact(scheduled, user_id="owner", project="personal", now=JAN)
        assert (await gate.get_checkpoint("scheduled", user_id="owner")).fact.id == predecessor
        before = conn.serialize()
        with pytest.raises(StaleCheckpoint):
            await gate.put_checkpoint(
                state(),
                checkpoint_id="scheduled",
                context=CheckpointContext(user_id="owner"),
                support_event_ids=["A"],
            )
        assert conn.serialize() == before
        updated = await gate.put_checkpoint(
            state(status="blocked"),
            checkpoint_id="scheduled",
            context=CheckpointContext(user_id="owner"),
            support_event_ids=["A"],
            expected_fact_id=predecessor,
        )
        assert (await gate.get_checkpoint("scheduled", user_id="owner")).fact.id == updated
    finally:
        conn.close()


async def test_overlapping_imported_finite_heads_refuse_resumption(tmp_path):
    conn = open_db(str(tmp_path / "overlapping.db"))
    gate = build(conn)
    try:
        for identity, start in [
            ("older", datetime(2025, 11, 1, tzinfo=UTC)),
            ("newer", datetime(2025, 12, 1, tzinfo=UTC)),
        ]:
            await gate.upsert_fact(
                TemporalFact(
                    id=identity,
                    user_id="owner",
                    subject="checkpoint:legacy",
                    predicate="resumable_state_v1",
                    object=state().encode([]),
                    valid_from=start,
                    valid_to=JUN,
                )
            )
        before = conn.serialize()
        with pytest.raises(
            AmbiguousCheckpoint, match="multiple effective checkpoint facts"
        ) as error:
            await gate.get_checkpoint("legacy", user_id="owner")
        assert error.value.fact_ids == ("newer", "older")
        assert conn.serialize() == before
    finally:
        conn.close()


@pytest.mark.parametrize("fields", [{"unknown": True}, {"user_id": 1}, {"scope": "invalid"}])
def test_checkpoint_context_rejects_invalid_fields(fields):
    with pytest.raises(ValidationError):
        CheckpointContext.model_validate({"user_id": "owner", **fields})


def test_checkpoint_context_is_immutable_with_personal_defaults():
    context = CheckpointContext(user_id="owner")
    assert context.project == "personal" and context.author_id == ""
    with pytest.raises(ValidationError):
        context.user_id = "other"


@pytest.mark.parametrize("operation", ["ADD", "UPDATE", "DELETE"])
async def test_consolidation_cannot_bypass_checkpoint_cas(tmp_path, operation):

    conn = open_db(str(tmp_path / "consolidation.db"))
    gate = build(conn)
    try:
        await source(gate)
        identity = await gate.put_checkpoint(
            state(),
            checkpoint_id="reading",
            context=CheckpointContext(user_id="owner"),
            support_event_ids=["A"],
        )
        prepared = await gate.capture_consolidation_basis(
            user_id="owner",
            project="personal",
            event_ids=["A"],
            generation=gate.capture_erasure_generation(),
        )
        batch = FactOpBatch(
            ops=[
                FactOp(
                    op="ADD",
                    subject="ordinary",
                    predicate="plan",
                    object="new",
                    support_event_ids=["A"],
                ),
                FactOp(
                    op=operation,
                    subject="checkpoint:reading",
                    predicate="resumable_state_v1",
                    object="destroyed json",
                    support_event_ids=["A"],
                ),
            ]
        )
        worker = MemoryConsolidator(
            gate=gate, client=FakeChatClient(reply='{"ops":[]}'), model="fake", clock=lambda: JAN
        )
        before = conn.serialize()
        with pytest.raises(UnsupportedConsolidationOperation, match="checkpoint"):
            await worker.apply("owner", batch, project="personal", basis=prepared.basis)
        assert conn.serialize() == before
        assert (await gate.get_checkpoint("reading", user_id="owner")).fact.id == identity
    finally:
        conn.close()


async def test_checkpoint_json_is_not_consolidation_model_input(tmp_path):

    conn = open_db(str(tmp_path / "prompt.db"))
    gate = build(conn)
    try:
        await source(gate)
        await gate.put_checkpoint(
            state(),
            checkpoint_id="reading",
            context=CheckpointContext(user_id="owner"),
            support_event_ids=["A"],
        )
        facts = await gate.current_facts(user_id="owner")
        client = FakeChatClient(reply='{"ops":[]}')
        worker = MemoryConsolidator(gate=gate, client=client, model="fake", clock=lambda: JAN)
        await worker.propose("owner", [], facts)
        assert "checkpoint:reading" not in client.last_messages[1].content
        assert "resumable_state_v1" not in client.last_messages[1].content
        assert "Practise daily" not in client.last_messages[1].content
        await worker.consolidate("owner", project="personal")
        assert client.calls == 2
        assert "checkpoint:reading" not in client.last_messages[1].content
    finally:
        conn.close()


async def test_checkpoint_context_routes_provenance_without_changing_subject(tmp_path):
    conn = open_db(str(tmp_path / "context.db"))
    gate = build(conn)
    try:
        await source(gate, project="optional-project")
        context = CheckpointContext(
            user_id="owner",
            project="optional-project",
            author_id="reported-agent",
            scope=Scope.PRIVATE,
        )
        await gate.put_checkpoint(
            state(subject_entity_id="person:reader", applies_to=["agent:reader"]),
            checkpoint_id="reading",
            context=context,
            support_event_ids=["A"],
        )
        assert await gate.get_checkpoint("reading", user_id="owner") is None
        result = await gate.get_checkpoint("reading", user_id="owner", project="optional-project")
        assert result.fact.author_id == "reported-agent" and result.fact.scope is Scope.PRIVATE
        assert result.fact.user_id == "owner" and result.fact.project == "optional-project"
        assert result.state.subject_entity_id == "person:reader"
        assert result.state.applies_to == ["agent:reader"]
    finally:
        conn.close()
