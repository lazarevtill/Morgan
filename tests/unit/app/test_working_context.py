"""Real scoped storage, bounded model selection and atomic lifecycle checks; no endpoints."""

import json
from contextlib import suppress
from datetime import UTC, datetime, timedelta

import pytest

from morgan_brain.app.working_context import WorkingContextService
from morgan_brain.composition import build_memory_module
from morgan_brain.memory.checkpoints import CheckpointContext, StaleCheckpoint
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.errors import EvidenceChanged
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.knowledge.consolidation import MemoryConsolidator
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.erasure import StoreInterruptedByForget
from morgan_brain.memory.working_context import WorkingContextDraft
from morgan_brain.models import Memory, MemoryQuery, MemorySource, TemporalFact
from morgan_brain.providers.wire import ChatResult

NOW = datetime(2026, 10, 1, tzinfo=UTC)


class Client:
    def __init__(self, draft):
        self.draft = draft
        self.calls = []

    async def agenerate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return ChatResult(text=json.dumps(self.draft, ensure_ascii=False), model="fake")


def draft(identity="decision", quote="Choose ceramic"):
    return {
        "title": "Unverified workspace",
        "decisions": [
            {
                "choice": {"event_id": identity, "quote": quote},
                "reason": None,
            }
        ],
    }


def stack(path=":memory:", clock=lambda: NOW, selection=None):
    conn = open_db(str(path))
    gate = MemoryGate(build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4, clock=clock))
    client = Client(selection or draft())
    service = WorkingContextService(gate=gate, client=client, model="fake")
    return conn, gate, client, service


async def event(gate, identity="decision", text="Choose ceramic because it lasts", **kwargs):
    await gate.store(
        Memory(
            id=identity,
            user_id="owner",
            project="personal",
            content=text,
            source=MemorySource.USER_STATED,
            author_id="person",
            created_at=NOW,
            **kwargs,
        )
    )


async def test_personal_continuation_keeps_exact_reasons_roles_and_bounded_old_view(tmp_path):
    selection = draft()
    selection["decisions"][0]["reason"] = {"event_id": "decision", "quote": "because it lasts"}
    selection["progress"] = [{"event_id": "progress", "quote": "I tested the sample"}]
    conn, gate, client, service = stack(tmp_path / "context.db", selection=selection)
    context = CheckpointContext(user_id="owner", author_id="model:fake")
    try:
        await event(gate, text="Choose ceramic because it lasts. EXTRANEOUS_OLD_BODY")
        await gate.store(
            Memory(
                id="progress",
                user_id="owner",
                project="personal",
                content="I tested the sample",
                source="agent_inferred",
                author_id="agent:test",
                created_at=NOW,
            )
        )
        preview = await service.preview(
            "ceramic", context=context, event_ids=["decision", "progress"]
        )
        assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
        head = await service.apply(preview)
        view = await gate.get_working_context("ceramic", user_id="owner")
        assert view.fact_id == head and view.eligibility == "current"
        assert view.state.classification == "unverified_model_selection"
        assert view.state.progress_verification == "unverified_report"
        assert view.sources[-1].source is MemorySource.AGENT_INFERRED
        assert view.sources[-1].author_id == "agent:test"
        sources = await gate.current_facts(user_id="owner")
        assert sources[0].support_event_ids == ["decision"]
        await event(gate, "question", "Which glaze should be tested?")
        client.draft["open_questions"] = [
            {"event_id": "question", "quote": "Which glaze should be tested?"}
        ]
        update = await service.preview("ceramic", context=context, event_ids=["question"])
        assert update.expected_fact_id == head
        payload = json.loads(client.calls[-1][0][-1].content)
        assert payload["old_view"]["sources"][-1]["source"] == "agent_inferred"
        assert [record["id"] for record in payload["sources"]] == ["question"]
        assert "EXTRANEOUS_OLD_BODY" not in client.calls[-1][0][-1].content
        await service.apply(update)
        before = conn.total_changes
        await gate.get_working_context("ceramic", user_id="owner")
        assert conn.total_changes == before
    finally:
        conn.close()
    conn, gate, _, _ = stack(tmp_path / "context.db")
    try:
        resumed = await gate.get_working_context("ceramic", user_id="owner")
        assert resumed.eligibility == "current" and len(resumed.state.open_questions) == 1
        assert await gate.get_working_context("ceramic", user_id="foreign") is None
        assert await gate.get_working_context("ceramic", user_id="owner", project="other") is None
        recalled = await gate.recall(MemoryQuery(user_id="owner", text="ceramic"))
        assert resumed.fact_id not in {record.id for record in recalled.memories}
    finally:
        conn.close()


@pytest.mark.parametrize("rebuild", [False, True])
@pytest.mark.parametrize("predecessor", [False, True], ids=["future-only", "replacement"])
async def test_pending_working_context_head_refuses_before_model(tmp_path, rebuild, predecessor):
    conn, gate, client, service = stack(tmp_path / "pending-head.db")
    context = CheckpointContext(user_id="owner")
    try:
        await event(gate)
        initial = await service.preview("ctx", context=context, event_ids=["decision"])
        if predecessor:
            await service.apply(initial)
            original = (await gate.current_facts(user_id="owner"))[0]
        else:
            from morgan_brain.memory.working_context import working_fact

            original = working_fact(initial)
        pending = original.model_copy(
            update={
                "id": "pending-head",
                "object": original.object.replace("Unverified workspace", "Scheduled workspace"),
                "valid_from": NOW + timedelta(seconds=2),
            }
        )
        await gate.upsert_fact(pending)
        view = await gate.get_working_context("ctx", user_id="owner")
        assert (view.fact_id if view else None) == (original.id if predecessor else None)
        calls_before = len(client.calls)
        before = conn.serialize()
        preview = None
        with suppress(StaleCheckpoint):
            preview = await service.preview(
                "ctx", context=context, event_ids=["decision"], rebuild=rebuild
            )
        if preview is not None:
            # Baseline generates against the effective predecessor/None, then structural CAS fails.
            with pytest.raises(StaleCheckpoint):
                await service.apply(preview)
        assert conn.serialize() == before
        assert len(client.calls) == calls_before
        assert preview is None
        gate._store._clock = lambda: NOW + timedelta(seconds=2)
        effective = await service.preview(
            "ctx", context=context, event_ids=["decision"], rebuild=rebuild
        )
        assert effective.expected_fact_id == pending.id
        await service.apply(effective)
    finally:
        conn.close()


@pytest.mark.parametrize("unrelated", ["subject", "project", "owner", "predicate"])
async def test_unrelated_future_head_does_not_block_working_context_preview(unrelated):
    conn, gate, _, service = stack()
    context = CheckpointContext(user_id="owner")
    try:
        await event(gate)
        preview = await service.preview("ctx", context=context, event_ids=["decision"])
        identity = await service.apply(preview)
        fields = {
            "user_id": "owner",
            "project": "personal",
            "subject": "working_checkpoint:ctx",
            "predicate": "working_context_v1",
        }
        fields[
            {
                "subject": "subject",
                "project": "project",
                "owner": "user_id",
                "predicate": "predicate",
            }[unrelated]
        ] = "unrelated"
        await gate.upsert_fact(
            TemporalFact(
                **fields, object="Future unrelated fact", valid_from=NOW + timedelta(seconds=2)
            )
        )
        update = await service.preview("ctx", context=context, event_ids=["decision"])
        assert update.expected_fact_id == identity
        await service.apply(update)
    finally:
        conn.close()


async def test_cancelled_future_head_allows_effective_predecessor_preview():
    conn, gate, _, service = stack()
    context = CheckpointContext(user_id="owner")
    try:
        await event(gate)
        first = await service.preview("ctx", context=context, event_ids=["decision"])
        identity = await service.apply(first)
        old = (await gate.current_facts(user_id="owner"))[0]
        pending = old.model_copy(
            update={
                "id": "cancelled-head",
                "object": old.object.replace("Unverified workspace", "Scheduled workspace"),
                "valid_from": NOW + timedelta(seconds=2),
            }
        )
        await gate.upsert_fact(pending)
        await gate.close_fact(pending.id, user_id="owner", project="personal", now=NOW)
        update = await service.preview("ctx", context=context, event_ids=["decision"])
        assert update.expected_fact_id == identity
        await service.apply(update)
    finally:
        conn.close()


async def test_unicode_quotes_get_server_offsets_and_ambiguous_quote_refuses():
    conn, gate, client, service = stack(selection=draft("decision", "Выбрать керамику"))
    try:
        await event(gate, text="Решение: Выбрать керамику, потому что она прочная.")
        preview = await service.preview(
            "ru", context=CheckpointContext(user_id="owner"), event_ids=["decision"]
        )
        span = preview.state.decisions[0].choice
        assert span.start == len("Решение: ")
        assert span.end == span.start + len("Выбрать керамику")
        await event(gate, "repeated", "да да")
        client.draft = draft("repeated", "да")
        with pytest.raises(ValueError, match="unique exact"):
            await service.preview(
                "repeat", context=CheckpointContext(user_id="owner"), event_ids=["repeated"]
            )
        assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    finally:
        conn.close()


@pytest.mark.parametrize("change", ["fork", "forget", "cas"])
async def test_proposal_commit_is_atomic_against_fork_forget_and_competing_head(change):
    conn, gate, _, service = stack()
    try:
        await event(gate)
        preview = await service.preview(
            "context", context=CheckpointContext(user_id="owner"), event_ids=["decision"]
        )
        if change == "fork":
            await event(gate, "left", "Choose steel", revises_event_ids=["decision"])
            await event(gate, "right", "Choose wood", revises_event_ids=["decision"])
            error = EvidenceChanged
        elif change == "forget":
            await gate.forget(user_id="owner", project="personal")
            error = StoreInterruptedByForget
        else:
            await service.apply(preview)
            error = StaleCheckpoint
        with pytest.raises(error):
            await service.apply(preview)
        assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == (
            1 if change == "cas" else 0
        )
    finally:
        conn.close()


async def test_decision_revision_hides_old_view_then_explicit_rebuild_replaces_head():
    conn, gate, client, service = stack()
    context = CheckpointContext(user_id="owner")
    try:
        await event(gate)
        original = await service.apply(
            await service.preview("context", context=context, event_ids=["decision"])
        )
        await event(gate, "updated", "Choose steel", revises_event_ids=["decision"])
        view = await gate.get_working_context("context", user_id="owner")
        assert view.eligibility == "needs_rebuild" and view.state is None and view.sources == []
        listed = await gate.list_working_contexts(user_id="owner")
        assert [item.context_id for item in listed.items] == ["context"]
        assert listed.items[0].view.eligibility == "needs_rebuild"
        assert listed.items[0].view.state is None and listed.items[0].view.sources == []
        assert not listed.truncated
        assert (await gate.list_working_contexts(user_id="foreign")).items == []
        assert (await gate.list_working_contexts(user_id="owner", project="other")).items == []
        with pytest.raises(ValueError, match="rebuilding"):
            await service.preview("context", context=context, event_ids=["updated"])
        client.draft = draft("updated", "Choose steel")
        rebuilt = await service.preview(
            "context", context=context, event_ids=["updated"], rebuild=True
        )
        assert rebuilt.expected_fact_id == original
        assert "Choose ceramic" not in client.calls[-1][0][-1].content
        new_head = await service.apply(rebuilt)
        assert new_head != original
        assert (await gate.get_working_context("context", user_id="owner")).eligibility == "current"
    finally:
        conn.close()


@pytest.mark.parametrize("invalid", ["unknown_id", "invented_quote", "empty"])
async def test_model_proposal_cannot_invent_sources_or_quotes_or_empty_state(invalid):
    selected = draft(
        "absent" if invalid == "unknown_id" else "decision",
        "Invented meaning" if invalid == "invented_quote" else "Choose ceramic",
    )
    if invalid == "empty":
        selected = {"title": "No sources"}
    conn, gate, _, service = stack(selection=selected)
    try:
        await event(gate)
        with pytest.raises(ValueError):
            await service.preview(
                "context", context=CheckpointContext(user_id="owner"), event_ids=["decision"]
            )
        assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    finally:
        conn.close()


async def test_future_and_foreign_sources_refuse_before_model_and_agent_revision_hides_view():
    conn, gate, client, service = stack(selection=draft("agent", "I finished the draft"))
    try:
        await gate.store(
            Memory(
                id="agent",
                user_id="owner",
                content="I finished the draft",
                source="agent_inferred",
                author_id="agent",
                created_at=NOW,
            )
        )
        context = CheckpointContext(user_id="owner")
        await service.apply(await service.preview("ctx", context=context, event_ids=["agent"]))
        await gate.store(
            Memory(
                id="agent-revised",
                user_id="owner",
                content="Draft needs work",
                source="agent_inferred",
                author_id="agent",
                created_at=NOW,
                revises_event_ids=["agent"],
            )
        )
        view = await gate.get_working_context("ctx", user_id="owner")
        assert view.eligibility == "needs_rebuild" and view.state is None
        await gate.store(
            Memory(
                id="future",
                user_id="owner",
                content="Future assertion",
                created_at=NOW + timedelta(days=1),
            )
        )
        count = len(client.calls)
        for ids, owner in [(["future"], "owner"), (["agent"], "foreign")]:
            with pytest.raises(ValueError):
                await service.preview(
                    "fresh", context=CheckpointContext(user_id=owner), event_ids=ids
                )
        assert len(client.calls) == count
    finally:
        conn.close()


async def test_working_selection_is_not_factual_recall_or_consolidation_input():
    conn, gate, client, service = stack()
    try:
        await event(gate)
        client.draft["title"] = "VIEW_ONLY_MARKER"
        preview = await service.preview(
            "ctx", context=CheckpointContext(user_id="owner"), event_ids=["decision"]
        )
        identity = await service.apply(preview)
        recalled = await gate.recall(MemoryQuery(user_id="owner", text="ceramic"))
        assert identity not in [record.id for record in recalled.memories]
        client.draft = {"ops": []}
        consolidator = MemoryConsolidator(gate=gate, client=client, model="fake", clock=lambda: NOW)
        assert await consolidator.consolidate("owner", project="personal") == []
        assert "VIEW_ONLY_MARKER" not in "\n".join(m.content for m in client.calls[-1][0])
        assert len(client.calls) == 2
    finally:
        conn.close()


async def test_replaceable_proposer_receives_actual_bounded_prompt_without_model_client():
    conn, gate, client, _ = stack()
    seen = []

    async def proposer(messages):
        seen.extend(messages)
        return WorkingContextDraft.model_validate(draft())

    service = WorkingContextService(gate=gate, client=client, model="unused", proposer=proposer)
    try:
        await event(gate)
        await service.preview(
            "ctx", context=CheckpointContext(user_id="owner"), event_ids=["decision"]
        )
        assert client.calls == [] and len(seen) == 2
        assert "untrusted data" in seen[0].content
    finally:
        conn.close()


async def test_twenty_new_sources_and_sixteen_old_ids_use_only_two_bounded_reads():
    conn, gate, client, service = stack()
    try:
        for index in range(36):
            await event(gate, f"source-{index}", f"Exact source {index}")
        client.draft = {
            "title": "Sixteen source view",
            "decisions": [
                {
                    "choice": {"event_id": f"source-{i}", "quote": f"Exact source {i}"},
                    "reason": None,
                }
                for i in range(4)
            ],
            "open_questions": [
                {"event_id": f"source-{i}", "quote": f"Exact source {i}"} for i in range(4, 8)
            ],
            "intentions": [
                {"event_id": f"source-{i}", "quote": f"Exact source {i}"} for i in range(8, 12)
            ],
            "progress": [
                {"event_id": f"source-{i}", "quote": f"Exact source {i}"} for i in range(12, 16)
            ],
        }
        context = CheckpointContext(user_id="owner")
        first = await service.preview(
            "ctx", context=context, event_ids=[f"source-{i}" for i in range(16)]
        )
        await service.apply(first)
        reads = []
        original = gate._store.evidence

        async def recorded(**fields):
            reads.append(fields["evidence_ids"])
            return await original(**fields)

        gate._store.evidence = recorded
        client.draft = draft("source-35", "Exact source 35")
        second = await service.preview(
            "ctx", context=context, event_ids=[f"source-{i}" for i in range(16, 36)]
        )
        assert [len(ids) for ids in reads] == [16, 32, 4]
        assert len(json.loads(client.calls[-1][0][-1].content)["sources"]) == 20
        assert len(second.evidence_basis) == 36
        assert second.evidence_basis[0].embedding is None
        await service.apply(second)
        assert [len(ids) for ids in reads] == [16, 32, 4, 32, 4]
    finally:
        conn.close()


@pytest.mark.parametrize("corrected_index", [16, 34], ids=["first-chunk", "second-chunk"])
@pytest.mark.parametrize(
    "active_at_cutoff", [False, True], ids=["activates-later", "already-active"]
)
async def test_proposal_preparation_sources_share_cutoff_before_model(
    tmp_path, corrected_index, active_at_cutoff
):
    conn, gate, client, service = stack(tmp_path / "preparation-cutoff.db")
    try:
        for index in range(36):
            await event(gate, f"source-{index}", f"Exact source {index}")
        client.draft = {
            "title": "Sixteen source view",
            "decisions": [
                {"choice": {"event_id": f"source-{i}", "quote": f"Exact source {i}"}}
                for i in range(4)
            ],
            "open_questions": [
                {"event_id": f"source-{i}", "quote": f"Exact source {i}"} for i in range(4, 8)
            ],
            "intentions": [
                {"event_id": f"source-{i}", "quote": f"Exact source {i}"} for i in range(8, 12)
            ],
            "progress": [
                {"event_id": f"source-{i}", "quote": f"Exact source {i}"} for i in range(12, 16)
            ],
        }
        context = CheckpointContext(user_id="owner")
        first = await service.preview(
            "ctx", context=context, event_ids=[f"source-{i}" for i in range(16)]
        )
        await service.apply(first)
        await gate.store(
            Memory(
                id="scheduled-correction",
                user_id="owner",
                project="personal",
                content="Corrected measurement",
                source=MemorySource.USER_STATED,
                author_id="person",
                created_at=NOW + timedelta(seconds=2),
                revises_event_ids=[f"source-{corrected_index}"],
            )
        )
        cutoff = NOW + timedelta(seconds=3 if active_at_cutoff else 1)
        clock_reads = []

        def advancing_clock():
            value = cutoff if not clock_reads else NOW + timedelta(seconds=4)
            clock_reads.append(value)
            return value

        original_view = gate._store.working_context_view

        async def advance_after_view(**fields):
            view = await original_view(**fields)
            assert view.eligibility == "current"
            gate._store._clock = advancing_clock
            return view

        gate._store.working_context_view = advance_after_view
        reads = []
        original_evidence = gate._store.evidence

        async def recorded(**fields):
            resolved = await original_evidence(**fields)
            if len(fields["evidence_ids"]) != 16:
                reads.append((len(fields["evidence_ids"]), fields.get("effective_at")))
            return resolved

        gate._store.evidence = recorded
        client.draft = draft("source-35", "Exact source 35")
        calls_before = len(client.calls)
        before = conn.serialize()
        if active_at_cutoff:
            with pytest.raises(ValueError, match="preview source unavailable"):
                await service.preview(
                    "ctx", context=context, event_ids=[f"source-{i}" for i in range(16, 36)]
                )
            assert len(client.calls) == calls_before
        else:
            preview = await service.preview(
                "ctx", context=context, event_ids=[f"source-{i}" for i in range(16, 36)]
            )
            assert len(client.calls) == calls_before + 1
            assert len(preview.evidence_basis) == 36 and preview.input_seal
            assert preview.state.event_ids() == ["source-35"]
        assert conn.serialize() == before
        assert clock_reads == [cutoff]
        assert reads == [(32, cutoff), (4, cutoff)]
        if not active_at_cutoff:
            # Preparation is consistent at its early cutoff; apply rechecks the later instant.
            with pytest.raises(EvidenceChanged):
                await service.apply(preview)
            assert conn.serialize() == before
    finally:
        conn.close()


@pytest.mark.parametrize("corrected_index", [16, 34], ids=["first-chunk", "second-chunk"])
@pytest.mark.parametrize(
    "active_at_cutoff", [False, True], ids=["activates-later", "already-active"]
)
async def test_complete_proposal_basis_and_fact_share_apply_cutoff(
    tmp_path, corrected_index, active_at_cutoff
):
    conn, gate, client, service = stack(tmp_path / "cutoff.db")
    try:
        for index in range(36):
            await event(gate, f"source-{index}", f"Exact source {index}")
        client.draft = {
            "title": "Sixteen source view",
            "decisions": [
                {"choice": {"event_id": f"source-{i}", "quote": f"Exact source {i}"}}
                for i in range(4)
            ],
            "open_questions": [
                {"event_id": f"source-{i}", "quote": f"Exact source {i}"} for i in range(4, 8)
            ],
            "intentions": [
                {"event_id": f"source-{i}", "quote": f"Exact source {i}"} for i in range(8, 12)
            ],
            "progress": [
                {"event_id": f"source-{i}", "quote": f"Exact source {i}"} for i in range(12, 16)
            ],
        }
        context = CheckpointContext(user_id="owner")
        first = await service.preview(
            "ctx", context=context, event_ids=[f"source-{i}" for i in range(16)]
        )
        predecessor = await service.apply(first)
        client.draft = draft("source-35", "Exact source 35")
        preview = await service.preview(
            "ctx", context=context, event_ids=[f"source-{i}" for i in range(16, 36)]
        )
        assert len(preview.evidence_basis) == 36
        corrected_id = f"source-{corrected_index}"
        assert corrected_id not in preview.state.event_ids()
        assert preview.input_seal
        # Removing an unselected input cannot evade the real database-keyed seal.
        forged = preview.model_copy(
            update={"evidence_basis": [r for r in preview.evidence_basis if r.id != corrected_id]}
        )
        sealed_before = conn.serialize()
        with pytest.raises(ValueError, match="proposal input basis changed"):
            await service.apply(forged)
        assert conn.serialize() == sealed_before
        activation = NOW + timedelta(seconds=2)
        await gate.store(
            Memory(
                id="scheduled-correction",
                user_id="owner",
                project="personal",
                content="Corrected measurement",
                source=MemorySource.USER_STATED,
                author_id="person",
                created_at=activation,
                revises_event_ids=[corrected_id],
            )
        )
        cutoff = NOW + timedelta(seconds=3 if active_at_cutoff else 1)
        clock_reads = []

        def advancing_clock():
            value = cutoff if not clock_reads else NOW + timedelta(seconds=4)
            clock_reads.append(value)
            return value

        gate._store._clock = advancing_clock
        reads = []
        original = gate._store.evidence

        async def recorded(**fields):
            reads.append((len(fields["evidence_ids"]), fields["effective_at"]))
            return await original(**fields)

        gate._store.evidence = recorded
        before = conn.serialize()
        if active_at_cutoff:
            with pytest.raises(EvidenceChanged):
                await service.apply(preview)
            assert conn.serialize() == before
            assert reads == (
                [(32, cutoff)] if corrected_index == 16 else [(32, cutoff), (4, cutoff)]
            )
        else:
            identity = await service.apply(preview)
            row = conn.execute("SELECT * FROM facts WHERE id=?", (identity,)).fetchone()
            assert datetime.fromisoformat(row["valid_from"]) == cutoff
            old = conn.execute("SELECT * FROM facts WHERE id=?", (predecessor,)).fetchone()
            assert datetime.fromisoformat(old["valid_to"]) == cutoff
            assert old["superseded_by"] == identity
            assert reads == [(32, cutoff), (4, cutoff)]
        assert clock_reads == [cutoff]
    finally:
        conn.close()


@pytest.mark.parametrize("operation", ["ADD", "UPDATE", "DELETE"])
async def test_automatic_consolidation_cannot_bypass_working_context_cas(operation):
    from morgan_brain.memory.knowledge.basis import UnsupportedConsolidationOperation
    from morgan_brain.memory.knowledge.consolidation import FactOp, FactOpBatch

    conn, gate, client, _ = stack()
    try:
        await event(gate)
        prepared = await gate.capture_consolidation_basis(
            user_id="owner",
            project="personal",
            event_ids=["decision"],
            generation=gate.capture_erasure_generation(),
        )
        consolidator = MemoryConsolidator(gate=gate, client=client, model="fake", clock=lambda: NOW)
        op = FactOp(
            op=operation,
            subject="working_checkpoint:ctx",
            predicate="working_context_v1",
            object="not a validated view",
            support_event_ids=["decision"],
        )
        with pytest.raises(UnsupportedConsolidationOperation, match="compare-and-swap"):
            await consolidator.apply(
                "owner", FactOpBatch(ops=[op]), project="personal", basis=prepared.basis
            )
        assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    finally:
        conn.close()


async def test_structural_list_bounds_sorting_and_read_only_behavior():
    conn, gate, _, service = stack()
    try:
        await event(gate)
        for identity in ("zeta", "alpha", "middle"):
            await service.apply(
                await service.preview(
                    identity, context=CheckpointContext(user_id="owner"), event_ids=["decision"]
                )
            )
        changes = conn.total_changes
        listed = await gate.list_working_contexts(user_id="owner", limit=2)
        assert [item.context_id for item in listed.items] == ["alpha", "middle"]
        assert listed.truncated and all(item.view.eligibility == "current" for item in listed.items)
        assert conn.total_changes == changes
        for invalid in (True, 0, 33, 1.5):
            with pytest.raises(ValueError, match="limit"):
                await gate.list_working_contexts(user_id="owner", limit=invalid)
    finally:
        conn.close()


async def test_empty_persisted_context_is_invalid_in_show_and_list():
    from morgan_brain.memory.working_context import WORKING_CONTEXT_PREDICATE, WorkingContext

    conn, gate, _, _ = stack()
    try:
        identity = await gate.upsert_fact(
            TemporalFact(
                user_id="owner",
                project="personal",
                subject="working_checkpoint:empty",
                predicate=WORKING_CONTEXT_PREDICATE,
                object=WorkingContext(title="Empty legacy head").model_dump_json(),
                valid_from=NOW,
            ),
            now=NOW,
        )
        view = await gate.get_working_context("empty", user_id="owner", project="personal")
        assert view.fact_id == identity and view.eligibility == "invalid_state"
        assert view.state is None and view.sources == []
        listed = await gate.list_working_contexts(user_id="owner", project="personal")
        assert listed.items[0].view.eligibility == "invalid_state"
    finally:
        conn.close()


@pytest.mark.parametrize("change", ["correction", "fork"])
async def test_unselected_model_input_change_refuses_apply_atomically(change):
    conn, gate, _, service = stack()
    try:
        await event(gate)
        await event(gate, "candidate", "Measured width is 44 cm")
        preview = await service.preview(
            "context",
            context=CheckpointContext(user_id="owner"),
            event_ids=["decision", "candidate"],
        )
        assert preview.state.event_ids() == ["decision"]
        assert {r.id for r in preview.evidence_basis} == {"decision", "candidate"}
        await event(gate, "updated", "Measured width is 77 cm", revises_event_ids=["candidate"])
        if change == "fork":
            await event(gate, "other", "Measured width is 88 cm", revises_event_ids=["candidate"])
        with pytest.raises(EvidenceChanged):
            await service.apply(preview)
        assert await gate.get_working_context("context", user_id="owner") is None
    finally:
        conn.close()
