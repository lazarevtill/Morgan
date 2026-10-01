"""Fixed parser responses paused across real two-connection preparation barriers."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.knowledge.basis import (
    ConsolidationInputLimit,
    StaleConsolidationProposal,
    UnsupportedConsolidationOperation,
)
from morgan_brain.memory.knowledge.consolidation import MemoryConsolidator
from morgan_brain.memory.knowledge.fact_ops import FactOp, FactOpBatch
from morgan_brain.memory.store.db import open_db, write_transaction
from morgan_brain.models import Memory, MemorySource, TemporalFact
from morgan_brain.providers.structured import StructuredError
from tests.fakes import FakeChatClient

NOW = datetime(2026, 10, 1, tzinfo=UTC)
JAN = datetime(2026, 1, 1, tzinfo=UTC)
JUN = datetime(2026, 6, 1, tzinfo=UTC)
FUTURE = datetime(2027, 1, 1, tzinfo=UTC)


async def test_consolidation_refuses_outer_transaction_before_external_work(tmp_path):
    class CountingEmbedder(FakeEmbedder):
        calls = 0

        async def embed(self, text):
            self.calls += 1
            return await super().embed(text)

    conn = open_db(str(tmp_path / "preparation.db"))
    embedder = CountingEmbedder(dim=4)
    gate = MemoryGate(build_memory_module(conn, embedder=embedder, dim=4, clock=lambda: NOW))
    client = FakeChatClient(reply=response())
    try:
        await gate.store(source())
        embedder.calls = 0
        with write_transaction(conn):
            conn.execute("UPDATE memories SET content = ? WHERE id = ?", ("Pending edit", "A"))
            before = conn.serialize()
            with pytest.raises(ValueError, match="Preparation requires no active transaction"):
                await worker(gate, {"now": NOW}, client).consolidate("owner", project="personal")
            assert conn.in_transaction
            assert conn.serialize() == before
            assert embedder.calls == client.calls == 0
        assert conn.execute("SELECT content FROM memories WHERE id = 'A'").fetchone()[0] == (
            "Pending edit"
        )
    finally:
        conn.close()


class PausedClient(FakeChatClient):
    def __init__(self, reply):
        super().__init__(reply=reply)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def agenerate(self, *args, **kwargs):
        self.entered.set()
        await self.release.wait()
        return await super().agenerate(*args, **kwargs)


def source(identity="A", **changes):
    return Memory.model_validate(
        {
            "id": identity,
            "user_id": "owner",
            "content": "Synthetic old preference coffee",
            "source": "user_stated",
            "author_id": "person:owner",
            "created_at": JAN,
            **changes,
        }
    )


def inference(identity="existing", **changes):
    return TemporalFact.model_validate(
        {
            "id": identity,
            "user_id": "owner",
            "subject": "user",
            "predicate": "drink",
            "object": "tea",
            "source": "agent_inferred",
            **changes,
        }
    )


def response(**changes):
    return json.dumps(
        {
            "ops": [
                {
                    "op": "ADD",
                    "subject": "user",
                    "predicate": "drink",
                    "object": "coffee",
                    "support_event_ids": ["A"],
                    **changes,
                }
            ]
        }
    )


@pytest.fixture
def stack(tmp_path):
    path = str(tmp_path / "synthetic.db")
    clock = {"now": NOW}
    conn = open_db(path)
    second = open_db(path)
    gate = MemoryGate(
        build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4, clock=lambda: clock["now"])
    )
    writer = MemoryGate(
        build_memory_module(second, embedder=FakeEmbedder(dim=4), dim=4, clock=lambda: clock["now"])
    )
    yield conn, second, gate, writer, clock
    conn.close()
    second.close()


def worker(gate, clock, client):
    return MemoryConsolidator(
        gate=gate, client=client, model="fixture-model", clock=lambda: clock["now"]
    )


async def prepared(gate, clock, reply):
    client = PausedClient(reply)
    task = asyncio.create_task(worker(gate, clock, client).consolidate("owner", project="personal"))
    await asyncio.wait_for(client.entered.wait(), 2)
    return client, task


@pytest.mark.parametrize(
    "mutation",
    [
        "forget",
        "foreign_forget",
        "correction",
        "fork",
        "new_fact",
        "confidence",
        "confirmed",
        "scheduled",
    ],
)
async def test_stale_proposal_never_writes_across_barrier(stack, mutation):
    conn, second, gate, writer, clock = stack
    await writer.store(source())
    if mutation in ("confidence", "confirmed"):
        await writer.upsert_fact(inference())
    if mutation == "scheduled":
        await writer.store(source("B", created_at=FUTURE, revises_event_ids=["A"]))
    client, task = await prepared(
        gate, clock, response(op="UPDATE" if mutation == "new_fact" else "ADD")
    )
    try:
        if mutation.endswith("forget"):
            await writer.forget(
                user_id="other" if mutation == "foreign_forget" else "owner", project="personal"
            )
        elif mutation in ("correction", "fork"):
            await writer.store(source("B", created_at=JUN, revises_event_ids=["A"]))
            if mutation == "fork":
                await writer.store(source("C", created_at=JUN, revises_event_ids=["A"]))
        elif mutation == "new_fact":
            await writer.upsert_fact(inference())
        elif mutation == "confidence":
            await writer.set_confidence("existing", user_id="owner", project="personal", value=0.2)
        elif mutation == "confirmed":
            with write_transaction(second):
                second.execute(
                    "UPDATE facts SET last_confirmed=? WHERE id='existing'", (FUTURE.isoformat(),)
                )
        else:
            clock["now"] = FUTURE
        before = conn.serialize()
        client.release.set()
        with pytest.raises(StaleConsolidationProposal):
            await task
        assert conn.serialize() == before
        assert client.calls == 1  # no old-batch automatic retry or extra generation
        if mutation == "forget":
            assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
    finally:
        client.release.set()
        if not task.done():
            task.cancel()


async def test_fresh_preparation_after_forget_stores_declared_support_only(stack):
    conn, _, gate, writer, clock = stack
    await writer.store(source())
    await writer.forget(user_id="owner", project="personal")
    await writer.store(source("fresh"))
    await writer.store(source("unrelated", content="Another unrelated synthetic episode"))
    client = FakeChatClient(reply=response(support_event_ids=["fresh"]))
    applied = await worker(gate, clock, client).consolidate("owner", project="personal")
    assert len(applied) == 1
    facts = await gate.current_facts(user_id="owner", project="personal")
    assert facts[0].support_event_ids == ["fresh"]
    assert facts[0].source is MemorySource.AGENT_INFERRED
    assert facts[0].author_id == "model:fixture-model"
    assert '"id": "fresh"' in client.last_messages[-1].content
    assert '"id": "unrelated"' in client.last_messages[-1].content
    assert "untrusted quoted data" in client.last_messages[0].content
    assert conn.execute("SELECT generation FROM erasure_state").fetchone()[0] == 1


async def test_failed_forget_rolls_back_generation_and_allows_prepared_write(stack, monkeypatch):
    conn, _, gate, writer, clock = stack
    await writer.store(source())
    client, task = await prepared(gate, clock, response())

    def fail(conn, erasure):
        assert conn.execute("SELECT generation FROM erasure_state").fetchone()[0] == 1
        raise RuntimeError("synthetic deletion failure")

    with monkeypatch.context() as patch:
        patch.setitem(
            __import__("morgan_brain.memory.module", fromlist=["_DELETERS"])._DELETERS,
            "facts",
            fail,
        )
        with pytest.raises(RuntimeError, match="synthetic deletion"):
            await writer.forget(user_id="owner", project="personal")
    assert conn.execute("SELECT generation FROM erasure_state").fetchone()[0] == 0
    client.release.set()
    assert len(await task) == 1
    assert (await gate.current_facts(user_id="owner", project="personal"))[0].support_event_ids == [
        "A"
    ]


@pytest.mark.parametrize("supports", [[], ["invented"], ["foreign"], ["unknown"]])
async def test_unsupported_real_parser_batch_is_rejected_without_any_write(stack, supports):
    conn, _, gate, writer, clock = stack
    await writer.store(source())
    await writer.store(source("foreign", project="work"))
    await writer.store(source("unknown", source="unknown"))
    reply = json.dumps(
        {
            "ops": [
                {
                    "op": "ADD",
                    "subject": "valid",
                    "predicate": "drink",
                    "object": "coffee",
                    "support_event_ids": ["A"],
                },
                {
                    "op": "ADD",
                    "subject": "invalid",
                    "predicate": "drink",
                    "object": "tea",
                    "support_event_ids": supports,
                },
            ]
        }
    )
    before = conn.serialize()
    with pytest.raises(UnsupportedConsolidationOperation):
        await worker(gate, clock, FakeChatClient(reply=reply)).consolidate(
            "owner", project="personal"
        )
    assert conn.serialize() == before


@pytest.mark.parametrize("supports", [["A", "A"], ["A"] * 33, [None], [""], "A"])
async def test_malformed_supports_fail_actual_parser_without_writes(stack, supports):
    conn, _, gate, writer, clock = stack
    await writer.store(source())
    before = conn.serialize()
    client = FakeChatClient(reply=response(support_event_ids=supports))
    with pytest.raises(StructuredError):
        await worker(gate, clock, client).consolidate("owner", project="personal")
    assert conn.serialize() == before
    assert client.calls == 3  # bounded syntax repairs, never apply/retry a stale proposal


async def test_apply_without_basis_refuses_but_manual_fact_contract_remains(stack):
    _, _, gate, _, clock = stack
    engine = worker(gate, clock, FakeChatClient())
    batch = FactOpBatch(ops=[FactOp(op="ADD", subject="user", predicate="drink")])
    with pytest.raises(UnsupportedConsolidationOperation, match="prepared basis"):
        await engine.apply("owner", batch, project="personal")
    assert (
        await engine.apply(
            "owner",
            FactOpBatch(ops=[FactOp(op="NOOP", subject="user", predicate="drink")]),
            project="personal",
        )
        == []
    )
    await gate.upsert_fact(inference())
    assert len(await gate.current_facts(user_id="owner", project="personal")) == 1


async def test_basis_refuses_oversized_fact_input_without_calling_model(stack):
    conn, _, gate, writer, clock = stack
    await writer.store(source())
    with write_transaction(conn):
        for index in range(257):
            await gate.upsert_fact(inference(str(index), subject=str(index)))
    client = FakeChatClient(reply=response())
    with pytest.raises(ConsolidationInputLimit, match="256"):
        await worker(gate, clock, client).consolidate("owner", project="personal")
    assert client.calls == 0


async def test_clock_boundary_after_cas_cannot_retarget_future_fact(stack, monkeypatch):
    conn, _, gate, writer, clock = stack
    await writer.store(source())
    await writer.upsert_fact(inference("old", support_event_ids=["A"]))
    await writer.upsert_fact(inference("future", valid_from=FUTURE, support_event_ids=["A"]))
    original = gate.check_consolidation_basis

    async def crossing(basis, *, effective_at):
        await original(basis, effective_at=effective_at)
        clock["now"] = FUTURE

    monkeypatch.setattr(gate, "check_consolidation_basis", crossing)
    before = conn.serialize()
    with pytest.raises(StaleConsolidationProposal, match="effective fact"):
        await worker(gate, clock, FakeChatClient(reply=response(op="DELETE"))).consolidate(
            "owner", project="personal"
        )
    assert conn.serialize() == before
    assert conn.execute("SELECT valid_to FROM facts WHERE id='future'").fetchone()[0] is None


async def test_capture_uses_one_clock_cutoff_for_sources_and_facts():
    conn = open_db(":memory:")
    state = {"advance": False, "calls": 0}

    def clock():
        state["calls"] += 1
        return FUTURE if state["advance"] and state["calls"] > 1 else NOW

    gate = MemoryGate(build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4, clock=clock))
    try:
        await gate.store(source())
        await gate.upsert_fact(inference("old", support_event_ids=["A"]))
        await gate.upsert_fact(inference("future", valid_from=FUTURE, support_event_ids=["A"]))
        state.update(advance=True, calls=0)
        inputs = await gate.capture_consolidation_basis(
            user_id="owner",
            project="personal",
            event_ids=["A"],
            generation=gate.capture_erasure_generation(),
        )
        assert state["calls"] == 1
        assert inputs.basis.prepared_at == NOW
        assert inputs.basis.fact_ids == ("old",)
        assert [fact.id for fact in inputs.facts] == ["old"]
    finally:
        conn.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"source": "unknown"},
        {"source": "agent_inferred"},
        {"status": "quarantined"},
        {"project": "work"},
        {"user_id": "other"},
        {"kind": "semantic"},
        {"created_at": FUTURE},
    ],
)
async def test_capture_refuses_untrusted_or_unavailable_source(stack, changes):
    conn, _, gate, writer, _ = stack
    await writer.store(source(**changes))
    before = conn.serialize()
    with pytest.raises(StaleConsolidationProposal):
        await gate.capture_consolidation_basis(
            user_id="owner",
            project="personal",
            event_ids=["A"],
            generation=gate.capture_erasure_generation(),
        )
    assert conn.serialize() == before


async def test_unresolved_fork_abstains_then_explicit_resolution_is_usable(stack):
    _, _, gate, writer, clock = stack
    await writer.store(source())
    await writer.store(source("B", created_at=JUN, revises_event_ids=["A"]))
    await writer.store(source("C", created_at=JUN, revises_event_ids=["A"]))
    client = FakeChatClient(reply=response(support_event_ids=["B"]))
    assert await worker(gate, clock, client).consolidate("owner", project="personal") == []
    assert client.calls == 0
    with pytest.raises(StaleConsolidationProposal, match="conflict"):
        await gate.capture_consolidation_basis(
            user_id="owner",
            project="personal",
            event_ids=["B"],
            generation=gate.capture_erasure_generation(),
        )
    await writer.store(source("D", created_at=JUN, revises_event_ids=["B", "C"]))
    fresh = FakeChatClient(reply=response(support_event_ids=["D"]))
    assert len(await worker(gate, clock, fresh).consolidate("owner", project="personal")) == 1


async def test_forget_during_retrieval_cancels_before_generation(tmp_path):
    class PausedEmbedder(FakeEmbedder):
        def __init__(self):
            super().__init__(dim=4)
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.paused = False

        async def embed(self, text):
            if self.paused:
                self.entered.set()
                await self.release.wait()
            return await super().embed(text)

    path = str(tmp_path / "retrieval.db")
    conn, second = open_db(path), open_db(path)
    embedder = PausedEmbedder()
    gate = MemoryGate(build_memory_module(conn, embedder=embedder, dim=4, clock=lambda: NOW))
    writer = MemoryGate(
        build_memory_module(second, embedder=FakeEmbedder(dim=4), dim=4, clock=lambda: NOW)
    )
    try:
        await writer.store(source())
        embedder.paused = True
        client = FakeChatClient(reply=response())
        task = asyncio.create_task(
            worker(gate, {"now": NOW}, client).consolidate("owner", project="personal")
        )
        await asyncio.wait_for(embedder.entered.wait(), 2)
        await writer.forget(user_id="owner", project="personal")
        embedder.release.set()
        with pytest.raises(StaleConsolidationProposal, match="forget"):
            await task
        assert client.calls == 0
        assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
    finally:
        embedder.release.set()
        conn.close()
        second.close()
