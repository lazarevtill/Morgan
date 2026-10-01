"""Useful continuation persists independently and rejects changes during generation."""

from datetime import UTC, datetime

import pytest

from morgan_brain.app.continuation import ContinuationRequest, resume_work
from morgan_brain.composition import build_memory_module
from morgan_brain.memory.checkpoints import CheckpointContext
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.errors import EvidenceChanged
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.erasure import StoreInterruptedByForget
from morgan_brain.memory.store.history import SessionHistoryStore, session_key
from morgan_brain.memory.working_context import WorkingContextDraft, WorkingContextPreview
from morgan_brain.models import Memory, MemorySource, Message, Role
from morgan_brain.providers.wire import ChatResult

NOW = datetime(2026, 10, 1, tzinfo=UTC)


class DraftClient:
    def __init__(self, conn, hook=None, finish="stop"):
        self.conn, self.hook, self.finish = conn, hook, finish
        self.messages = []

    async def agenerate(self, messages, *, model):
        assert not self.conn.in_transaction
        self.messages = messages
        if self.hook:
            await self.hook()
        return ChatResult(text="A usable four-panel draft.", model=model, finish_reason=self.finish)


async def prepared(conn):
    gate = MemoryGate(
        build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4, clock=lambda: NOW)
    )
    history = SessionHistoryStore(conn, clock=lambda: NOW)
    await gate.store(
        Memory(
            id="choice",
            user_id="owner",
            content="Use four panels.",
            source=MemorySource.USER_STATED,
            author_id="person:owner",
        )
    )
    records = (
        await gate.evidence(user_id="owner", project="personal", evidence_ids=["choice"])
    ).records
    state = WorkingContextDraft.model_validate(
        {
            "title": "Gift",
            "intentions": [{"event_id": "choice", "quote": "Use four panels."}],
        }
    ).normalize(records)
    proposal = WorkingContextPreview(
        context_id="gift",
        context=CheckpointContext(user_id="owner", author_id="agent:planner"),
        expected_fact_id=None,
        generation=gate.capture_erasure_generation(),
        state=state,
        evidence_basis=records,
    )
    await gate.put_working_context(proposal)
    return gate, history, proposal


async def run(gate, history, client, session="new-agent-session"):
    return await resume_work(
        gate=gate,
        history=history,
        client=client,
        clock=lambda: NOW,
        request=ContinuationRequest(
            model="synthetic",
            context_id="gift",
            user_id="owner",
            project="personal",
            text="Continue our gift.",
            session_id=session,
            caller_client="independent-agent",
            source=MemorySource.USER_STATED,
            author_id="person:owner",
        ),
    )


async def test_new_agent_session_draft_and_provenance_survive_restart(tmp_path):
    path = str(tmp_path / "synthetic.db")
    conn = open_db(path)
    gate, history, _ = await prepared(conn)
    client = DraftClient(conn)
    result = await run(gate, history, client)
    assert result.response == "A usable four-panel draft."
    assert result.actions_executed is False
    assert result.source_event_ids == ["choice"]
    assert '"source":"user_stated"' in client.messages[1].content
    assert "no action tools" in client.messages[0].content
    conn.close()
    reopened = open_db(path)
    try:
        rows = SessionHistoryStore(reopened, clock=lambda: NOW).recent(
            session_key("owner", "new-agent-session"), project="personal", user_id="owner"
        )
        assert [row.content for row in rows] == ["Continue our gift.", result.response]
        stored = reopened.execute(
            "SELECT source,author_id,client,session_id FROM memories WHERE content=?",
            (result.response,),
        ).fetchone()
        assert tuple(stored) == (
            "agent_inferred",
            "model:synthetic",
            "independent-agent",
            "new-agent-session",
        )
    finally:
        reopened.close()


@pytest.mark.parametrize("change", ["head", "revision", "forget"])
async def test_changed_basis_during_generation_never_commits_old_draft(change):
    conn = open_db(":memory:")
    try:
        gate, history, proposal = await prepared(conn)

        async def modify():
            if change == "head":
                old = await gate.get_working_context("gift", user_id="owner")
                await gate.put_working_context(
                    proposal.model_copy(
                        update={
                            "expected_fact_id": old.fact_id,
                            "state": proposal.state.model_copy(
                                update={"title": "Gift, revised organization"}
                            ),
                        }
                    )
                )
            elif change == "revision":
                await gate.store(
                    Memory(
                        id="correction",
                        user_id="owner",
                        content="Use two panels instead.",
                        source=MemorySource.USER_STATED,
                        author_id="person:owner",
                        created_at=NOW,
                        revises_event_ids=["choice"],
                    )
                )
            else:
                await gate.forget(user_id="owner", project="personal")

        error = StoreInterruptedByForget if change == "forget" else EvidenceChanged
        with pytest.raises(error):
            await run(gate, history, DraftClient(conn, hook=modify))
        assert not history.recent(
            session_key("owner", "new-agent-session"), project="personal", user_id="owner"
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM memories WHERE content=?", ("A usable four-panel draft.",)
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()


async def test_truncated_generation_is_not_committed_as_a_useful_draft():
    conn = open_db(":memory:")
    try:
        gate, history, _ = await prepared(conn)
        with pytest.raises(ValueError, match="generation incomplete"):
            await run(gate, history, DraftClient(conn, finish="length"))
        assert not history.recent(
            session_key("owner", "new-agent-session"), project="personal", user_id="owner"
        )
    finally:
        conn.close()


@pytest.mark.parametrize("session", ["default", "occupied"])
async def test_reserved_or_occupied_session_rejected_before_model(session):
    conn = open_db(":memory:")
    try:
        gate, history, _ = await prepared(conn)
        if session == "occupied":
            history.append(
                session_key("owner", session),
                Message(
                    user_id="owner",
                    project="personal",
                    role=Role.USER,
                    content="Existing unrelated work",
                ),
                project="personal",
            )
        client = DraftClient(conn)
        with pytest.raises(ValueError, match="session"):
            await run(gate, history, client, session=session)
        assert client.messages == []
    finally:
        conn.close()


async def test_concurrent_session_occupancy_rejected_without_partial_draft():
    conn = open_db(":memory:")
    try:
        gate, history, _ = await prepared(conn)

        async def occupy():
            history.append(
                session_key("owner", "new-agent-session"),
                Message(
                    user_id="owner", project="personal", role=Role.USER, content="Other caller won"
                ),
                project="personal",
            )

        with pytest.raises(ValueError, match="already occupied"):
            await run(gate, history, DraftClient(conn, hook=occupy))
        assert [
            m.content
            for m in history.recent(
                session_key("owner", "new-agent-session"), project="personal", user_id="owner"
            )
        ] == ["Other caller won"]
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM memories WHERE content=?", ("A usable four-panel draft.",)
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()
