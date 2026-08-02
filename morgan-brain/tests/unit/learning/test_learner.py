"""Unit tests for ConsolidationLearner — Phase 2B.

Verifies that ConsolidationLearner:
  - Implements the Learner Protocol.
  - process_session() stores episodics (MinimalLearner parity).
  - consolidate() applies MemoryConsolidator ops.
  - user_model() returns a default UserModel.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from morgan_brain.interfaces.learning import Learner
from morgan_brain.learning.consolidation import FactOp, FactOpBatch, FactOpKind
from morgan_brain.learning.learner import ConsolidationLearner
from morgan_brain.models.memory import Memory, MemoryKind, MemoryQuery, MemorySource
from morgan_brain.models.message import Conversation, Message, Role
from morgan_brain.models.user import RelationshipStage
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.episodic import EpisodicStore
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter
from morgan_brain.security.memory_gate import MemoryGate
from morgan_brain.learning.consolidation import MemoryConsolidator

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _build_learner(
    fake_replies: list[str],
) -> tuple[ConsolidationLearner, MemoryGate, SqliteTemporalStore]:
    temporal = SqliteTemporalStore(":memory:")
    embedder = FakeEmbedder(dim=16)
    vector_index = InMemoryVectorIndex()
    conn = open_db(":memory:")
    memory_module = MemoryModule(
        embedder=embedder,
        vectors=vector_index,
        temporal=temporal,
        clock=lambda: T0,
        fts=FtsIndex(conn),
        entities=EntityIndex(conn),
        episodics=EpisodicStore(conn),
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
        clock=lambda: T0,
        role="strong",
    )
    learner = ConsolidationLearner(
        consolidator=consolidator,
        gate=gate,
        clock=lambda: T0,
    )
    return learner, gate, temporal


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_consolidation_learner_satisfies_learner_protocol() -> None:
    learner, _, _ = _build_learner([])
    assert isinstance(learner, Learner)


# ---------------------------------------------------------------------------
# process_session — episodic storage (MinimalLearner parity)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_session_stores_episodics() -> None:
    learner, gate, _ = _build_learner([])

    convo = Conversation(
        user_id="u1",
        session_id="s1",
        messages=[
            Message(user_id="u1", role=Role.USER, content="I enjoy hiking"),
            Message(user_id="u1", role=Role.ASSISTANT, content="That's great!"),
        ],
    )
    await learner.process_session(convo)

    memories = await gate.recall(MemoryQuery(user_id="u1", text="hiking", top_k=10))
    contents = {m.content for m in memories}
    assert "I enjoy hiking" in contents


@pytest.mark.asyncio
async def test_process_session_attributes_sources_correctly() -> None:
    learner, gate, _ = _build_learner([])

    convo = Conversation(
        user_id="u1",
        session_id="s1",
        messages=[
            Message(user_id="u1", role=Role.USER, content="user msg"),
            Message(user_id="u1", role=Role.ASSISTANT, content="assistant msg"),
        ],
    )
    await learner.process_session(convo)

    memories = await gate.recall(MemoryQuery(user_id="u1", text="msg", top_k=10))
    by_content = {m.content: m.source for m in memories if "msg" in m.content}
    # User-sourced memories should carry USER_STATED
    if "user msg" in by_content:
        assert by_content["user msg"] is MemorySource.USER_STATED
    if "assistant msg" in by_content:
        assert by_content["assistant msg"] is MemorySource.AGENT_INFERRED


# ---------------------------------------------------------------------------
# user_model — default
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_model_returns_default() -> None:
    learner, _, _ = _build_learner([])
    um = await learner.user_model("u1")
    assert um.user_id == "u1"
    assert um.relationship_stage is RelationshipStage.NEW


# ---------------------------------------------------------------------------
# consolidate — calls MemoryConsolidator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consolidate_applies_add_op() -> None:
    batch = FactOpBatch(
        ops=[FactOp(op=FactOpKind.ADD, subject="user", predicate="hobby", object="climbing")]
    )
    reply = batch.model_dump_json()

    learner, gate, temporal = _build_learner([reply])

    # Seed episodic so recall returns something
    await gate.store(
        Memory(
            user_id="u1",
            kind=MemoryKind.EPISODIC,
            content="I like climbing",
            source=MemorySource.USER_STATED,
            created_at=T0,
        )
    )

    await learner.consolidate("u1")

    current = await temporal.current_facts(user_id="u1")
    hobbies = [f for f in current if f.predicate == "hobby"]
    assert len(hobbies) == 1
    assert hobbies[0].object == "climbing"


@pytest.mark.asyncio
async def test_consolidate_returns_none_not_ops() -> None:
    """consolidate() matches the Protocol return type (None / no return value)."""
    batch = FactOpBatch(ops=[])
    learner, gate, _ = _build_learner([batch.model_dump_json()])

    result = await learner.consolidate("u1")
    assert result is None
