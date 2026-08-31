"""The upper index, end to end through the real composition wiring.

The unit tests prove each piece. This proves they are actually connected: a turn stored
through the cold path files its entities, and a later recall routes through them --
neither of which happens if the index is built but never read, or read but never built.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from morgan_brain.learning.consolidation import MemoryConsolidator
from morgan_brain.learning.learner import ConsolidationLearner
from morgan_brain.learning.semantic_index_builder import (
    KeywordSchemaClassifier,
    SemanticIndexBuilder,
)
from morgan_brain.models.memory import MemoryQuery
from morgan_brain.models.message import Conversation, Message, Role
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.retrieval.semantic_index import SemanticIndex
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.episodic import EpisodicStore
from morgan_brain.modules.memory.stores.sqlite_vector import SqliteVectorIndex
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter
from morgan_brain.security.memory_gate import MemoryGate

U = "u1"
P = "acme"
T0 = datetime(2026, 8, 31, tzinfo=UTC)


@pytest.fixture
def stack(tmp_path):
    conn = open_db(str(tmp_path / "morgan.db"))
    semantic = SemanticIndex(conn)
    module = MemoryModule(
        embedder=FakeEmbedder(dim=16),
        vectors=SqliteVectorIndex(conn, dim=16),
        temporal=SqliteTemporalStore(conn=conn),
        clock=lambda: T0,
        fts=FtsIndex(conn),
        entities=EntityIndex(conn),
        episodics=EpisodicStore(conn),
        semantic=semantic,
    )
    gate = MemoryGate(module)
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
        reg=reg, bindings={"strong": [Binding("fake", "test-model", FakeChatClient(replies=[]))]}
    )
    learner = ConsolidationLearner(
        consolidator=MemoryConsolidator(
            gate=gate, router=router, capability_registry=reg, clock=lambda: T0
        ),
        gate=gate,
        clock=lambda: T0,
        index_builder=SemanticIndexBuilder(semantic=semantic, classifier=KeywordSchemaClassifier()),
    )
    yield module, semantic, learner, conn
    conn.close()


async def _turn(learner, text: str) -> None:
    await learner.process_session(
        Conversation(
            user_id=U,
            project=P,
            session_id="s1",
            messages=[Message(user_id=U, role=Role.USER, content=text)],
        )
    )


async def test_a_turn_files_its_entities_into_the_index(stack):
    _module, semantic, learner, _conn = stack
    await _turn(learner, "Harbor blocked the deploy again")
    assert semantic.schema_of(user_id=U, project=P, entity="harbor") == "work"


async def test_recall_routes_through_what_the_turn_filed(stack):
    """Two unrelated turns, one query. Without routing the fused ranking would offer
    both; the pool leaves only the one the query points at."""
    module, _semantic, learner, _conn = stack
    await _turn(learner, "Harbor blocked the deploy again")
    await _turn(learner, "the Dentist appointment moved to the gym slot")

    hits = await module.recall(MemoryQuery(user_id=U, project=P, text="Harbor", top_k=8))
    contents = [m.content for m in hits]
    assert "Harbor blocked the deploy again" in contents
    assert "the Dentist appointment moved to the gym slot" not in contents


async def test_cross_project_recall_is_never_routed(stack):
    """all_projects=True is the explicit escape hatch. The index is per project, so
    routing it would make the escape hatch stricter than the default."""
    module, _semantic, learner, _conn = stack
    await _turn(learner, "Harbor blocked the deploy again")
    await _turn(learner, "the Dentist appointment moved to the gym slot")

    hits = await module.recall(
        MemoryQuery(user_id=U, project=P, all_projects=True, text="Harbor", top_k=8)
    )
    assert len(hits) == 2


async def test_an_unrouted_query_still_recalls_everything(stack):
    module, _semantic, learner, _conn = stack
    await _turn(learner, "Harbor blocked the deploy again")
    await _turn(learner, "the Dentist appointment moved to the gym slot")

    hits = await module.recall(MemoryQuery(user_id=U, project=P, text="moved slot", top_k=8))
    assert any("Dentist" in m.content for m in hits)
