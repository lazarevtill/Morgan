"""Anti-amnesia guard (the retention half of the hoarding-vs-amnesia tradeoff).

A fact the user explicitly stated is high-importance by construction. It must survive:
  (1) an inferred DELETE proposed by the consolidator, and
  (2) aggressive confidence decay (a confidence floor),
so low-frequency, high-importance user statements ("never deploy on Friday", "allergic to
penicillin") are never silently lost — the failure mode the 2026 memory literature names.
Agent-inferred facts remain freely deletable and decayable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from morgan_brain.learning.consolidation import (
    FactOp,
    FactOpBatch,
    FactOpKind,
    MemoryConsolidator,
)
from morgan_brain.models.memory import MemorySource, TemporalFact
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

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
USER = "u1"


def _consolidator() -> tuple[MemoryConsolidator, MemoryGate, SqliteTemporalStore]:
    temporal = SqliteTemporalStore(":memory:")
    conn = open_db(":memory:")
    mm = MemoryModule(
        embedder=FakeEmbedder(dim=16),
        vectors=InMemoryVectorIndex(),
        temporal=temporal,
        clock=lambda: T0,
        fts=FtsIndex(conn),
        entities=EntityIndex(conn),
        episodics=EpisodicStore(conn),
    )
    gate = MemoryGate(mm)
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
        reg=reg, bindings={"strong": [Binding("fake", "test-model", FakeChatClient(reply="{}"))]}
    )
    cons = MemoryConsolidator(gate=gate, router=router, capability_registry=reg, clock=lambda: T0)
    return cons, gate, temporal


async def _seed(
    gate: MemoryGate,
    predicate: str,
    obj: str,
    source: MemorySource,
    *,
    conf: float = 1.0,
    when: datetime = T0,
) -> None:
    await gate.upsert_fact(
        TemporalFact(
            user_id=USER,
            subject="user",
            predicate=predicate,
            object=obj,
            source=source,
            confidence=conf,
            last_confirmed=when,
            valid_from=when,
            created_at=when,
        )
    )


@pytest.mark.asyncio
async def test_inferred_delete_cannot_erase_user_stated_fact() -> None:
    cons, gate, temporal = _consolidator()
    await _seed(gate, "deploy_policy", "never on Friday", MemorySource.USER_STATED)

    await cons.apply(
        USER,
        FactOpBatch(ops=[FactOp(op=FactOpKind.DELETE, subject="user", predicate="deploy_policy")]),
        project="default",
    )

    facts = await temporal.current_facts(user_id=USER)
    assert any(f.predicate == "deploy_policy" and f.object == "never on Friday" for f in facts), (
        "user-stated fact was erased by an inferred DELETE"
    )


@pytest.mark.asyncio
async def test_inferred_delete_still_removes_agent_inferred_fact() -> None:
    cons, gate, temporal = _consolidator()
    await _seed(gate, "likes", "coffee", MemorySource.AGENT_INFERRED)

    await cons.apply(
        USER,
        FactOpBatch(ops=[FactOp(op=FactOpKind.DELETE, subject="user", predicate="likes")]),
        project="default",
    )

    facts = await temporal.current_facts(user_id=USER)
    assert not any(f.predicate == "likes" for f in facts), (
        "agent-inferred fact should remain deletable"
    )


@pytest.mark.asyncio
async def test_user_stated_facts_resist_decay_floor() -> None:
    cons, gate, temporal = _consolidator()
    await _seed(gate, "allergy", "penicillin", MemorySource.USER_STATED, conf=1.0)
    await _seed(gate, "mood", "happy", MemorySource.AGENT_INFERRED, conf=1.0)

    later = T0 + timedelta(days=365)  # a year of aggressive decay
    await cons.decay_confidence(
        USER, project="default", now=later, half_life_days=30.0, protected_floor=0.5
    )

    conf = {f.predicate: f.confidence for f in await temporal.current_facts(user_id=USER)}
    assert conf["allergy"] >= 0.5, "user-stated fact decayed below the protected floor"
    assert conf["mood"] < 0.5, "agent-inferred fact should decay freely"
