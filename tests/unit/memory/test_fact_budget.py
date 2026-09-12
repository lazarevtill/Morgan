"""Facts must never crowd episodic memory out of recall.

`recall` prepended every currently-valid fact, unranked, ahead of the fused episodics and
then truncated to top_k. Once a project held top_k facts, no episodic memory could be
returned at all, however well it matched -- and `current_facts` has no limit, so the fact
count only grows as consolidation runs. The probe harness stores no facts, so the
measurement could never see it.

The evidence says the union is right and the eviction is not: verbatim episodics beat
extracted artifacts on both LoCoMo and LongMemEval, and the most heavily structured
retrieval path in the published comparisons regressed on raw-utterance questions.
"""

from __future__ import annotations

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemoryKind, MemoryQuery, MemorySource, TemporalFact


@pytest.fixture
def gate(tmp_path):
    conn = open_db(str(tmp_path / "morgan.db"))
    return MemoryGate(build_memory_module(conn=conn, embedder=FakeEmbedder(dim=16), dim=16))


async def _add_facts(gate: MemoryGate, n: int, *, subject: str = "unrelated") -> None:
    for i in range(n):
        await gate.upsert_fact(
            TemporalFact(
                user_id="u",
                project="p",
                subject=f"{subject}{i}",
                predicate="is",
                object=f"value{i}",
                source=MemorySource.USER_STATED,
            )
        )


async def test_an_exactly_matching_memory_survives_a_project_full_of_facts(gate):
    await gate.store(
        Memory(
            user_id="u",
            project="p",
            kind=MemoryKind.EPISODIC,
            content="the harbor mirror blocked the deploy",
            source=MemorySource.USER_STATED,
        )
    )
    await _add_facts(gate, 12)

    found = await gate.recall(MemoryQuery(user_id="u", project="p", text="harbor", top_k=8))

    assert any("harbor" in m.content for m in found), [m.content for m in found]


async def test_facts_still_fill_the_window_when_there_is_little_else(gate):
    """The budget reserves room for episodics; it must not waste it when none exist."""
    await _add_facts(gate, 12)

    found = await gate.recall(MemoryQuery(user_id="u", project="p", text="value3", top_k=8))

    assert len(found) == 8


async def test_the_facts_that_survive_the_budget_are_the_ones_the_query_asked_about(gate):
    """Cutting the fact block by arbitrary order would drop the relevant fact as readily as
    an irrelevant one, trading one silent failure for another."""
    # Subjects chosen to sort and insert ahead of the relevant one, so a cut that keeps
    # whichever facts happen to come first drops exactly the fact the query asked for.
    await _add_facts(gate, 12, subject="aaa")
    await gate.upsert_fact(
        TemporalFact(
            user_id="u",
            project="p",
            subject="harbor",
            predicate="mirror_state",
            object="stale",
            source=MemorySource.USER_STATED,
        )
    )
    for i in range(8):
        await gate.store(
            Memory(
                user_id="u",
                project="p",
                kind=MemoryKind.EPISODIC,
                content=f"an unrelated note number {i}",
                source=MemorySource.USER_STATED,
            )
        )

    found = await gate.recall(MemoryQuery(user_id="u", project="p", text="harbor", top_k=8))

    facts = [m.content for m in found if m.kind is MemoryKind.SEMANTIC]
    assert any("harbor" in c for c in facts), facts
