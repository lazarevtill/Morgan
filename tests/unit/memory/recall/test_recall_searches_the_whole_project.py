"""Recall searches every memory in the project, whatever the question names.

Narrowing the search to the memories that carry a name from the question, before ranking,
cut the answer out of the questions it narrowed on the owner's archive and raised recall on
none. A memory that answers a question without repeating its words is the case semantic
search exists for.
"""

from __future__ import annotations

from morgan_brain.memory.gate import MemoryGate
from morgan_brain.models import Memory, MemoryQuery
from tests.unit.memory.conftest import build_memory_module as _module


async def test_a_memory_that_does_not_carry_the_name_in_the_question_is_still_searched(tmp_path):
    gate = MemoryGate(_module(str(tmp_path / "m.db")))
    for content in (
        "yesterday Harbor blocked the deploy again",
        "the Dentist appointment moved to the gym slot",
    ):
        await gate.store(Memory(user_id="u", project="p", content=content))

    hits = (
        await gate.recall(MemoryQuery(user_id="u", project="p", text="Harbor", top_k=8))
    ).memories

    assert {m.content for m in hits} == {
        "yesterday Harbor blocked the deploy again",
        "the Dentist appointment moved to the gym slot",
    }
