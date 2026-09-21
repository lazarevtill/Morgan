"""Recall ranks by meaning and by keyword; a stored name does not add a third vote.

An exact name match is a keyword match too: the name is in the memory's text, and the keyword
index finds it. Fusing the entity ranking as well counted the same evidence twice, and on the
owner's archive that double vote pushed paraphrased answers down -- recall@8 0.83 with it, 0.90
without, and MRR 0.59 against 0.72. Lookups by name found their answer either way and ranked it
somewhat higher with it (MRR 0.94 against 0.88). The entity index still tells the relevance
floor whether a question named something the vector search ranked.
"""

from __future__ import annotations

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemoryQuery

#: Lowercase, so no entity: the vector search ranks it first and so does the keyword search,
#: which prefers the shorter of two texts that mention "harbor" once.
PLAIN = "harbor was quiet"
#: Capitalised mid-sentence, so "Harbor" is stored as an entity; ranked second by both searches.
NAMED = "we moved Harbor again today"


class PlacedEmbedder:
    """Puts each text where the test says, so the ranking recall fuses is chosen, not hashed."""

    def __init__(self, placed: dict[str, list[float]]) -> None:
        self._placed = placed

    async def embed(self, text: str) -> list[float]:
        return self._placed.get(text, [1.0, 0.0, 0.0, 0.0])

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


async def test_a_stored_name_does_not_outrank_what_meaning_and_keywords_both_rank_higher(tmp_path):
    placed = {PLAIN: [0.99, 0.14, 0.0, 0.0], NAMED: [0.5, 0.866, 0.0, 0.0]}
    gate = MemoryGate(
        build_memory_module(open_db(str(tmp_path / "m.db")), embedder=PlacedEmbedder(placed), dim=4)
    )
    for content in (NAMED, PLAIN):
        await gate.store(Memory(user_id="u", project="p", content=content))

    hits = (
        await gate.recall(MemoryQuery(user_id="u", project="p", text="Harbor", top_k=8))
    ).memories

    assert [m.content for m in hits] == [PLAIN, NAMED]
