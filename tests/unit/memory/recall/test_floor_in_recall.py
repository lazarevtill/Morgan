"""When an exact word match may overrule the floor.

The override exists for a real hit an embedding is bad at: an identifier or a surname whose
memory sits flat against its neighbours. It used to fire on *any* query word that was stored
as an entity *anywhere* in the project. On the owner's real archive that was 31 of 38
questions the corpus cannot answer -- a question about Minecraft on Ubuntu matched "ubuntu" on
an unrelated memory -- so the floor silenced almost nothing.

A match now counts only when the memory it found is also one the vector search ranked among
the results. A genuine identifier hit is; a word that merely occurs somewhere is not.
"""

from __future__ import annotations

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemoryKind, MemoryQuery, MemorySource

QUERY = "minecraft server on ubuntu"
FLAT = [0.5, 0.866, 0.0, 0.0]  # cosine 0.5 to the query: an equally mediocre neighbour


class PlacedEmbedder:
    """Puts each text where the test says, so the scores recall sees are chosen, not hashed."""

    def __init__(self, placed: dict[str, list[float]]) -> None:
        self._placed = placed

    async def embed(self, text: str) -> list[float]:
        return self._placed.get(text, [1.0, 0.0, 0.0, 0.0])

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


#: Equally mediocre neighbours, so the vector search has a background to judge the margin
#: against: the floor needs at least five results before it judges at all.
BACKGROUND = [f"the Office grocery list number {n} for the weekend" for n in range(6)]


async def _gate(tmp_path, *, ubuntu_vector: list[float]) -> MemoryGate:
    ubuntu = "the Office printer runs on Ubuntu now"
    placed = dict.fromkeys(BACKGROUND, FLAT) | {ubuntu: ubuntu_vector}
    gate = MemoryGate(
        build_memory_module(
            open_db(str(tmp_path / "morgan.db")),
            embedder=PlacedEmbedder(placed),
            dim=4,
            floor_margin=0.1,
        )
    )
    for text in [*BACKGROUND, ubuntu]:
        await gate.store(
            Memory(
                user_id="u",
                project="p",
                kind=MemoryKind.EPISODIC,
                content=text,
                source=MemorySource.USER_STATED,
            )
        )
    return gate


async def test_a_word_stored_on_an_unrelated_memory_does_not_overrule_the_floor(tmp_path):
    # The Ubuntu memory points away from the query, so the vector search ranks it last and it
    # falls outside the results; the six it does return are equally mediocre.
    gate = await _gate(tmp_path, ubuntu_vector=[0.0, 0.0, 1.0, 0.0])

    found = (await gate.recall(MemoryQuery(user_id="u", project="p", text=QUERY, top_k=3))).memories

    assert found == [], [m.content for m in found]


async def test_an_exact_match_the_vector_search_also_ranked_still_answers(tmp_path):
    # Same flat margin, but now the matching memory is among the vector results: the case the
    # override is for.
    gate = await _gate(tmp_path, ubuntu_vector=FLAT)

    found = (await gate.recall(MemoryQuery(user_id="u", project="p", text=QUERY, top_k=8))).memories

    assert any("Ubuntu" in m.content for m in found), [m.content for m in found]
