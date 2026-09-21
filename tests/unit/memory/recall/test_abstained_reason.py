"""An empty answer says which kind of empty it is.

"No memories found" covered four different situations: nothing stored, the floor declining,
too few results to judge, and no floor configured at all. The owner cannot act on the first
without being able to tell it from the second.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from morgan_brain.composition import build_memory_module as build_placed
from morgan_brain.memory.module import MemoryModule
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemoryQuery, TemporalFact
from tests.unit.memory.conftest import build_memory_module

#: Where every text the placed embedder is not told about lands -- queries included.
_QUERY_VECTOR = [1.0, 0.0, 0.0, 0.0]
#: Cosine 0.5 to the query: an equally mediocre neighbour.
_FLAT = [0.5, 0.866, 0.0, 0.0]
#: Enough equally mediocre neighbours for the floor to judge: it needs five results.
_BACKGROUND = [f"the Office grocery list number {n} for the weekend" for n in range(6)]


class _PlacedEmbedder:
    """Puts each text where the test says, so the scores recall sees are chosen, not hashed."""

    def __init__(self, placed: dict[str, list[float]]) -> None:
        self._placed = placed

    async def embed(self, text: str) -> list[float]:
        return self._placed.get(text, _QUERY_VECTOR)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


async def _store(module: MemoryModule, texts: list[str]) -> None:
    for text in texts:
        await module.store(Memory(user_id="u", project="p", content=text))


async def _module_with(tmp_path: Path, *, count: int, floor_margin: float | None) -> MemoryModule:
    module = build_memory_module(str(tmp_path / "m.db"), floor_margin=floor_margin)
    await _store(module, [f"the Harbor mirror note number {n}" for n in range(count)])
    return module


async def _placed_module(tmp_path: Path, *, extra: list[str]) -> MemoryModule:
    """The background at cosine 0.5 to every query, plus *extra* texts placed on the query.

    Built as ``test_floor_in_recall.py`` builds its decline.
    """
    module = build_placed(
        open_db(str(tmp_path / "m.db")),
        embedder=_PlacedEmbedder(dict.fromkeys(_BACKGROUND, _FLAT)),
        dim=4,
        floor_margin=0.1,
    )
    await _store(module, [*_BACKGROUND, *extra])
    return module


async def _module_whose_floor_declines(tmp_path: Path) -> MemoryModule:
    # Nothing stands above the background, and no query word is an entity of any memory.
    return await _placed_module(tmp_path, extra=[])


async def _module_whose_floor_answers(tmp_path: Path) -> MemoryModule:
    # One memory sits on the query itself, half a unit of cosine above the background.
    return await _placed_module(tmp_path, extra=["the answer everyone was looking for"])


async def _a_current_fact(module: MemoryModule) -> None:
    await module.upsert_fact(
        TemporalFact(
            user_id="u", project="p", subject="deploy", predicate="blocked_by", object="mirror"
        )
    )


async def test_an_empty_project_says_empty(tmp_path):
    module = build_memory_module(str(tmp_path / "m.db"))
    out = await module.recall(MemoryQuery(user_id="u", project="p", text="anything"))
    assert (out.abstained, out.reason, out.memories) == (True, "empty", [])


async def test_three_memories_are_returned_unjudged(tmp_path):
    module = await _module_with(tmp_path, count=3, floor_margin=0.11)
    out = await module.recall(MemoryQuery(user_id="u", project="p", text="Harbor"))
    assert out.abstained is False and out.reason == "too_few_to_judge" and out.memories


async def test_a_decline_says_declined_and_returns_no_facts(tmp_path):
    module = await _module_whose_floor_declines(tmp_path)
    await _a_current_fact(module)
    # The fact is current: a recall that had not declined would have returned it.
    assert await module.current_facts(user_id="u", project="p")

    out = await module.recall(MemoryQuery(user_id="u", project="p", text="unanswerable"))

    assert (out.abstained, out.reason) == (True, "declined")
    assert out.memories == []


async def test_no_floor_configured_says_so(tmp_path):
    module = await _module_with(tmp_path, count=9, floor_margin=None)
    out = await module.recall(MemoryQuery(user_id="u", project="p", text="Harbor"))
    assert out.abstained is False and out.reason == "no_floor"


async def test_a_judged_answer_carries_no_reason(tmp_path):
    module = await _module_whose_floor_answers(tmp_path)

    out = await module.recall(MemoryQuery(user_id="u", project="p", text="unanswerable"))

    assert (out.abstained, out.reason) == (False, None)
    assert out.memories[0].content == "the answer everyone was looking for"


async def test_facts_alone_are_an_answer_not_empty(tmp_path):
    # "empty" means nothing came back at all; a project holding only facts returns them.
    module = build_memory_module(str(tmp_path / "m.db"))
    await _a_current_fact(module)

    out = await module.recall(MemoryQuery(user_id="u", project="p", text="deploy"))

    assert (out.abstained, out.reason) == (False, "no_floor")
    assert [m.content for m in out.memories] == ["deploy blocked by mirror"]


async def _empty(tmp_path: Path) -> MemoryModule:
    return build_memory_module(str(tmp_path / "m.db"))


async def _too_few(tmp_path: Path) -> MemoryModule:
    return await _module_with(tmp_path, count=3, floor_margin=0.11)


async def _no_floor(tmp_path: Path) -> MemoryModule:
    return await _module_with(tmp_path, count=9, floor_margin=None)


@pytest.mark.parametrize(
    ("build", "reason"),
    [
        (_empty, "empty"),
        (_too_few, "too_few_to_judge"),
        (_no_floor, "no_floor"),
        (_module_whose_floor_declines, "declined"),
        (_module_whose_floor_answers, None),
    ],
)
async def test_recall_done_carries_the_reason_recall_returned(
    tmp_path: Path, build: Callable[[Path], Awaitable[MemoryModule]], reason: str | None
):
    module = await build(tmp_path)

    with capture_logs() as logs:
        out = await module.recall(MemoryQuery(user_id="u", project="p", text="unanswerable"))

    assert out.reason == reason
    assert [e["reason"] for e in logs if e["event"] == "recall.done"] == [reason]
