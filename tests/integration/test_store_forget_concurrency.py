"""A project erased while a memory is being stored leaves no piece of that memory behind.

`morgan forget` and `morgan-mcp` share one database. Storing a memory writes five indexes --
the episodic row, its vector, its keyword row, its entity rows and the semantic index -- and
`forget()` erases a project in one locked transaction. Unless the store is one transaction
too, an erasure can land between two of its writes and leave the rest behind for a memory
that no longer exists.

Schema classification is the widest such gap: it is an awaited seam (keyword-based today, a
model call on the roadmap), and it ran between the index writes. The classifier here hands
control to the other process and waits until that process has erased the project, so the
erasure lands inside the store at every step -- no timing luck involved. Awaiting it while
holding the write lock would fail too: the erasure could never get the lock, and the forget
times out.

Whatever is left must be whole: every index row belongs to a stored memory, every stored
memory is in every index, and the memory stored after the last erasure is there.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.knowledge.schema_classifier import (
    KeywordSchemaClassifier,
    SemanticIndexBuilder,
)
from morgan_brain.memory.module import MemoryModule
from morgan_brain.memory.recall.semantic_index import SemanticIndex
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.entities import EntityIndex
from morgan_brain.memory.store.episodic import EpisodicStore
from morgan_brain.memory.store.fts import FtsIndex
from morgan_brain.memory.store.temporal import SqliteTemporalStore
from morgan_brain.memory.store.vectors import SqliteVectorIndex
from morgan_brain.models import Entity, Memory, MemoryKind

if TYPE_CHECKING:
    import sqlite3
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event

_DIM = 8
_STEPS = 5
_TIMEOUT_S = 60

#: Each query counts rows that break "whole": an index row without its memory, or a memory
#: missing from an index.
_BROKEN = {
    "keyword rows without a memory": (
        "SELECT COUNT(*) FROM fts_memories WHERE memory_id NOT IN (SELECT id FROM memories)"
    ),
    "entity rows without a memory": (
        "SELECT COUNT(*) FROM memory_entities WHERE memory_id NOT IN (SELECT id FROM memories)"
    ),
    "vectors without a memory": (
        "SELECT COUNT(*) FROM vec_meta WHERE id NOT IN (SELECT id FROM memories)"
    ),
    "embeddings without metadata": (
        "SELECT COUNT(*) FROM vec_items WHERE rowid NOT IN (SELECT rowid FROM vec_meta)"
    ),
    "memories without a vector": (
        "SELECT COUNT(*) FROM memories WHERE id NOT IN (SELECT id FROM vec_meta)"
    ),
    "memories without a keyword row": (
        "SELECT COUNT(*) FROM memories WHERE id NOT IN (SELECT memory_id FROM fts_memories)"
    ),
    "memories without entity rows": (
        "SELECT COUNT(*) FROM memories WHERE id NOT IN (SELECT memory_id FROM memory_entities)"
    ),
    "entity nodes without their schema": (
        "SELECT COUNT(*) FROM mem_entity_nodes n WHERE NOT EXISTS (SELECT 1 FROM mem_schemas s "
        "WHERE s.user_id = n.user_id AND s.project = n.project AND s.name = n.schema_name)"
    ),
    "entity edges without their nodes": (
        "SELECT COUNT(*) FROM mem_entity_edges e WHERE NOT EXISTS (SELECT 1 FROM "
        "mem_entity_nodes n WHERE n.user_id = e.user_id AND n.project = e.project "
        "AND n.name = e.src) OR NOT EXISTS (SELECT 1 FROM mem_entity_nodes n "
        "WHERE n.user_id = e.user_id AND n.project = e.project AND n.name = e.dst)"
    ),
}


class _ClassifierThatWaitsForAnErasure:
    """Classifies like the keyword classifier, after the other process has erased the project."""

    def __init__(self, classifying: Event, erased: Event) -> None:
        self._classifying = classifying
        self._erased = erased
        self._inner = KeywordSchemaClassifier()

    async def classify(
        self, names: list[str], *, schemas: list[str], samples: dict[str, str]
    ) -> dict[str, str]:
        self._erased.clear()
        self._classifying.set()
        # Suspends like a model call would, until the erasure has committed.
        await asyncio.to_thread(self._erased.wait, _TIMEOUT_S)
        return await self._inner.classify(names, schemas=schemas, samples=samples)


def _module(conn: sqlite3.Connection, classifier: _ClassifierThatWaitsForAnErasure) -> MemoryModule:
    semantic = SemanticIndex(conn)
    return MemoryModule(
        embedder=FakeEmbedder(dim=_DIM),
        vectors=SqliteVectorIndex(conn, dim=_DIM),
        temporal=SqliteTemporalStore(conn=conn),
        clock=lambda: datetime.now(UTC),
        fts=FtsIndex(conn),
        entities=EntityIndex(conn),
        episodics=EpisodicStore(conn),
        semantic=semantic,
        index_builder=SemanticIndexBuilder(semantic=semantic, classifier=classifier),
    )


def _storer(
    path: str, classifying: Event, erased: Event, finished: Event, outcomes: Queue[str]
) -> None:
    """Runs in a spawned process: store one memory per step."""
    try:
        conn = open_db(path)
        module = _module(conn, _ClassifierThatWaitsForAnErasure(classifying, erased))

        async def store_every_step() -> None:
            for i in range(_STEPS):
                await module.store(
                    Memory(
                        user_id="u",
                        project="p",
                        content=f"note {i}: the Harbor mirror blocked the Kafka deploy",
                        kind=MemoryKind.EPISODIC,
                        # New names at every step: only unfiled entities are classified.
                        entities=[Entity(name=f"Harbor{i}"), Entity(name=f"Kafka{i}")],
                    )
                )

        asyncio.run(store_every_step())
        conn.close()
    except BaseException as exc:
        outcomes.put(f"storer: {type(exc).__name__}: {exc}")
        raise
    finally:
        finished.set()
    outcomes.put("storer: ok")


def _forgetter(
    path: str, classifying: Event, erased: Event, finished: Event, outcomes: Queue[str]
) -> None:
    """Runs in a spawned process: erase the project whenever the storer is classifying."""
    try:
        conn = open_db(path)
        module = build_memory_module(conn, embedder=FakeEmbedder(dim=_DIM), dim=_DIM)

        async def erase_while_classifying() -> int:
            erasures = 0
            while erasures < _STEPS:
                if not classifying.wait(0.1):
                    if finished.is_set():
                        break
                    continue
                classifying.clear()
                await module.forget(user_id="u", project="p")
                erasures += 1
                erased.set()
            return erasures

        erasures = asyncio.run(erase_while_classifying())
        conn.close()
    except BaseException as exc:
        erased.set()
        outcomes.put(f"forgetter: {type(exc).__name__}: {exc}")
        raise
    outcomes.put(f"forgetter: {erasures} erasures")


def test_a_project_erased_mid_store_leaves_only_whole_memories(tmp_path: Path) -> None:
    path = str(tmp_path / "morgan.db")
    setup = open_db(path)
    build_memory_module(setup, embedder=FakeEmbedder(dim=_DIM), dim=_DIM)
    setup.close()

    ctx = mp.get_context("spawn")
    classifying = ctx.Event()
    erased = ctx.Event()
    finished = ctx.Event()
    outcomes: Queue[str] = ctx.Queue()
    workers = [
        ctx.Process(target=_storer, args=(path, classifying, erased, finished, outcomes)),
        ctx.Process(target=_forgetter, args=(path, classifying, erased, finished, outcomes)),
    ]
    for p in workers:
        p.start()
    try:
        reported = sorted(outcomes.get(timeout=_TIMEOUT_S * 2) for _ in workers)
        for p in workers:
            p.join(timeout=_TIMEOUT_S)
    finally:
        for p in workers:
            if p.is_alive():
                p.terminate()

    # Every step's store was interrupted by an erasure, so every step tested the gap.
    assert reported == [f"forgetter: {_STEPS} erasures", "storer: ok"]
    assert [p.exitcode for p in workers] == [0, 0]

    conn = open_db(path)
    try:
        broken = {name: conn.execute(sql).fetchone()[0] for name, sql in _BROKEN.items()}
        memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    finally:
        conn.close()
    assert {name: n for name, n in broken.items() if n} == {}
    assert memories == 1, "the memory stored after the last erasure should be there, whole"
