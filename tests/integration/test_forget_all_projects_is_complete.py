"""`forget --all-projects` must not report a clean sweep while leaving data behind.

Project enumeration read `memories` alone, but `facts`, `interaction_signals` and
`session_history` are independently project-keyed. `Orchestrator._persist_turn` writes history
and the base signal synchronously in brain-api while the episodic memory is written by the
worker off the bus -- so if the worker is down, or the bounded in-proc queue drops the event, a
project holds full transcripts and signals with zero memory rows. Such a project was skipped
entirely, unnamed in `projects`, uncounted, and absent from `warnings`.

`_forget_result`'s own docstring calls itself "the one place that must not lie".
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from morgan_brain.learning.signals import InteractionSignal, SignalStore
from morgan_brain.models.memory import Memory, MemoryKind, TemporalFact
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.episodic import EpisodicStore
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex
from morgan_brain.security.memory_gate import MemoryGate


def _now() -> datetime:
    return datetime.now(UTC)


@pytest.fixture
def gate_and_conn(tmp_path: Path) -> tuple[MemoryGate, object]:
    """One database, every store sharing the connection -- the production arrangement."""
    path = str(tmp_path / "morgan.db")
    conn = open_db(path)
    module = MemoryModule(
        embedder=FakeEmbedder(dim=16),
        vectors=InMemoryVectorIndex(),
        temporal=SqliteTemporalStore(path),
        clock=_now,
        fts=FtsIndex(conn),
        entities=EntityIndex(conn),
        episodics=EpisodicStore(conn),
    )
    return MemoryGate(module), conn


async def test_a_facts_only_project_is_enumerated(
    gate_and_conn: tuple[MemoryGate, object],
) -> None:
    gate, _ = gate_and_conn
    await gate.store(
        Memory(user_id="u", project="has-memory", content="a note", kind=MemoryKind.EPISODIC)
    )
    await gate.upsert_fact(
        TemporalFact(
            user_id="u",
            project="facts-only",
            subject="user",
            predicate="lives_in",
            object="Moscow",
            valid_from=_now(),
        )
    )

    projects = await gate.distinct_projects("u")

    assert projects == ["facts-only", "has-memory"], projects


async def test_a_signals_only_project_is_enumerated(
    gate_and_conn: tuple[MemoryGate, object],
) -> None:
    """The realistic case: brain-api wrote the signal, the worker never wrote the memory."""
    gate, conn = gate_and_conn
    store = SignalStore(conn, clock=_now)  # type: ignore[arg-type]
    await store.record(
        InteractionSignal(
            user_id="u",
            project="signals-only",
            session_id="s",
            turn_id="t",
            query="q",
            original_reply="r",
        )
    )

    assert await gate.distinct_projects("u") == ["signals-only"]


async def test_the_full_wipe_leaves_nothing_behind(
    gate_and_conn: tuple[MemoryGate, object],
) -> None:
    """Erase every enumerated project, then assert no project-keyed table has a row left."""
    gate, conn = gate_and_conn
    await gate.store(
        Memory(user_id="u", project="has-memory", content="a note", kind=MemoryKind.EPISODIC)
    )
    await gate.upsert_fact(
        TemporalFact(
            user_id="u",
            project="facts-only",
            subject="user",
            predicate="lives_in",
            object="Moscow",
            valid_from=_now(),
        )
    )

    swept = []
    for project in await gate.distinct_projects("u"):
        await gate.forget(user_id="u", project=project)
        swept.append(project)

    assert swept == ["facts-only", "has-memory"], swept
    surviving = conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) AS n FROM facts WHERE user_id = 'u'"
    ).fetchone()
    assert surviving["n"] == 0, "a fact survived a full wipe"
    assert await gate.distinct_projects("u") == []
