"""`forget --all-projects` must not report a clean sweep while leaving data behind.

Project enumeration used to read `memories` alone, but `facts` and `session_history` are
independently project-keyed: a project can hold facts with zero memory rows, and such a
project was skipped entirely, unnamed in `projects`, uncounted, and absent from `warnings`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.db import open_db
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.models import Memory, MemoryKind, TemporalFact


def _now() -> datetime:
    return datetime.now(UTC)


@pytest.fixture
def gate_and_conn(tmp_path: Path) -> tuple[MemoryGate, object]:
    """One database, every store sharing the connection -- the production arrangement."""
    conn = open_db(str(tmp_path / "morgan.db"))
    module = build_memory_module(conn, embedder=FakeEmbedder(dim=16), dim=16, clock=_now)
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
