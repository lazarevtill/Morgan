"""`forget()` either erases everything or nothing.

Atomicity is the stated justification for the whole shared-single-connection design
(`memory/module.py`), and this is the destructive operation. It had no committed test:
replacing the exception handler's `conn.rollback()` with `conn.commit()` left the suite green,
so a refactor that half-erased a project would have shipped.
"""

from __future__ import annotations

import sqlite3
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


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    out: dict[str, int] = {}
    for table in ("memories", "fts_memories", "memory_entities", "facts"):
        row = conn.execute(
            {
                "memories": "SELECT COUNT(*) AS n FROM memories",
                "fts_memories": "SELECT COUNT(*) AS n FROM fts_memories",
                "memory_entities": "SELECT COUNT(*) AS n FROM memory_entities",
                "facts": "SELECT COUNT(*) AS n FROM facts",
            }[table]
        ).fetchone()
        out[table] = row["n"]
    return out


@pytest.fixture
def gate_and_conn(tmp_path: Path) -> tuple[MemoryGate, sqlite3.Connection]:
    conn = open_db(str(tmp_path / "morgan.db"))
    module = build_memory_module(conn, embedder=FakeEmbedder(dim=16), dim=16, clock=_now)
    return MemoryGate(module), conn


async def _seed(gate: MemoryGate) -> None:
    await gate.store(
        Memory(user_id="u", project="p", content="Harbor mirror note", kind=MemoryKind.EPISODIC)
    )
    await gate.upsert_fact(
        TemporalFact(
            user_id="u",
            project="p",
            subject="user",
            predicate="lives_in",
            object="Moscow",
            valid_from=_now(),
        )
    )


class _FailsOnFactsDelete:
    """Delegates to a real connection but raises on the facts DELETE.

    ``sqlite3.Connection.execute`` is read-only, so the failure is injected with a proxy rather
    than a monkeypatch. Everything else -- including ``rollback`` -- reaches the real
    connection, which is the point: the rollback under test must be a real one.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
        if sql.strip().upper().startswith("DELETE FROM FACTS"):
            raise sqlite3.OperationalError("disk I/O error (injected)")
        return self._conn.execute(sql, *args)

    def __getattr__(self, name: str) -> object:
        return getattr(self._conn, name)


async def test_a_mid_transaction_failure_rolls_everything_back(
    gate_and_conn: tuple[MemoryGate, sqlite3.Connection],
) -> None:
    gate, conn = gate_and_conn
    await _seed(gate)
    before = _counts(conn)
    assert before["memories"] == 1 and before["facts"] == 1, before

    # Fail partway: let the memory/FTS/entity deletes through, then blow up on the facts
    # DELETE -- the worst case, with rows already removed inside the open transaction.
    episodics = gate._store._episodics  # type: ignore[attr-defined]
    episodics._conn = _FailsOnFactsDelete(conn)
    try:
        with pytest.raises(sqlite3.OperationalError):
            await gate.forget(user_id="u", project="p")
    finally:
        episodics._conn = conn

    assert _counts(conn) == before, "forget() left the database half-erased"


async def test_the_happy_path_still_erases_everything(
    gate_and_conn: tuple[MemoryGate, sqlite3.Connection],
) -> None:
    """The other half of the guarantee -- otherwise a no-op forget() would pass the test above."""
    gate, conn = gate_and_conn
    await _seed(gate)

    await gate.forget(user_id="u", project="p")

    assert _counts(conn) == {
        "memories": 0,
        "fts_memories": 0,
        "memory_entities": 0,
        "facts": 0,
    }
