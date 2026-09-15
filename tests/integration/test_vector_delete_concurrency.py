"""Deleting a vector never removes another id's vector, even while a second process writes.

`vec_meta.rowid` is an ``INTEGER PRIMARY KEY`` without ``AUTOINCREMENT``, so SQLite hands a
deleted row's rowid to the next insert when it was the highest. A delete that looks the rowid
up and deletes by it later can therefore erase whatever was stored under that rowid in between:
process A reads ``d``'s rowid, process B deletes ``d`` and stores ``k`` -- which reuses the
rowid -- and A's delete then removes ``k``.

The two writers move in lockstep, one id at a time, so every step offers that window: B
creates ``d-i``, then A deletes ``d-i`` while B deletes it too and stores ``k-i``.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import struct
from pathlib import Path
from typing import TYPE_CHECKING

from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.vectors import SqliteVectorIndex, VectorRecord

if TYPE_CHECKING:
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Barrier

_DIM = 4
_STEPS = 50
_TIMEOUT_S = 120


def _vector(i: int) -> list[float]:
    # Small integers survive the float32 round trip exactly.
    return [float(i), 1.0, 0.0, 0.0]


def _deleter(path: str, step: Barrier, outcomes: Queue[str]) -> None:
    """Runs in a spawned process: delete ``d-i`` at every step."""
    try:
        conn = open_db(path)
        index = SqliteVectorIndex(conn, dim=_DIM)

        async def run() -> None:
            for i in range(_STEPS):
                step.wait(timeout=_TIMEOUT_S)
                await index.delete([f"d-{i}"])

        asyncio.run(run())
        conn.close()
    except BaseException as exc:
        outcomes.put(f"deleter: {type(exc).__name__}: {exc}")
        raise
    outcomes.put("deleter: ok")


def _reinserter(path: str, step: Barrier, outcomes: Queue[str]) -> None:
    """Runs in a spawned process: create ``d-i``, then delete it and store ``k-i``."""
    try:
        conn = open_db(path)
        index = SqliteVectorIndex(conn, dim=_DIM)

        async def run() -> None:
            for i in range(_STEPS):
                await index.upsert(VectorRecord(id=f"d-{i}", user_id="u", vector=_vector(i)))
                step.wait(timeout=_TIMEOUT_S)
                await index.delete([f"d-{i}"])
                await index.upsert(VectorRecord(id=f"k-{i}", user_id="u", vector=_vector(i)))

        asyncio.run(run())
        conn.close()
    except BaseException as exc:
        outcomes.put(f"reinserter: {type(exc).__name__}: {exc}")
        raise
    outcomes.put("reinserter: ok")


def test_a_delete_racing_a_reinsert_never_removes_the_new_id(tmp_path: Path) -> None:
    path = str(tmp_path / "morgan.db")
    setup = open_db(path)
    SqliteVectorIndex(setup, dim=_DIM)
    setup.close()

    ctx = mp.get_context("spawn")
    step = ctx.Barrier(2)
    outcomes: Queue[str] = ctx.Queue()
    workers = [
        ctx.Process(target=_deleter, args=(path, step, outcomes)),
        ctx.Process(target=_reinserter, args=(path, step, outcomes)),
    ]
    for p in workers:
        p.start()
    try:
        reported = sorted(outcomes.get(timeout=_TIMEOUT_S) for _ in workers)
        for p in workers:
            p.join(timeout=_TIMEOUT_S)
    finally:
        for p in workers:
            if p.is_alive():
                p.terminate()

    assert reported == ["deleter: ok", "reinserter: ok"]
    assert [p.exitcode for p in workers] == [0, 0]

    conn = open_db(path)
    try:
        records = conn.execute(
            "SELECT m.id AS id, v.embedding AS embedding "
            "FROM vec_meta m JOIN vec_items v ON v.rowid = m.rowid"
        ).fetchall()
        item_rows = conn.execute("SELECT COUNT(*) FROM vec_items").fetchone()[0]
    finally:
        conn.close()

    stored = {r["id"]: list(struct.unpack(f"{_DIM}f", r["embedding"])) for r in records}
    lost = [f"k-{i}" for i in range(_STEPS) if stored.get(f"k-{i}") != _vector(i)]
    assert lost == [], f"{len(lost)} of {_STEPS} stored ids were erased by another id's delete"
    assert sorted(stored) == sorted(f"k-{i}" for i in range(_STEPS))
    assert item_rows == _STEPS
