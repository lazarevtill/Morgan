"""Two processes upserting the same ids into one database both finish, with whole records.

`morgan-mcp` holds the database open while the `morgan` CLI or `morgan import` writes to it,
each through its own `open_db` connection. `SqliteVectorIndex.upsert` looks an id up and then
writes, so unless the lookup and the write are one atomic step, two processes storing the same
id both see it absent, both insert, and the second dies on `UNIQUE constraint failed:
vec_meta.id`.

The writers are spawned rather than forked, so each opens an independent connection on every
platform. Each writer tags its vector and its payload with its own number: were both to write
identical records, a row whose metadata came from one writer and whose embedding came from the
other would pass unnoticed.
"""

from __future__ import annotations

import asyncio
import json
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
_IDS = 300
_WRITERS = 2
_TIMEOUT_S = 120


def _vector(i: int, writer: int) -> list[float]:
    # Small integers survive the float32 round trip exactly, so the comparison can be exact.
    return [float(i), float(writer), 1.0, 0.0]


def _write_all(path: str, writer: int, start: Barrier, outcomes: Queue[str]) -> None:
    """Runs in a spawned process: upsert every id, then report how it went."""
    try:
        conn = open_db(path)
        index = SqliteVectorIndex(conn, dim=_DIM)
        start.wait(timeout=_TIMEOUT_S)

        async def upsert_every_id() -> None:
            for i in range(_IDS):
                await index.upsert(
                    VectorRecord(
                        id=f"m-{i}",
                        user_id="u",
                        vector=_vector(i, writer),
                        payload={"writer": writer},
                    )
                )

        asyncio.run(upsert_every_id())
        conn.close()
    except BaseException as exc:
        outcomes.put(f"writer {writer}: {type(exc).__name__}: {exc}")
        raise
    outcomes.put(f"writer {writer}: ok")


def test_two_processes_upserting_the_same_ids_both_finish_with_whole_records(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "morgan.db")
    # The schema exists before the writers start, so the race under test is the upsert, not
    # two processes creating tables at once.
    setup = open_db(path)
    SqliteVectorIndex(setup, dim=_DIM)
    setup.close()

    ctx = mp.get_context("spawn")
    start = ctx.Barrier(_WRITERS)
    outcomes: Queue[str] = ctx.Queue()
    writers = [
        ctx.Process(target=_write_all, args=(path, w, start, outcomes)) for w in range(_WRITERS)
    ]
    for p in writers:
        p.start()
    try:
        reported = sorted(outcomes.get(timeout=_TIMEOUT_S) for _ in writers)
        for p in writers:
            p.join(timeout=_TIMEOUT_S)
    finally:
        for p in writers:
            if p.is_alive():
                p.terminate()

    assert reported == [f"writer {w}: ok" for w in range(_WRITERS)]
    assert [p.exitcode for p in writers] == [0] * _WRITERS

    conn = open_db(path)
    try:
        meta_rows = conn.execute("SELECT COUNT(*) FROM vec_meta").fetchone()[0]
        item_rows = conn.execute("SELECT COUNT(*) FROM vec_items").fetchone()[0]
        records = conn.execute(
            "SELECT m.id AS id, m.payload AS payload, v.embedding AS embedding "
            "FROM vec_meta m JOIN vec_items v ON v.rowid = m.rowid"
        ).fetchall()
    finally:
        conn.close()

    # One record per id, and no embedding left behind without its metadata row.
    assert meta_rows == item_rows == _IDS
    assert sorted(r["id"] for r in records) == sorted(f"m-{i}" for i in range(_IDS))
    # Every stored vector is the one written for its id by the writer its payload names.
    torn = [
        r["id"]
        for r in records
        if list(struct.unpack(f"{_DIM}f", r["embedding"]))
        != _vector(int(r["id"].removeprefix("m-")), json.loads(r["payload"])["writer"])
    ]
    assert torn == []
