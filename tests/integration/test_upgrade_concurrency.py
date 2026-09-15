"""Two processes opening one out-of-date database at once both start, and upgrade it once.

`morgan` and `morgan-mcp` open the same file, and each runs the upgrade when it does. The
count of steps done is read again once the write lock is held, so the process that waited
for the other's upgrade finds nothing left to do. Without that second read both would
re-extract every memory, one after the other.

Each process counts the extractions it ran and reports the count; together they must add up
to one pass over the memories.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory

if TYPE_CHECKING:
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Barrier

_DIM = 8
_MEMORIES = 400
_TIMEOUT_S = 60


async def _stale_database(path: str) -> None:
    """Memories whose stored entities an older rule wrote, from before the index was removed."""
    conn = open_db(path)
    module = build_memory_module(conn, embedder=FakeEmbedder(dim=_DIM), dim=_DIM)
    for i in range(_MEMORIES):
        await module.store(
            Memory(user_id="u", project="p", content=f"note {i}: the Harbor{i} mirror blocked it")
        )
    conn.execute('UPDATE memories SET entities = \'[{"name": "Note", "type": "unknown"}]\'')
    conn.execute("DELETE FROM memory_entities")
    conn.execute(
        "INSERT INTO memory_entities (memory_id, user_id, project, name) "
        "SELECT id, user_id, project, 'note' FROM memories"
    )
    conn.execute("CREATE TABLE mem_entity_nodes (user_id TEXT, project TEXT, name TEXT)")
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()


def _opener(path: str, start: Barrier, outcomes: Queue[tuple[str, int]]) -> None:
    """Runs in a spawned process: open the database the way every entrypoint does."""
    extractions = 0
    try:
        from morgan_brain.memory import migrations

        extract = migrations.extract_entity_names

        def counted(text: str) -> list[str]:
            nonlocal extractions
            extractions += 1
            return extract(text)

        migrations.extract_entity_names = counted
        conn = open_db(path)
        start.wait(_TIMEOUT_S)
        build_memory_module(conn, embedder=FakeEmbedder(dim=_DIM), dim=_DIM)
        conn.close()
    except BaseException as exc:
        outcomes.put((f"{type(exc).__name__}: {exc}", extractions))
        raise
    outcomes.put(("ok", extractions))


async def test_two_processes_opening_a_stale_database_upgrade_it_once(tmp_path: Path) -> None:
    path = str(tmp_path / "morgan.db")
    await _stale_database(path)

    ctx = mp.get_context("spawn")
    start = ctx.Barrier(2)
    outcomes: Queue[tuple[str, int]] = ctx.Queue()
    workers = [ctx.Process(target=_opener, args=(path, start, outcomes)) for _ in range(2)]
    for p in workers:
        p.start()
    try:
        reported = [outcomes.get(timeout=_TIMEOUT_S * 2) for _ in workers]
        for p in workers:
            p.join(timeout=_TIMEOUT_S)
    finally:
        for p in workers:
            if p.is_alive():
                p.terminate()

    assert [status for status, _ in reported] == ["ok", "ok"], reported
    assert sum(n for _, n in reported) == _MEMORIES, reported

    conn = open_db(path)
    try:
        names = {r[0] for r in conn.execute("SELECT DISTINCT name FROM memory_entities")}
        leftover = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'mem_entity_nodes'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert names == {f"harbor{i}" for i in range(_MEMORIES)}
    assert leftover == 0
