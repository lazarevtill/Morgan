"""After forget, the forgotten words are nowhere in the database's files.

A DELETE leaves a row's words behind in more places than its table: FTS5 keeps a deleted
row's terms in its segment b-tree until a merge, and a write-ahead log keeps the frames that
wrote them until it is truncated. The owner's promise is that forget forgets, so these tests
read the raw bytes of the database file and its ``-wal`` file, not the tables.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.history import SessionHistoryStore
from morgan_brain.models import Memory, Message, Role, TemporalFact
from tests.unit.memory.conftest import build_memory_module

#: A token in no other fixture, which FTS5 indexes as one term.
_MARKER = "zqxjvforgetmarker"


def _clock() -> datetime:
    return datetime.now(UTC)


def _files_holding(path: Path, needle: bytes) -> list[str]:
    """The database file and its ``-wal`` file, whichever exist, that contain *needle*. Read
    as bytes from disk, never through a connection."""
    found = []
    for candidate in (path, path.with_name(path.name + "-wal")):
        if candidate.exists() and needle in candidate.read_bytes():
            found.append(candidate.name)
    return found


async def _remember(tmp_path: Path):
    """Memories, a fact, a history turn and the recorded repository carrying the marker in
    ``p``, and some memories in ``q``.

    The ``projects`` row is in this too because its ``remote`` and ``root`` are the owner's
    data as much as a memory's words are: a remote URL names the host they push to, and may
    carry a token."""
    path = tmp_path / "m.db"
    module = build_memory_module(str(path))
    conn = module._conn
    history = SessionHistoryStore(conn, clock=_clock)
    for n in range(3):
        await module.store(
            Memory(user_id="u", project="p", content=f"note {n}: the {_MARKER} is in the harbor")
        )
        await module.store(Memory(user_id="u", project="q", content=f"note {n}: the tide is out"))
    await module.upsert_fact(
        TemporalFact(user_id="u", project="p", subject="user", predicate="keeps", object=_MARKER)
    )
    history.append("u:s", Message(user_id="u", role=Role.USER, content=_MARKER), project="p")
    recorded = await module.record_project(
        "p",
        classification="work",
        remote=f"https://{_MARKER}.work.example/team/p.git",
        root=f"/src/{_MARKER}",
    )
    assert recorded, "the marker never reached the projects row"
    assert _files_holding(path, _MARKER.encode()), "the marker never reached the files"
    return path, module


@pytest.mark.parametrize("other_connection", [False, True], ids=["alone", "beside-morgan-mcp"])
async def test_the_forgotten_words_are_in_neither_the_file_nor_the_wal(tmp_path, other_connection):
    """``beside-morgan-mcp`` holds a second connection open, idle between requests, as a
    running ``morgan-mcp`` does."""
    path, module = await _remember(tmp_path)
    other = open_db(str(path)) if other_connection else None
    if other is not None:
        assert other.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 6

    await module.forget(user_id="u", project="p")

    assert _files_holding(path, _MARKER.encode()) == []
    assert _files_holding(path, b"the tide is out"), "q's memories went too"
    if other is not None:
        other.close()


async def test_a_wal_another_connection_is_reading_is_named_and_forget_still_succeeds(tmp_path):
    """A connection in the middle of a read keeps the log from being truncated. forget has
    already erased and committed by then, so it succeeds, and says which log it could not
    truncate."""
    path, module = await _remember(tmp_path)
    reader = open_db(str(path))
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM memories").fetchone()
    module._conn.execute("PRAGMA busy_timeout = 50")

    with capture_logs() as logs:
        report = await module.forget(user_id="u", project="p")

    assert report.memories == 3
    busy = [entry for entry in logs if entry["event"] == "forget.wal-not-truncated"]
    assert len(busy) == 1
    assert busy[0]["log_level"] == "warning"
    assert busy[0]["wal"] == str(path) + "-wal"
    reader.rollback()
    reader.close()
