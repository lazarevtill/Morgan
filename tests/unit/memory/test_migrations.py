"""A database written by an older Morgan is brought up to date when it is opened.

Entities are derived from a memory's content once, when it is stored. Changing the rule changes
nothing already stored: without an upgrade, every memory written before the change keeps the
old names -- sentence openers included -- for as long as it exists.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from morgan_brain.composition import build_memory_context, migration_stores
from morgan_brain.config import Settings
from morgan_brain.memory import migrations
from morgan_brain.memory.store import projects as projects_store
from morgan_brain.memory.store.db import SQLiteTooOld, open_readonly
from morgan_brain.models import Memory
from tests.unit.memory.conftest import (
    a_version_seven_database,
    a_version_two_database,
)
from tests.unit.memory.conftest import (
    build_memory_module as _module,
)

CONTENTS = ["Проверь конфиг. Теперь Harbor упал.", "Install the chart. Then ask Kafka again."]
#: What the rule before this one extracted from each: every sentence opener was a name.
OLD_NAMES = [["Проверь", "Теперь", "Harbor"], ["Install", "Kafka"]]
NEW_NAMES = [["Harbor"], ["Kafka"]]


async def _written_under_the_old_rule(path: str) -> list[str]:
    """Store the memories, then put back what the old rule wrote, as a database from before."""
    module = _module(path)
    ids = [await module.store(Memory(user_id="u", project="p", content=c)) for c in CONTENTS]
    conn = module._conn
    conn.execute("DELETE FROM memory_entities")
    for memory_id, names in zip(ids, OLD_NAMES, strict=True):
        conn.executemany(
            "INSERT INTO memory_entities (memory_id, user_id, project, name) "
            "VALUES (?, 'u', 'p', ?)",
            [(memory_id, n.lower()) for n in names],
        )
        conn.execute(
            "UPDATE memories SET entities = ? WHERE id = ?",
            (json.dumps([{"name": n, "type": "unknown"} for n in names]), memory_id),
        )
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()
    return ids


def _indexed_names(module, memory_id: str) -> list[str]:
    rows = module._conn.execute(
        "SELECT name FROM memory_entities WHERE memory_id = ? ORDER BY name", (memory_id,)
    )
    return [r["name"] for r in rows]


async def test_opening_an_old_database_re_extracts_every_memory(tmp_path):
    path = str(tmp_path / "m.db")
    ids = await _written_under_the_old_rule(path)

    module = _module(path)

    for memory_id, names in zip(ids, NEW_NAMES, strict=True):
        assert _indexed_names(module, memory_id) == sorted(n.lower() for n in names)
        stored = await module.get(memory_id, user_id="u")
        assert stored is not None
        assert [e.name for e in stored.entities] == names


async def test_an_upgrade_that_fails_part_way_leaves_the_old_rows(tmp_path, monkeypatch):
    path = str(tmp_path / "m.db")
    ids = await _written_under_the_old_rule(path)
    calls = 0

    def fails_on_the_second_memory(text: str) -> list[str]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("extraction failed")
        return ["Replaced"]

    monkeypatch.setattr(migrations, "extract_entity_names", fails_on_the_second_memory)
    with pytest.raises(RuntimeError, match="extraction failed"):
        _module(path)
    monkeypatch.undo()

    # One memory was rewritten before the failure; that rewrite did not survive it.
    raw = sqlite3.connect(path)
    for memory_id, names in zip(ids, OLD_NAMES, strict=True):
        indexed = raw.execute(
            "SELECT name FROM memory_entities WHERE memory_id = ? ORDER BY name", (memory_id,)
        ).fetchall()
        stored = raw.execute("SELECT entities FROM memories WHERE id = ?", (memory_id,)).fetchone()
        assert [r[0] for r in indexed] == sorted(n.lower() for n in names)
        assert [e["name"] for e in json.loads(stored[0])] == names
    raw.close()

    module = _module(path)  # the next open runs the whole upgrade again, with the real rule
    for memory_id, names in zip(ids, NEW_NAMES, strict=True):
        assert _indexed_names(module, memory_id) == sorted(n.lower() for n in names)


#: The retired semantic index's tables, as a database written before its removal holds them.
_SEMANTIC_INDEX_TABLES = (
    "CREATE TABLE mem_schemas (user_id TEXT, project TEXT, name TEXT)",
    "CREATE TABLE mem_entity_nodes (user_id TEXT, project TEXT, name TEXT, schema_name TEXT)",
    "CREATE TABLE mem_entity_edges (user_id TEXT, project TEXT, src TEXT)",
    "CREATE TABLE mem_schema_edges (user_id TEXT, project TEXT, src TEXT)",
)


async def test_opening_a_database_drops_the_retired_semantic_index(tmp_path):
    """Its entity names are the owner's words. Nothing reads or erases them any more, so a
    table left behind would keep them on disk through every ``forget``."""
    path = str(tmp_path / "m.db")
    module = _module(path)
    conn = module._conn
    for create in _SEMANTIC_INDEX_TABLES:
        conn.execute(create)
    conn.execute("INSERT INTO mem_entity_nodes VALUES ('u', 'p', 'harbor', 'work')")
    conn.execute("PRAGMA user_version = 1")  # re-extracted, from before the index was removed
    conn.commit()
    conn.close()

    reopened = _module(path)._conn
    left = reopened.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'mem\\_%' ESCAPE '\\'"
    ).fetchall()
    assert [r["name"] for r in left] == []


async def test_the_upgrade_runs_once(tmp_path, monkeypatch):
    path = str(tmp_path / "m.db")
    await _written_under_the_old_rule(path)
    _module(path)._conn.close()

    def must_not_run(text: str) -> list[str]:
        raise AssertionError("an upgraded database was re-extracted again")

    monkeypatch.setattr(migrations, "extract_entity_names", must_not_run)
    _module(path)


ARCHIVE_TABLES = (
    "sessions",
    "turns",
    "turns_fts",
    "corrections_fts",
    "turn_links",
    "capture_cursors",
    "capture_exclusions",
    "capture_pauses",
    "capture_state",
    "call_log",
    "digests",
    "digest_refs",
    "digest_ratings",
    "link_ratings",
)
#: Step 8's columns: the last of each table, in this order.
NEW_COLUMNS = {
    "projects": ["retention_confirmed"],
    "memories": ["redactions", "flags"],
    "facts": ["redactions", "flags"],
}


def _settings(tmp_path, name: str) -> Settings:
    return Settings(data_dir=str(tmp_path / name), embedding_backend="hash")


def _version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")]


def _shape(conn: sqlite3.Connection) -> dict[str, list[tuple[object, ...]]]:
    """Every table's ``(cid, name, type, notnull, dflt_value, pk)`` rows, plus the index names,
    for a table-by-table comparison of a fresh file with a migrated one (the ``sql`` text of an
    ``ALTER``ed table differs in whitespace from a ``CREATE``, its columns do not)."""
    tables = sorted(
        str(r["name"])
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    )
    shape = {
        table: [tuple(r) for r in conn.execute(f"PRAGMA table_info({table})")] for table in tables
    }
    shape["<indexes>"] = [
        (str(r["name"]),)
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' ORDER BY name")
    ]
    return shape


def _fts_option(conn: sqlite3.Connection, table: str) -> int | None:
    # `table` is one of the two FTS table names, never caller input.
    row = conn.execute(f"SELECT v FROM {table}_config WHERE k = 'secure-delete'").fetchone()  # noqa: S608
    return None if row is None else int(row[0])


def _fresh_shape(tmp_path) -> dict[str, list[tuple[object, ...]]]:
    fresh = build_memory_context(_settings(tmp_path, "fresh"))
    try:
        assert _version(fresh.conn) == 8
        return _shape(fresh.conn)
    finally:
        fresh.conn.close()


def test_a_version_seven_file_reaches_eight_on_open_with_the_tables_and_the_columns(tmp_path):
    """The owner's live path: phase 0's `migrate` leaves the file at 7, and the first open of
    this build runs step 8 itself, after the SQLite check."""
    a_version_seven_database(str(tmp_path / "seven" / "morgan.db"))
    ctx = build_memory_context(_settings(tmp_path, "seven"))
    try:
        conn = ctx.conn
        assert _version(conn) == 8
        assert ctx.gate.read_only_reason is None
        names = {str(r["name"]) for r in conn.execute("SELECT name FROM sqlite_master")}
        assert set(ARCHIVE_TABLES) <= names
        for table, columns in NEW_COLUMNS.items():
            assert _column_names(conn, table)[-len(columns) :] == columns
        assert _fts_option(conn, "turns_fts") == 1 and _fts_option(conn, "corrections_fts") == 1
        assert conn.execute("SELECT retention_confirmed FROM projects").fetchone()[0] == 0
        assert tuple(conn.execute("SELECT redactions, flags FROM memories").fetchone()) == (
            "[]",
            "[]",
        )
        migrated = _shape(conn)
    finally:
        ctx.conn.close()
    assert migrated == _fresh_shape(tmp_path)


def test_a_second_open_of_the_migrated_file_changes_nothing(tmp_path):
    a_version_seven_database(str(tmp_path / "seven" / "morgan.db"))
    build_memory_context(_settings(tmp_path, "seven")).conn.close()
    raw = sqlite3.connect(str(tmp_path / "seven" / "morgan.db"))
    before = raw.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
    raw.close()
    again = build_memory_context(_settings(tmp_path, "seven"))
    try:
        assert _version(again.conn) == 8
        after = again.conn.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
        assert [tuple(r) for r in after] == before
        assert again.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    finally:
        again.conn.close()


def test_a_version_two_file_opens_read_only_at_three_and_reaches_eight_under_migrate(tmp_path):
    a_version_two_database(str(tmp_path / "two" / "morgan.db"))
    ctx = build_memory_context(_settings(tmp_path, "two"))
    try:
        conn = ctx.conn
        assert _version(conn) == 3
        assert ctx.gate.read_only_reason is not None
        assert "morgan migrate" in ctx.gate.read_only_reason
        assert [s.number for s in migrations.pending(conn)] == [4, 5, 6, 7, 8]
        assert "redactions" not in _column_names(conn, "memories")
        assert "retention_confirmed" in _column_names(conn, "projects")  # ProjectStore made it

        applied = migrations.migrate(conn, migration_stores(conn))

        assert [step.number for step, _ in applied] == [4, 5, 6, 7, 8]
        assert _version(conn) == 8
        for table, columns in NEW_COLUMNS.items():
            assert _column_names(conn, table)[-len(columns) :] == columns
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1
        migrated = _shape(conn)
    finally:
        ctx.conn.close()
    assert migrated == _fresh_shape(tmp_path)


def test_a_project_row_without_the_column_reads_as_not_confirmed(tmp_path):
    """``doctor`` reads a version-7 file over ``open_readonly`` and runs no step."""
    path = str(tmp_path / "seven" / "morgan.db")
    a_version_seven_database(path)
    conn = open_readonly(path)
    try:
        [project] = projects_store.list_all(conn)
        assert (project.name, project.retention_confirmed) == ("p", False)
        assert _version(conn) == 7
    finally:
        conn.close()


def test_an_old_sqlite_is_refused_by_name_before_any_step(tmp_path, monkeypatch):
    a_version_seven_database(str(tmp_path / "seven" / "morgan.db"))
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 40, 1))
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.40.1")
    with pytest.raises(SQLiteTooOld) as raised:
        build_memory_context(_settings(tmp_path, "seven"))
    assert "3.40.1" in str(raised.value) and "3.42.0" in str(raised.value)
    monkeypatch.undo()
    raw = sqlite3.connect(str(tmp_path / "seven" / "morgan.db"))
    assert raw.execute("PRAGMA user_version").fetchone()[0] == 7
    raw.close()
