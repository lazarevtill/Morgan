"""A database written by an older Morgan is brought up to date when it is opened.

Entities are derived from a memory's content once, when it is stored. Changing the rule changes
nothing already stored: without an upgrade, every memory written before the change keeps the
old names -- sentence openers included -- for as long as it exists.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from morgan_brain.memory import migrations
from morgan_brain.models import Memory
from tests.unit.memory.conftest import build_memory_module as _module

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
