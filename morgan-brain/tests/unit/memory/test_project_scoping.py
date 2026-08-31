"""Project scoping: the dimension that keeps a company-repo memory unreachable from a personal
one and vice versa. A leak here is worse than losing a memory (see task-12 brief) — these tests
cover the domain model, the KNN-level vector scoping, and the store-level migration path."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from morgan_brain.models.memory import Memory, MemoryQuery, TemporalFact
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.sqlite_vector import SqliteVectorIndex
from morgan_brain.modules.memory.stores.vector import VectorRecord
from tests.unit.memory.conftest import build_memory_module as _module


async def test_recall_defaults_to_the_query_project(tmp_path):
    path = str(tmp_path / "m.db")
    m = _module(path)
    await m.store(Memory(user_id="u", project="acme", content="harbor mirror note"))
    await m.store(Memory(user_id="u", project="personal", content="harbor sailing note"))
    got = await m.recall(MemoryQuery(user_id="u", project="acme", text="harbor"))
    assert [x.content for x in got] == ["harbor mirror note"]


async def test_all_projects_crosses_the_boundary(tmp_path):
    path = str(tmp_path / "m.db")
    m = _module(path)
    await m.store(Memory(user_id="u", project="acme", content="harbor mirror note"))
    await m.store(Memory(user_id="u", project="personal", content="harbor sailing note"))
    got = await m.recall(MemoryQuery(user_id="u", text="harbor", all_projects=True, top_k=10))
    assert len(got) == 2


async def test_project_is_required_to_be_non_empty():
    with pytest.raises(ValidationError):
        Memory(user_id="u", project="", content="x")


async def test_recall_defaults_to_the_default_project_when_unspecified(tmp_path):
    """Memories stored without an explicit project land in DEFAULT_PROJECT, and a query that
    also doesn't specify a project only ever sees that project — not a project picked at
    random by another caller."""
    path = str(tmp_path / "m.db")
    m = _module(path)
    await m.store(Memory(user_id="u", content="unscoped note about harbor"))
    await m.store(Memory(user_id="u", project="acme", content="scoped note about harbor"))
    got = await m.recall(MemoryQuery(user_id="u", text="harbor"))
    assert [x.content for x in got] == ["unscoped note about harbor"]


async def test_facts_are_project_scoped(tmp_path):
    path = str(tmp_path / "m.db")
    m = _module(path)
    await m.upsert_fact(
        TemporalFact(
            user_id="u", project="acme", subject="user", predicate="works_on", object="ACME"
        )
    )
    await m.upsert_fact(
        TemporalFact(
            user_id="u", project="personal", subject="user", predicate="works_on", object="a boat"
        )
    )
    acme_facts = await m.current_facts(user_id="u", project="acme")
    assert [f.object for f in acme_facts] == ["ACME"]
    all_facts = await m.current_facts(user_id="u", project=None)
    assert {f.object for f in all_facts} == {"ACME", "a boat"}


def test_vector_scoping_happens_inside_the_knn_not_after(tmp_path):
    """Regression for the vec0 crowding trap (same shape as Task 8's user_id regression, but
    for project): with two projects interleaved and top_k=2, a correct implementation returns
    both of the query project's vectors, not one plus a vector belonging to the other project."""
    idx = SqliteVectorIndex(open_db(str(tmp_path / "m.db")), dim=4)

    async def run() -> None:
        await idx.upsert(
            VectorRecord(id="acme-exact", user_id="u", project="acme", vector=[1, 0, 0, 0])
        )
        await idx.upsert(
            VectorRecord(id="personal-exact", user_id="u", project="personal", vector=[1, 0, 0, 0])
        )
        await idx.upsert(
            VectorRecord(id="acme-near", user_id="u", project="acme", vector=[0.9, 0.1, 0, 0])
        )
        await idx.upsert(
            VectorRecord(
                id="personal-near", user_id="u", project="personal", vector=[0.9, 0.1, 0, 0]
            )
        )
        hits = await idx.search(user_id="u", vector=[1, 0, 0, 0], top_k=2, project="acme")
        assert [h.id for h in hits] == ["acme-exact", "acme-near"]

    import asyncio

    asyncio.run(run())


async def test_migration_backfills_default_project_on_a_pre_existing_database(tmp_path):
    """A database written by pre-Task-12 Morgan has no project column anywhere. Reopening it
    with the new store classes must migrate in place -- ALTER for regular tables, drop+rebuild
    for the FTS5/vec0 virtual tables -- and existing data must land in DEFAULT_PROJECT and stay
    fully queryable, including the vector signal."""
    path = str(tmp_path / "m.db")
    conn = open_db(path)
    conn.executescript(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL,
            source TEXT NOT NULL, content TEXT NOT NULL, importance REAL NOT NULL,
            entities TEXT NOT NULL, created_at TEXT
        );
        INSERT INTO memories VALUES
            ('m1', 'u', 'episodic', 'user_stated', 'old harbor note', 0.5, '[]',
             '2026-01-01T00:00:00');

        CREATE TABLE facts (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, subject TEXT NOT NULL,
            predicate TEXT NOT NULL, object TEXT NOT NULL, source TEXT NOT NULL,
            confidence REAL NOT NULL, valid_from TEXT, valid_to TEXT,
            superseded_by TEXT, last_confirmed TEXT
        );
        INSERT INTO facts VALUES
            ('f1', 'u', 'user', 'lives_in', 'Berlin', 'user_stated', 1.0,
             '2026-01-01T00:00:00', NULL, NULL, '2026-01-01T00:00:00');

        CREATE VIRTUAL TABLE fts_memories USING fts5(
            memory_id UNINDEXED, user_id UNINDEXED, content,
            tokenize = 'unicode61 remove_diacritics 2'
        );
        INSERT INTO fts_memories (memory_id, user_id, content)
        VALUES ('m1', 'u', 'old harbor note');

        CREATE TABLE memory_entities (
            memory_id TEXT NOT NULL, user_id TEXT NOT NULL, name TEXT NOT NULL,
            PRIMARY KEY (memory_id, name)
        );

        CREATE TABLE vec_meta (
            rowid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL,
            user_id TEXT NOT NULL, payload TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE vec_items USING vec0(
            embedding float[4] distance_metric=cosine,
            user_id TEXT
        );
        """
    )
    conn.commit()
    import json
    import struct

    vec = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
    conn.execute(
        "INSERT INTO vec_meta (rowid, id, user_id, payload) VALUES (1, 'm1', 'u', ?)",
        (json.dumps({"content": "old harbor note", "user_id": "u"}),),
    )
    conn.execute("INSERT INTO vec_items (rowid, embedding, user_id) VALUES (1, ?, 'u')", (vec,))
    conn.commit()
    conn.close()

    m = _module(path)
    got = await m.recall(MemoryQuery(user_id="u", text="harbor"))
    assert "old harbor note" in [x.content for x in got]
    facts = await m.current_facts(user_id="u")
    assert [f.object for f in facts] == ["Berlin"]
