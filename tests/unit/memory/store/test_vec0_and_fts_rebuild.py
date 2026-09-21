"""The vector and keyword tables are rebuilt once, with the columns 1b will filter on.

FTS5 and vec0 cannot be ALTERed, so a column added later is another full rewrite. The blobs are
the only copy of every vector: a rebuild that loses one loses it for good.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from morgan_brain.composition import migration_stores
from morgan_brain.memory import migrations
from morgan_brain.memory.store import spaces, vectors
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemoryQuery, MemoryStatus, Scope
from tests.unit.memory.conftest import a_version_five_database, build_memory_module

_DIM = 4
_NOW = datetime(2026, 9, 21, tzinfo=UTC)


def test_the_rebuild_keeps_every_vector_byte_for_byte(tmp_path):
    conn = _a_version_five_database_with(tmp_path, memories=3)
    before = _embeddings(conn)

    applied = migrations.migrate(conn, _stores(conn))

    after = _embeddings(conn)
    assert after == before and len(after) == 3
    counts = {step.number: c for step, c in applied}
    assert counts[6] == {"vec_items": 3, "fts_memories": 3}


def test_the_new_columns_are_there_and_carry_the_memory_row(tmp_path):
    conn = _a_version_five_database_with(tmp_path, memories=1, scope="private", author="u")

    migrations.migrate(conn, _stores(conn))

    row = conn.execute("SELECT status, scope, author_id FROM vec_items").fetchone()
    assert (row["status"], row["scope"], row["author_id"]) == ("stored", "private", "u")
    assert {c["name"] for c in conn.execute("PRAGMA table_info(fts_memories)")} >= {
        "status",
        "scope",
        "author_id",
    }


def test_the_new_columns_are_copied_from_the_memory_not_defaulted(tmp_path):
    """Values no default produces, so a rebuild that wrote constants fails here."""
    conn = _a_version_five_database_with(
        tmp_path, memories=1, scope="shared", author="a-colleague", status="quarantined"
    )

    migrations.migrate(conn, _stores(conn))

    vec = conn.execute("SELECT status, scope, author_id FROM vec_items").fetchone()
    fts = conn.execute("SELECT status, scope, author_id FROM fts_memories").fetchone()
    assert tuple(vec) == tuple(fts) == ("quarantined", "shared", "a-colleague")


def test_a_fresh_database_gets_the_same_ddl_as_a_migrated_one(tmp_path):
    fresh = build_memory_module(str(tmp_path / "fresh.db"), dim=_DIM)._conn
    migrated = _a_version_five_database_with(tmp_path, memories=0)
    migrations.migrate(migrated, _stores(migrated))

    assert _ddl(fresh, "vec_items") == _ddl(migrated, "vec_items")
    assert _ddl(fresh, "fts_memories") == _ddl(migrated, "fts_memories")
    assert vectors.declared_width(migrated, table_name="vec_items") == _DIM


def test_the_step_registers_no_space_it_cannot_name(tmp_path):
    """A step is handed the connection and the stores, never the settings, so it cannot know
    which model wrote the vectors. ``morgan migrate`` registers the settings' space at the end
    of its wave (``tests/integration/test_migrate_cli.py``); a bare wave leaves none."""
    conn = _a_version_five_database_with(tmp_path, memories=1)

    migrations.migrate(conn, _stores(conn))

    assert spaces.active(conn) is None


def test_a_failure_later_in_the_wave_leaves_the_old_tables_and_every_vector(tmp_path):
    conn = _a_version_five_database_with(tmp_path, memories=3)
    vec_ddl, fts_ddl = _ddl(conn, "vec_items"), _ddl(conn, "fts_memories")
    before = _embeddings(conn)

    def fail(conn: sqlite3.Connection, stores: migrations.Stores) -> None:
        raise RuntimeError("a later step failed")

    with pytest.raises(RuntimeError, match="a later step failed"):
        migrations.migrate(
            conn,
            _stores(conn),
            steps=(*migrations._STEPS, migrations.Step(99, "fails", True, fail)),
        )

    assert (_ddl(conn, "vec_items"), _ddl(conn, "fts_memories")) == (vec_ddl, fts_ddl)
    assert _embeddings(conn) == before
    assert conn.execute("SELECT COUNT(*) FROM fts_memories").fetchone()[0] == 3
    assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 5


def test_a_failure_inside_step_six_leaves_the_old_tables_and_every_vector(tmp_path, monkeypatch):
    """The failure this time is *inside* step 6 itself, not a later step: ``vec_items`` has
    already been dropped, recreated and reinserted, and ``fts_memories`` has too, when the
    count check on ``fts_memories`` raises. The whole wave -- including the already-rebuilt
    ``vec_items`` -- must still come back exactly as it was, not just from the point of the
    raise, and the connection must be out of its transaction afterwards."""
    conn = _a_version_five_database_with(tmp_path, memories=3)
    vec_ddl, fts_ddl = _ddl(conn, "vec_items"), _ddl(conn, "fts_memories")
    before_vectors = _embeddings(conn)
    before_fts = _fts_rows(conn)
    before_version = int(conn.execute("PRAGMA user_version").fetchone()[0])

    real_require_every_row = migrations._require_every_row

    def _fail_for_fts_memories(conn: sqlite3.Connection, table: str, expected: int) -> int:
        if table == "fts_memories":
            raise RuntimeError("fts_memories short by design (injected for the test)")
        return real_require_every_row(conn, table, expected)

    monkeypatch.setattr(migrations, "_require_every_row", _fail_for_fts_memories)

    with pytest.raises(RuntimeError, match="injected for the test"):
        migrations.migrate(conn, _stores(conn))

    assert (_ddl(conn, "vec_items"), _ddl(conn, "fts_memories")) == (vec_ddl, fts_ddl)
    assert _embeddings(conn) == before_vectors
    assert _fts_rows(conn) == before_fts
    assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == before_version
    assert conn.in_transaction is False


async def test_recall_answers_from_both_rebuilt_indexes(tmp_path):
    conn = _a_version_five_database_with(tmp_path, memories=3)
    migrations.migrate(conn, _stores(conn))
    conn.close()
    module = build_memory_module(str(tmp_path / "old.db"), dim=_DIM)

    vector_hits = await module._vectors.search(
        user_id="u", vector=_vector("memory number 1"), top_k=1, project="p"
    )
    keyword_hits = module._fts.search("number", user_id="u", top_k=3, project="p")
    recalled = await module.recall(
        MemoryQuery(user_id="u", project="p", text="memory number 1", top_k=3)
    )

    assert [h.id for h in vector_hits] == ["m1"]
    assert sorted(keyword_hits) == ["m0", "m1", "m2"]
    assert "m1" in [m.id for m in recalled]


async def test_a_stored_memory_writes_its_status_scope_and_author_to_both_indexes(tmp_path):
    module = build_memory_module(str(tmp_path / "m.db"), dim=_DIM)

    memory_id = await module.store(
        Memory(
            user_id="u",
            project="p",
            content="the harbor mirror",
            scope=Scope.SHARED,
            author_id="a-colleague",
            status=MemoryStatus.QUARANTINED,
        )
    )

    expected = ("quarantined", "shared", "a-colleague")
    vec = module._conn.execute(
        "SELECT v.status, v.scope, v.author_id FROM vec_items v "
        "JOIN vec_meta m ON m.rowid = v.rowid WHERE m.id = ?",
        (memory_id,),
    ).fetchone()
    fts = module._conn.execute(
        "SELECT status, scope, author_id FROM fts_memories WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    assert tuple(vec) == expected and tuple(fts) == expected


def test_a_database_without_the_vector_table_samples_nothing_and_holds_no_vector(tmp_path):
    """``morgan migrate`` registers the settings' space on a file that may never have had a
    vector table; the first-call check then reads a table that is not there yet."""
    conn = open_db(str(tmp_path / "bare.db"))

    assert vectors.stored_sample(conn, table_name="vec_items", n=5) == []
    assert vectors.holds_vectors(conn, table_name="vec_items") is False


def _a_version_five_database_with(
    tmp_path: Path,
    *,
    memories: int,
    scope: str = "private",
    author: str = "u",
    status: str = "stored",
) -> sqlite3.Connection:
    return a_version_five_database(
        str(tmp_path / "old.db"),
        dim=_DIM,
        memories=[
            Memory(
                id=f"m{i}",
                user_id="u",
                project="p",
                content=f"memory number {i}",
                created_at=_NOW,
                scope=Scope(scope),
                author_id=author,
                status=MemoryStatus(status),
            )
            for i in range(memories)
        ],
        vector=_vector,
    )


def _vector(text: str) -> list[float]:
    """A distinct direction per memory: ``memory number i`` points along axis *i*."""
    axis = int(text.rsplit(" ", 1)[-1]) % _DIM
    return [1.0 if i == axis else 0.1 for i in range(_DIM)]


def _embeddings(conn: sqlite3.Connection) -> dict[int, bytes]:
    return {
        r["rowid"]: r["embedding"] for r in conn.execute("SELECT rowid, embedding FROM vec_items")
    }


def _fts_rows(conn: sqlite3.Connection) -> dict[int, tuple[str, str, str, str]]:
    return {
        r["rowid"]: (r["memory_id"], r["user_id"], r["project"], r["content"])
        for r in conn.execute(
            "SELECT rowid, memory_id, user_id, project, content FROM fts_memories"
        )
    }


def _stores(conn: sqlite3.Connection) -> migrations.Stores:
    """The stores ``morgan migrate`` opens before it runs the steps -- and no others."""
    return migration_stores(conn)


def _ddl(conn: sqlite3.Connection, table: str) -> str:
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (table,)).fetchone()
    return str(row["sql"])
