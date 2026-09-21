"""The embedding space a database's vectors belong to is recorded, not assumed.

Morgan checks only the vector width today, so swapping one 4,096-wide model for another leaves
every stored vector searched by a model that never wrote them: wrong answers, no error.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from morgan_brain.composition import migration_stores
from morgan_brain.memory import migrations
from morgan_brain.memory.store import spaces
from morgan_brain.memory.store.db import open_db
from tests.unit.memory.conftest import build_memory_module


def _clock() -> datetime:
    return datetime(2026, 9, 21, tzinfo=UTC)


def test_one_space_can_be_active(tmp_path):
    conn = build_memory_module(str(tmp_path / "m.db"))._conn
    spaces.register(
        conn, model="qwen3-embedding:8b", dims=4096, table_name="vec_items", clock=_clock
    )

    # The partial unique index on status = 'active' is what enforces this, not application code.
    with pytest.raises(sqlite3.IntegrityError):
        spaces.register(conn, model="other", dims=1024, table_name="vec_items_2", clock=_clock)


def test_a_shadow_space_is_allowed_beside_the_active_one(tmp_path):
    conn = build_memory_module(str(tmp_path / "m.db"))._conn
    spaces.register(
        conn, model="qwen3-embedding:8b", dims=4096, table_name="vec_items", clock=_clock
    )
    spaces.register(
        conn,
        model="local-0.6b",
        dims=1024,
        table_name="vec_items_2",
        status="shadow",
        clock=_clock,
    )

    assert spaces.active(conn).model == "qwen3-embedding:8b"


def test_a_fingerprint_round_trips(tmp_path):
    conn = build_memory_module(str(tmp_path / "m.db"))._conn
    space = spaces.register(conn, model="m", dims=3, table_name="vec_items", clock=_clock)

    spaces.record_fingerprint(conn, space.id, [[0.1, 0.2, 0.3]] * 5, clock=_clock)

    stored = spaces.active(conn)
    assert stored.fingerprint_recorded_at is not None
    assert spaces.unpack(stored.fingerprint, dims=3)[0] == pytest.approx([0.1, 0.2, 0.3])


def test_set_status_frees_the_active_slot_for_a_new_registration(tmp_path):
    conn = build_memory_module(str(tmp_path / "m.db"))._conn
    first = spaces.register(conn, model="m1", dims=4, table_name="vec_items", clock=_clock)

    spaces.set_status(conn, first.id, "retired")

    assert spaces.active(conn) is None
    second = spaces.register(conn, model="m2", dims=4, table_name="vec_items_2", clock=_clock)
    assert spaces.active(conn).id == second.id


def test_the_projects_table_exists_with_its_defaults(tmp_path):
    conn = build_memory_module(str(tmp_path / "m.db"))._conn
    conn.execute("INSERT INTO projects (name, created_at) VALUES ('p', '2026-09-21')")
    row = conn.execute("SELECT * FROM projects WHERE name = 'p'").fetchone()
    assert (row["classification"], row["capture_enabled"], row["consolidate_enabled"]) == (
        "unclassified",
        1,
        1,
    )


def _table_sql(conn, name: str) -> str | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return None if row is None else str(row["sql"])


def _index_sql(conn, name: str) -> str | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?", (name,)
    ).fetchone()
    return None if row is None else str(row["sql"])


def _old_database_missing_both_tables(tmp_path, name: str) -> str:
    """A version-2 database with neither table -- what step 3 must build from nothing."""
    path = str(tmp_path / name)
    build_memory_module(path)._conn.close()
    setup = open_db(path)
    setup.executescript(
        "DROP INDEX IF EXISTS idx_embedding_spaces_one_active;"
        "DROP TABLE IF EXISTS embedding_spaces;"
        "DROP TABLE IF EXISTS projects;"
    )
    setup.execute("PRAGMA user_version = 2")
    setup.commit()
    setup.close()
    return path


def test_a_fresh_database_and_a_migrated_one_end_with_the_same_schema(tmp_path):
    """Step 3 creates ``embedding_spaces`` and ``projects`` on a database opened for the first
    time (``stamp_if_new`` skips every step, so the stores must build the same tables step 3
    would have built) and on an old database that runs step 3 through ``morgan migrate`` --
    a raw connection with only the migration's own stores (``episodics``, ``entities``), the
    shape ``cmd_migrate`` opens; neither table is part of that trio, so step 3 alone must be
    able to create them from nothing. Both paths must produce byte-identical DDL, or a live
    upgrade would diverge from a new install.
    """
    fresh = build_memory_module(str(tmp_path / "fresh.db"))._conn

    old_path = _old_database_missing_both_tables(tmp_path, "old.db")
    migrated = open_db(old_path)
    # Up to step 3 only: this is step 3's test, and the heavy steps after it are left pending
    # by ``upgrade``, so the pending list would name them too.
    through_three = migrations._STEPS[:3]
    assert [s.number for s in migrations.pending(migrated, through_three)] == [3]

    migrations.upgrade(migrated, migration_stores(migrated), steps=through_three)

    assert migrations.pending(migrated, through_three) == ()
    for table in ("embedding_spaces", "projects"):
        assert _table_sql(fresh, table) == _table_sql(migrated, table)
        assert _table_sql(fresh, table) is not None
    assert _index_sql(fresh, "idx_embedding_spaces_one_active") == _index_sql(
        migrated, "idx_embedding_spaces_one_active"
    )
    assert _index_sql(fresh, "idx_embedding_spaces_one_active") is not None


def test_a_later_failing_step_rolls_step_threes_ddl_back_too(tmp_path):
    """``create_schema`` must join the migration's own transaction rather than commit one of
    its own. ``conn.executescript`` (an early implementation this test caught) issues an
    implicit ``COMMIT`` before it runs anything -- inside ``write_transaction``'s ``BEGIN
    IMMEDIATE`` that ends the whole wave's transaction early, so a later step's failure rolls
    back everything from that point on but leaves step 3's tables and ``user_version``
    committed regardless. ``morgan migrate`` runs steps 3 to 7 in exactly one transaction
    (SPEC-phase0 §4), so a failure in a later step must undo step 3 as cleanly as any other.
    """
    old_path = _old_database_missing_both_tables(tmp_path, "old.db")
    conn = open_db(old_path)
    stores = migration_stores(conn)

    def _fail(c: sqlite3.Connection, s: migrations.Stores) -> None:
        raise RuntimeError("a later step failed")

    steps = (*migrations._STEPS[:3], migrations.Step(4, "fail", True, _fail))

    with pytest.raises(RuntimeError, match="a later step failed"):
        migrations.migrate(conn, stores, steps=steps)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert (
        conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'embedding_spaces'").fetchone()
        is None
    )
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'projects'").fetchone() is None
