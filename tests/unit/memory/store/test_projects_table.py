"""``projects`` -- seeded from what a database already holds, one row per repository.

Migration step 7 (light: it only inserts into an empty table) seeds a row, classification
``unclassified``, for every project named in ``memories``, ``facts`` or ``session_history`` --
whichever of the three a database actually has. The disk walk that would fill ``remote`` and
``root`` is phase 1a; here the row exists so the owner's per-project switches
(``capture_enabled``, ``retention_days``, ``consolidate_enabled``) have something to attach to
before that walk ever runs.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from morgan_brain.composition import build_memory_context, migration_stores
from morgan_brain.config import Settings
from morgan_brain.memory import migrations
from morgan_brain.memory.knowledge.consolidation import MemoryConsolidator
from morgan_brain.memory.store import projects as projects_store
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory
from morgan_brain.surfaces.cli.commands import cmd_consolidate


def _clock() -> datetime:
    return datetime(2026, 9, 21, tzinfo=UTC)


def _a_version_six_database(
    tmp_path,
    *,
    memories_projects: list[str] = (),
    facts_projects: list[str] | None = None,
    session_history_projects: list[str] | None = None,
):
    """A database at ``user_version`` 6: every step but 7 already ran. ``memories`` always
    exists (every Morgan has one); ``facts`` and ``session_history`` only when given, so a
    step 7 run here also exercises the "table not present" skip.
    """
    conn = open_db(str(tmp_path / "m.db"))
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, user_id TEXT, project TEXT)")
    for i, project in enumerate(memories_projects):
        conn.execute(
            "INSERT INTO memories (id, user_id, project) VALUES (?, 'u', ?)", (f"m{i}", project)
        )
    if facts_projects is not None:
        conn.execute("CREATE TABLE facts (id TEXT PRIMARY KEY, user_id TEXT, project TEXT)")
        for i, project in enumerate(facts_projects):
            conn.execute(
                "INSERT INTO facts (id, user_id, project) VALUES (?, 'u', ?)", (f"f{i}", project)
            )
    if session_history_projects is not None:
        conn.execute(
            "CREATE TABLE session_history "
            "(id INTEGER PRIMARY KEY, session_id TEXT, user_id TEXT, project TEXT)"
        )
        for i, project in enumerate(session_history_projects):
            conn.execute(
                "INSERT INTO session_history (id, session_id, user_id, project) "
                "VALUES (?, 's', 'u', ?)",
                (i, project),
            )
    # Step 3 already ran on a version-6 database: `projects` exists, empty.
    projects_store.create_schema(conn)
    conn.execute("PRAGMA user_version = 6")
    conn.commit()
    return conn


def test_step_seven_seeds_one_row_per_distinct_project_and_skips_absent_tables(tmp_path):
    """``session_history`` never exists here -- a version-6 database that has never opened one
    -- so it is skipped rather than erroring; ``facts`` and ``memories`` between them name three
    distinct projects, deduplicated."""
    conn = _a_version_six_database(
        tmp_path,
        memories_projects=["Morgan", "personal"],
        facts_projects=["Morgan", "other"],
    )
    stores = migration_stores(conn)

    migrations.upgrade(conn, stores)

    assert migrations.pending(conn) == ()
    rows = {p.name: p for p in projects_store.all(conn)}
    assert sorted(rows) == ["Morgan", "other", "personal"]
    assert all(p.classification == "unclassified" for p in rows.values())


def test_seeding_twice_inserts_nothing_the_second_time(tmp_path):
    conn = _a_version_six_database(tmp_path, memories_projects=["Morgan"])

    first = projects_store.seed(conn, clock=_clock)
    second = projects_store.seed(conn, clock=_clock)

    assert (first, second) == (1, 0)
    assert [p.name for p in projects_store.all(conn)] == ["Morgan"]


def test_seed_returns_zero_on_a_database_with_none_of_the_source_tables(tmp_path):
    conn = open_db(str(tmp_path / "empty.db"))
    projects_store.create_schema(conn)
    conn.commit()

    assert projects_store.seed(conn, clock=_clock) == 0
    assert projects_store.all(conn) == []


def test_capture_enabled_and_retention_days_round_trip(tmp_path):
    conn = open_db(str(tmp_path / "m.db"))
    projects_store.create_schema(conn)
    conn.execute(
        "INSERT INTO projects (name, capture_enabled, retention_days, created_at) "
        "VALUES ('Morgan', 0, 30, '2026-09-21T00:00:00+00:00')"
    )
    conn.commit()

    project = projects_store.get(conn, "Morgan")

    assert project is not None
    assert project.capture_enabled is False
    assert project.retention_days == 30


def test_get_returns_none_for_a_project_with_no_row(tmp_path):
    conn = open_db(str(tmp_path / "m.db"))
    projects_store.create_schema(conn)
    conn.commit()

    assert projects_store.get(conn, "no-such-project") is None


async def test_consolidate_all_projects_skips_a_project_with_consolidate_disabled(
    tmp_path, monkeypatch
):
    """The owner turned consolidation off for one project; ``--all-projects`` must not spend a
    model call proposing facts for it. A project the seed never reached (no ``projects`` row)
    is consolidated as it always was -- absence is not the same as being switched off."""
    settings = Settings(data_dir=str(tmp_path), embedding_backend="hash")
    ctx = build_memory_context(settings)
    await ctx.gate.store(Memory(user_id=settings.owner_user_id, project="on", content="a"))
    await ctx.gate.store(Memory(user_id=settings.owner_user_id, project="off", content="b"))
    await ctx.gate.store(Memory(user_id=settings.owner_user_id, project="unseeded", content="c"))
    projects_store.seed(ctx.conn, clock=_clock)
    ctx.conn.execute("UPDATE projects SET consolidate_enabled = 0 WHERE name = 'off'")
    ctx.conn.execute("DELETE FROM projects WHERE name = 'unseeded'")
    ctx.conn.commit()
    ctx.conn.close()

    seen: list[str] = []

    async def fake_consolidate(self, user_id: str, *, project: str):
        seen.append(project)
        return []

    monkeypatch.setattr(MemoryConsolidator, "consolidate", fake_consolidate)

    await cmd_consolidate(argparse.Namespace(all_projects=True), settings, "on")

    assert sorted(seen) == ["on", "unseeded"]
