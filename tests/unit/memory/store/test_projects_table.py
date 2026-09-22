"""``projects`` -- one row per repository: seeded from what a database already holds,
registered by every write after that, and classified by a CLI write from inside the repository.

Migration step 7 (light: it only inserts into an empty table) seeds a row, classification
``unclassified``, for every project named in ``memories``, ``facts`` or ``session_history`` --
whichever of the three a database actually has. After it, a memory, a fact or a session-history
row registers its project in its own write transaction, so no project written to later goes
without a row. A recording fills ``classification``, ``remote`` and ``root`` and nothing else:
the owner's per-project switches (``capture_enabled``, ``retention_days``, ``paused_until``,
``consolidate_enabled``) are theirs.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

import pytest

from morgan_brain.composition import build_memory_context, build_memory_module, migration_stores
from morgan_brain.config import Settings
from morgan_brain.memory import migrations
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.knowledge.consolidation import MemoryConsolidator
from morgan_brain.memory.store import projects as projects_store
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.fts import FtsIndex
from morgan_brain.memory.store.history import SessionHistoryStore
from morgan_brain.models import Memory, Message, Role, TemporalFact
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
    rows = {p.name: p for p in projects_store.list_all(conn)}
    assert sorted(rows) == ["Morgan", "other", "personal"]
    assert all(p.classification == "unclassified" for p in rows.values())


def test_seeding_twice_inserts_nothing_the_second_time(tmp_path):
    conn = _a_version_six_database(tmp_path, memories_projects=["Morgan"])

    first = projects_store.seed(conn, clock=_clock)
    second = projects_store.seed(conn, clock=_clock)

    assert (first, second) == (1, 0)
    assert [p.name for p in projects_store.list_all(conn)] == ["Morgan"]


def test_seed_returns_zero_on_a_database_with_none_of_the_source_tables(tmp_path):
    conn = open_db(str(tmp_path / "empty.db"))
    projects_store.create_schema(conn)
    conn.commit()

    assert projects_store.seed(conn, clock=_clock) == 0
    assert projects_store.list_all(conn) == []


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


class _Clock:
    """A clock the test moves, so a second write is seen to happen later than the first."""

    def __init__(self) -> None:
        self.now = datetime(2026, 9, 22, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _writing_stack(tmp_path, clock: _Clock, *, read_only_reason: str | None = None):
    """The gate and the session history over one fresh database, as ``build_memory_context``
    wires them, with *clock* injected into both."""
    conn = open_db(str(tmp_path / "m.db"))
    module = build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4, clock=clock)
    gate = MemoryGate(module, read_only_reason=read_only_reason)
    return gate, SessionHistoryStore(conn, clock=clock), conn


async def test_a_memory_a_fact_and_a_history_row_each_register_their_project(tmp_path):
    """Three projects never written to before, one per project-keyed write. Each gets a row the
    moment it is written, ``unclassified`` and stamped by the injected clock: no remote or root,
    because nothing below the gate knows where a repository is."""
    clock = _Clock()
    gate, history, conn = _writing_stack(tmp_path, clock)
    assert projects_store.list_all(conn) == []

    await gate.store(Memory(user_id="u", project="alpha", content="The Harbor mirror is slow."))
    await gate.upsert_fact(
        TemporalFact(user_id="u", project="beta", subject="deploy", predicate="uses", object="x")
    )
    history.append("u:default", Message(user_id="u", role=Role.USER, content="hi"), project="gamma")

    rows = {p.name: p for p in projects_store.list_all(conn)}
    assert sorted(rows) == ["alpha", "beta", "gamma"]
    for row in rows.values():
        assert (row.classification, row.remote, row.root) == ("unclassified", None, None)
        assert row.created_at == clock.now.isoformat()
        assert (row.capture_enabled, row.consolidate_enabled) == (True, True)


async def test_writing_to_a_registered_project_again_changes_nothing_in_its_row(tmp_path):
    """``INSERT OR IGNORE``: a later write -- a day later, after the row was classified --
    neither restamps ``created_at`` nor resets what was recorded."""
    clock = _Clock()
    gate, history, conn = _writing_stack(tmp_path, clock)
    await gate.store(Memory(user_id="u", project="alpha", content="first"))
    await gate.record_project(
        user_id="u",
        project="alpha",
        classification="work",
        remote="https://gitlab.work.example/team/alpha.git",
        root="/src/alpha",
    )
    before = projects_store.get(conn, "alpha")

    clock.now += timedelta(days=1)
    await gate.store(Memory(user_id="u", project="alpha", content="second"))
    await gate.upsert_fact(
        TemporalFact(user_id="u", project="alpha", subject="deploy", predicate="uses", object="x")
    )
    history.append("u:default", Message(user_id="u", role=Role.USER, content="hi"), project="alpha")

    assert projects_store.get(conn, "alpha") == before
    assert [p.name for p in projects_store.list_all(conn)] == ["alpha"]


async def test_a_write_that_fails_leaves_no_row_behind(tmp_path, monkeypatch):
    """The row is written in the memory's own transaction: a store that fails part-way rolls
    the registration back with everything else, so no project is left with a row and no data."""
    gate, _history, conn = _writing_stack(tmp_path, _Clock())

    def failing_add(self, *args, **kwargs):
        raise RuntimeError("the keyword index failed")

    monkeypatch.setattr(FtsIndex, "add", failing_add)
    with pytest.raises(RuntimeError, match="keyword index"):
        await gate.store(Memory(user_id="u", project="alpha", content="lost"))

    assert projects_store.get(conn, "alpha") is None


async def test_a_forgotten_project_written_to_again_is_registered_again(tmp_path):
    """``forget`` deletes the row once the project is empty; the next write brings back a
    fresh ``unclassified`` one, with nothing the forgotten row carried."""
    clock = _Clock()
    gate, _history, conn = _writing_stack(tmp_path, clock)
    await gate.store(Memory(user_id="u", project="alpha", content="first"))
    await gate.record_project(
        user_id="u",
        project="alpha",
        classification="work",
        remote="https://gitlab.work.example/team/alpha.git",
        root="/src/alpha",
    )

    await gate.forget(user_id="u", project="alpha")
    assert projects_store.get(conn, "alpha") is None

    clock.now += timedelta(days=1)
    await gate.store(Memory(user_id="u", project="alpha", content="again"))
    row = projects_store.get(conn, "alpha")
    assert row is not None
    assert (row.classification, row.remote, row.root) == ("unclassified", None, None)
    assert row.created_at == clock.now.isoformat()


async def test_recording_updates_the_classification_remote_and_root_and_nothing_else(tmp_path):
    """The owner's switches belong to the owner; a recording recomputes only what derives
    from the repository and the settings."""
    gate, _history, conn = _writing_stack(tmp_path, _Clock())
    await gate.store(Memory(user_id="u", project="alpha", content="first"))
    conn.execute(
        "UPDATE projects SET capture_enabled = 0, retention_days = 30, "
        "paused_until = '2026-10-01T00:00:00+00:00', consolidate_enabled = 0 WHERE name = 'alpha'"
    )
    conn.commit()
    before = projects_store.get(conn, "alpha")

    recorded = await gate.record_project(
        user_id="u",
        project="alpha",
        classification="personal",
        remote="https://example.com/someone/alpha.git",
        root="/src/alpha",
    )

    after = projects_store.get(conn, "alpha")
    assert recorded is True
    assert (after.classification, after.remote, after.root) == (
        "personal",
        "https://example.com/someone/alpha.git",
        "/src/alpha",
    )
    assert (
        after.capture_enabled,
        after.retention_days,
        after.paused_until,
        after.consolidate_enabled,
        after.created_at,
    ) == (
        before.capture_enabled,
        before.retention_days,
        before.paused_until,
        before.consolidate_enabled,
        before.created_at,
    )


async def test_recording_a_project_that_has_no_row_creates_none(tmp_path):
    """A recording classifies the row a write registered; it never creates one. A command
    whose write wrote nothing -- or a project another process forgot in between -- is left
    without a row, so no remote or root outlives a forget."""
    gate, _history, conn = _writing_stack(tmp_path, _Clock())

    recorded = await gate.record_project(
        user_id="u",
        project="never-written",
        classification="work",
        remote="https://gitlab.work.example/team/never-written.git",
        root="/src/never-written",
    )

    assert recorded is False
    assert projects_store.get(conn, "never-written") is None


async def test_recording_is_refused_on_a_database_waiting_for_migrate(tmp_path):
    reason = "writes are blocked until `morgan migrate` runs: 1 step pending (8 a heavy step)"
    gate, _history, conn = _writing_stack(tmp_path, _Clock())
    await gate.store(Memory(user_id="u", project="alpha", content="first"))
    before = projects_store.get(conn, "alpha")
    read_only = MemoryGate(gate._store, read_only_reason=reason)

    with pytest.raises(migrations.DatabaseNeedsMigration) as exc:
        await read_only.record_project(
            user_id="u",
            project="alpha",
            classification="work",
            remote="https://gitlab.work.example/team/alpha.git",
            root="/src/alpha",
        )

    assert exc.value.reason == reason
    assert projects_store.get(conn, "alpha") == before
