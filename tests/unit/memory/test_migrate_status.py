"""Heavy steps do not run because a client opened the database.

A step that rewrites rows inside a client's open has no snapshot behind it and no way back.
Until it is run by morgan migrate, the database opens read-only and every write says so.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from morgan_brain.composition import build_memory_context
from morgan_brain.config import Settings
from morgan_brain.memory import migrations
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.entities import EntityIndex
from morgan_brain.memory.store.episodic import EpisodicStore
from morgan_brain.models import Memory, TemporalFact
from tests.unit.memory.conftest import build_memory_module

_RAN: list[str] = []

LIGHT = migrations.Step(3, "a light step", False, lambda c, s: _RAN.append("light"))
HEAVY = migrations.Step(4, "a heavy step", True, lambda c, s: _RAN.append("heavy"))


def test_a_light_step_runs_on_open_and_a_heavy_one_waits(tmp_path):
    _RAN.clear()
    conn = build_memory_module(str(tmp_path / "m.db"))._conn
    conn.execute("PRAGMA user_version = 2")

    migrations.upgrade(conn, _stores(conn), steps=(*_existing(), LIGHT, HEAVY))

    assert _RAN == ["light"]
    assert _version(conn) == 3
    assert [s.name for s in migrations.pending(conn, steps=(*_existing(), LIGHT, HEAVY))] == [
        "a heavy step"
    ]


async def test_a_write_on_a_read_only_context_says_why(tmp_path):
    gate = MemoryGate(
        build_memory_module(str(tmp_path / "m.db")),
        read_only_reason="writes are blocked until `morgan migrate` runs: 1 step pending "
        "(4 a heavy step)",
    )

    with pytest.raises(migrations.DatabaseNeedsMigration) as exc:
        await gate.store(Memory(user_id="u", project="p", content="x"))

    assert "morgan migrate" in str(exc.value)
    assert "4 a heavy step" in str(exc.value)


async def test_recall_still_answers_on_the_old_schema(tmp_path):
    module = build_memory_module(str(tmp_path / "m.db"))
    await MemoryGate(module).store(Memory(user_id="u", project="p", content="Harbor upgrade"))
    read_only = MemoryGate(module, read_only_reason="1 step pending")

    from morgan_brain.models import MemoryQuery

    assert await read_only.recall(MemoryQuery(user_id="u", project="p", text="Harbor"))


def test_the_steps_are_numbered_from_one_and_the_first_two_are_light():
    """Step *n* brings a database from ``user_version`` *n - 1* to *n*, so a gap or a repeat
    would leave a step that no database is ever at the version to run."""
    assert [s.number for s in migrations._STEPS] == list(range(1, len(migrations._STEPS) + 1))
    assert [(s.number, s.name, s.heavy) for s in _existing()] == [
        (1, "reextract entities", False),
        (2, "drop the semantic index", False),
    ]


def test_a_light_step_behind_a_heavy_one_waits_with_it(tmp_path):
    """Steps run strictly in order: a light step written for the schema a heavy step leaves
    cannot run before it."""
    _RAN.clear()
    conn = build_memory_module(str(tmp_path / "m.db"))._conn
    conn.execute("PRAGMA user_version = 2")
    steps = (
        *_existing(),
        migrations.Step(3, "a heavy step", True, lambda c, s: _RAN.append("heavy")),
        migrations.Step(4, "a light step", False, lambda c, s: _RAN.append("light")),
    )

    migrations.upgrade(conn, _stores(conn), steps=steps)

    assert _RAN == []
    assert _version(conn) == 2
    assert [s.number for s in migrations.pending(conn, steps=steps)] == [3, 4]


def test_opening_a_blocked_database_does_not_wait_for_the_write_lock(tmp_path):
    """An open that has nothing it may run must not queue behind another process's write,
    or every client of a database awaiting `morgan migrate` stalls on each open."""
    path = str(tmp_path / "m.db")
    conn = build_memory_module(path)._conn
    conn.execute("PRAGMA user_version = 2")
    conn.execute("PRAGMA busy_timeout = 0")
    stores = _stores(conn)
    writer = sqlite3.connect(path)
    writer.execute("BEGIN IMMEDIATE")
    try:
        migrations.upgrade(conn, stores, steps=(*_existing(), HEAVY))
    finally:
        writer.rollback()
        writer.close()

    assert _version(conn) == 2


async def test_migrate_runs_every_pending_step_and_returns_what_each_touched(tmp_path):
    module = build_memory_module(str(tmp_path / "m.db"))
    await module.store(Memory(user_id="u", project="p", content="Harbor upgrade"))
    conn = module._conn
    conn.execute("PRAGMA user_version = 2")

    def touch_every_memory(c: sqlite3.Connection, s: migrations.Stores) -> dict[str, int]:
        return {"memories": c.execute("UPDATE memories SET importance = importance").rowcount}

    steps = (
        *_existing(),
        migrations.Step(3, "touch every memory", True, touch_every_memory),
        migrations.Step(4, "count nothing", False, lambda c, s: None),
    )

    applied = migrations.migrate(conn, _stores(conn), steps=steps)

    assert [(s.number, counts) for s, counts in applied] == [(3, {"memories": 1}), (4, {})]
    assert _version(conn) == 4
    assert migrations.pending(conn, steps=steps) == ()


def test_a_failing_step_rolls_the_whole_wave_back(tmp_path):
    conn = build_memory_module(str(tmp_path / "m.db"))._conn
    conn.execute("PRAGMA user_version = 2")

    def fails(c: sqlite3.Connection, s: migrations.Stores) -> None:
        raise RuntimeError("step four failed")

    steps = (
        *_existing(),
        migrations.Step(3, "make a table", False, lambda c, s: c.execute("CREATE TABLE t (x)")),
        migrations.Step(4, "fail", True, fails),
    )

    with pytest.raises(RuntimeError, match="step four failed"):
        migrations.migrate(conn, _stores(conn), steps=steps)

    assert _version(conn) == 2
    made = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 't'")
    assert made.fetchone() is None


_HEAVY_THREE = migrations.Step(3, "a heavy step", True, lambda c, s: None)
_LIGHT_FOUR = migrations.Step(4, "a light step", False, lambda c, s: None)


@pytest.mark.parametrize(
    ("injected", "reason"),
    [
        (
            (_HEAVY_THREE,),
            "writes are blocked until `morgan migrate` runs: 1 step pending (3 a heavy step)",
        ),
        (
            (_HEAVY_THREE, _LIGHT_FOUR),
            "writes are blocked until `morgan migrate` runs: 2 steps pending "
            "(3 a heavy step, 4 a light step)",
        ),
    ],
)
async def test_a_blocked_context_names_every_pending_step(tmp_path, monkeypatch, injected, reason):
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    monkeypatch.setattr(migrations, "_STEPS", _existing())
    build_memory_context(Settings()).conn.close()  # a database at user_version 2
    monkeypatch.setattr(migrations, "_STEPS", (*_existing(), *injected))

    ctx = build_memory_context(Settings())
    try:
        with pytest.raises(migrations.DatabaseNeedsMigration) as exc:
            await ctx.gate.store(Memory(user_id="u", project="p", content="x"))
    finally:
        ctx.conn.close()

    assert str(exc.value) == reason
    assert exc.value.reason == reason


async def test_every_write_through_a_read_only_gate_is_refused(tmp_path):
    gate = MemoryGate(build_memory_module(str(tmp_path / "m.db")), read_only_reason="blocked")
    writes: list[Callable[[], Awaitable[Any]]] = [
        lambda: gate.store(Memory(user_id="u", project="p", content="x")),
        lambda: gate.upsert_fact(
            TemporalFact(user_id="u", project="p", subject="s", predicate="is", object="o")
        ),
        lambda: gate.close_fact("f", user_id="u", project="p"),
        lambda: gate.set_confidence("f", user_id="u", project="p", value=0.5),
        lambda: gate.forget(user_id="u", project="p"),
    ]

    for write in writes:
        with pytest.raises(migrations.DatabaseNeedsMigration, match="blocked"):
            await write()


def _add_a_column_the_schema_already_has(c: sqlite3.Connection, s: migrations.Stores) -> None:
    """What a provenance step does to a database written before it: fails on a new one."""
    c.execute("ALTER TABLE memories ADD COLUMN content TEXT")


async def test_a_new_database_starts_at_the_codes_version_and_opens_writable(tmp_path, monkeypatch):
    """A new file is created by this code at the schema its last step leaves. No step has
    anything to re-derive in it, and a heavy one would open it read-only on its first day and
    then fail on the columns the stores had just created."""
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    heavy = migrations.Step(3, "a heavy step", True, _add_a_column_the_schema_already_has)
    monkeypatch.setattr(migrations, "_STEPS", (*_existing(), heavy))

    ctx = build_memory_context(Settings())
    try:
        assert ctx.gate.read_only_reason is None
        assert _version(ctx.conn) == 3
        assert migrations.pending(ctx.conn) == ()
        await ctx.gate.store(Memory(user_id="u", project="p", content="the first day"))
    finally:
        ctx.conn.close()


def test_a_version_zero_database_that_holds_tables_still_gets_its_steps(tmp_path, monkeypatch):
    """``user_version`` 0 alone does not say "new": a database written before step 1 existed
    reads 0 too, and it is the one every step was written for."""
    _RAN.clear()
    path = str(tmp_path / "m.db")
    conn = build_memory_module(path)._conn
    conn.execute("PRAGMA user_version = 0")
    conn.close()
    monkeypatch.setattr(migrations, "_STEPS", (*_existing(), LIGHT))

    reopened = build_memory_module(path)._conn

    assert _RAN == ["light"]
    assert _version(reopened) == 3


def _existing() -> tuple[migrations.Step, ...]:
    """The steps every database written before phase 0 has been through: ``user_version`` 2.

    A slice, not the whole of ``_STEPS``: the steps later tasks append must not collide with
    the numbers these tests give the steps they inject after these two."""
    return migrations._STEPS[:2]


def _stores(conn: sqlite3.Connection) -> migrations.Stores:
    return migrations.Stores(episodics=EpisodicStore(conn), entities=EntityIndex(conn))


def _version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])
