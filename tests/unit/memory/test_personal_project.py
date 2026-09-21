"""Outside a repository, memories land in 'personal', and the result says so.

'default' named nothing: a chat with no folder, a command run from home and a misconfigured
client all landed in one project whose name told the owner nothing. Migration step 5 moves
every row filed under it to 'personal'; both surfaces report when a write landed there
because the caller named nothing, not because they asked for it.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from morgan_brain.composition import migration_stores
from morgan_brain.memory import migrations
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.entities import EntityIndex
from morgan_brain.memory.store.episodic import EpisodicStore
from morgan_brain.memory.store.fts import FtsIndex
from morgan_brain.memory.store.history import SessionHistoryStore
from morgan_brain.memory.store.tables import PROJECT_TABLES
from morgan_brain.memory.store.temporal import SqliteTemporalStore
from morgan_brain.memory.store.vectors import SqliteVectorIndex, VectorRecord
from morgan_brain.models import (
    PERSONAL_PROJECT,
    Memory,
    MemoryQuery,
    Message,
    Role,
    TemporalFact,
)
from morgan_brain.surfaces.cli.commands import cmd_remember
from morgan_brain.surfaces.cli.project import detect_project
from morgan_brain.surfaces.mcp_server import build_server
from tests.unit.memory.conftest import build_memory_module

_DIM = 4
_NOW = datetime(2026, 9, 1, tzinfo=UTC)


def test_outside_a_repository_detect_project_returns_none(tmp_path):
    """``detect_project`` only detects -- it never resolves to ``PERSONAL_PROJECT`` itself,
    or a repository literally named ``personal`` would be indistinguishable from no
    repository at all. Each surface resolves ``None`` at its own one call site."""
    assert detect_project(tmp_path) is None


async def test_remember_says_when_it_defaulted(tmp_path, settings_for_tmp):
    defaulted = await cmd_remember(argparse.Namespace(text="x"), settings_for_tmp, None)
    named = await cmd_remember(argparse.Namespace(text="y"), settings_for_tmp, "Morgan")

    assert defaulted["project"] == "personal" and defaulted["project_defaulted"] is True
    assert named["project"] == "Morgan" and named["project_defaulted"] is False


async def test_mcp_remember_without_a_project_reports_it_defaulted(settings_for_tmp):
    """The same rule on the second surface: an MCP client that sends no ``project`` gets
    back ``project_defaulted: true``, never a silent ``personal``."""
    server = build_server(settings_for_tmp)

    defaulted = await server.call_tool("remember", {"text": "x"})
    named = await server.call_tool("remember", {"text": "y", "project": "Morgan"})

    assert defaulted["project"] == "personal" and defaulted["project_defaulted"] is True
    assert named["project"] == "Morgan" and named["project_defaulted"] is False


async def test_mcp_remember_with_an_empty_project_lands_in_personal(settings_for_tmp):
    """An empty string is what some client sends for "no project", same as omitting the
    argument -- ``Memory.project`` rejects it outright (``min_length=1``) rather than
    defaulting it, so it must be folded into the same "named nothing" case as ``None``."""
    server = build_server(settings_for_tmp)

    result = await server.call_tool("remember", {"text": "x", "project": ""})

    assert result["project"] == "personal" and result["project_defaulted"] is True


def _stores(conn: sqlite3.Connection) -> migrations.Stores:
    """The stores ``morgan migrate`` opens before it runs the steps -- and no others."""
    return migration_stores(conn)


def _a_version_four_database_with_default_rows(tmp_path: Path) -> sqlite3.Connection:
    """Every project-keyed table, at the schema step 4 leaves (provenance columns already
    added, nothing renamed yet), with two rows filed under the old implicit project
    ``'default'`` in each -- ``vec_items`` and ``fts_memories`` included -- and one row under
    a real project, so the test can tell a row the rename moved from one it must leave alone.

    Built through the real store classes rather than raw SQL: today's stores already create
    step 4's schema (the provenance columns are in every ``CREATE TABLE``), so this is exactly
    what opening a pre-Task-12 database with this code looks like, one store at a time.
    """
    path = str(tmp_path / "old.db")
    conn = open_db(path)
    episodics = EpisodicStore(conn)
    entities = EntityIndex(conn)
    fts = FtsIndex(conn)
    vectors = SqliteVectorIndex(conn, dim=_DIM)
    temporal = SqliteTemporalStore(conn=conn)
    history = SessionHistoryStore(conn)

    async def _seed() -> None:
        for i in range(2):
            memory_id = f"m{i}"
            episodics.put(
                Memory(
                    id=memory_id,
                    user_id="u",
                    project="default",
                    content=f"harbor plan {i}",
                    created_at=_NOW,
                )
            )
            fts.add(memory_id, f"harbor plan {i}", user_id="u", project="default")
            entities.add(memory_id, ["Harbor"], user_id="u", project="default")
            await vectors.upsert(
                VectorRecord(
                    id=memory_id, user_id="u", project="default", vector=[1.0, 0.0, 0.0, 0.0]
                )
            )
            await temporal.upsert_fact(
                TemporalFact(
                    user_id="u",
                    project="default",
                    subject="user",
                    predicate="lives_in",
                    object=f"City{i}",
                ),
                now=_NOW,
            )
            history.append(
                "s1", Message(user_id="u", role=Role.USER, content=f"turn {i}"), project="default"
            )

    asyncio.run(_seed())

    # One row per table left under a real project -- proves the rename is scoped to
    # 'default' and doesn't sweep every row it touches.
    episodics.put(
        Memory(id="m-morgan", user_id="u", project="Morgan", content="kept", created_at=_NOW)
    )

    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    return conn


def test_step_five_moves_every_default_row_and_counts_them(tmp_path):
    conn = _a_version_four_database_with_default_rows(tmp_path)

    applied = migrations.migrate(conn, _stores(conn))

    counts = next(c for s, c in applied if s.number == 5)
    assert counts["memories"] == 2 and counts["fts_memories"] == 2 and counts["vec_items"] == 2
    remaining = conn.execute("SELECT COUNT(*) FROM memories WHERE project = 'default'").fetchone()[
        0
    ]
    assert remaining == 0


def test_step_five_moves_default_rows_in_every_project_keyed_table(tmp_path):
    """Every table ``PROJECT_TABLES`` names, not just the three the brief's own test checks --
    ``facts``, ``memory_entities`` and ``vec_meta`` included."""
    conn = _a_version_four_database_with_default_rows(tmp_path)

    applied = migrations.migrate(conn, _stores(conn))
    counts = next(c for s, c in applied if s.number == 5)

    assert counts == dict.fromkeys(PROJECT_TABLES, 2)
    for table in PROJECT_TABLES:
        remaining = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE project = 'default'"  # noqa: S608
        ).fetchone()[0]
        assert remaining == 0, table
    assert conn.execute("SELECT COUNT(*) FROM memories WHERE project = 'Morgan'").fetchone()[0] == 1


def test_recall_finds_the_moved_memories_under_personal(tmp_path):
    """The point of moving the rows, not just counting them: a query scoped to ``personal``
    after ``migrate`` finds what a query scoped to ``default`` used to find.

    Sync, like the vector-scoping test above -- ``_a_version_four_database_with_default_rows``
    already runs its own event loop to seed the async stores, and pytest-asyncio's own loop
    for an ``async def`` test can't host a second ``asyncio.run()`` inside it.
    """
    conn = _a_version_four_database_with_default_rows(tmp_path)
    migrations.migrate(conn, _stores(conn))
    conn.close()

    module = build_memory_module(str(tmp_path / "old.db"), dim=_DIM)

    async def run() -> list[str]:
        got = await module.recall(
            MemoryQuery(user_id="u", project=PERSONAL_PROJECT, text="harbor plan", top_k=8)
        )
        assert all(m.project == PERSONAL_PROJECT for m in got)
        return [m.id for m in got]

    # A superset, not an equality: the moved facts are surfaced too (alongside episodics,
    # never instead of them -- see module.py's `recall`), and the second upsert's key
    # collides with the first's, so exactly one of them stays current.
    assert {"m0", "m1"} <= set(asyncio.run(run()))
