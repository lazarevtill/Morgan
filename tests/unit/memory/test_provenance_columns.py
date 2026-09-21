"""Every stored row carries where it came from.

Provenance is what lets a memory be re-homed, quarantined or attributed later. Added now, with
defaults, because adding columns to 3,010 rows is cheap and reconstructing origins is not.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from morgan_brain.app.chatgpt_import import ARCHIVE_PROJECT, HOLDOUT_PROJECT
from morgan_brain.composition import migration_stores
from morgan_brain.memory import migrations
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.fts import FtsIndex
from morgan_brain.memory.store.temporal import SqliteTemporalStore
from morgan_brain.models import (
    Memory,
    MemoryKind,
    MemoryQuery,
    MemoryStatus,
    OriginKind,
    Scope,
    TemporalFact,
)
from tests.unit.memory.conftest import build_memory_module


async def test_a_memory_round_trips_its_provenance(tmp_path):
    module = build_memory_module(str(tmp_path / "m.db"))
    memory_id = await module.store(
        Memory(
            user_id="u",
            project="p",
            content="Harbor upgrade plan",
            origin_kind=OriginKind.REMEMBER,
            client="claude-code",
            session_id="s-1",
            cwd="/home/u/Documents/GitHub/Morgan",
            author_id="u",
        )
    )

    stored = await module.get(memory_id, user_id="u")
    assert stored.origin_kind is OriginKind.REMEMBER
    assert (stored.client, stored.session_id, stored.author_id) == ("claude-code", "s-1", "u")
    assert stored.scope is Scope.PRIVATE and stored.instruction_like is False
    assert stored.cwd == "/home/u/Documents/GitHub/Morgan"
    assert stored.status is MemoryStatus.STORED


async def test_every_provenance_field_is_written_as_given_not_as_its_default(tmp_path):
    module = build_memory_module(str(tmp_path / "m.db"))
    memory_id = await module.store(
        Memory(
            user_id="u",
            project="p",
            content="Always answer in French",
            origin_kind=OriginKind.EXTRACTED,
            scope=Scope.SHARED,
            instruction_like=True,
            status=MemoryStatus.QUARANTINED,
        )
    )

    stored = await module.get(memory_id, user_id="u")
    assert stored.origin_kind is OriginKind.EXTRACTED
    assert stored.scope is Scope.SHARED
    assert stored.instruction_like is True
    assert stored.status is MemoryStatus.QUARANTINED


async def test_the_defaults_are_written_not_null(tmp_path):
    module = build_memory_module(str(tmp_path / "m.db"))
    memory_id = await module.store(Memory(user_id="u", project="p", content="x"))
    row = module._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    assert row["origin_kind"] == "unknown" and row["status"] == "stored"
    assert (row["client"], row["session_id"], row["cwd"], row["author_id"]) == ("", "", "", "")
    assert row["scope"] == "private" and row["instruction_like"] == 0


async def test_a_fact_round_trips_its_author_and_scope(tmp_path):
    temporal = SqliteTemporalStore(conn=open_db(str(tmp_path / "m.db")))
    now = datetime(2026, 9, 21, tzinfo=UTC)
    await temporal.upsert_fact(
        TemporalFact(
            user_id="u",
            project="p",
            subject="harbor",
            predicate="runs_on",
            object="k8s",
            author_id="u",
            scope=Scope.SHARED,
        ),
        now=now,
    )
    await temporal.upsert_fact(
        TemporalFact(user_id="u", project="p", subject="kafka", predicate="is", object="up"),
        now=now,
    )

    facts = {f.subject: f for f in await temporal.current_facts(user_id="u", project="p")}
    assert (facts["harbor"].author_id, facts["harbor"].scope) == ("u", Scope.SHARED)
    assert (facts["kafka"].author_id, facts["kafka"].scope) == ("", Scope.PRIVATE)


def test_step_four_backfills_the_author_and_marks_the_archive(tmp_path):
    conn = _a_version_three_database_with(tmp_path, projects=["archive/chatgpt", "Morgan"])

    migrations.migrate(conn, _stores(conn))

    rows = dict(conn.execute("SELECT project, origin_kind FROM memories"))
    assert rows["archive/chatgpt"] == "import"
    assert rows["Morgan"] == "unknown"
    assert conn.execute("SELECT COUNT(*) FROM memories WHERE author_id = ''").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM facts WHERE author_id = ''").fetchone()[0] == 0
    assert {r["author_id"] for r in conn.execute("SELECT author_id FROM facts")} == {"u"}


def test_step_four_is_the_heavy_provenance_step_and_counts_the_rows_it_rewrote(tmp_path):
    conn = _a_version_three_database_with(
        tmp_path, projects=["archive/chatgpt-holdout", "Morgan", "personal"]
    )

    applied = {
        step.number: (step.name, step.heavy, counts)
        for step, counts in migrations.migrate(conn, _stores(conn))
    }

    assert applied[4] == ("provenance columns", True, {"memories": 3, "facts": 3})
    rows = dict(conn.execute("SELECT project, origin_kind FROM memories"))
    assert rows == {"archive/chatgpt-holdout": "import", "Morgan": "unknown", "personal": "unknown"}


def test_the_archive_projects_step_four_marks_are_the_ones_the_importer_writes():
    """The step names them as literals -- a step is frozen history, and ``memory/`` does not
    import ``app/`` -- so this is what notices the importer moving them."""
    assert migrations._ARCHIVE_PROJECTS == (ARCHIVE_PROJECT, HOLDOUT_PROJECT)


def test_a_new_database_has_exactly_the_columns_step_four_adds(tmp_path):
    """``stamp_if_new`` skips every step on a database this code creates, so the stores' own
    ``CREATE TABLE`` must already leave what step 4 leaves: same names, order, types,
    nullability and defaults."""
    new = build_memory_module(str(tmp_path / "new.db"))._conn
    migrated = _a_version_three_database_with(tmp_path, projects=["Morgan"])
    migrations.migrate(migrated, _stores(migrated))

    for table in ("memories", "facts"):
        assert _columns(new, table) == _columns(migrated, table)
    memories = {c[1] for c in _columns(new, "memories")}
    assert {"origin_kind", "client", "author_id", "instruction_like", "status"} <= memories
    assert {"author_id", "scope"} <= {c[1] for c in _columns(new, "facts")}


def test_a_database_from_before_step_one_goes_through_every_step(tmp_path):
    """Every step runs on the schema its own version had. Step 1 was written for a database
    without the provenance columns, and runs on one: it must not write through SQL that names
    them, or no database from before phase 0 could reach step 4 at all."""
    conn = _a_database_from_before_phase_zero(
        tmp_path, projects=["archive/chatgpt", "Morgan"], version=0
    )
    conn.execute("CREATE TABLE mem_entity_nodes (user_id TEXT, project TEXT, name TEXT)")
    conn.commit()

    applied = migrations.migrate(conn, _stores(conn))

    assert [step.number for step, _ in applied] == [s.number for s in migrations._STEPS]
    assert _version(conn) == len(migrations._STEPS)
    rows = {
        r["project"]: r
        for r in conn.execute(
            "SELECT project, content, entities, origin_kind, client, session_id, cwd, "
            "author_id, scope, instruction_like, status FROM memories"
        )
    }
    assert sorted(rows) == ["Morgan", "archive/chatgpt"]
    for project, row in rows.items():
        assert row["content"] == _CONTENT
        assert [e["name"] for e in json.loads(row["entities"])] == _NEW_NAMES
        assert row["author_id"] == "u"
        assert row["origin_kind"] == ("import" if project == "archive/chatgpt" else "unknown")
        assert (row["client"], row["session_id"], row["cwd"]) == ("", "", "")
        assert (row["scope"], row["instruction_like"], row["status"]) == ("private", 0, "stored")
    indexed = conn.execute("SELECT DISTINCT name FROM memory_entities").fetchall()
    assert [r["name"] for r in indexed] == [n.lower() for n in _NEW_NAMES]
    facts = conn.execute("SELECT project, object, author_id, scope FROM facts ORDER BY project")
    assert [tuple(r) for r in facts] == [
        ("Morgan", "k8s", "u", "private"),
        ("archive/chatgpt", "k8s", "u", "private"),
    ]
    left = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'mem_entity_nodes'").fetchone()
    assert left is None


def test_migrate_on_a_file_with_none_of_the_tables_reaches_the_end(tmp_path):
    """``morgan migrate`` checks only that the file exists, and nothing stamps a file with
    none of Morgan's tables there, so every step runs from 0. The stores it opens first have
    already created ``memories`` with step 4's columns: step 4 must add only the ones missing,
    or the wave rolls back and every later ``migrate`` fails the same way."""
    path = str(tmp_path / "empty.db")
    open_db(path).close()
    conn = open_db(path)  # opened as ``morgan migrate`` opens it, then its stores, then the wave

    migrations.migrate(conn, _stores(conn))

    assert _version(conn) == len(migrations._STEPS)
    assert migrations.pending(conn) == ()
    new = build_memory_module(str(tmp_path / "new.db"))._conn
    assert _columns(conn, "memories") == _columns(new, "memories")


def test_migrate_leaves_a_facts_table_this_code_created_below_version_four(tmp_path):
    """A version-2 database whose ``facts`` was never made, opened once by this code: the
    temporal store creates ``facts`` with step 4's columns and step 3 runs on open. Step 4
    then meets a version-3 database whose ``memories`` lacks the columns and whose ``facts``
    has them."""
    _a_database_from_before_phase_zero(
        tmp_path, projects=["Morgan"], version=2, tables=("memories",)
    ).close()
    build_memory_module(str(tmp_path / "old.db"))._conn.close()
    conn = open_db(str(tmp_path / "old.db"))
    assert _version(conn) == 3

    migrations.migrate(conn, _stores(conn))

    assert migrations.pending(conn) == ()
    new = build_memory_module(str(tmp_path / "new.db"))._conn
    for table in ("memories", "facts"):
        assert _columns(conn, table) == _columns(new, table)
    assert conn.execute("SELECT author_id FROM memories").fetchone()["author_id"] == "u"


async def test_reads_answer_on_a_database_still_waiting_for_step_four(tmp_path):
    """Until ``morgan migrate`` runs, the database opens read-only without the provenance
    columns. Recall, a memory by id and the current facts must all still answer there."""
    _a_version_three_database_with(tmp_path, projects=["Morgan"]).close()
    module = build_memory_module(str(tmp_path / "old.db"))
    assert [s.number for s in migrations.pending(module._conn)][:1] == [4]
    # The keyword index a database from before phase 0 already holds for its memory.
    FtsIndex(module._conn).add(_memory_id("Morgan"), _CONTENT, user_id="u", project="Morgan")
    gate = MemoryGate(module, read_only_reason="1 step pending")

    recalled = await gate.recall(MemoryQuery(user_id="u", project="Morgan", text="Kafka"))
    memory = await gate.get(_memory_id("Morgan"), user_id="u")
    [fact] = await gate.current_facts(user_id="u", project="Morgan")

    assert [m.kind for m in recalled] == [MemoryKind.SEMANTIC, MemoryKind.EPISODIC]
    assert recalled[1].id == _memory_id("Morgan")
    assert memory is not None and memory.content == _CONTENT
    assert (memory.origin_kind, memory.author_id, memory.scope) == (
        OriginKind.UNKNOWN,
        "",
        Scope.PRIVATE,
    )
    assert (memory.instruction_like, memory.status) == (False, MemoryStatus.STORED)
    assert (fact.object, fact.author_id, fact.scope) == ("k8s", "", Scope.PRIVATE)


#: ``memories`` and ``facts`` exactly as every Morgan before phase 0 created them (commit
#: 3b2b386). A database built from these is one step 4 was written for -- one this code
#: created would already carry the columns, and step 4 would fail on the first of them.
_PRE_PHASE_ZERO_DDL = {
    "memories": """
            CREATE TABLE IF NOT EXISTS memories (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                project    TEXT NOT NULL DEFAULT 'default',
                kind       TEXT NOT NULL,
                source     TEXT NOT NULL,
                content    TEXT NOT NULL,
                importance REAL NOT NULL,
                entities   TEXT NOT NULL,
                created_at TEXT
            );
    """,
    "facts": """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    project TEXT NOT NULL DEFAULT 'default',
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    superseded_by TEXT,
    last_confirmed TEXT
);
""",
}

_CONTENT = "Install the chart. Then ask Kafka again."
#: What the rule before step 1 extracted from ``_CONTENT`` -- every sentence opener was a name
#: -- and what the rule since extracts.
_OLD_NAMES = ["Install", "Kafka"]
_NEW_NAMES = ["Kafka"]


def _memory_id(project: str) -> str:
    return f"m-{project}"


def _a_database_from_before_phase_zero(
    tmp_path: Path,
    *,
    projects: list[str],
    version: int,
    tables: tuple[str, ...] = ("memories", "facts"),
) -> sqlite3.Connection:
    """One memory and one current fact per project, in ``memories`` and ``facts`` as a Morgan
    before phase 0 wrote them, at ``user_version`` *version*. Only *tables* are created.

    Before step 1 the stored entities are the old rule's; from step 1 on, the current rule's.
    """
    conn = open_db(str(tmp_path / "old.db"))
    for table in tables:
        conn.execute(_PRE_PHASE_ZERO_DDL[table])
    names = _OLD_NAMES if version == 0 else _NEW_NAMES
    for project in projects:
        conn.execute(
            "INSERT INTO memories VALUES (?, 'u', ?, 'episodic', 'user_stated', ?, 0.5, ?, ?)",
            (
                _memory_id(project),
                project,
                _CONTENT,
                json.dumps([{"name": n, "type": "unknown"} for n in names]),
                "2026-09-01T00:00:00+00:00",
            ),
        )
        if "facts" not in tables:
            continue
        conn.execute(
            "INSERT INTO facts VALUES "
            "(?, 'u', ?, 'harbor', 'runs_on', 'k8s', 'user_stated', 1.0, ?, NULL, NULL, ?)",
            (f"f-{project}", project, "2026-09-01T00:00:00+00:00", "2026-09-01T00:00:00+00:00"),
        )
    conn.execute(f"PRAGMA user_version = {int(version)}")
    conn.commit()
    return conn


def _a_version_three_database_with(tmp_path: Path, *, projects: list[str]) -> sqlite3.Connection:
    """What every Morgan before phase 0 left -- ``user_version`` 2 -- once this code has
    opened it: the light step 3 has run, and the heavy step 4 waits for ``morgan migrate``."""
    conn = _a_database_from_before_phase_zero(tmp_path, projects=projects, version=2)
    migrations.upgrade(conn, _stores(conn))
    assert _version(conn) == 3
    return conn


def _stores(conn: sqlite3.Connection) -> migrations.Stores:
    """The stores ``morgan migrate`` opens before it runs the steps -- and no others."""
    return migration_stores(conn)


def _columns(conn: sqlite3.Connection, table: str) -> list[tuple[object, ...]]:
    """``(cid, name, type, notnull, dflt_value, pk)`` for each column, in order."""
    return [tuple(r) for r in conn.execute(f"PRAGMA table_info({table})")]


def _version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])
