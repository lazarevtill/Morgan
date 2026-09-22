"""Bring a database written by an older Morgan up to what this one writes.

Some stored data is derived by code rather than given by the caller -- a memory's entities
are extracted from its content when it is stored. When that code changes, nothing already
stored changes with it, so each such change adds a numbered step here that re-derives what
is stored.

A step is light or heavy. A light step only adds a table or a defaulted column, and runs when
the database is opened (``upgrade``). A heavy step rewrites, moves or deletes rows, and runs
only under ``morgan migrate`` (``migrate``), behind a snapshot: a client that merely opened
the file has no snapshot behind it and no way back. Steps run strictly in order, so a light
step queued behind a heavy one waits for ``migrate`` with it; until then the database opens
read-only and every write raises ``DatabaseNeedsMigration``.

A database this code creates starts at the code's version (``stamp_if_new``): the stores
write the schema the last step leaves, so no step has anything to do in it.

The number of steps a database has been through is SQLite's own ``user_version`` header
field. It is read and advanced inside the same write transaction as the steps, so two
processes opening one file at once run each step once, and a step that fails leaves both
the data and the counter as they were, for the next open or ``morgan migrate`` to retry.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from itertools import takewhile
from typing import NamedTuple

from morgan_brain.memory.knowledge.extract import extract_entity_names
from morgan_brain.memory.store import projects, spaces, vectors
from morgan_brain.memory.store.db import write_transaction
from morgan_brain.memory.store.entities import EntityIndex
from morgan_brain.memory.store.episodic import EpisodicStore
from morgan_brain.memory.store.tables import PROJECT_TABLES, project_tables
from morgan_brain.models import PERSONAL_PROJECT, Entity


class Stores(NamedTuple):
    """The stores a step reads and writes, already open, so no step creates a table mid-upgrade."""

    episodics: EpisodicStore
    entities: EntityIndex


class Step(NamedTuple):
    """One numbered change to what is stored.

    *run* returns the rows it touched, per table, or ``None`` when it counts nothing;
    ``migrate`` reports them. *heavy* is true for any step that rewrites, moves or deletes
    rows -- those run only under ``morgan migrate``.
    """

    number: int
    name: str
    heavy: bool
    run: Callable[[sqlite3.Connection, Stores], dict[str, int] | None]


class DatabaseNeedsMigration(Exception):
    """A write reached a database whose next step is heavy and has not been run.

    *reason* names the pending steps and the command that runs them. It is the whole message,
    and both surfaces show it to the owner: the CLI as its error, an MCP client as the tool's.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _reextract_entities(conn: sqlite3.Connection, stores: Stores) -> None:
    """Re-extract every memory's entities under the current rule.

    Both copies are rewritten: the entity index recall searches, and the list stored on the
    memory itself, which is what a memory read back carries. Rewriting one would leave the two
    disagreeing about the same memory.

    The memory's row is rewritten in its ``entities`` column only (``set_entities``), never
    through ``put``: this step runs on the schema of version 0, which lacks every column a
    later step added and ``put`` names.
    """
    episodics, entities = stores
    for memory_id in episodics.ids():
        memory = episodics.get(memory_id)
        if memory is None:
            continue
        names = extract_entity_names(memory.content)
        episodics.set_entities(memory.id, [Entity(name=n) for n in names])
        entities.add(memory.id, names, user_id=memory.user_id, project=memory.project)


def _drop_the_semantic_index(conn: sqlite3.Connection, stores: Stores) -> None:
    """Drop the tables of the semantic upper index, which recall no longer has.

    Nothing reads, writes or erases them now, and the entity names in them are the owner's
    words: left in place they would outlive every ``forget``.
    """
    conn.execute("DROP TABLE IF EXISTS mem_entity_edges")
    conn.execute("DROP TABLE IF EXISTS mem_schema_edges")
    conn.execute("DROP TABLE IF EXISTS mem_entity_nodes")
    conn.execute("DROP TABLE IF EXISTS mem_schemas")


def _create_embedding_spaces_and_projects(conn: sqlite3.Connection, stores: Stores) -> None:
    """Add ``embedding_spaces`` and ``projects``. Light: two ``CREATE TABLE``s, no row moved.

    *stores* is unused -- neither table is part of the episodics/entities pair every step
    receives. Calls each store's ``create_schema`` directly, not its class -- the class also
    commits, and this runs inside ``upgrade``/``migrate``'s ``write_transaction``: a commit
    here would end that transaction early, so a later step's failure could no longer roll
    this one back with it. The same two functions run at open (via ``EmbeddingSpaceStore`` and
    ``ProjectStore``, which do commit, outside any transaction), so a fresh database and an
    old one upgraded through this step end up with identical schema either way.
    """
    spaces.create_schema(conn)
    projects.create_schema(conn)


#: The projects ``app/chatgpt_import.py`` writes: the only rows whose origin is known. Named
#: here as literals because a step is frozen history, and ``memory/`` does not import ``app/``;
#: a test holds them equal to ``ARCHIVE_PROJECT`` and ``HOLDOUT_PROJECT``.
_ARCHIVE_PROJECTS = ("archive/chatgpt", "archive/chatgpt-holdout")

#: Step 4's columns, in the order it adds them, each with a constant default. The stores'
#: ``CREATE TABLE`` ends with the same columns in the same order, so a database this code
#: creates and one migrated through step 4 have the same schema; a test compares the two.
_PROVENANCE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("memories", "origin_kind", "TEXT NOT NULL DEFAULT 'unknown'"),
    ("memories", "client", "TEXT NOT NULL DEFAULT ''"),
    ("memories", "session_id", "TEXT NOT NULL DEFAULT ''"),
    ("memories", "cwd", "TEXT NOT NULL DEFAULT ''"),
    ("memories", "author_id", "TEXT NOT NULL DEFAULT ''"),
    ("memories", "scope", "TEXT NOT NULL DEFAULT 'private'"),
    ("memories", "instruction_like", "INTEGER NOT NULL DEFAULT 0"),
    ("memories", "status", "TEXT NOT NULL DEFAULT 'stored'"),
    ("facts", "author_id", "TEXT NOT NULL DEFAULT ''"),
    ("facts", "scope", "TEXT NOT NULL DEFAULT 'private'"),
)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[0]) for r in conn.execute("SELECT name FROM pragma_table_info(?)", (table,))}


def _add_provenance(conn: sqlite3.Connection, stores: Stores) -> dict[str, int]:
    """Give every memory and fact its provenance columns, and fill in what is known.

    Heavy: the backfill rewrites every memory and fact. Each existing row's author is its
    owner. A memory's origin is known only for the two archive projects, which only the
    ChatGPT import writes; every other memory stays ``unknown``, because guessing an origin
    would be worse than saying none was recorded.

    A column is added only where it is missing. The stores create the latest schema whenever
    they create a table, so a table this code made while the database was still below
    version 4 -- ``facts`` on an old database opened once, ``memories`` in a file ``morgan
    migrate`` found empty -- already has step 4's columns; adding them again would fail the
    wave, and every later ``migrate`` with it. ``memories`` always exists here, because
    *stores* opened it; ``facts`` exists only if its store ever opened this file, and when it
    is made later its ``CREATE TABLE`` carries the columns. Returns the rows each backfill
    rewrote, per table.
    """
    for table, column, definition in _PROVENANCE_COLUMNS:
        if not _table_exists(conn, table) or column in _column_names(conn, table):
            continue
        # Names from the constant above, never from a caller: DDL takes no bound parameters.
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    memories = conn.execute("UPDATE memories SET author_id = user_id WHERE author_id = ''")
    counts = {"memories": memories.rowcount}
    conn.execute(
        "UPDATE memories SET origin_kind = 'import' WHERE project IN (?, ?)", _ARCHIVE_PROJECTS
    )
    counts["facts"] = (
        conn.execute("UPDATE facts SET author_id = user_id WHERE author_id = ''").rowcount
        if _table_exists(conn, "facts")
        else 0
    )
    return counts


#: The project every memory, fact and index row landed in before this step existed -- named
#: nothing, told the owner nothing. Frozen as a literal, like ``_ARCHIVE_PROJECTS`` above: a
#: step is history, and the constant it moves rows *to* is ``PERSONAL_PROJECT``, not this one.
_OLD_DEFAULT_PROJECT = "default"


def _rename_default_project(conn: sqlite3.Connection, stores: Stores) -> dict[str, int]:
    """Move every row filed under ``'default'`` to ``PERSONAL_PROJECT``. Heavy: it rewrites
    every project-keyed table.

    Walks ``tables.project_tables(conn)`` rather than a fixed list, so a table this database
    has never created -- the version-0 test runs every step on one that lacks ``vec_items``,
    ``fts_memories`` and ``session_history`` -- is skipped instead of raising, and a table
    ``PROJECT_TABLES`` does not yet know about (a later embedding space, once one is
    registered) is still reached.

    ``vec_items`` is a sqlite-vec vec0 virtual table with ``project`` as a metadata column,
    and ``fts_memories`` is FTS5 with ``project`` UNINDEXED -- both cannot be ``ALTER``ed, the
    reason every other project-column migration in this package reads a table's rows out,
    drops it and reinserts them. Renaming a value needs none of that: verified in a scratch
    database on this repository's sqlite-vec (0.1.9), a plain ``UPDATE ... SET project = ?
    WHERE project = ?`` rewrites the metadata of a vec0 table and an UNINDEXED FTS5 column in
    place, rowids and embeddings untouched, and both tables still answer a KNN / MATCH query
    correctly afterwards. So one statement, the same shape for every table, does the whole
    step. Returns the rows moved, per table -- ``0`` for a table with no ``'default'`` row,
    which is every table on the owner's live archive today: that is what makes this step
    worth counting rather than assuming.
    """
    counts: dict[str, int] = {}
    for table in project_tables(conn):
        if not _table_exists(conn, table):
            continue
        # `table` is never caller-supplied: it comes from `PROJECT_TABLES` or from
        # `embedding_spaces.table_name`, itself written only by this package's own migrations
        # and stores. DML takes no bound parameters for an identifier either way.
        cur = conn.execute(
            f"UPDATE {table} SET project = ? WHERE project = ?",  # noqa: S608 # nosec B608
            (PERSONAL_PROJECT, _OLD_DEFAULT_PROJECT),
        )
        counts[table] = cur.rowcount
    return counts


#: ``vec_items`` and ``fts_memories`` as step 6 recreates them. Frozen here, like
#: ``_PROVENANCE_COLUMNS``: a step is history, and a later change to either table is a later
#: step. ``SqliteVectorIndex`` and ``FtsIndex`` create a new database's tables with the same
#: statements (plus ``IF NOT EXISTS``, which SQLite does not record), so the DDL of a new
#: database and a migrated one is the same text; a test compares the two. ``project`` stays a
#: metadata column rather than a ``PARTITION KEY``: ``store/vectors.py``'s docstring has the
#: measurements that decided it.
_VEC_ITEMS_AT_STEP_SIX = """CREATE VIRTUAL TABLE vec_items USING vec0(
    embedding float[{dims}] distance_metric=cosine,
    user_id TEXT,
    project TEXT,
    status TEXT,
    scope TEXT,
    author_id TEXT
)"""
_FTS_MEMORIES_AT_STEP_SIX = """CREATE VIRTUAL TABLE fts_memories USING fts5(
    memory_id UNINDEXED,
    user_id   UNINDEXED,
    project   UNINDEXED,
    content,
    status    UNINDEXED,
    scope     UNINDEXED,
    author_id UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
)"""

#: The columns step 6 adds to both tables. A table that has them all was created by this code
#: -- a store opened before ``morgan migrate`` ran makes a missing table at the latest DDL --
#: and step 6 leaves it as it is.
_STEP_SIX_COLUMNS = frozenset({"status", "scope", "author_id"})


def _rebuild_vec0_and_fts5(conn: sqlite3.Connection, stores: Stores) -> dict[str, int]:
    """Recreate ``vec_items`` and ``fts_memories`` with ``status``, ``scope`` and
    ``author_id``, copying every row. Heavy: every vector blob is read and written again.

    Neither vec0 nor FTS5 can be ``ALTER``ed, so each table's rows are read out, the table is
    dropped and recreated, and the rows are reinserted under their own rowids -- the vectors
    byte for byte, since nothing is embedded. The new columns come from the memory's own row
    in ``memories`` (step 4 gave it all three); a row whose memory is gone gets the values
    step 4 gave every existing memory: ``stored``, ``private``, and its owner as its author.
    Each table's count is checked against the rows read before this step returns, inside the
    wave's transaction: a shortfall raises, and the whole wave rolls back to the old tables.

    A table that is not there is skipped -- the version-0 test runs every step on a database
    without either, and the store creates it at the latest DDL when it first opens it -- and
    so is one that already has the columns. The width of ``vec_items`` is the one its old
    DDL declares. Registers no embedding space: a step is not given the settings, so it
    cannot know which model wrote the vectors, and ``morgan migrate`` registers the settings'
    space once its wave has committed. Returns the rows copied, per table rebuilt.
    """
    counts: dict[str, int] = {}
    # None when the table is not there, and nothing to rebuild.
    dims = vectors.declared_width(conn, table_name="vec_items")
    if dims is not None and _lacks_step_six_columns(conn, "vec_items"):
        rows = [
            tuple(r)
            for r in conn.execute(
                "SELECT v.rowid, v.embedding, v.user_id, v.project, "
                "COALESCE(e.status, 'stored'), COALESCE(e.scope, 'private'), "
                "COALESCE(e.author_id, v.user_id) "
                "FROM vec_items v "
                "LEFT JOIN vec_meta m ON m.rowid = v.rowid "
                "LEFT JOIN memories e ON e.id = m.id"
            )
        ]
        conn.execute("DROP TABLE vec_items")
        # The width is an int parsed from the old DDL; DDL takes no bound parameters.
        conn.execute(_VEC_ITEMS_AT_STEP_SIX.format(dims=int(dims)))
        conn.executemany(
            "INSERT INTO vec_items (rowid, embedding, user_id, project, status, scope, author_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        counts["vec_items"] = _require_every_row(conn, "vec_items", len(rows))
    if _table_exists(conn, "fts_memories") and _lacks_step_six_columns(conn, "fts_memories"):
        rows = [
            tuple(r)
            for r in conn.execute(
                "SELECT f.rowid, f.memory_id, f.user_id, f.project, f.content, "
                "COALESCE(e.status, 'stored'), COALESCE(e.scope, 'private'), "
                "COALESCE(e.author_id, f.user_id) "
                "FROM fts_memories f LEFT JOIN memories e ON e.id = f.memory_id"
            )
        ]
        conn.execute("DROP TABLE fts_memories")
        conn.execute(_FTS_MEMORIES_AT_STEP_SIX)
        conn.executemany(
            "INSERT INTO fts_memories "
            "(rowid, memory_id, user_id, project, content, status, scope, author_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        counts["fts_memories"] = _require_every_row(conn, "fts_memories", len(rows))
    return counts


def _lacks_step_six_columns(conn: sqlite3.Connection, table: str) -> bool:
    return not _STEP_SIX_COLUMNS.issubset(_column_names(conn, table))


def _require_every_row(conn: sqlite3.Connection, table: str, expected: int) -> int:
    """The rows *table* holds, which must be the *expected* rows read out of it before it was
    dropped; anything else raises, and the wave rolls back with the old table in place."""
    # `table` is one of step 6's two literals, never caller input.
    found = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # noqa: S608 # nosec B608
    if found != expected:
        raise RuntimeError(
            f"step 6 read {expected} rows out of {table} and the rebuilt table holds {found}; "
            "the wave is rolled back and the old table kept"
        )
    return found


def _utcnow() -> datetime:
    """A step is not given the settings or the caller's clock -- ``composition.py``'s
    ``utcnow`` would be a step importing the module that imports this one -- so this is its
    own, the same one second's shape everywhere else in this package uses."""
    return datetime.now(UTC)


def _seed_projects(conn: sqlite3.Connection, stores: Stores) -> dict[str, int]:
    """One ``unclassified`` row per distinct project already named in ``memories``, ``facts``
    or ``session_history``. Light: it only inserts into an empty table -- ``projects`` has no
    row before this step runs, on a fresh database or an upgraded one alike, so there is
    nothing here to rewrite, move or delete.
    """
    return {"projects": projects.seed(conn, clock=_utcnow)}


#: In order. Step *n* brings a database from ``user_version`` *n - 1* to *n*; append only.
#: Steps 1 and 2 rewrite and drop, yet stay light: they predate the split, and every Morgan
#: that shipped them already ran them on open.
_STEPS: tuple[Step, ...] = (
    # A capital counts as a name only where its position does not explain it.
    Step(1, "reextract entities", False, _reextract_entities),
    Step(2, "drop the semantic index", False, _drop_the_semantic_index),
    Step(3, "create embedding_spaces and projects", False, _create_embedding_spaces_and_projects),
    Step(4, "provenance columns", True, _add_provenance),
    Step(5, "rename default to personal", True, _rename_default_project),
    Step(6, "rebuild vec0 and FTS5", True, _rebuild_vec0_and_fts5),
    Step(7, "seed projects", False, _seed_projects),
)


def code_version() -> int:
    """The ``user_version`` a database is at once it has been through every step this build
    knows. Read from ``_STEPS`` when called, like ``pending``'s default."""
    return len(_STEPS)


def _version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _advance(conn: sqlite3.Connection, step: Step) -> None:
    # PRAGMA takes no bound parameters; the value is an int a Step carries.
    conn.execute(f"PRAGMA user_version = {int(step.number)}")


def pending(conn: sqlite3.Connection, steps: Sequence[Step] | None = None) -> tuple[Step, ...]:
    """The steps the database behind *conn* has not been through, in order.

    *steps* defaults to ``_STEPS``, looked up when called rather than bound when defined, so
    opening, ``morgan migrate`` and ``restore``'s version check all read the one list.
    """
    done = _version(conn)
    return tuple(s for s in (_STEPS if steps is None else steps) if s.number > done)


def holds_morgan_tables(conn: sqlite3.Connection) -> bool:
    """Whether the database behind *conn* holds any table Morgan keys by project."""
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    return not names.isdisjoint(PROJECT_TABLES)


def stamp_if_new(conn: sqlite3.Connection, steps: Sequence[Step] | None = None) -> bool:
    """Record a database this code is about to create as through every step; ``True`` if so.

    A file holding none of Morgan's tables has nothing a step could re-derive, and the stores
    about to open it create the schema the last step leaves. Started at ``user_version`` 0
    instead, its first heavy step would open it read-only on its first day and then fail on
    the columns its own ``CREATE TABLE`` had just made. ``user_version`` 0 alone does not say
    "new": a database written before step 1 existed reads 0 too, and holds tables. Call it
    before any store opens the connection.
    """
    if holds_morgan_tables(conn):
        return False
    resolved = _STEPS if steps is None else steps
    with write_transaction(conn):
        # Again under the lock: another process may have created the tables since.
        if holds_morgan_tables(conn):
            return False
        # PRAGMA takes no bound parameters; the value is a length this function took.
        conn.execute(f"PRAGMA user_version = {len(resolved)}")
    return True


def _light_prefix(steps: Sequence[Step]) -> tuple[Step, ...]:
    """The steps before the first heavy one: all that may run on open."""
    return tuple(takewhile(lambda s: not s.heavy, steps))


def upgrade(
    conn: sqlite3.Connection, stores: Stores, *, steps: Sequence[Step] | None = None
) -> None:
    """Run the consecutive pending light steps, stopping at the first heavy one.

    Nothing to run means no write lock: a database waiting for ``morgan migrate`` is opened
    by every client, and none of them should queue behind another process's write to find
    that out.
    """
    if not _light_prefix(pending(conn, steps)):
        return
    with write_transaction(conn):
        # Read again under the lock: another process may have upgraded since the check above.
        for step in _light_prefix(pending(conn, steps)):
            step.run(conn, stores)
            _advance(conn, step)


def migrate(
    conn: sqlite3.Connection, stores: Stores, *, steps: Sequence[Step] | None = None
) -> list[tuple[Step, dict[str, int]]]:
    """Run every pending step, light and heavy, in one write transaction.

    ``user_version`` advances after each step, inside the transaction, so a step that raises
    rolls back the whole wave: the data and the counter stay where they were before this call.
    Returns each step run with the rows it reported touching (``{}`` when it counts nothing).
    The caller takes the snapshot first -- ``VACUUM INTO`` cannot run inside a transaction.
    """
    applied: list[tuple[Step, dict[str, int]]] = []
    with write_transaction(conn):
        # Read under the lock, for the same reason as ``upgrade``.
        for step in pending(conn, steps):
            counts = step.run(conn, stores)
            _advance(conn, step)
            applied.append((step, dict(counts or {})))
    return applied
