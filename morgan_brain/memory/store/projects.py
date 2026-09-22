"""``projects`` -- one row per repository Morgan has ever written to: classification, the
repository it is, and the per-project capture/consolidate switches.

Three functions write to it, and they divide the work:

- ``seed`` is migration step 7: one ``unclassified`` row per project named in ``memories``,
  ``facts`` or ``session_history`` -- whichever of the three a database actually has. It runs
  once, for what a database already held.
- ``register`` is every project-keyed write after that, in that write's own transaction, so a
  project first written to after the migration has a row too.
- ``record`` is a CLI write from inside a repository: the classification, remote and root that
  repository has right now. It updates a row and never inserts one -- what a write registers is
  what gets classified, so no remote or root is left behind for a project ``forget`` emptied.

The disk walk over ``MORGAN_CODE_ROOTS`` that would fill ``remote`` and ``root`` for a project
no CLI write ever reached is phase 1a.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from morgan_brain.memory.store.db import write_transaction
from morgan_brain.memory.store.tables import Erasure, project_tables

#: A single statement, run with plain ``execute`` rather than ``executescript`` -- the latter
#: issues an implicit ``COMMIT`` before it runs anything, which would end migration step 3's
#: enclosing ``write_transaction`` partway through a wave that a later step then fails.
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS projects (
        name                TEXT PRIMARY KEY,
        classification      TEXT NOT NULL DEFAULT 'unclassified',
        remote              TEXT,
        root                TEXT,
        capture_enabled     INTEGER NOT NULL DEFAULT 1,
        retention_days      INTEGER,
        paused_until        TEXT,
        consolidate_enabled INTEGER NOT NULL DEFAULT 1,
        created_at          TEXT NOT NULL
    )
    """,
)


def create_schema(conn: sqlite3.Connection) -> None:
    """Create ``projects``, joining *conn*'s current transaction rather than committing one
    of its own. Called by ``ProjectStore`` at open and by migration step 3, so both leave the
    same DDL -- and step 3's DDL rolls back with the rest of its wave when a later step fails.
    """
    for statement in _SCHEMA_STATEMENTS:
        conn.execute(statement)


class ProjectStore:
    """Creates the ``projects`` table. Every query over it is one of the module-level
    functions below, which take a plain connection rather than this store -- migration step 7
    runs ``seed`` directly, on a connection this class never opens."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        create_schema(conn)
        conn.commit()


@dataclass
class Project:
    name: str
    classification: str
    remote: str | None
    root: str | None
    capture_enabled: bool
    retention_days: int | None
    paused_until: str | None
    consolidate_enabled: bool
    created_at: str


def _row_to_project(row: sqlite3.Row) -> Project:
    return Project(
        name=row["name"],
        classification=row["classification"],
        remote=row["remote"],
        root=row["root"],
        capture_enabled=bool(row["capture_enabled"]),
        retention_days=row["retention_days"],
        paused_until=row["paused_until"],
        consolidate_enabled=bool(row["consolidate_enabled"]),
        created_at=row["created_at"],
    )


def get(conn: sqlite3.Connection, name: str) -> Project | None:
    """The one row for *name*, or ``None`` when it was never seeded and the owner has not
    set it up by hand -- absence is not the same as a project switched off."""
    row = conn.execute("SELECT * FROM projects WHERE name = ?", (name,)).fetchone()
    return None if row is None else _row_to_project(row)


def all(conn: sqlite3.Connection) -> list[Project]:
    """Every project Morgan has a row for, by name."""
    return [_row_to_project(r) for r in conn.execute("SELECT * FROM projects ORDER BY name")]


def register(conn: sqlite3.Connection, project: str, *, now: datetime) -> None:
    """Give *project* a row if it has none: ``unclassified``, no remote, no root, created at
    *now*. One statement, joining the caller's transaction rather than opening one.

    Every project-keyed write calls this from inside its own write transaction -- the memory
    store, a fact upsert, a session-history row -- so the row appears and disappears with the
    data it describes: rolled back with a write that failed, and deleted by ``forget`` once
    nothing of the project is left (``delete_project``).

    ``INSERT OR IGNORE``, so the row a project already has is left exactly as it is: neither
    restamped nor reset to ``unclassified`` by the next write to a classified project.
    """
    conn.execute(
        "INSERT OR IGNORE INTO projects (name, classification, created_at) "
        "VALUES (?, 'unclassified', ?)",
        (project, now.isoformat()),
    )


def record(
    conn: sqlite3.Connection,
    project: str,
    *,
    classification: str,
    remote: str | None,
    root: str | None,
) -> bool:
    """Record what repository *project* is, and return whether a row was updated.

    Only the three columns that derive from the repository and the settings; the owner's
    switches (``capture_enabled``, ``retention_days``, ``paused_until``,
    ``consolidate_enabled``) and the row's ``created_at`` are never touched here. The label is
    recomputed on every CLI write rather than stored once, because a repository's remote and
    ``MORGAN_WORK_REMOTE_GLOBS`` both change without Morgan hearing about it.

    An ``UPDATE``, never an insert: the write this rides with has already registered the
    project, so a missing row means there is nothing to describe -- a command that wrote
    nothing, or a project another process forgot in between. Inserting one would put the
    owner's remote and root back into a database their ``forget`` just took them out of.
    """
    cursor = conn.execute(
        "UPDATE projects SET classification = ?, remote = ?, root = ? WHERE name = ?",
        (classification, remote, root, project),
    )
    return cursor.rowcount > 0


#: The tables ``seed`` reads distinct projects from, per SPEC-phase0 §3.7. A table absent on
#: an old or freshly-created database -- the version-0 migration test runs every step on one
#: that lacks ``facts`` and ``session_history``, and a fresh database has no ``session_history``
#: until ``build_memory_context`` opens one -- is skipped, not an error.
_SOURCE_TABLES: tuple[str, ...] = ("memories", "facts", "session_history")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


def _distinct_projects(conn: sqlite3.Connection) -> set[str]:
    found: set[str] = set()
    for table in _SOURCE_TABLES:
        if not _table_exists(conn, table):
            continue
        # `table` is one of the three literals above, never caller input.
        for row in conn.execute(
            f"SELECT DISTINCT project FROM {table}"  # noqa: S608 # nosec B608
        ):
            found.add(str(row["project"]))
    return found


def seed(conn: sqlite3.Connection, clock: Callable[[], datetime]) -> int:
    """Insert one ``unclassified`` row per distinct project found in ``memories``, ``facts``
    and ``session_history`` -- whichever exist -- and return how many rows this call inserted.

    Idempotent: ``INSERT OR IGNORE`` against the primary key, so a project already seeded (by
    an earlier run of this same step, or set up by the owner ahead of it) costs nothing and is
    not recounted. Wraps its own ``write_transaction`` -- migration step 7 already holds the
    wave's write lock when it calls this, and a nested block joins it as a savepoint rather
    than starting a second one (``store/db.py::write_transaction``), so this is exactly as
    safe to call from inside a step as it is on its own, the way ``spaces.register`` does.

    Assumes ``projects`` already exists. Steps run strictly in order and the counter advances
    only once a step's own transaction commits (``pending``/``upgrade``/``migrate``), so no
    database that ever really went through migration step 3 can reach step 7 without the
    table step 3 creates -- only a fixture that hand-stamps ``user_version`` past a step it
    never ran could get here without it, and that is a fixture bug to fix, not a case for this
    function to paper over.
    """
    created_at = clock().isoformat()
    inserted = 0
    with write_transaction(conn):
        for name in sorted(_distinct_projects(conn)):
            cursor = conn.execute(
                "INSERT OR IGNORE INTO projects (name, classification, created_at) "
                "VALUES (?, 'unclassified', ?)",
                (name, created_at),
            )
            if cursor.rowcount > 0:
                inserted += 1
    return inserted


def _holds_rows_of(conn: sqlite3.Connection, project: str) -> bool:
    """Whether any registered project-keyed table still holds a row of *project*, from any
    owner."""
    for table in project_tables(conn):
        if not _table_exists(conn, table):
            continue
        # `table` comes from `project_tables`: `PROJECT_TABLES` or a name Morgan wrote to
        # `embedding_spaces` itself, never caller input.
        sql = f"SELECT EXISTS (SELECT 1 FROM {table} WHERE project = ?)"  # noqa: S608 # nosec B608
        if conn.execute(sql, (project,)).fetchone()[0]:
            return True
    return False


def delete_project(conn: sqlite3.Connection, erasure: Erasure) -> int:
    """`forget()`'s deleter for ``projects``: the erased project's own row, once no row of the
    project is left in any registered project-keyed table, from any owner.

    The row has no owner. Its remote URL, root path and switches belong to everyone with data
    in the project, so one owner's ``forget`` keeps it while another's rows remain, and the
    last one's removes it. ``forget()`` runs this after every project-keyed table's deleter
    (``_erasure_plan`` puts the name-keyed tables last), so the check sees what the erasure
    left, not the rows it is about to delete.
    """
    if _holds_rows_of(conn, erasure.project):
        return 0
    return conn.execute("DELETE FROM projects WHERE name = ?", (erasure.project,)).rowcount
