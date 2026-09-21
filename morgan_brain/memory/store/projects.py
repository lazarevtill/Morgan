"""``projects`` -- one row per repository Morgan has ever written to: classification and the
per-project capture/consolidate switches.

Only the schema lives here for now. Task 17 seeds the table from what a database already
holds and adds the queries over it (``get``, ``upsert``, ``list``); until then this module
exists so ``ProjectStore`` -- like every other store -- owns its own DDL, and migration step 3
can create the table on an old database the same way this class creates it on a new one.
"""

from __future__ import annotations

import sqlite3

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
    """Creates the ``projects`` table. Task 17 adds the queries over it."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        create_schema(conn)
        conn.commit()
