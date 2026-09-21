"""``projects`` -- one row per repository Morgan has ever written to: classification and the
per-project capture/consolidate switches.

Only the schema lives here for now. Task 17 seeds the table from what a database already
holds and adds the queries over it (``get``, ``upsert``, ``list``); until then this module
exists so ``ProjectStore`` -- like every other store -- owns its own DDL, and migration step 3
can create the table on an old database the same way this class creates it on a new one.
"""

from __future__ import annotations

import sqlite3

_SCHEMA = """
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
);
"""


class ProjectStore:
    """Creates the ``projects`` table. Task 17 adds the queries over it."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        conn.executescript(_SCHEMA)
        conn.commit()
