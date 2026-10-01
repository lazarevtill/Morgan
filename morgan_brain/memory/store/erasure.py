"""One global erasure generation, without owner/context tombstones or deleted data."""

from __future__ import annotations

import sqlite3


class StoreInterruptedByForget(RuntimeError):
    """An explicit caller retry is required after forget invalidates preparation."""


def create_schema(conn: sqlite3.Connection) -> None:
    """Create a singleton scalar; the caller owns the write transaction."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS erasure_state ("
        "singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
        "generation INTEGER NOT NULL CHECK(typeof(generation)='integer' AND generation>=0))"
    )
    conn.execute("INSERT OR IGNORE INTO erasure_state VALUES (1, 0)")


def read_generation(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT generation FROM erasure_state WHERE singleton=1").fetchone()
    if row is None:
        raise RuntimeError("Morgan erasure generation metadata is missing")
    return int(row[0])


def advance_generation(conn: sqlite3.Connection) -> None:
    """Invalidate prepared stores in the same transaction as an erasure."""
    if (
        conn.execute("UPDATE erasure_state SET generation=generation+1 WHERE singleton=1").rowcount
        != 1
    ):
        raise RuntimeError("Morgan erasure generation metadata is missing")


def require_generation(conn: sqlite3.Connection, expected: int) -> None:
    if read_generation(conn) != expected:
        raise StoreInterruptedByForget(
            "Store preparation crossed a committed forget; retry explicitly only if "
            "you still intend to store this memory"
        )
