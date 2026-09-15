"""The one SQLite connection factory, and the one way to write.

Every store in the memory subsystem shares a single database file so that erasure is one
transaction and at-rest encryption is one volume. WAL mode lets several processes -- the MCP
server alongside the CLI -- read concurrently; the busy timeout absorbs writer contention.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

import sqlite_vec  # type: ignore[import-untyped]

_BUSY_TIMEOUT_MS = 5000


def open_db(path: str) -> sqlite3.Connection:
    """Open (or create) the Morgan database with WAL, a busy timeout, and sqlite-vec loaded."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)

    # ":memory:" has no journal to switch; WAL is meaningless and PRAGMA returns "memory".
    if path != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()
    return conn


@contextmanager
def write_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Run the block as one atomic write, holding the database write lock from the start.

    The lock is taken with ``BEGIN IMMEDIATE`` before the block's first statement, not at its
    first write. Another process shares this file, and a store that looks something up and
    then writes on the strength of it is only correct if nothing can change in between: a
    bare SELECT followed by a write let two processes insert the same vector id, or leave a
    fact key with two current values.

    A block opened while this connection is already inside one joins it as a savepoint, so a
    store method is atomic on its own and also composes into a larger write -- storing a
    memory writes four indexes as one unit. A nested block that raises undoes only its own
    statements; the outer block decides whether the whole write commits.

    Nothing inside may await real I/O. The connection is shared, so a coroutine that ran
    while the lock was held would have its statements folded into this transaction.
    """
    if conn.in_transaction:
        conn.execute("SAVEPOINT nested_write")
        try:
            yield
        except BaseException:
            conn.execute("ROLLBACK TO nested_write")
            conn.execute("RELEASE nested_write")
            raise
        conn.execute("RELEASE nested_write")
        return

    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.rollback()
        raise
    conn.commit()
