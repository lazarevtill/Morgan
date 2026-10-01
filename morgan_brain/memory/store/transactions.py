"""SQLite transaction primitives, independent of database opening and migrations."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def read_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Keep multi-statement source reads on one snapshot, without a writer lock.

    An existing caller-owned transaction is joined unchanged. No model or external
    I/O may await inside this block; SQLite-only async methods do not suspend.
    """
    if conn.in_transaction:
        yield
        return
    conn.execute("BEGIN")
    try:
        yield
    except BaseException:
        conn.rollback()
        raise
    conn.commit()


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
