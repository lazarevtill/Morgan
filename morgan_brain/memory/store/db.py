"""The one SQLite connection factory, and the one way to write.

Every store in the memory subsystem shares a single database file so that erasure is one
transaction and at-rest encryption is one volume. WAL mode lets several processes -- the MCP
server alongside the CLI -- read concurrently; the busy timeout absorbs writer contention.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec  # type: ignore[import-untyped]


def open_db(path: str, *, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    """Open (or create) the Morgan database with WAL, a busy timeout, and sqlite-vec loaded.

    *busy_timeout_ms* is how long a statement waits on another process's lock before it fails
    with "database is locked". Every caller in Morgan that opens the database file passes
    ``Settings.db_busy_timeout_ms``; the default, that setting's own, serves an in-memory
    database and the tests.
    """
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
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()
    return conn


def readonly_uri(path: str | Path) -> str:
    """The SQLite URI that opens *path* read-only.

    Built by ``Path.as_uri``, which percent-encodes the path. In a bare ``file:`` URI a ``#``
    or ``?`` in a directory name starts the fragment or the query: SQLite opens the path cut
    off there -- read-write, creating it -- and ``mode=ro`` never arrives.
    """
    return f"{Path(path).resolve().as_uri()}?mode=ro"


def open_readonly(path: str, *, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    """Open an existing Morgan database only to read it: SQLite's read-only mode, sqlite-vec
    loaded, and no pragma that writes.

    ``open_db`` sets ``journal_mode=WAL``, which rewrites the header of a file in rollback-
    journal mode -- a snapshot, or one ``morgan restore`` just put in place -- and so cannot be
    what a command that only looks uses. A read-only connection never creates the file and
    never changes it. On a WAL database it reads the ``-wal`` as every reader does, which needs
    the ``-shm``: SQLite makes both when they are missing, and a read-only connection cannot
    remove them again; the next writer to close does.

    SQLite opens lazily, so one read here makes a file that cannot be read -- not a database,
    or a WAL database whose ``-wal`` cannot be opened -- raise ``sqlite3.Error`` from this call
    rather than from the caller's first query. A blocked ``-shm`` is not one of them
    everywhere: with a directory in its place Windows refuses the open, while Linux opens that
    directory read-only, keeps a wal-index of its own and reads the database and its log
    anyway. ``:memory:`` is a new, empty database with nothing on disk to protect, and opens
    as ``open_db`` opens it.
    """
    if path == ":memory:":
        return open_db(path, busy_timeout_ms=busy_timeout_ms)
    # The busy timeout is the connection's own, not a write: a reader can meet a checkpoint.
    conn = sqlite3.connect(
        readonly_uri(path), uri=True, timeout=busy_timeout_ms / 1000, check_same_thread=False
    )
    try:
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
        conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
    except BaseException:
        conn.close()
        raise
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
