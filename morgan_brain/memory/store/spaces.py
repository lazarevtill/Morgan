"""``embedding_spaces`` -- which model wrote the vectors a database holds, recorded rather
than assumed.

Morgan checks only a vector's width today: two models of the same width are indistinguishable
to `vectors.py`, so swapping one in leaves every stored vector searched by a model that never
wrote it -- wrong answers, no error. One row per model Morgan has embedded with; the partial
unique index keeps exactly one of them ``active`` at a time, so `recall` never has to choose
between two. A ``shadow`` row may sit beside it -- a candidate replacement, embedding the same
content without yet being searched.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from morgan_brain.memory import fingerprint
from morgan_brain.memory.store.db import write_transaction

#: Individual statements, run with plain ``execute`` rather than ``executescript`` -- the
#: latter issues an implicit ``COMMIT`` before it runs anything, which would end migration
#: step 3's enclosing ``write_transaction`` partway through a wave that a later step then
#: fails: the DDL would stay committed while ``user_version`` rolled back to before it.
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS embedding_spaces (
        id                      INTEGER PRIMARY KEY,
        model                   TEXT NOT NULL,
        quant                   TEXT NOT NULL DEFAULT '',
        dims                    INTEGER NOT NULL,
        query_prefix            TEXT NOT NULL DEFAULT '',
        document_prefix         TEXT NOT NULL DEFAULT '',
        fingerprint             BLOB,
        fingerprint_recorded_at TEXT,
        floor_margin            REAL,
        status                  TEXT NOT NULL
                                    CHECK (status IN ('active', 'shadow', 'retired', 'mismatch')),
        table_name              TEXT NOT NULL UNIQUE,
        created_at              TEXT NOT NULL
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_embedding_spaces_one_active
        ON embedding_spaces (status) WHERE status = 'active'
    """,
)


def create_schema(conn: sqlite3.Connection) -> None:
    """Create ``embedding_spaces`` and its partial unique index, joining *conn*'s current
    transaction rather than committing one of its own. Called by ``EmbeddingSpaceStore`` at
    open and by migration step 3, so both leave the same DDL -- and step 3's DDL rolls back
    with the rest of its wave when a later step in it fails."""
    for statement in _SCHEMA_STATEMENTS:
        conn.execute(statement)


class EmbeddingSpaceStore:
    """Creates ``embedding_spaces`` and its partial unique index.

    Every query over the table is one of the module-level functions below, which take a plain
    connection rather than this store: migration step 3 runs this same schema directly, on a
    connection this class never opens, so an old database and a fresh one end with identical
    DDL either way.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        create_schema(conn)
        conn.commit()


@dataclass
class EmbeddingSpace:
    id: int
    model: str
    quant: str
    dims: int
    query_prefix: str
    document_prefix: str
    fingerprint: bytes | None
    fingerprint_recorded_at: str | None
    floor_margin: float | None
    status: str
    table_name: str
    created_at: str


def _row_to_space(row: sqlite3.Row) -> EmbeddingSpace:
    return EmbeddingSpace(
        id=row["id"],
        model=row["model"],
        quant=row["quant"],
        dims=row["dims"],
        query_prefix=row["query_prefix"],
        document_prefix=row["document_prefix"],
        fingerprint=row["fingerprint"],
        fingerprint_recorded_at=row["fingerprint_recorded_at"],
        floor_margin=row["floor_margin"],
        status=row["status"],
        table_name=row["table_name"],
        created_at=row["created_at"],
    )


def active(conn: sqlite3.Connection) -> EmbeddingSpace | None:
    """The one space in use, or ``None`` before any has been registered."""
    row = conn.execute("SELECT * FROM embedding_spaces WHERE status = 'active' LIMIT 1").fetchone()
    return None if row is None else _row_to_space(row)


def register(
    conn: sqlite3.Connection,
    *,
    model: str,
    dims: int,
    table_name: str,
    quant: str = "",
    status: str = "active",
    clock: Callable[[], datetime],
) -> EmbeddingSpace:
    """Record a new embedding space. Registering a second ``active`` one fails on the partial
    unique index -- retiring the first is the caller's explicit act, not this function's."""
    created_at = clock().isoformat()
    with write_transaction(conn):
        cursor = conn.execute(
            """
            INSERT INTO embedding_spaces (model, quant, dims, status, table_name, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (model, quant, dims, status, table_name, created_at),
        )
        space_id = int(cursor.lastrowid or 0)
    return EmbeddingSpace(
        id=space_id,
        model=model,
        quant=quant,
        dims=dims,
        query_prefix="",
        document_prefix="",
        fingerprint=None,
        fingerprint_recorded_at=None,
        floor_margin=None,
        status=status,
        table_name=table_name,
        created_at=created_at,
    )


def set_status(conn: sqlite3.Connection, space_id: int, status: str) -> None:
    """Move a space to *status*. Setting ``active`` while one already is fails on the same
    partial unique index ``register`` does -- the caller retires the old one first."""
    with write_transaction(conn):
        conn.execute("UPDATE embedding_spaces SET status = ? WHERE id = ?", (status, space_id))


def _pack(vectors: list[list[float]], *, dims: int) -> bytes:
    """Validate every vector is *dims*-wide, then delegate the actual packing to
    ``fingerprint.pack`` -- little-endian float32, same format ``record_fingerprint`` has
    always stored."""
    for vector in vectors:
        if len(vector) != dims:
            raise ValueError(f"expected a {dims}-wide vector, got {len(vector)}")
    return fingerprint.pack(vectors)


def unpack(blob: bytes, *, dims: int) -> list[list[float]]:
    """The inverse of the packing ``record_fingerprint`` does. Delegates to
    ``fingerprint.unpack``, which uses the same little-endian float32 layout."""
    return fingerprint.unpack(blob, dims=dims)


def record_fingerprint(
    conn: sqlite3.Connection,
    space_id: int,
    vectors: list[list[float]],
    clock: Callable[[], datetime],
) -> None:
    """Pack *vectors* -- the embeddings of ``fingerprint.STRINGS``'s five fixed strings, in
    that order -- into one blob and store it against *space_id*, timestamped now. A later
    embed of the same strings that falls outside the measured tolerance of this blob is how a
    same-width model swap is caught."""
    row = conn.execute("SELECT dims FROM embedding_spaces WHERE id = ?", (space_id,)).fetchone()
    if row is None:
        raise ValueError(f"no embedding space with id {space_id}")
    blob = _pack(vectors, dims=int(row["dims"]))
    recorded_at = clock().isoformat()
    with write_transaction(conn):
        conn.execute(
            "UPDATE embedding_spaces SET fingerprint = ?, fingerprint_recorded_at = ? WHERE id = ?",
            (blob, recorded_at, space_id),
        )
