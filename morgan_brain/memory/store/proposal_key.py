"""One durable database-local input-attestation key, never source data or permission."""

from __future__ import annotations

import secrets
import sqlite3


def create_schema(conn: sqlite3.Connection) -> None:
    if not conn.in_transaction:
        raise ValueError("proposal key initialization requires a write transaction")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS proposal_integrity ("
        "id INTEGER PRIMARY KEY CHECK(id=1), "
        "key BLOB NOT NULL CHECK(length(key)=32))"
    )
    conn.execute(
        "INSERT OR IGNORE INTO proposal_integrity (id,key) VALUES (1,?)", (secrets.token_bytes(32),)
    )


def read_key(conn: sqlite3.Connection) -> bytes:
    try:
        row = conn.execute("SELECT key FROM proposal_integrity WHERE id=1").fetchone()
    except sqlite3.OperationalError:
        raise ValueError(
            "proposal input attestation unavailable; database needs upgrading"
        ) from None
    if row is None or not isinstance(row[0], bytes) or len(row[0]) != 32:
        raise ValueError("proposal input attestation key unavailable")
    return row[0]
