"""Persistent vector index backed by sqlite-vec, inside the one Morgan database.

Scoping happens **inside** the KNN via vec0 metadata columns, not by over-fetching and
filtering afterwards. This is not a style choice — post-filtering is incorrect. vec0 selects
its ``k`` nearest neighbours globally, before any join or WHERE on a joined table, so with
several users or projects in one store the caller's own near neighbours can be crowded out
and never returned at all. Verified on sqlite-vec 0.1.9: with two users interleaved and
``k=2``, an unscoped query returned only the *other* user's rows, while the metadata-scoped
query returned the correct two.

vec0 also defaults to L2. The index this replaces ranked by cosine (``_cosine`` in vector.py,
and Qdrant's ``Distance.COSINE``), so ``distance_metric=cosine`` is set explicitly — otherwise
ranking silently changes for unnormalised llama-server embeddings.
"""

from __future__ import annotations

import json
import sqlite3
import struct

from morgan_brain.modules.memory.stores.vector import VectorHit, VectorRecord


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


class SqliteVectorIndex:
    def __init__(self, conn: sqlite3.Connection, *, dim: int) -> None:
        self._conn = conn
        self._dim = dim
        conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS vec_meta (
                rowid   INTEGER PRIMARY KEY,
                id      TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vec_meta_id ON vec_meta (id);
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
                embedding float[{dim}] distance_metric=cosine,
                user_id TEXT
            );
            """
        )
        conn.commit()

    async def upsert(self, record: VectorRecord) -> None:
        if len(record.vector) != self._dim:
            raise ValueError(
                f"embedding dimension {len(record.vector)} does not match store dimension "
                f"{self._dim}"
            )
        cur = self._conn.execute("SELECT rowid FROM vec_meta WHERE id = ?", (record.id,))
        row = cur.fetchone()
        if row is not None:
            rowid = row["rowid"]
            self._conn.execute("DELETE FROM vec_items WHERE rowid = ?", (rowid,))
            self._conn.execute(
                "UPDATE vec_meta SET user_id = ?, payload = ? WHERE rowid = ?",
                (record.user_id, json.dumps(record.payload), rowid),
            )
        else:
            cur = self._conn.execute(
                "INSERT INTO vec_meta (id, user_id, payload) VALUES (?, ?, ?)",
                (record.id, record.user_id, json.dumps(record.payload)),
            )
            rowid = int(cur.lastrowid or 0)
        self._conn.execute(
            "INSERT INTO vec_items (rowid, embedding, user_id) VALUES (?, ?, ?)",
            (rowid, _pack(record.vector), record.user_id),
        )
        self._conn.commit()

    async def search(self, *, user_id: str, vector: list[float], top_k: int) -> list[VectorHit]:
        # user_id is a vec0 metadata column, so the filter applies INSIDE the KNN.
        rows = self._conn.execute(
            """
            SELECT m.id AS id, m.payload AS payload, v.distance AS distance
            FROM vec_items v
            JOIN vec_meta m ON m.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = ? AND v.user_id = ?
            ORDER BY v.distance
            """,
            (_pack(vector), top_k, user_id),
        ).fetchall()
        # vec0's cosine distance is (1 - cosine_similarity), on 0..2. Convert back to
        # similarity on -1..1 so the score scale matches InMemoryVectorIndex (_cosine) and
        # QdrantVectorIndex (Qdrant's own cosine score) — negating distance would give -2..0.
        return [
            VectorHit(
                id=r["id"], score=1.0 - float(r["distance"]), payload=json.loads(r["payload"])
            )
            for r in rows
        ]

    async def delete(self, ids: list[str]) -> None:
        """Protocol-level delete, for callers outside the single-database path.

        ``forget()`` deletes these rows with plain SQL inside its own transaction instead —
        see Task 16 — because committing here would break its atomicity.
        """
        for mid in ids:
            row = self._conn.execute("SELECT rowid FROM vec_meta WHERE id = ?", (mid,)).fetchone()
            if row is None:
                continue
            self._conn.execute("DELETE FROM vec_items WHERE rowid = ?", (row["rowid"],))
            self._conn.execute("DELETE FROM vec_meta WHERE rowid = ?", (row["rowid"],))
        self._conn.commit()
