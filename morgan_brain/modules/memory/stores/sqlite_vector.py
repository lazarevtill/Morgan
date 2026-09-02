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

from morgan_brain.models.memory import DEFAULT_PROJECT
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
                user_id TEXT,
                project TEXT
            );
            """
        )
        conn.commit()
        self._migrate_project_column(dim)

    def _migrate_project_column(self, dim: int) -> None:
        """Idempotent upgrade for a database written before project scoping existed.

        ``vec_meta`` is a regular table, so a plain ``ALTER TABLE`` covers it. ``vec_items`` is
        a vec0 virtual table -- like FTS5, it cannot be ``ALTER``ed -- so its rows (the packed
        embedding blobs, which have no other source of truth) are read out, the table is
        dropped and recreated with the ``project`` metadata column, and the rows are
        reinserted with ``DEFAULT_PROJECT`` backfilled.
        """
        meta_cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(vec_meta)")}
        if "project" not in meta_cols:
            self._conn.execute(
                f"ALTER TABLE vec_meta ADD COLUMN project TEXT NOT NULL DEFAULT '{DEFAULT_PROJECT}'"
            )
            self._conn.commit()

        item_cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(vec_items)")}
        if "project" not in item_cols:
            rows = self._conn.execute("SELECT rowid, embedding, user_id FROM vec_items").fetchall()
            self._conn.execute("DROP TABLE vec_items")
            self._conn.execute(
                f"""
                CREATE VIRTUAL TABLE vec_items USING vec0(
                    embedding float[{dim}] distance_metric=cosine,
                    user_id TEXT,
                    project TEXT
                )
                """
            )
            for r in rows:
                self._conn.execute(
                    "INSERT INTO vec_items (rowid, embedding, user_id, project) "
                    "VALUES (?, ?, ?, ?)",
                    (r["rowid"], r["embedding"], r["user_id"], DEFAULT_PROJECT),
                )
            self._conn.commit()

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
                "UPDATE vec_meta SET user_id = ?, project = ?, payload = ? WHERE rowid = ?",
                (record.user_id, record.project, json.dumps(record.payload), rowid),
            )
        else:
            cur = self._conn.execute(
                "INSERT INTO vec_meta (id, user_id, project, payload) VALUES (?, ?, ?, ?)",
                (record.id, record.user_id, record.project, json.dumps(record.payload)),
            )
            rowid = int(cur.lastrowid or 0)
        self._conn.execute(
            "INSERT INTO vec_items (rowid, embedding, user_id, project) VALUES (?, ?, ?, ?)",
            (rowid, _pack(record.vector), record.user_id, record.project),
        )
        self._conn.commit()

    async def search(
        self,
        *,
        user_id: str,
        vector: list[float],
        top_k: int,
        project: str | None = DEFAULT_PROJECT,
        restrict_ids: list[str] | None = None,
    ) -> list[VectorHit]:
        # user_id and project are both vec0 metadata columns, so the filter applies INSIDE the
        # KNN -- see the module docstring for why post-filtering would silently drop results.
        sql = """
            SELECT m.id AS id, m.payload AS payload, v.distance AS distance
            FROM vec_items v
            JOIN vec_meta m ON m.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = ? AND v.user_id = ?
            """
        params: list[object] = [_pack(vector), top_k, user_id]
        if project is not None:
            sql += " AND v.project = ?"
            params.append(project)
        if restrict_ids is not None:
            # `rowid IN (...)` constrains the KNN itself in sqlite-vec, exactly like the
            # metadata columns above -- the k nearest are chosen from inside the pool
            # rather than from the whole table and then filtered. That distinction is the
            # whole value of routing: with a pool of {m3, m4} and k=2 this returns m3 and
            # m4 even when m0 and m1 are globally nearer, which a post-filter could not.
            sql += " AND v.rowid IN (SELECT rowid FROM vec_meta WHERE id IN "
            sql += "(SELECT value FROM json_each(?)))"
            params.append(json.dumps(restrict_ids))
        sql += " ORDER BY v.distance"
        rows = self._conn.execute(sql, params).fetchall()
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
