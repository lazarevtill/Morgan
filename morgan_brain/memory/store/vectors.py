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

``project`` is a metadata column, not a vec0 ``PARTITION KEY``. The baseline
(``scripts/measure_partition_key.py``, ``docs/measurements/2026-09-phase0-baseline.md`` §4)
times per-project queries: 42 ms partitioned against 59 ms. The same script also measures the
all-projects scope and the on-disk cost (raw output
``2026-09-21-partition-key-all-projects.txt``, named in full in the baseline doc's §4): an
all-projects query -- the scope the relevance floor was fitted on -- takes about 76 ms
partitioned against about 61 ms metadata-scoped; each project gets vector chunks of its own,
1,024 vectors wide whatever it holds, so twenty projects of one memory each fill 336.4 MB
partitioned against 16.9 MB metadata-scoped; and sqlite-vec 0.1.9
refuses an ``UPDATE`` of a partition key, the statement migration step 5 renames a
project with. ``status`` is stored and not yet filtered on: recall does not filter on it in
phase 0, and a vec0 table cannot be altered to add it later.
"""

from __future__ import annotations

import json
import re
import sqlite3
import struct
from dataclasses import dataclass, field
from typing import Any

from morgan_brain.memory.store.db import write_transaction
from morgan_brain.models import PERSONAL_PROJECT, MemoryStatus, Scope


@dataclass
class VectorRecord:
    """One memory's vector and the metadata stored beside it: whose it is, which project,
    and the memory's status, scope and author, as its row in ``memories`` has them."""

    id: str
    user_id: str
    vector: list[float]
    project: str = PERSONAL_PROJECT
    payload: dict[str, Any] = field(default_factory=dict)
    status: MemoryStatus = MemoryStatus.STORED
    scope: Scope = Scope.PRIVATE
    author_id: str = ""


@dataclass
class VectorHit:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    """The inverse of ``_pack``, in the same native byte order. ``fingerprint.pack`` writes
    little-endian explicitly; the two agree on every platform Morgan runs on."""
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


#: ``vec_items`` as this code creates it. Migration step 6 recreates an older table from its
#: own frozen copy of this statement (``migrations._VEC_ITEMS_AT_STEP_SIX``), so a new database
#: and a migrated one end with the same DDL; a test holds the two equal. SQLite does not record
#: ``IF NOT EXISTS`` as part of the DDL.
_CREATE_VEC_ITEMS = """CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
    embedding float[{dim}] distance_metric=cosine,
    user_id TEXT,
    project TEXT,
    status TEXT,
    scope TEXT,
    author_id TEXT
)"""


class SqliteVectorIndex:
    def __init__(self, conn: sqlite3.Connection, *, dim: int) -> None:
        self._conn = conn
        self._dim = dim
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vec_meta (
                rowid   INTEGER PRIMARY KEY,
                id      TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vec_meta_id ON vec_meta (id);
            """
        )
        conn.execute(_CREATE_VEC_ITEMS.format(dim=int(dim)))
        conn.commit()
        self._migrate_project_column(dim)

    def _migrate_project_column(self, dim: int) -> None:
        """Idempotent upgrade for a database written before project scoping existed.

        ``vec_meta`` is a regular table, so a plain ``ALTER TABLE`` covers it. ``vec_items`` is
        a vec0 virtual table -- like FTS5, it cannot be ``ALTER``ed -- so its rows (the packed
        embedding blobs, which have no other source of truth) are read out, the table is
        dropped and recreated with the ``project`` metadata column, and the rows are
        reinserted with ``PERSONAL_PROJECT`` backfilled.

        The table is recreated as it stood when ``project`` was added, without the columns
        migration step 6 adds: a database this old was written before phase 0, opens
        read-only until ``morgan migrate`` runs, and gets them from step 6 then. A table that
        has ``project`` -- every one this code or step 6 created -- is left alone.
        """
        meta_cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(vec_meta)")}
        if "project" not in meta_cols:
            self._conn.execute(
                "ALTER TABLE vec_meta ADD COLUMN project TEXT NOT NULL "
                f"DEFAULT '{PERSONAL_PROJECT}'"
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
                    (r["rowid"], r["embedding"], r["user_id"], PERSONAL_PROJECT),
                )
            self._conn.commit()

    async def upsert(self, record: VectorRecord) -> None:
        if len(record.vector) != self._dim:
            raise ValueError(
                f"embedding dimension {len(record.vector)} does not match store dimension "
                f"{self._dim}"
            )
        # The id is looked up inside the write transaction, which holds the lock from its
        # first statement. Several processes share this database file -- morgan-mcp stays open
        # while the CLI or `morgan import` writes -- and a lookup made before the lock leaves a
        # window in which another one stores the same id: both see it absent, both insert, and
        # the second dies on `UNIQUE constraint failed: vec_meta.id`.
        with write_transaction(self._conn):
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
                "INSERT INTO vec_items "
                "(rowid, embedding, user_id, project, status, scope, author_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    rowid,
                    _pack(record.vector),
                    record.user_id,
                    record.project,
                    record.status.value,
                    record.scope.value,
                    record.author_id,
                ),
            )

    async def search(
        self,
        *,
        user_id: str,
        vector: list[float],
        top_k: int,
        project: str | None = PERSONAL_PROJECT,
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
        """Delete the vectors stored under *ids*; ids with no vector are ignored.

        The rows are found by id inside the same locked statements that delete them, never by
        a rowid read beforehand. ``vec_meta.rowid`` has no ``AUTOINCREMENT``, so SQLite gives
        the highest deleted rowid to the next insert: a rowid looked up first could belong to
        a different id by the time it was deleted, and the delete would erase that id's
        vector instead.
        """
        id_json = json.dumps(ids)
        with write_transaction(self._conn):
            self._conn.execute(
                "DELETE FROM vec_items WHERE rowid IN "
                "(SELECT rowid FROM vec_meta WHERE id IN (SELECT value FROM json_each(?)))",
                (id_json,),
            )
            self._conn.execute(
                "DELETE FROM vec_meta WHERE id IN (SELECT value FROM json_each(?))", (id_json,)
            )


#: A vec0 table name is interpolated into SQL, so it must be a plain identifier. It comes from
#: ``embedding_spaces.table_name``, which Morgan writes itself; this keeps it that way.
_TABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _vector_table(table_name: str) -> str:
    if not _TABLE_NAME.fullmatch(table_name):
        raise ValueError(f"not a vector table name: {table_name!r}")
    return table_name


def _exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)
    ).fetchone()
    return row is not None


def _text_and_vector(
    conn: sqlite3.Connection, *, table: str, rowid: int, memory_id: str
) -> tuple[str, list[float]] | None:
    """One candidate's stored text and vector, or ``None`` when either is missing.

    The point lookup ``stored_sample`` and ``audit_sample`` both repeat per candidate: the
    vec0 embedding by rowid -- one lookup per row, because vec0 answers ``rowid = ?`` from its
    rowid index while ``rowid IN (...)`` scans every stored vector, 49 MB on a 3,000-memory
    archive -- and the memory's own text by id.
    """
    lookup = f"SELECT embedding FROM {table} WHERE rowid = ?"  # noqa: S608 # nosec B608
    hit = conn.execute(lookup, (rowid,)).fetchone()
    if hit is None:
        return None
    memory = conn.execute("SELECT content FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if memory is None:
        return None
    return str(memory["content"]), _unpack(hit["embedding"])


def stored_sample(
    conn: sqlite3.Connection, *, table_name: str, n: int
) -> list[tuple[str, list[float]]]:
    """Up to *n* stored memories, chosen at random, each as its text and the vector stored for
    it in *table_name* -- the active embedding space's vec0 table. Reads only.

    ``memory/checked_embedder.py`` re-embeds these texts when the active space has no
    fingerprint yet, and records one only if the fresh vectors match these: whatever model
    answers first must prove it is the one that wrote the rows, not be trusted for going first.
    Random rather than the newest or oldest, so a process that checks sees any part of the
    archive. A memory whose vector is missing is passed over for the next one, so fewer than
    *n* come back only when fewer than *n* memories have a vector at all -- and none when the
    table does not exist yet, as in a file ``morgan migrate`` registered a space on before any
    store created the table.
    """
    table = _vector_table(table_name)
    if n <= 0 or not _exists(conn, table):
        return []
    # Ids only in the shuffle: the texts are read for the memories actually taken.
    candidates = conn.execute(
        "SELECT m.rowid AS rowid, m.id AS id FROM vec_meta m JOIN memories e ON e.id = m.id "
        "ORDER BY random()"
    ).fetchall()
    pairs: list[tuple[str, list[float]]] = []
    for candidate in candidates:
        found = _text_and_vector(
            conn, table=table, rowid=candidate["rowid"], memory_id=candidate["id"]
        )
        if found is None:
            continue
        pairs.append(found)
        if len(pairs) == n:
            break
    return pairs


def audit_sample(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    n: int,
    user_id: str,
    project: str,
    all_projects: bool,
) -> list[tuple[str, str, list[float]]]:
    """Up to *n* stored memories in scope, spread evenly across *table_name*'s rowids (and so
    across its vec0 chunks), each as its id, its text and the vector stored for it. Reads only.

    A sibling of ``stored_sample`` above, for ``doctor --vectors``: that one is random and
    unscoped, right for what ``checked_embedder.py`` uses it for -- any part of the archive,
    proving whoever answers wrote it, before a project even exists to scope by. This one
    reports its findings by id, so it must carry them, and it is scoped like every other
    doctor read -- *project*, or every project with *all_projects* -- so a caller auditing one
    project's vectors does not have another project's rows silently mixed in.

    Evenly spread rather than random: a fixed stride across the ordered candidates touches the
    whole table on every run, not a different random slice each time, which is what a repeat
    run comparing today against last week wants. An id whose vector or memory row is missing
    is skipped rather than replaced by a neighbour, so fewer than *n* rows come back only when
    fewer than *n* memories in scope have both -- as with ``stored_sample``, never an error.
    """
    table = _vector_table(table_name)
    if n <= 0 or not _exists(conn, table):
        return []
    sql = (
        "SELECT m.rowid AS rowid, m.id AS id FROM vec_meta m "
        "JOIN memories e ON e.id = m.id WHERE m.user_id = ?"
    )
    params: list[object] = [user_id]
    if not all_projects:
        sql += " AND m.project = ?"
        params.append(project)
    sql += " ORDER BY m.rowid"
    candidates = conn.execute(sql, params).fetchall()
    total = len(candidates)
    if total == 0:
        return []
    take = min(n, total)
    # `take` indices spread evenly across [0, total) -- a stride sample, not a prefix, so a
    # sample smaller than the table touches all of it rather than only its oldest rows.
    indices = sorted({(i * total) // take for i in range(take)})
    triples: list[tuple[str, str, list[float]]] = []
    for idx in indices:
        candidate = candidates[idx]
        found = _text_and_vector(
            conn, table=table, rowid=candidate["rowid"], memory_id=candidate["id"]
        )
        if found is None:
            continue
        text, vector = found
        triples.append((candidate["id"], text, vector))
    return triples


#: The embedding column of a vec0 table's DDL, as ``SqliteVectorIndex`` writes it.
_DECLARED_WIDTH = re.compile(r"\bembedding\s+float\[(\d+)\]")


def declared_width(conn: sqlite3.Connection, *, table_name: str) -> int | None:
    """The width *table_name*'s vec0 DDL declares, or ``None`` when the table does not exist.

    Read from the schema, not from a vector: the table keeps the width it was created at,
    whatever width was set when it is next opened -- ``CREATE ... IF NOT EXISTS`` does not
    change it, and nothing else checks it until a vector of another width is written or
    searched with.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (_vector_table(table_name),),
    ).fetchone()
    if row is None:
        return None
    found = _DECLARED_WIDTH.search(row["sql"])
    if found is None:
        raise ValueError(f"the schema of {table_name} declares no embedding width: {row['sql']}")
    return int(found.group(1))


def holds_vectors(conn: sqlite3.Connection, *, table_name: str) -> bool:
    """Whether *table_name* stores any vector at all; ``False`` when it does not exist yet. A
    vec0 scan that reads no vector column stops at the first row: under a millisecond on a
    3,000-row, 4,096-wide table."""
    table = _vector_table(table_name)
    if not _exists(conn, table):
        return False
    row = conn.execute(f"SELECT EXISTS (SELECT 1 FROM {table})").fetchone()  # noqa: S608 # nosec B608
    return bool(row[0])
