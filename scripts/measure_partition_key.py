"""Does a vec0 `PARTITION KEY` on `project` beat the current metadata-column table? -- the
numbers the `PARTITION KEY` verdict is set from (`docs/measurements/2026-09-phase0-baseline.md`
§4, `morgan_brain/memory/store/vectors.py`'s module docstring).

`morgan_brain/memory/store/vectors.py` scopes every KNN query by `user_id`/`project` as vec0
*metadata* columns, inside the search. sqlite-vec 0.1.9 also supports declaring one metadata
column a `PARTITION KEY`, which shards the index by that column instead of filtering a shared
one -- possibly faster when a query always scopes to exactly one partition, at the cost of a
schema only migration step 6 can decide whether to adopt.

One run reports, on a copy of a database it is given:

* **per-project** KNN time for both layouts -- the query scoped to `user_id` *and* `project`,
  the shape `SqliteVectorIndex.search` always uses today;
* **all-projects** KNN time for both layouts -- the query scoped to `user_id` only, the shape
  `search` uses when the caller passes `project=None` (`all_projects=True` in `MemoryGate`).
  Against the partitioned table this forces vec0 to scan every partition instead of one;
* whether the two layouts return the **same ids, in the same order** at each scope --
  `id_order_mismatches` (order differs) and `id_set_mismatches` (the set itself differs, a
  strict superset of the order count);
* the **on-disk size** of each layout, isolated into its own single-table file so the number
  is not entangled with anything else the source database holds: once for the rows `--db`
  holds (labelled "same rows" below), and once for `--synthetic-projects` synthetic
  one-memory projects -- a project holding a single vector still pays for a whole partition
  chunk, which is the cost that scales with the number of projects rather than the number of
  vectors.

This script never sends anything over the network: it uses each sampled memory's own stored
vector as its query vector, so there is no embedding call and no model server involved. It
copies *--db* first (default: a temp file, deleted after unless `--keep-copy`) and only ever
builds the partitioned table and writes to that copy -- the source is opened read-only and
never touched. The synthetic-projects measurement never reads *--db*'s rows, only the vector
width sampled from it, and writes to its own throwaway files under a temp directory.

    measure_partition_key.py --db SNAPSHOT.db --queries 200 --k 16 --runs 2 \\
        --synthetic-projects 20 --results-json OUT.json

Nothing about the database, host, or model is hardcoded -- everything comes from *--db* and
the numeric options above.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import statistics
import struct
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlite_vec  # type: ignore[import-untyped]

PARTITIONED_TABLE = "vec_items_pk"
_SOLO_TABLE = "vec_solo"


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)
    return conn


def build_partition_table(conn: sqlite3.Connection, dim: int) -> int:
    """Create ``vec_items_pk`` -- the existing schema plus ``project`` as a `PARTITION KEY`
    instead of a plain metadata column -- and copy every row from ``vec_items`` into it.
    Returns the row count copied, for a sanity check against the source table."""
    conn.executescript(
        f"""
        DROP TABLE IF EXISTS {PARTITIONED_TABLE};
        CREATE VIRTUAL TABLE {PARTITIONED_TABLE} USING vec0(
            embedding float[{dim}] distance_metric=cosine,
            user_id TEXT,
            project TEXT PARTITION KEY
        );
        """
    )
    conn.commit()
    rows = conn.execute("SELECT rowid, embedding, user_id, project FROM vec_items").fetchall()
    conn.executemany(
        # PARTITIONED_TABLE is the module constant above, never caller input.
        f"INSERT INTO {PARTITIONED_TABLE} (rowid, embedding, user_id, project) "  # noqa: S608
        "VALUES (?, ?, ?, ?)",
        [(r["rowid"], r["embedding"], r["user_id"], r["project"]) for r in rows],
    )
    conn.commit()
    return len(rows)


def sample_queries(conn: sqlite3.Connection, n: int) -> list[sqlite3.Row]:
    """*n* rows spread evenly across ``vec_items``, each supplying its own vector as the
    query and its own ``user_id``/``project`` as the scope -- deterministic in (table
    contents, *n*), like ``measure_repeat_distribution.py``'s sampling."""
    all_rows = conn.execute(
        "SELECT rowid, embedding, user_id, project FROM vec_items ORDER BY rowid"
    ).fetchall()
    total = len(all_rows)
    if total == 0:
        raise SystemExit("no rows found in vec_items")
    picked = min(n, total)
    if picked == total:
        indices: Sequence[int] = range(total)
    else:
        indices = [int(i * total / picked) for i in range(picked)]
    return [all_rows[i] for i in indices]


def query_once(
    conn: sqlite3.Connection,
    table: str,
    row: sqlite3.Row,
    k: int,
    *,
    scoped_to_project: bool,
) -> tuple[list[int], float]:
    """*table* is always one of the two module-level table names below, never caller input.

    *scoped_to_project* mirrors the two shapes ``SqliteVectorIndex.search`` queries in: with
    ``project`` bound (the normal, per-project case) and without it (``all_projects=True``).
    """
    sql = f"SELECT rowid FROM {table} WHERE embedding MATCH ? AND k = ? AND user_id = ?"  # noqa: S608
    params: list[object] = [row["embedding"], k, row["user_id"]]
    if scoped_to_project:
        sql += " AND project = ?"
        params.append(row["project"])
    sql += " ORDER BY distance"
    start = time.perf_counter()
    got = conn.execute(sql, params).fetchall()
    elapsed = time.perf_counter() - start
    return [r["rowid"] for r in got], elapsed


@dataclass
class RunResult:
    scope: str
    metadata_ms: list[float]
    partition_ms: list[float]
    order_mismatches: int
    set_mismatches: int
    n_queries: int


def compare(
    conn: sqlite3.Connection,
    queries: Sequence[sqlite3.Row],
    k: int,
    *,
    scoped_to_project: bool,
) -> RunResult:
    metadata_ms: list[float] = []
    partition_ms: list[float] = []
    order_mismatches = 0
    set_mismatches = 0
    for row in queries:
        meta_ids, meta_t = query_once(
            conn, "vec_items", row, k, scoped_to_project=scoped_to_project
        )
        pk_ids, pk_t = query_once(
            conn, PARTITIONED_TABLE, row, k, scoped_to_project=scoped_to_project
        )
        metadata_ms.append(meta_t * 1000)
        partition_ms.append(pk_t * 1000)
        if meta_ids != pk_ids:
            order_mismatches += 1
            if set(meta_ids) != set(pk_ids):
                set_mismatches += 1
    return RunResult(
        scope="per-project" if scoped_to_project else "all-projects",
        metadata_ms=metadata_ms,
        partition_ms=partition_ms,
        order_mismatches=order_mismatches,
        set_mismatches=set_mismatches,
        n_queries=len(queries),
    )


def _stats(ms: list[float]) -> dict[str, float]:
    return {
        "min": min(ms),
        "median": statistics.median(ms),
        "mean": statistics.mean(ms),
        "max": max(ms),
    }


def build_isolated_table(
    path: Path,
    dim: int,
    rows: Sequence[tuple[int, bytes, str, str]],
    *,
    partitioned: bool,
) -> int:
    """A fresh file holding exactly one vec0 table -- metadata-column or `PARTITION KEY` --
    populated with *rows* (``rowid``, embedding blob, ``user_id``, ``project``). Returns the
    file size in bytes after ``VACUUM``, so the number reflects committed pages only, not
    incidental slack. The caller deletes *path* when it is done with it."""
    path.unlink(missing_ok=True)
    conn = connect(path)
    try:
        kind = "project TEXT PARTITION KEY" if partitioned else "project TEXT"
        conn.executescript(
            f"""
            CREATE VIRTUAL TABLE {_SOLO_TABLE} USING vec0(
                embedding float[{dim}] distance_metric=cosine,
                user_id TEXT,
                {kind}
            );
            """
        )
        conn.commit()
        conn.executemany(
            f"INSERT INTO {_SOLO_TABLE} (rowid, embedding, user_id, project) "  # noqa: S608
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.execute("VACUUM")
        return path.stat().st_size
    finally:
        conn.close()


def measure_storage(
    tmp_dir: Path,
    dim: int,
    rows: Sequence[tuple[int, bytes, str, str]],
    *,
    label: str,
) -> dict[str, int]:
    """The isolated on-disk size of each layout for the same *rows*, as bytes."""
    md_path = tmp_dir / f"solo-metadata-{label}.db"
    pk_path = tmp_dir / f"solo-partition-{label}.db"
    try:
        metadata_bytes = build_isolated_table(md_path, dim, rows, partitioned=False)
        partition_bytes = build_isolated_table(pk_path, dim, rows, partitioned=True)
    finally:
        md_path.unlink(missing_ok=True)
        pk_path.unlink(missing_ok=True)
    return {"metadata_column_bytes": metadata_bytes, "partition_key_bytes": partition_bytes}


def synthetic_one_memory_projects(n: int, dim: int) -> list[tuple[int, bytes, str, str]]:
    """*n* projects of one memory each, one row per project -- every row a distinct unit
    vector, one user. What a partition pays for holding a single vector does not depend on
    what that vector is, only on there being one project per chunk."""
    rows = []
    for i in range(n):
        vector = [1.0 if j == i % dim else 0.0 for j in range(dim)]
        blob = struct.pack(f"{dim}f", *vector)
        rows.append((i + 1, blob, "synthetic-user", f"synthetic-project-{i}"))
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--db", required=True, type=Path, help="Source snapshot (read-only).")
    parser.add_argument(
        "--work-copy",
        type=Path,
        default=None,
        help="Where to copy --db before building the partitioned table. Defaults to a "
        "temp file, deleted afterward unless --keep-copy.",
    )
    parser.add_argument(
        "--keep-copy",
        action="store_true",
        help="Don't delete --work-copy afterward (to inspect the partitioned table).",
    )
    parser.add_argument("--queries", type=int, default=200, help="Queries, spread evenly.")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--runs", type=int, default=2, help="Independent repeats, for stability.")
    parser.add_argument(
        "--synthetic-projects",
        type=int,
        default=20,
        help="One-memory projects to build for the storage-per-project measurement.",
    )
    parser.add_argument("--results-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    work_copy = args.work_copy
    owns_temp = work_copy is None
    if owns_temp:
        fd, name = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        work_copy = Path(name)
    storage_dir = Path(tempfile.mkdtemp(prefix="measure-partition-key-storage-"))

    try:
        shutil.copy2(args.db, work_copy)
        conn = connect(work_copy)
        try:
            dim = conn.execute("SELECT length(embedding) FROM vec_items LIMIT 1").fetchone()[0] // 4
            source_rows = conn.execute("SELECT COUNT(*) FROM vec_items").fetchone()[0]
            copied = build_partition_table(conn, dim)
            if copied != source_rows:
                raise SystemExit(f"copied {copied} rows but vec_items has {source_rows}")

            queries = sample_queries(conn, args.queries)
            runs: list[dict[str, Any]] = []
            for i in range(1, args.runs + 1):
                for scoped_to_project in (True, False):
                    result = compare(conn, queries, args.k, scoped_to_project=scoped_to_project)
                    run_record = {
                        "run": i,
                        "scope": result.scope,
                        "n_queries": result.n_queries,
                        "k": args.k,
                        "metadata_column_ms": _stats(result.metadata_ms),
                        "partition_key_ms": _stats(result.partition_ms),
                        "id_order_mismatches": result.order_mismatches,
                        "id_set_mismatches": result.set_mismatches,
                    }
                    runs.append(run_record)
                    m, p = run_record["metadata_column_ms"], run_record["partition_key_ms"]
                    print(
                        f"run {i} [{result.scope}]: metadata median={m['median']:.3f}ms "
                        f"mean={m['mean']:.3f}ms | partition median={p['median']:.3f}ms "
                        f"mean={p['mean']:.3f}ms | order mismatches="
                        f"{result.order_mismatches}/{result.n_queries} | set mismatches="
                        f"{result.set_mismatches}/{result.n_queries}"
                    )

            same_rows = [
                (r["rowid"], r["embedding"], r["user_id"], r["project"])
                for r in conn.execute("SELECT rowid, embedding, user_id, project FROM vec_items")
            ]
            same_rows_storage = measure_storage(storage_dir, dim, same_rows, label="same-rows")
            print(
                f"storage, same rows ({len(same_rows)}): "
                f"metadata={same_rows_storage['metadata_column_bytes']:,}B "
                f"partition={same_rows_storage['partition_key_bytes']:,}B"
            )

            synthetic_rows = synthetic_one_memory_projects(args.synthetic_projects, dim)
            synthetic_storage = measure_storage(storage_dir, dim, synthetic_rows, label="synthetic")
            print(
                f"storage, {args.synthetic_projects} one-memory projects: "
                f"metadata={synthetic_storage['metadata_column_bytes']:,}B "
                f"partition={synthetic_storage['partition_key_bytes']:,}B"
            )
        finally:
            conn.close()
    finally:
        if owns_temp and not args.keep_copy:
            work_copy.unlink(missing_ok=True)
        elif args.keep_copy:
            print(f"kept work copy at {work_copy}")
        shutil.rmtree(storage_dir, ignore_errors=True)

    if args.results_json:
        payload = {
            "db": str(args.db),
            "dim": dim,
            "source_rows": source_rows,
            "k": args.k,
            "queries": args.queries,
            "synthetic_projects": args.synthetic_projects,
            "runs": runs,
            "storage_same_rows_bytes": same_rows_storage,
            "storage_synthetic_one_memory_projects_bytes": synthetic_storage,
        }
        args.results_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"wrote {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
