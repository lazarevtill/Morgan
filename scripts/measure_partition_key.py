"""Does a vec0 `PARTITION KEY` on `project` beat the current metadata-column table? -- the
number the `PARTITION KEY` verdict (Task 16) is set from.

`morgan_brain/memory/store/vectors.py` scopes every KNN query by `user_id`/`project` as vec0
*metadata* columns, inside the search. sqlite-vec 0.1.9 also supports declaring one metadata
column a `PARTITION KEY`, which shards the index by that column instead of filtering a shared
one -- possibly faster when a query always scopes to exactly one partition, at the cost of a
schema only Task 16 can decide whether to adopt.

This script never sends anything over the network: it uses each sampled memory's own stored
vector as its query vector, so there is no embedding call and no model server involved. It
copies *--db* first (default: a temp file, deleted after unless `--keep-copy`) and only ever
builds the partitioned table and writes to that copy -- the source is opened read-only and
never touched.

    measure_partition_key.py --db SNAPSHOT.db --queries 200 --k 8 --runs 2 \\
        --results-json OUT.json

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
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlite_vec  # type: ignore[import-untyped]

PARTITIONED_TABLE = "vec_items_pk"


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
    conn: sqlite3.Connection, table: str, row: sqlite3.Row, k: int
) -> tuple[list[int], float]:
    """*table* is always one of the two module-level table names below, never caller input."""
    sql = (
        f"SELECT rowid FROM {table} "  # noqa: S608
        "WHERE embedding MATCH ? AND k = ? AND user_id = ? AND project = ? "
        "ORDER BY distance"
    )
    params = (row["embedding"], k, row["user_id"], row["project"])
    start = time.perf_counter()
    got = conn.execute(sql, params).fetchall()
    elapsed = time.perf_counter() - start
    return [r["rowid"] for r in got], elapsed


@dataclass
class RunResult:
    metadata_ms: list[float]
    partition_ms: list[float]
    mismatches: int
    n_queries: int


def compare(conn: sqlite3.Connection, queries: Sequence[sqlite3.Row], k: int) -> RunResult:
    metadata_ms: list[float] = []
    partition_ms: list[float] = []
    mismatches = 0
    for row in queries:
        meta_ids, meta_t = query_once(conn, "vec_items", row, k)
        pk_ids, pk_t = query_once(conn, PARTITIONED_TABLE, row, k)
        metadata_ms.append(meta_t * 1000)
        partition_ms.append(pk_t * 1000)
        if meta_ids != pk_ids:
            mismatches += 1
    return RunResult(
        metadata_ms=metadata_ms,
        partition_ms=partition_ms,
        mismatches=mismatches,
        n_queries=len(queries),
    )


def _stats(ms: list[float]) -> dict[str, float]:
    return {
        "min": min(ms),
        "median": statistics.median(ms),
        "mean": statistics.mean(ms),
        "max": max(ms),
    }


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
                result = compare(conn, queries, args.k)
                run_record = {
                    "run": i,
                    "n_queries": result.n_queries,
                    "k": args.k,
                    "metadata_column_ms": _stats(result.metadata_ms),
                    "partition_key_ms": _stats(result.partition_ms),
                    "id_mismatches": result.mismatches,
                }
                runs.append(run_record)
                m, p = run_record["metadata_column_ms"], run_record["partition_key_ms"]
                print(
                    f"run {i}: metadata median={m['median']:.3f}ms mean={m['mean']:.3f}ms | "
                    f"partition median={p['median']:.3f}ms mean={p['mean']:.3f}ms | "
                    f"mismatches={result.mismatches}/{result.n_queries}"
                )
        finally:
            conn.close()
    finally:
        if owns_temp and not args.keep_copy:
            work_copy.unlink(missing_ok=True)
        elif args.keep_copy:
            print(f"kept work copy at {work_copy}")

    if args.results_json:
        payload = {"db": str(args.db), "dim": dim, "source_rows": source_rows, "runs": runs}
        args.results_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"wrote {args.results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
