"""Synthetic-only benchmark of two frozen temporal store implementations.

Run from a Morgan checkout with PYTHONPATH=. and the development Python. Inputs
are source files, never existing databases. The output directory must not exist.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import platform
import sqlite3
import statistics
import sys
import time
import tracemalloc
from datetime import UTC, datetime, timedelta
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(source: Path, output: Path):
    before = digest(source)
    output.write_bytes(source.read_bytes())
    require(digest(source) == before == digest(output), "Source changed while freezing")
    spec = importlib.util.spec_from_file_location(output.stem, output)
    require(spec is not None and spec.loader is not None, "Cannot load source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, before


def populate(module, path: Path, count: int) -> datetime:
    require(not path.exists(), "Refusing to open an existing database")
    store = module.SqliteTemporalStore(str(path))
    conn = store._conn
    intervals = count // 10
    rows = []
    for owner, project in (("owner", "personal"), ("other", "personal"), ("owner", "work")):
        for key in range(10):
            for revision in range(intervals):
                identity = f"{owner}/{project}/{key}/{revision}"
                start = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=revision)
                end = start + timedelta(days=1) if revision + 1 < intervals else None
                following = f"{owner}/{project}/{key}/{revision + 1}" if end else None
                rows.append(
                    (
                        identity,
                        owner,
                        project,
                        f"subject{key}",
                        "preference",
                        "fixed synthetic value",
                        start.isoformat(),
                        end.isoformat() if end else None,
                        following,
                    )
                )
    conn.executemany(
        "INSERT INTO facts (id,user_id,project,subject,predicate,object,source,confidence,"
        "valid_from,valid_to,superseded_by,last_confirmed,author_id,scope) "
        "VALUES (?,?,?,?,?,?,'user_stated',1.0,?,?,?,NULL,'synthetic','private')",
        rows,
    )
    conn.commit()
    conn.close()
    return datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=intervals - 1, hours=12)


async def measure(modules, path: Path, count: int, at: datetime) -> dict:
    before = digest(path)
    expected = {f"owner/personal/{key}/{count // 10 - 1}" for key in range(10)}
    repeats = []
    for repeat in range(3):
        connections = {
            arm: sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) for arm in modules
        }
        try:
            stores = {
                arm: module.SqliteTemporalStore(conn=connections[arm])
                for arm, module in modules.items()
            }

            async def read(arm, stores=stores):
                return await stores[arm].current_facts(user_id="owner", project="personal", at=at)

            arms = list(modules)
            if repeat % 2:
                arms.reverse()
            for arm in arms:
                for _ in range(5):
                    require(
                        {fact.id for fact in await read(arm)} == expected, "Incorrect warmup result"
                    )
            timings = {arm: [] for arm in arms}
            for iteration in range(30):
                for arm in arms if iteration % 2 == 0 else list(reversed(arms)):
                    start = time.perf_counter_ns()
                    result = await read(arm)
                    elapsed = time.perf_counter_ns() - start
                    require({fact.id for fact in result} == expected, "Incorrect result")
                    timings[arm].append(elapsed / 1_000_000)
            peaks = {}
            for arm in arms:
                tracemalloc.start()
                result = await read(arm)
                _, peaks[arm] = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                require({fact.id for fact in result} == expected, "Incorrect result")
            repeats.append(
                {
                    "repeat": repeat,
                    "initial_order": arms,
                    "arms": {
                        arm: {
                            "median_ms": statistics.median(values),
                            "p95_ms": sorted(values)[math.ceil(0.95 * len(values)) - 1],
                            "timings_ms": values,
                            "traced_python_peak_bytes": peaks[arm],
                            "selected_rows": len(expected),
                        }
                        for arm, values in timings.items()
                    },
                }
            )
        finally:
            for conn in connections.values():
                conn.close()
    after = digest(path)
    require(before == after, "Synthetic database changed during reads")
    return {
        "facts_in_scope": count,
        "total_facts": count * 3,
        "db_sha256_before": before,
        "db_sha256_after": after,
        "repeats": repeats,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    modules, sources = {}, {}
    for arm, source in (("original", args.baseline_file), ("candidate", args.candidate_file)):
        frozen = output / f"{arm}_temporal.py"
        modules[arm], before = freeze(source.resolve(), frozen)
        sources[arm] = {
            "input_path": str(source.resolve()),
            "frozen_file": str(frozen),
            "sha256_before": before,
        }
    data = {
        "schema_version": 2,
        "sources": sources,
        "python": sys.version,
        "sqlite_version": sqlite3.sqlite_version,
        "platform": platform.platform(),
        "script_sha256": digest(Path(__file__)),
        "model_requests": 0,
        "limits": [
            "Traced Python allocations, not total RSS",
            "OS caches not flushed",
            "Direct effective-read latency, not complete recall or model latency",
            "Fixed synthetic fixtures; no general model or workload claim",
        ],
        "measurements": [],
    }
    for count in (100, 1000, 10000):
        path = output / f"synthetic-{count}.db"
        at = populate(modules["original"], path, count)
        data["measurements"].append(await measure(modules, path, count, at))
        print(f"completed {count} scoped synthetic facts", flush=True)
    for record in sources.values():
        record["sha256_after"] = digest(Path(record["frozen_file"]))
        require(record["sha256_after"] == record["sha256_before"], "Frozen source changed")
    (output / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
