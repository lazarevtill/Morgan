"""Bounded synthetic capture/check costs; no model, network or real memory database."""

import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db, write_transaction
from morgan_brain.models import Memory, TemporalFact

NOW = datetime(2026, 10, 1, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[1]


def sources():
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted([*(ROOT / "morgan_brain").rglob("*.py"), Path(__file__).resolve()])
    }


async def sample(call):
    for _ in range(5):
        await call()
    durations = []
    for _ in range(30):
        start = time.perf_counter_ns()
        await call()
        durations.append((time.perf_counter_ns() - start) / 1_000_000)
    tracemalloc.start()
    await call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "wall_ms": durations,
        "median_wall_ms": statistics.median(durations),
        "python_allocation_peak_bytes": peak,
    }


async def repeated_samples(arms):
    repeats = []
    for repeat in range(3):
        ordered = arms if repeat % 2 == 0 else list(reversed(arms))
        repeats.append({name: await sample(call) for name, call in ordered})
    return repeats


async def case(fact_count, source_count):
    conn = open_db(":memory:")
    gate = MemoryGate(
        build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4, clock=lambda: NOW)
    )
    ids = [f"source-{index}" for index in range(source_count)]
    try:
        for identity in ids:
            await gate.store(
                Memory(
                    id=identity,
                    user_id="synthetic",
                    content="Synthetic source",
                    source="user_stated",
                    author_id="person:synthetic",
                )
            )
        with write_transaction(conn):
            for index in range(fact_count):
                await gate.upsert_fact(
                    TemporalFact(
                        id=f"fact-{index}",
                        user_id="synthetic",
                        subject=f"entity-{index}",
                        predicate="state",
                        object="synthetic",
                        source="agent_inferred",
                        support_event_ids=[ids[0]],
                    )
                )
        before = hashlib.sha256(conn.serialize()).hexdigest()

        async def capture():
            return await gate.capture_consolidation_basis(
                user_id="synthetic",
                project="personal",
                event_ids=ids,
                generation=gate.capture_erasure_generation(),
            )

        inputs = await capture()

        async def check():
            with gate.write_transaction():
                await gate.check_consolidation_basis(inputs.basis, effective_at=NOW)

        async def read_only_baseline():
            await gate.current_facts(user_id="synthetic", project="personal")
            for offset in range(0, len(ids), 32):
                await gate.evidence(
                    user_id="synthetic",
                    project="personal",
                    evidence_ids=ids[offset : offset + 32],
                    effective_at=NOW,
                )

        repeats = await repeated_samples(
            [
                ("scoped_reads_without_cas", read_only_baseline),
                ("capture", capture),
                ("check_under_lock", check),
            ]
        )
        after = hashlib.sha256(conn.serialize()).hexdigest()
        if before != after:
            raise RuntimeError("Synthetic database changed during measurement")
        return {
            "fact_count": fact_count,
            "source_count": source_count,
            "db_sha256_before": before,
            "db_sha256_after": after,
            "repeats": repeats,
        }
    finally:
        conn.close()


async def measure(output_directory):
    output_directory.mkdir(parents=True, exist_ok=False)
    before = sources()
    result = {
        "scope": "Model-free synthetic resource accounting; baseline has no CAS guarantee",
        "python_version": platform.python_version(),
        "allocation_metric": "Python tracemalloc peak, not total process RSS",
        "warmups": 5,
        "timed_samples": 30,
        "repeats": 3,
        "cases": [await case(facts, roots) for facts, roots in ((10, 10), (100, 20), (256, 50))],
        "source_sha256_before": before,
        "source_sha256_after": sources(),
    }
    if result["source_sha256_before"] != result["source_sha256_after"]:
        raise RuntimeError("Source changed during measurement")
    with (output_directory / "results.json").open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2)
        target.write("\n")
    for item in result["cases"]:
        print(
            item["fact_count"],
            item["source_count"],
            [
                {name: round(values["median_wall_ms"], 3) for name, values in repeat.items()}
                for repeat in item["repeats"]
            ],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New synthetic result directory; never a memory DB path",
    )
    args = parser.parse_args()
    asyncio.run(measure(args.output_dir))


if __name__ == "__main__":
    main()
