"""Synthetic checkpoint overhead; execute with --output and an isolated --db path."""

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.checkpoints import Checkpoint
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemorySource


async def measure(db: Path, fixture_digest: str) -> dict:
    def clock():
        return datetime(2026, 9, 15, tzinfo=UTC)

    conn = open_db(str(db))
    gate = MemoryGate(build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4, clock=clock))
    try:
        await gate.store(
            Memory(
                id="source",
                user_id="synthetic",
                content="Read for 20 minutes",
                source=MemorySource.USER_STATED,
            )
        )
        state = Checkpoint(
            kind="goal", title="Read German", objective="Read independently by December"
        )
        timings = {"codec_ms": [], "read_ms": [], "cas_update_ms": []}
        identity = await gate.put_checkpoint(
            state, checkpoint_id="reading", user_id="synthetic", support_event_ids=["source"]
        )
        for _ in range(100):
            start = time.perf_counter()
            encoded = state.encode(["source"])
            timings["codec_ms"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            await gate.get_checkpoint("reading", user_id="synthetic")
            timings["read_ms"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            identity = await gate.put_checkpoint(
                state,
                checkpoint_id="reading",
                user_id="synthetic",
                support_event_ids=["source"],
                expected_fact_id=identity,
            )
            timings["cas_update_ms"].append((time.perf_counter() - start) * 1000)
        return {
            "version": "checkpoint.resource.v1",
            "fixture_sha256": fixture_digest,
            "conditions": (
                "100 warmed sequential repetitions; one source, one checkpoint, "
                "101 preserved fact versions; fake 4D embeddings; no model/network; SQLite WAL"
            ),
            "repetitions": 100,
            "serialized_state_bytes": len(encoded.encode("utf-8")),
            "median_ms": {key: statistics.median(values) for key, values in timings.items()},
            "samples_ms": timings,
            "limitations": (
                "Not a model or token benchmark; native/Python RAM, large history, file fsync "
                "and simultaneous writer contention are unmeasured. Plain files can encode "
                "the same state and implement identical lifecycle/CAS policies; "
                "no universal efficiency superiority claimed."
            ),
        }
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.db.exists():
        raise ValueError("measurement requires a new isolated database")
    digest = hashlib.sha256(
        Path(__file__).with_name("resume_cases_v1.json").read_bytes()
    ).hexdigest()
    args.output.write_text(
        json.dumps(asyncio.run(measure(args.db, digest)), indent=2) + "\n", encoding="utf-8"
    )
