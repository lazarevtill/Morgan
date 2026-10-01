"""Two consolidation runs applying the same facts at once add each fact once.

`morgan consolidate` applies a batch of fact operations against the facts it reads as current.
Two runs over one database -- the owner's cron and a manual run, say -- that each read the
current facts before applying anything both see a new fact as absent and both add it, so every
key ends with the fact twice: the second copy superseding an identical first one.

Both processes apply the same batch of ADDs at the same moment. Whichever applies second must
see what the first added and skip it, leaving one fact per key and each ADD applied once in all.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.knowledge.basis import StaleConsolidationProposal
from morgan_brain.memory.knowledge.consolidation import (
    FactOp,
    FactOpBatch,
    FactOpKind,
    MemoryConsolidator,
)
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemorySource
from tests.fakes import FakeChatClient

if TYPE_CHECKING:
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Barrier

_DIM = 4
_KEYS = 200
_RUNS = 2
_TIMEOUT_S = 120

_BATCH = FactOpBatch(
    ops=[
        FactOp(
            op=FactOpKind.ADD,
            subject=f"s{i}",
            predicate="lives_in",
            object="Berlin",
            support_event_ids=["source"],
        )
        for i in range(_KEYS)
    ]
)


def _apply(path: str, start: Barrier, outcomes: Queue[str]) -> None:
    """Runs in a spawned process: apply the batch once, report how many ops it applied."""
    try:
        conn = open_db(path)
        gate = MemoryGate(build_memory_module(conn, embedder=FakeEmbedder(dim=_DIM), dim=_DIM))
        consolidator = MemoryConsolidator(
            gate=gate,
            client=FakeChatClient(replies=[]),
            model="unused: apply() never calls the model",
            clock=lambda: datetime.now(UTC),
        )
        inputs = asyncio.run(
            gate.capture_consolidation_basis(
                user_id="u",
                project="p",
                event_ids=["source"],
                generation=gate.capture_erasure_generation(),
            )
        )
        start.wait(timeout=_TIMEOUT_S)
        try:
            applied = asyncio.run(consolidator.apply("u", _BATCH, project="p", basis=inputs.basis))
            result = str(len(applied))
        except StaleConsolidationProposal:
            result = "stale"
        conn.close()
    except BaseException as exc:
        outcomes.put(f"error: {type(exc).__name__}: {exc}")
        raise
    outcomes.put(result)


def test_two_runs_applying_the_same_facts_add_each_once(tmp_path: Path) -> None:
    path = str(tmp_path / "morgan.db")
    setup = open_db(path)
    gate = MemoryGate(build_memory_module(setup, embedder=FakeEmbedder(dim=_DIM), dim=_DIM))
    asyncio.run(
        gate.store(
            Memory(
                id="source",
                user_id="u",
                project="p",
                content="Synthetic authored contract: the 200 subjects live in Berlin",
                source=MemorySource.USER_STATED,
                author_id="person:u",
            )
        )
    )
    setup.close()

    ctx = mp.get_context("spawn")
    start = ctx.Barrier(_RUNS)
    outcomes: Queue[str] = ctx.Queue()
    runs = [ctx.Process(target=_apply, args=(path, start, outcomes)) for _ in range(_RUNS)]
    for p in runs:
        p.start()
    try:
        reported = [outcomes.get(timeout=_TIMEOUT_S) for _ in runs]
        for p in runs:
            p.join(timeout=_TIMEOUT_S)
    finally:
        for p in runs:
            if p.is_alive():
                p.terminate()

    assert not [r for r in reported if r.startswith("error")], reported
    assert [p.exitcode for p in runs] == [0] * _RUNS

    conn = open_db(path)
    try:
        per_key = conn.execute(
            "SELECT subject, COUNT(*) AS n FROM facts GROUP BY subject, predicate"
        ).fetchall()
    finally:
        conn.close()

    assert len(per_key) == _KEYS
    duplicated = [r["subject"] for r in per_key if r["n"] != 1]
    assert duplicated == [], f"{len(duplicated)} of {_KEYS} facts were added twice"
    assert sorted(reported) == [str(_KEYS), "stale"]
    # Fresh explicit preparation sees the winning state and deduplicates without a write.
    conn = open_db(path)
    try:
        gate = MemoryGate(build_memory_module(conn, embedder=FakeEmbedder(dim=_DIM), dim=_DIM))
        inputs = asyncio.run(
            gate.capture_consolidation_basis(
                user_id="u",
                project="p",
                event_ids=["source"],
                generation=gate.capture_erasure_generation(),
            )
        )
        consolidator = MemoryConsolidator(
            gate=gate, client=FakeChatClient(), model="unused", clock=lambda: datetime.now(UTC)
        )
        before = conn.serialize()
        assert asyncio.run(consolidator.apply("u", _BATCH, project="p", basis=inputs.basis)) == []
        assert conn.serialize() == before
    finally:
        conn.close()
