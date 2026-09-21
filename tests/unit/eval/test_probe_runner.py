"""Running probes against a real gate. Uses the hash embedder, so it checks the harness's
mechanics -- what was stored, how ids map back to probe keys -- and never its relevance,
which a hash embedder does not have."""

from __future__ import annotations

import pytest

from morgan_brain.composition import build_memory_module
from morgan_brain.eval.retrieval import Probe, ProbeKind, ProbeSet, run_probes
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db


@pytest.fixture
def gate(tmp_path):
    conn = open_db(str(tmp_path / "morgan.db"))
    return MemoryGate(build_memory_module(conn=conn, embedder=FakeEmbedder(dim=8), dim=8))


async def test_every_probe_produces_a_result_naming_the_keys_that_came_back(gate):
    probe_set = ProbeSet(
        corpus={"m1": "the deploy was blocked by a registry mirror", "m2": "unrelated"},
        probes=(
            Probe(id="p1", kind=ProbeKind.SINGLE_HOP, query="registry", expected=("m1",)),
            Probe(id="p2", kind=ProbeKind.TEMPORAL, query="unrelated", expected=("m2",)),
        ),
    )

    results = await run_probes(probe_set, gate=gate, user_id="owner", k=5)

    assert [r.probe.id for r in results] == ["p1", "p2"]
    assert all(set(r.retrieved) <= {"m1", "m2"} for r in results)


async def test_the_corpus_is_isolated_in_its_own_project(gate):
    """A measurement that shares a project with real memories measures those too, and a
    number that moves when the owner stores something unrelated is not a measurement."""
    probe_set = ProbeSet(
        corpus={"m1": "harbor mirror"},
        probes=(Probe(id="p1", kind=ProbeKind.SINGLE_HOP, query="harbor", expected=("m1",)),),
    )

    await run_probes(probe_set, gate=gate, user_id="owner", k=5, project="eval/run-1")

    from morgan_brain.models import MemoryQuery

    elsewhere = await gate.recall(
        MemoryQuery(user_id="owner", project="some-real-project", text="harbor", top_k=5)
    )
    assert elsewhere.memories == []


async def test_a_probe_that_retrieves_nothing_still_produces_a_result(gate):
    """Phase 4 adds a relevance floor, after which an honest run returns nothing for an
    unanswerable question. The harness must score that as a miss, not drop the probe."""
    probe_set = ProbeSet(
        corpus={"m1": "harbor mirror"},
        probes=(Probe(id="p1", kind=ProbeKind.SINGLE_HOP, query="harbor", expected=("m1",)),),
    )

    results = await run_probes(
        ProbeSet(corpus={}, probes=probe_set.probes), gate=gate, user_id="owner", k=5
    )

    assert len(results) == 1
    assert results[0].retrieved == ()
