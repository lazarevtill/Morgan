"""A scorecard carries what it was measured under.

Retrieval numbers move with the embedding model, the window, the relevance floor, the probe
file and the database's upgrade state. A card printed without them cannot be compared with
the next one, and the comparisons that decide what recall does were run as scripts whose
output recorded none of it.
"""

from __future__ import annotations

import json

from morgan_brain.composition import build_memory_module
from morgan_brain.config import Settings
from morgan_brain.eval.retrieval import describe_run, load_probe_set
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.store.db import open_db

ENDPOINT = "http://embeddings.internal.example:8083/v1"


def _probe_file(tmp_path):
    path = tmp_path / "probes.json"
    path.write_text(
        json.dumps(
            {
                "corpus": {"m1": "the deploy was blocked", "m2": "lunch at noon"},
                "probes": [
                    {"id": "p1", "kind": "single_hop", "query": "deploy", "expected": ["m1"]}
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _described(tmp_path, *, floor_margin):
    settings = Settings(
        embedding_backend="provider",
        embedding_model="qwen3-embedding",
        embedding_dim=1024,
        embedding_endpoint=ENDPOINT,
    )
    path = _probe_file(tmp_path)
    conn = open_db(str(tmp_path / "morgan.db"))
    build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4)
    return describe_run(
        settings=settings,
        probe_path=path,
        probe_set=load_probe_set(path),
        conn=conn,
        k=8,
        floor_margin=floor_margin,
        commit="6795b3d",
    )


def test_the_block_names_everything_the_numbers_depend_on(tmp_path):
    block = _described(tmp_path, floor_margin=0.11).format()

    for expected in (
        "qwen3-embedding",
        "dim 1024",
        "k=8",
        "floor=0.11",
        "probes.json@",
        "2 memories",
        "1 probes",
        "db-upgrade=2",
        "commit=6795b3d",
    ):
        assert expected in block, (expected, block)


def test_no_floor_reads_as_off(tmp_path):
    assert "floor=off" in _described(tmp_path, floor_margin=None).format()


def test_the_probe_file_is_identified_by_its_content(tmp_path):
    """Same name, edited labels: the two runs must not look comparable."""
    before = _described(tmp_path, floor_margin=None)
    path = tmp_path / "probes.json"
    path.write_text(path.read_text(encoding="utf-8").replace('["m1"]', '["m2"]'), "utf-8")

    after = describe_run(
        settings=Settings(embedding_backend="hash"),
        probe_path=path,
        probe_set=load_probe_set(path),
        conn=open_db(str(tmp_path / "morgan.db")),
        k=8,
        floor_margin=None,
        commit=None,
    )

    assert before.probe_digest != after.probe_digest


def test_the_hash_stub_is_not_named_after_a_model_it_does_not_call(tmp_path):
    path = _probe_file(tmp_path)
    conn = open_db(str(tmp_path / "morgan.db"))
    run = describe_run(
        settings=Settings(embedding_backend="hash", embedding_model="mxbai-embed-large"),
        probe_path=path,
        probe_set=load_probe_set(path),
        conn=conn,
        k=8,
        floor_margin=None,
        commit=None,
    )

    assert "mxbai" not in run.format()
    assert "embedding=hash stub" in run.format()


def test_the_endpoint_is_left_out(tmp_path):
    """A test log is the thing that gets pasted into an issue; the model identifies the run and
    the host would only publish where it runs."""
    block = _described(tmp_path, floor_margin=None).format()

    assert "internal.example" not in block
    assert "8083" not in block
