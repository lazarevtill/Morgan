"""Spec §7 milestone 1 acceptance: store a fact in one repo, restart, recall it from a
different repo -- with a real embedder, on disk.

Every command below runs the ``morgan`` CLI as a real subprocess, in real git repositories
under separate ``cwd``s that share nothing but ``MORGAN_DATA_DIR``. That is deliberate: it
proves what a human at a terminal would experience (project auto-detection from the git
root, durability across process restarts) rather than what a Python object graph does
sharing one in-memory connection.

Before this milestone Morgan lost two of its three retrieval signals (vectors, entities) on
every restart and returned nothing for Cyrillic queries. Nothing here may be weakened to make
it pass -- a green test that does not exercise the durable path is worse than no test at all.

**What this file can and cannot prove.** Recall has no relevance floor: every signal returns
its top-k regardless of score, so with one memory in the database the vector index returns
that memory for *any* query and RRF fuses it into the results. A CLI-level assertion of the
form "recall X returns the memory" therefore cannot distinguish a working keyword signal from
a broken one -- verified by reverting the tokenizer to its historical ``[a-z0-9]+`` bug, which
left every test here green. Signal-level correctness is asserted where it is decidable:
``tests/unit/memory/test_fts.py::test_finds_cyrillic_term`` fails on that same revert. What
this file uniquely proves is the end-to-end path -- process restart, git-root project
detection, on-disk durability, encoding integrity, and project isolation, which is a hard
filter rather than a ranked signal and so *is* decidable here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _morgan(
    args: list[str], *, cwd: Path, data_dir: Path, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "MORGAN_DATA_DIR": str(data_dir), "MORGAN_EMBEDDING_BACKEND": "hash"}
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "morgan_brain.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        check=False,
    )


def _init_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    return path


def test_cross_repo_recall_after_restart(tmp_path: Path) -> None:
    data = tmp_path / "brain"
    repo_a = _init_repo(tmp_path / "acme")
    repo_b = _init_repo(tmp_path / "personal")

    stored = _morgan(
        ["remember", "ACME-14802 blocked on the Harbor mirror, not the chart"],
        cwd=repo_a,
        data_dir=data,
    )
    assert stored.returncode == 0, stored.stderr

    # A second memory, in the second repository. Without it "the recall found the only row
    # in the database" and "the recall found the right row" are the same observation, and
    # the project-scoping assertion below has nothing it could fail on.
    decoy = _morgan(["remember", "жёлтая папка лежит на верхней полке"], cwd=repo_b, data_dir=data)
    assert decoy.returncode == 0, decoy.stderr

    # A separate process, in a separate repository -- nothing is shared but the database
    # on disk. This is the restart: no in-memory state survives between the two subprocess
    # invocations.
    out = _morgan(["recall", "harbor", "--all-projects", "--json"], cwd=repo_b, data_dir=data)
    assert out.returncode == 0, out.stderr
    results = json.loads(out.stdout)["results"]
    assert any("Harbor mirror" in r["content"] for r in results)
    assert results[0]["project"] == "acme"

    # The same query without --all-projects, from the same cwd. Project scoping is a hard
    # filter (inside the vec0 KNN and the FTS WHERE clause), not a ranked signal, so this
    # assertion is decidable regardless of the relevance-floor problem above: repo_b's
    # project must never see repo_a's memory.
    scoped = _morgan(["recall", "harbor", "--json"], cwd=repo_b, data_dir=data)
    assert scoped.returncode == 0, scoped.stderr
    scoped_results = json.loads(scoped.stdout)["results"]
    assert all(r["project"] == "personal" for r in scoped_results), scoped_results
    assert not any("Harbor mirror" in r["content"] for r in scoped_results), scoped_results


def test_vectors_are_actually_persisted_not_just_fts(tmp_path: Path) -> None:
    """Without this, the whole suite passes on the FTS signal alone.

    Under the hash embedding backend there is no semantic similarity, so a recall assertion
    alone cannot distinguish "vectors work" from "vectors were never persisted and FTS
    carried the result" -- which is exactly how a milestone declares itself done while
    production memory stays ephemeral. Assert the row count on disk instead.
    """
    data = tmp_path / "brain"
    repo = _init_repo(tmp_path / "repo")

    stored = _morgan(["remember", "the Harbor mirror blocked the deploy"], cwd=repo, data_dir=data)
    assert stored.returncode == 0, stored.stderr

    doctor = json.loads(_morgan(["doctor", "--json"], cwd=repo, data_dir=data).stdout)
    # `or 0` is not defensive padding: doctor reports vector_rows=None for a non-sqlite
    # backend, and a bare `None > 0` raises TypeError instead of failing this assertion with
    # the doctor payload attached. The misconfiguration is exactly what the test is for.
    assert (doctor["vector_rows"] or 0) > 0, doctor
    assert doctor["database"].endswith("morgan.db"), doctor


@pytest.mark.live
def test_real_embedder_round_trip(tmp_path: Path) -> None:
    """The spec's acceptance criterion says "with a real embedder". The deterministic hash
    stub used everywhere else in this file has no semantic similarity and cannot satisfy
    that -- this is the one test in the suite that actually proves it.

    Skipped by default (see tests/conftest.py); run with ``pytest --live`` against a
    reachable embedding endpoint (``MORGAN_LLM_ENDPOINT``, default
    ``http://localhost:8081/v1``).
    """
    data = tmp_path / "brain"
    repo = _init_repo(tmp_path / "repo")
    provider = {"MORGAN_EMBEDDING_BACKEND": "provider"}

    # Decoys first, so ranking has to mean something. With a single memory stored, the
    # unfloored top-k returns it for any query and the assertion below would hold even
    # against a random embedding.
    for decoy in (
        "жёлтая папка лежит на верхней полке",
        "the espresso machine on the second floor takes coins",
        "Anna prefers code review comments in English",
    ):
        assert (
            _morgan(["remember", decoy], cwd=repo, data_dir=data, extra_env=provider).returncode
            == 0
        )

    stored = _morgan(
        ["remember", "our deploy was blocked by a registry mirror"],
        cwd=repo,
        data_dir=data,
        extra_env=provider,
    )
    assert stored.returncode == 0, stored.stderr

    # Query and target share no token: {what, stopped, yesterdays, rollout} against
    # {our, deploy, was, blocked, by, a, registry, mirror}. The keyword signal therefore
    # cannot produce this hit -- only a genuine embedding model bridges it, and it must
    # bridge it well enough to outrank three unrelated memories.
    out = _morgan(
        ["recall", "what stopped yesterdays rollout?", "--json"],
        cwd=repo,
        data_dir=data,
        extra_env=provider,
    )
    assert out.returncode == 0, out.stderr
    results = json.loads(out.stdout)["results"]
    assert results, results
    assert "registry mirror" in results[0]["content"], results


def test_cyrillic_survives_the_same_round_trip(tmp_path: Path) -> None:
    """Cyrillic survives the CLI -> SQLite -> CLI round trip byte-for-byte.

    This asserts *encoding integrity* across the process boundary -- argv decoding, the
    SQLite text round trip, and JSON output -- not tokenizer correctness. It cannot assert
    the latter: with the tokenizer reverted to its historical ``[a-z0-9]+`` bug this test
    still passes, because the unfloored vector top-k returns the only stored row anyway.
    ``tests/unit/memory/test_fts.py::test_finds_cyrillic_term`` is the guard that fails on
    that revert; mojibake is what would slip past it and get caught here.
    """
    data = tmp_path / "brain"
    repo = _init_repo(tmp_path / "repo")
    content = "Ромашка сохранила образец в архиве"

    stored = _morgan(["remember", content], cwd=repo, data_dir=data)
    assert stored.returncode == 0, stored.stderr

    out = _morgan(["recall", "образец", "--json"], cwd=repo, data_dir=data)
    assert out.returncode == 0, out.stderr
    results = json.loads(out.stdout)["results"]
    assert results
    assert results[0]["content"] == content, results[0]["content"]
