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
    repo_a = _init_repo(tmp_path / "plata")
    repo_b = _init_repo(tmp_path / "personal")

    stored = _morgan(
        ["remember", "RISKDEV-14802 blocked on the Harbor mirror, not the chart"],
        cwd=repo_a,
        data_dir=data,
    )
    assert stored.returncode == 0, stored.stderr

    # A separate process, in a separate repository -- nothing is shared but the database
    # on disk. This is the restart: no in-memory state survives between the two subprocess
    # invocations.
    out = _morgan(["recall", "harbor", "--all-projects", "--json"], cwd=repo_b, data_dir=data)
    assert out.returncode == 0, out.stderr
    results = json.loads(out.stdout)["results"]
    assert any("Harbor mirror" in r["content"] for r in results)
    assert results[0]["project"] == "plata"


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
    assert doctor["vector_rows"] > 0, doctor
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

    # No MORGAN_EMBEDDING_BACKEND override here -- the config default ("provider") is a real
    # embedding model behind MORGAN_LLM_ENDPOINT.
    stored = _morgan(
        ["remember", "the deploy was blocked by the registry mirror"],
        cwd=repo,
        data_dir=data,
        extra_env={"MORGAN_EMBEDDING_BACKEND": "provider"},
    )
    assert stored.returncode == 0, stored.stderr

    # A semantic match with no shared keywords ("what stopped the release?" vs. "the deploy
    # was blocked by the registry mirror") -- only a genuine embedding model can bridge that.
    out = _morgan(
        ["recall", "what stopped the release?", "--json"],
        cwd=repo,
        data_dir=data,
        extra_env={"MORGAN_EMBEDDING_BACKEND": "provider"},
    )
    assert out.returncode == 0, out.stderr
    results = json.loads(out.stdout)["results"]
    assert any("registry mirror" in r["content"] for r in results), results


def test_cyrillic_survives_the_same_round_trip(tmp_path: Path) -> None:
    """Keyword recall for Russian has never worked in this project's history until this
    milestone -- it is the failure most likely to regress silently."""
    data = tmp_path / "brain"
    repo = _init_repo(tmp_path / "repo")

    stored = _morgan(["remember", "реестр Harbor заблокировал деплой"], cwd=repo, data_dir=data)
    assert stored.returncode == 0, stored.stderr

    out = _morgan(["recall", "реестр", "--json"], cwd=repo, data_dir=data)
    assert out.returncode == 0, out.stderr
    results = json.loads(out.stdout)["results"]
    assert results
    assert "реестр" in results[0]["content"]
