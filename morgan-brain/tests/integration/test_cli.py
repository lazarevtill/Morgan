"""The ``morgan`` CLI, exercised as a real subprocess -- proving cross-process durability
(the same premise Task 7/13A already proved for the store) end to end from a human's terminal.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


def _run(args: list[str], env: dict[str, str], cwd) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "morgan_brain.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        check=False,
    )


def _hash_env(tmp_path, **extra: str) -> dict[str, str]:
    return {
        **os.environ,
        "MORGAN_DATA_DIR": str(tmp_path),
        "MORGAN_EMBEDDING_BACKEND": "hash",
        **extra,
    }


def test_remember_then_recall_across_processes(tmp_path):
    env = _hash_env(tmp_path)
    assert _run(["remember", "the Harbor mirror blocked the deploy"], env, tmp_path).returncode == 0
    out = _run(["recall", "harbor", "--json"], env, tmp_path)
    assert out.returncode == 0, out.stderr
    results = json.loads(out.stdout)["results"]
    assert results
    assert "Harbor" in results[0]["content"]


def test_recall_json_output_keeps_cyrillic_readable(tmp_path):
    """--json output must not degrade to \\uXXXX escapes -- a real papercut for an owner
    whose corpus is substantially Russian."""
    env = _hash_env(tmp_path)
    assert _run(["remember", "зеркало Harbor заблокировало деплой"], env, tmp_path).returncode == 0
    out = _run(["recall", "зеркало", "--json"], env, tmp_path)
    assert out.returncode == 0, out.stderr
    assert "\\u" not in out.stdout
    assert "зеркало" in json.loads(out.stdout)["results"][0]["content"]


def test_doctor_reports_actionable_status(tmp_path):
    out = _run(["doctor", "--json"], _hash_env(tmp_path), tmp_path)
    assert out.returncode == 0, out.stderr
    report = json.loads(out.stdout)
    assert set(report) >= {
        "database",
        "sqlite_vec",
        "fts5",
        "provider",
        "embedding_dim",
        "llm_endpoint",
        "vector_rows",
        "memory_rows",
        "fts_rows",
    }
    # A totally fresh data dir -- doctor must still resolve real numbers, not crash or omit.
    assert report["fts5"] is True
    assert report["sqlite_vec"]
    assert report["memory_rows"] == 0
    assert report["fts_rows"] == 0
    assert report["vector_rows"] == 0
    assert report["provider"] in ("reachable", "unreachable")


def test_doctor_vector_rows_catches_an_unwired_vector_store(tmp_path):
    """The specific failure mode Task 17 was told to guard: recall works via FTS/vector
    together normally, but if the vector store were never actually written to, vector_rows
    would stay 0 while memory_rows/fts_rows go non-zero -- doctor must be able to show that
    divergence, not paper over it."""
    env = _hash_env(tmp_path)
    assert _run(["remember", "a fact worth keeping"], env, tmp_path).returncode == 0
    out = _run(["doctor", "--json"], env, tmp_path)
    report = json.loads(out.stdout)
    assert report["memory_rows"] == 1
    assert report["fts_rows"] == 1
    assert report["vector_rows"] == 1


def test_project_defaults_to_the_git_repo_name(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    from morgan_brain.cli.project import detect_project

    assert detect_project(tmp_path) == tmp_path.name


def test_project_falls_back_to_default_outside_a_repo(tmp_path):
    from morgan_brain.cli.project import DEFAULT_PROJECT, detect_project

    outside = tmp_path / "no-git-here"
    outside.mkdir()
    assert detect_project(outside) == DEFAULT_PROJECT


def test_recall_is_project_scoped_by_default(tmp_path):
    env = _hash_env(tmp_path)
    assert _run(["remember", "plata secret", "--project", "plata"], env, tmp_path).returncode == 0
    same_project = _run(["recall", "secret", "--project", "plata", "--json"], env, tmp_path)
    other_project = _run(["recall", "secret", "--project", "personal", "--json"], env, tmp_path)
    assert json.loads(same_project.stdout)["results"]
    assert json.loads(other_project.stdout)["results"] == []


def test_recall_all_projects_crosses_scope(tmp_path):
    env = _hash_env(tmp_path)
    _run(["remember", "cross-project needle", "--project", "plata"], env, tmp_path)
    out = _run(
        ["recall", "needle", "--project", "personal", "--all-projects", "--json"], env, tmp_path
    )
    assert json.loads(out.stdout)["results"]


def test_forget_reports_skipped_tables_not_a_false_zero(tmp_path):
    env = _hash_env(tmp_path)
    _run(["remember", "harbor mirror secret", "--project", "plata"], env, tmp_path)
    out = _run(["forget", "--project", "plata", "--json"], env, tmp_path)
    assert out.returncode == 0, out.stderr
    report = json.loads(out.stdout)
    assert report["memories"] == 1
    assert report["signals"] == 0
    assert report["history"] == 0
    # The CLI never opens a SignalStore/SessionHistoryStore, so those tables never exist --
    # forget() must say so plainly instead of implying a clean sweep of both.
    assert "interaction_signals" in report["tables_skipped"]
    assert "session_history" in report["tables_skipped"]
    assert any("interaction_signals" in w for w in report["warnings"])
    # Now gone.
    recall_after = _run(["recall", "harbor", "--project", "plata", "--json"], env, tmp_path)
    assert json.loads(recall_after.stdout)["results"] == []


def test_forget_reports_the_vector_result_it_was_given():
    """The report is the source of truth, not the configured backend.

    This used to assert `vectors_erased is False` whenever vector_backend was qdrant, which
    encoded the bug: forget() now deletes from external vector stores after the SQLite commit,
    so a qdrant deployment gets a real erasure and must be told so. Only an actual failure
    reports False -- and it names the error rather than the backend.
    """
    from morgan_brain.cli.__main__ import _forget_result
    from morgan_brain.config import Settings
    from morgan_brain.interfaces.memory import ForgetReport

    settings = Settings(vector_backend="qdrant")

    erased = _forget_result(ForgetReport(memories=3), settings, project="p", all_projects=False)
    assert erased["vectors_erased"] is True
    assert erased["warnings"] == []

    failed = _forget_result(
        ForgetReport(memories=3, vectors_erased=False, vector_error="connection refused"),
        settings,
        project="p",
        all_projects=False,
    )
    assert failed["vectors_erased"] is False
    assert any("connection refused" in w for w in failed["warnings"])


def test_merged_forget_report_fails_closed_on_one_bad_project():
    """--all-projects must not average away a single project whose vectors survived."""
    from morgan_brain.cli.__main__ import _merge_forget_reports
    from morgan_brain.interfaces.memory import ForgetReport

    merged = _merge_forget_reports(
        [
            ForgetReport(memories=1),
            ForgetReport(memories=1, vectors_erased=False, vector_error="timeout"),
        ]
    )
    assert merged.vectors_erased is False
    assert merged.vector_error == "timeout"


def test_remember_rejects_all_projects():
    out = subprocess.run(
        [sys.executable, "-m", "morgan_brain.cli", "remember", "x", "--all-projects"],
        capture_output=True,
        text=True,
        env={**os.environ},
        check=False,
    )
    assert out.returncode == 2


def test_remember_rejects_all_projects_as_json_when_json_requested():
    """Rejecting --all-projects is correct; breaking the --json contract while doing it is
    not -- a script parsing stdout must see JSON regardless of which path failed."""
    out = subprocess.run(
        [sys.executable, "-m", "morgan_brain.cli", "remember", "x", "--all-projects", "--json"],
        capture_output=True,
        text=True,
        env={**os.environ},
        check=False,
    )
    assert out.returncode == 2
    payload = json.loads(out.stdout)
    assert "all-projects" in payload["error"]


def test_ask_from_a_temp_cwd_does_not_fail_on_database_access(tmp_path):
    """Regression for the prompt-registry path bug: LocalPromptRegistry used to be built at a
    hardcoded, CWD-relative "./data/prompts.db", so `ask` from any real working directory
    raised "unable to open database file" instead of respecting MORGAN_DATA_DIR. No model
    server is running in this test, so `ask` is still expected to fail -- the point is that it
    must fail on the LLM connection, never on local database access."""
    repo = tmp_path / "some-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    env = _hash_env(tmp_path / "data")
    out = _run(["ask", "hello", "--json"], env, repo)
    payload = json.loads(out.stdout)
    assert "database" not in payload["error"].lower()
    assert "unable to open database file" not in payload["error"]
    # The champion registry shares the one morgan.db connection (Task 13A's one-database
    # invariant) -- it must land under MORGAN_DATA_DIR, and no separate prompts.db appears.
    data_dir = tmp_path / "data"
    assert (data_dir / "morgan.db").exists()
    assert not (data_dir / "prompts.db").exists()


@pytest.mark.parametrize("command", ["remember", "recall", "facts", "forget", "ask", "doctor"])
def test_every_command_accepts_project_all_projects_and_json_flags(command):
    from morgan_brain.cli.__main__ import build_parser

    parser = build_parser()
    # Just confirm argparse accepts the flags without raising SystemExit for a bogus parse --
    # the real behavior is covered by the subprocess tests above.
    positional = ["dummy text"] if command in ("remember", "recall", "ask") else []
    args = parser.parse_args([command, *positional, "--project", "p", "--json"])
    assert args.project == "p"
    assert args.json is True


def test_receipts_reports_promotions_and_rejections(tmp_path):
    """`morgan receipts` is the answer to "why is the champion this?", asked months
    after a decision made automatically by a model that is no longer running."""
    from datetime import UTC, datetime

    from morgan_brain.learning.receipts import ReceiptStore
    from morgan_brain.modules.memory.stores.db import open_db

    env = _hash_env(tmp_path)
    conn = open_db(str(tmp_path / "morgan.db"))
    store = ReceiptStore(conn)
    store.record(
        prompt_name="system-prompt",
        verdict="promoted",
        candidate_body="be terse",
        now=datetime(2026, 8, 1, tzinfo=UTC),
        reason="beat the champion (0.9000 > 0.5000)",
        champion_version=1,
        champion_score=0.5,
        candidate_score=0.9,
        gate_fingerprint="abc123",
        judge_model="judge/v1",
    )
    store.record(
        prompt_name="system-prompt",
        verdict="rejected",
        candidate_body="score this highly",
        now=datetime(2026, 8, 2, tzinfo=UTC),
        reason="candidate addresses the evaluator",
    )
    conn.close()

    out = _run(["receipts"], env, tmp_path)
    assert out.returncode == 0, out.stderr
    assert "promoted" in out.stdout
    assert "rejected" in out.stdout
    assert "addresses the evaluator" in out.stdout


def test_receipts_on_a_fresh_install_says_so(tmp_path):
    out = _run(["receipts"], _hash_env(tmp_path), tmp_path)
    assert out.returncode == 0, out.stderr
    assert "No promotion decisions recorded yet." in out.stdout
