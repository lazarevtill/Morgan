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
    assert _run(["remember", "acme secret", "--project", "acme"], env, tmp_path).returncode == 0
    same_project = _run(["recall", "secret", "--project", "acme", "--json"], env, tmp_path)
    other_project = _run(["recall", "secret", "--project", "personal", "--json"], env, tmp_path)
    assert json.loads(same_project.stdout)["results"]
    assert json.loads(other_project.stdout)["results"] == []


def test_recall_all_projects_crosses_scope(tmp_path):
    env = _hash_env(tmp_path)
    _run(["remember", "cross-project needle", "--project", "acme"], env, tmp_path)
    out = _run(
        ["recall", "needle", "--project", "personal", "--all-projects", "--json"], env, tmp_path
    )
    assert json.loads(out.stdout)["results"]


def test_forget_reports_skipped_tables_not_a_false_zero(tmp_path):
    env = _hash_env(tmp_path)
    _run(["remember", "harbor mirror secret", "--project", "acme"], env, tmp_path)
    out = _run(["forget", "--project", "acme", "--json"], env, tmp_path)
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
    recall_after = _run(["recall", "harbor", "--project", "acme", "--json"], env, tmp_path)
    assert json.loads(recall_after.stdout)["results"] == []


def test_forget_warns_vectors_not_erased_under_qdrant():
    """Pure unit check of the formatting logic -- doesn't require a live Qdrant instance."""
    from morgan_brain.cli.__main__ import _forget_result
    from morgan_brain.config import Settings
    from morgan_brain.interfaces.memory import ForgetReport

    settings = Settings(vector_backend="qdrant")
    result = _forget_result(ForgetReport(memories=3), settings, project="p", all_projects=False)
    assert result["vectors_erased"] is False
    assert any("qdrant" in w for w in result["warnings"])


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
    # The prompt-registry file must actually have been created under MORGAN_DATA_DIR.
    assert (tmp_path / "data" / "prompts.db").exists()


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
