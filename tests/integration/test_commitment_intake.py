"""Real CLI intake, immutable completion and planning through the existing memory gate."""

import asyncio
import json
from datetime import date

from morgan_brain.composition import open_evidence_context
from morgan_brain.daily_brief import BriefRequest, prepare_brief
from morgan_brain.tasks import TaskKey
from tests.integration.test_cli import _hash_env, _run


def _arguments(identity="plants", status="open"):
    """Explicit human-reviewed fields with stable provenance for replay tests."""
    return [
        "commitment",
        "Полить растения",
        "--due",
        "2026-11-06",
        "--minutes",
        "10",
        "--priority",
        "normal",
        "--status",
        status,
        "--event-id",
        identity,
        "--source",
        "user_stated",
        "--author-id",
        "owner",
        "--project",
        "personal",
        "--effective-at",
        "2020-01-01T00:00:00Z",
        "--json",
    ]


def test_intake_replay_completion_and_plan(tmp_path):
    """A real commitment survives restart; its explicit completion excludes it from the plan."""
    env = _hash_env(tmp_path)
    first = _run(_arguments(), env, tmp_path)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["id"] == "plants"
    assert _run(_arguments(), env, tmp_path).returncode == 0
    completed = [*_arguments("plants-done", "done"), "--revises-event-id", "plants"]
    completed[completed.index("2020-01-01T00:00:00Z")] = "2021-01-01T00:00:00Z"
    result = _run(completed, env, tmp_path)
    assert result.returncode == 0, result.stderr
    context = open_evidence_context(tmp_path / "morgan.db")
    try:
        draft = asyncio.run(
            prepare_brief(
                context.gate,
                BriefRequest(
                    TaskKey("owner", "personal", "focus"), ("plants-done",), date(2026, 11, 6), 30
                ),
            )
        )
        assert draft.selected_ids == ()
        assert draft.excluded_ids == ("plants-done",)
    finally:
        context.conn.close()


def test_invalid_fields_do_not_open_database(tmp_path):
    """Typed validation happens before any memory context or provider is opened."""
    args = _arguments()
    args[args.index("10")] = "0"
    result = _run(args, _hash_env(tmp_path), tmp_path)
    assert result.returncode != 0
    assert not (tmp_path / "morgan.db").exists()


def test_commitment_cannot_cross_projects(tmp_path):
    """Structured intake preserves the single-project write restriction."""
    result = _run([*_arguments(), "--all-projects"], _hash_env(tmp_path), tmp_path)
    assert result.returncode != 0
    assert not (tmp_path / "morgan.db").exists()


def test_inference_is_not_promoted(tmp_path):
    """Caller-supplied inferred provenance remains inferred, without silent promotion."""
    args = _arguments()
    args[args.index("user_stated")] = "agent_inferred"
    result = _run(args, _hash_env(tmp_path), tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["source"] == "agent_inferred"
