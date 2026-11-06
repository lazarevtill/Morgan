"""Explicit zero-task review must support availability without inventing commitments."""

import json

import pytest

from tests.integration.test_guided_plan import inputs, run
from tests.integration.test_guided_status import durable_files, status


@pytest.mark.parametrize("selection", ["none", "нет задач"])
def test_empty_review_then_availability_survives_restart(tmp_path, selection):
    """A reviewed note with no obligations can still supply a sourced planning constraint."""
    note = "I already returned the book.\nI am unavailable during my appointment."
    args = inputs(tmp_path, note, [])
    answers = f"{selection}\nCONFIRM\nbusy\nI am unavailable during my appointment.\n"
    answers += "Europe/Belgrade\n2026-11-06T09:00+01:00 2026-11-07T00:00+01:00\nCONFIRM\nquit\n"
    result = run(tmp_path, answers, args)
    assert result.returncode == 0, result.stderr
    saved = json.loads((tmp_path / "session/session.json").read_text())
    assert saved["captured"] and saved["captures"] == []
    assert len(saved["unavailability_ids"]) == 1
    resumed = run(tmp_path, "quit\n")
    assert resumed.returncode == 0 and "Task numbers" not in resumed.stdout
    before = durable_files(tmp_path / "session")
    payload = json.loads(status(tmp_path).stdout)
    assert payload["state"] == "ready" and payload["tasks"] == []
    assert payload["planned_minutes"] == 0 and payload["timezone"] == "Europe/Belgrade"
    human = run(tmp_path, "", ["--status", "--human"])
    assert human.returncode == 0 and "день пересекает недоступность" in human.stdout
    assert "не означает местоположение" in human.stdout
    outside = run(tmp_path, "", ["--status", "--human", "--preview-day", "2026-11-07"])
    assert outside.returncode == 0 and "день не заблокирован этим интервалом" in outside.stdout
    assert durable_files(tmp_path / "session") == before


@pytest.mark.parametrize("answers", ["\n", "none\nNO\n", "нет задач\n"])
def test_exit_or_unconfirmed_empty_review_does_not_create_memory(tmp_path, answers):
    """No implicit acceptance follows blank input, a refusal, or EOF at confirmation."""
    args = inputs(tmp_path, "The appointment is over.", [])
    result = run(tmp_path, answers, args)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "session/memory").exists()
    payload = json.loads(status(tmp_path).stdout)
    assert payload["state"] == "needs_review" and not payload["plan_available"]


def test_reviewed_empty_selection_never_reads_unselected_memories(tmp_path):
    """An empty selection means zero evidence, never an implicit query over the whole scope."""
    script = """
from morgan_brain import guided_plan
from morgan_brain.memory.gate import MemoryGate
async def forbidden(*args, **kwargs):
    raise AssertionError('empty selection attempted evidence lookup')
MemoryGate.evidence = forbidden
guided_plan.main()
"""
    result = run(
        tmp_path, "none\nCONFIRM\nquit\n", inputs(tmp_path, "Already finished.", []), script=script
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(status(tmp_path).stdout)["tasks"] == []
