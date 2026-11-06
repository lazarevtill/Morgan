"""A read-only query surface for an existing personal session, with honest pending states."""

import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime

import pytest

from morgan_brain import guided_plan
from tests.integration.test_guided_availability import busy_answers, setup, task_answers
from tests.integration.test_guided_plan import run


def status(tmp_path, *, script=None):
    """Query the real CLI without supplying input or implicitly confirming anything."""
    return run(tmp_path, "", ["--status"], script=script)


def durable_files(root):
    """SQLite may maintain WAL coordination files; durable records and dialogue files must stay."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and not path.name.endswith(("-wal", "-shm"))
    }


def test_current_status_preserves_files_and_makes_no_model_or_capture_calls(tmp_path):
    """Status returns grounded current records without writes, prompts, embeddings or inference."""
    assert (
        run(tmp_path, task_answers() + busy_answers() + "quit\n", setup(tmp_path)).returncode == 0
    )
    before = durable_files(tmp_path / "session")
    script = """
from morgan_brain import guided_plan
from morgan_brain.local_notes import NoteWorkspace
def forbidden(*args, **kwargs):
    raise AssertionError('status attempted a write or model call')
guided_plan.cmd_remember = forbidden
guided_plan.managed_model = forbidden
NoteWorkspace.publish = forbidden
guided_plan.main()
"""
    result = status(tmp_path, script=script)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["state"] == "ready" and payload["plan_available"]
    assert payload["planned_minutes"] == 0 and len(payload["blocked_ids"]) == 1
    assert payload["owner"] == "pilot" and payload["project"] == "personal"
    assert payload["timezone"] == "Europe/Paris" and payload["planning_day"] == "2026-11-06"
    assert not payload["scope_complete"] and not payload["execution_authorized"]
    assert payload["tasks"][0]["evidence_id"] in payload["blocked_ids"]
    assert payload["unavailability"][0]["interval"]["source_ids"]
    assert durable_files(tmp_path / "session") == before


def test_status_follows_completion_and_cancellation_instead_of_saved_draft(tmp_path):
    """New status follows memory revisions even while older Markdown plans remain on disk."""
    assert (
        run(tmp_path, task_answers() + busy_answers() + "quit\n", setup(tmp_path)).returncode == 0
    )
    assert json.loads(status(tmp_path).stdout)["planned_minutes"] == 0
    changed = run(tmp_path, "cancel 1\nI cancel this interval.\nCONFIRM\ndone 1\nquit\n")
    assert changed.returncode == 0, changed.stderr
    # A deliberately misleading saved artifact must not be used as the source of truth.
    (tmp_path / "session/plan-latest.md").write_text("All tasks are open and approved.")
    result = status(tmp_path)
    payload = json.loads(result.stdout)
    assert result.returncode == 0 and payload["tasks"][0]["commitment"]["status"] == "done"
    assert payload["unavailability"][0]["interval"]["status"] == "cancelled"
    assert len(payload["excluded_ids"]) == 1 and payload["blocked_ids"] == []
    assert any(k != v for k, v in payload["resolved_sources"].items())


@pytest.mark.parametrize("reviewed", [False, True])
def test_unconfirmed_task_input_is_reported_without_opening_or_creating_memory(tmp_path, reviewed):
    """A snapshot must not turn proposed or unconfirmed tasks into confirmed obligations."""
    args = setup(tmp_path)
    answers = "1\n10 normal 2026-11-06\n" if reviewed else ""
    assert run(tmp_path, answers, args).returncode == 0
    before = durable_files(tmp_path / "session")
    payload = json.loads(status(tmp_path).stdout)
    assert payload["state"] == "needs_review" and not payload["plan_available"]
    assert payload["pending_reviews"] == ["task_confirmation" if reviewed else "task_selection"]
    assert "tasks" not in payload
    assert not (tmp_path / "session/memory").exists()
    assert durable_files(tmp_path / "session") == before


def test_pending_write_is_not_replayed_or_presented_as_ready(tmp_path):
    """A committed write with lost acknowledgement remains pending until interactive review."""
    assert run(tmp_path, task_answers() + "quit\n", setup(tmp_path)).returncode == 0
    script = """
import os
from morgan_brain import guided_plan
original = guided_plan._capture
async def crash(*args):
    await original(*args)
    os._exit(21)
guided_plan._capture = crash
guided_plan.main()
"""
    assert run(tmp_path, "done 1\n", script=script).returncode == 21
    before = durable_files(tmp_path / "session")
    payload = json.loads(status(tmp_path).stdout)
    assert payload["state"] == "needs_review"
    assert payload["pending_reviews"] == ["completion_confirmation"]
    assert "tasks" not in payload and not payload["plan_available"]
    assert durable_files(tmp_path / "session") == before
    assert run(tmp_path, "CONFIRM\nquit\n").returncode == 0
    assert json.loads(status(tmp_path).stdout)["tasks"][0]["commitment"]["status"] == "done"


def test_missing_session_or_lock_is_not_created_or_repaired(tmp_path):
    """A read request must not initialize an unknown session or manufacture its lock."""
    result = status(tmp_path)
    assert result.returncode == 2 and not (tmp_path / "session").exists()
    assert run(tmp_path, task_answers() + "quit\n", setup(tmp_path)).returncode == 0
    (tmp_path / "session/.guided.lock").unlink()
    assert status(tmp_path).returncode == 2
    assert not (tmp_path / "session/.guided.lock").exists()


def test_active_dialogue_refuses_status_without_waiting_or_interfering(tmp_path):
    """Do not return mixed state while an interactive owner holds the exclusive session lock."""
    args = setup(tmp_path)
    command = [sys.executable, "-m", "morgan_brain.guided_plan", str(tmp_path / "session"), *args]
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as active:
        try:
            deadline = time.monotonic() + 5
            while not (tmp_path / "session/session.json").exists():
                assert active.poll() is None and time.monotonic() < deadline
                time.sleep(0.01)
            result = status(tmp_path)
            assert result.returncode == 2 and "session is busy" in result.stderr
        finally:
            active.communicate("\n", timeout=5)


def test_changed_source_and_input_overrides_are_refused(tmp_path):
    """Status is a query, not a channel for silently changing inputs or adopting modified text."""
    assert run(tmp_path, task_answers() + "quit\n", setup(tmp_path)).returncode == 0
    overridden = run(tmp_path, "", ["--status", "--budget", "1"])
    assert overridden.returncode == 2
    (tmp_path / "session/source.txt").write_text("Changed source")
    assert status(tmp_path).returncode == 2


def test_fixed_planning_day_is_not_misrepresented_as_today():
    """Date freshness uses the explicit planning zone; a missing zone stays unknown."""
    session = guided_plan.Session(
        user="demo",
        project="pilot",
        day="2027-04-10",
        budget=30,
        note_sha256="note",
        proposal_sha256="proposal",
    )
    instant = datetime(2027, 4, 10, 23, tzinfo=UTC)
    assert guided_plan._day_relation(session, instant) == "unknown_timezone"
    session.timezone = "Europe/Paris"
    assert guided_plan._day_relation(session, instant) == "past"
    assert guided_plan._day_relation(session, datetime(2027, 4, 10, 12, tzinfo=UTC)) == "today"
    assert guided_plan._day_relation(session, datetime(2027, 4, 9, 12, tzinfo=UTC)) == "future"


def test_pending_interval_is_not_silently_confirmed_by_status(tmp_path):
    """A partial interval capture must remain pending and cannot become an authoritative plan."""
    assert run(tmp_path, task_answers() + "quit\n", setup(tmp_path)).returncode == 0
    script = """
import os
from morgan_brain import guided_plan
original = guided_plan.cmd_remember
async def crash(*args, **kwargs):
    result = await original(*args, **kwargs)
    if kwargs.get('client') == 'guided-availability-v1':
        os._exit(22)
    return result
guided_plan.cmd_remember = crash
guided_plan.main()
"""
    assert run(tmp_path, busy_answers(), script=script).returncode == 22
    before = durable_files(tmp_path / "session")
    payload = json.loads(status(tmp_path).stdout)
    assert payload["pending_reviews"] == ["unavailability_confirmation"]
    assert not payload["plan_available"] and "unavailability" not in payload
    assert durable_files(tmp_path / "session") == before
