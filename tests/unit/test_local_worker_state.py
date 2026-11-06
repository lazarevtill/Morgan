"""Diagnostic observations must not leak payloads or authorize/repeat effects."""

import asyncio
import json
import sqlite3
from dataclasses import asdict

import pytest

from morgan_brain.local_notes import ACTION, NoteWorkspace, RunLimits, approve, run_local
from morgan_brain.local_worker_state import RunObservation, read_observation, scoped_counts
from morgan_brain.tasks import TaskKey, TaskRequest


@pytest.fixture
def workspace(tmp_path):
    value = NoteWorkspace(tmp_path / "agent")
    yield value
    value.close()


def request(project="personal"):
    return TaskRequest(TaskKey("owner", project, "secret-task-id"), ACTION, "private note contents")


async def test_finished_diagnostics_are_payload_free_and_scoped(workspace):
    approve(workspace, [request()], ttl_seconds=60)
    await run_local(workspace)
    raw = workspace.read(".run.json", 16384)
    assert "private note contents" not in raw and "secret-task-id" not in raw
    state = read_observation(workspace, active=False)
    assert state["recorded_state"] == state["observed_state"] == "finished"
    assert state["reason"] == "cycle_limit" and state["actions"] == 1
    assert scoped_counts(workspace, user_id="owner", project="personal") == {"succeeded": 1}
    assert scoped_counts(workspace, user_id="other", project="personal") == {}
    assert scoped_counts(workspace, user_id="owner", project="other") == {}


def test_missing_diagnostics_do_not_create_journal(workspace):
    assert read_observation(workspace, active=False) is None
    assert scoped_counts(workspace, user_id="owner", project="personal") == {}
    assert not (workspace.path / ".tasks.db").exists()


def test_interrupted_is_observation_not_journal_mutation(workspace):
    RunObservation(workspace, asdict(RunLimits()))
    assert read_observation(workspace, active=False)["observed_state"] == "interrupted"
    assert read_observation(workspace, active=True)["observed_state"] == "running"
    assert json.loads(workspace.read(".run.json", 16384))["state"] == "running"


async def test_stop_has_persisted_reason(workspace):
    workspace.stop()
    result = await run_local(workspace)
    assert result.reason == "stopped"
    assert read_observation(workspace, active=False)["reason"] == "stopped"


async def test_cancellation_is_recorded_without_error_text(workspace):
    task = asyncio.create_task(run_local(workspace, RunLimits(max_cycles=60, poll_seconds=0.05)))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert read_observation(workspace, active=False)["reason"] == "cancelled"


async def test_diagnostic_failure_before_dispatch_does_not_run_task(workspace, monkeypatch):
    approve(workspace, [request()], ttl_seconds=60)
    original = NoteWorkspace.publish

    def fail(self, name, text, *, replace=False):
        if name == ".run.json":
            raise OSError("private path and input must never be recorded")
        return original(self, name, text, replace=replace)

    monkeypatch.setattr(NoteWorkspace, "publish", fail)
    with pytest.raises(OSError):
        await run_local(workspace)
    assert not list(workspace.path.glob("note-*.md"))
    assert not (workspace.path / ".tasks.db").exists()


async def test_failure_record_does_not_store_exception_message(workspace, monkeypatch):
    def fail(self):
        raise sqlite3.OperationalError("SECRET SQL PAYLOAD")

    monkeypatch.setattr(NoteWorkspace, "journal", fail)
    with pytest.raises(sqlite3.OperationalError):
        await run_local(workspace)
    assert "SECRET" not in workspace.read(".run.json", 16384)
    assert read_observation(workspace, active=False)["reason"] == "storage_error"


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "private contents"),
        ("reason", "private exception"),
        ("actions", "payload"),
        ("limits", {"payload": "secret"}),
        ("run_id", "secret"),
    ],
)
def test_tampered_metadata_is_rejected_before_display(workspace, field, value):
    RunObservation(workspace, asdict(RunLimits()))
    raw = json.loads(workspace.read(".run.json", 16384))
    raw[field] = value
    workspace.publish(".run.json", json.dumps(raw), replace=True)
    with pytest.raises(ValueError):
        read_observation(workspace, active=False)


def test_foreign_journal_and_symlink_are_refused(workspace, tmp_path):
    path = workspace.path / ".tasks.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE private(data TEXT)")
    path.chmod(0o600)
    with pytest.raises(ValueError):
        scoped_counts(workspace, user_id="owner", project="personal")
    target = tmp_path / "foreign.db"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(OSError):
        scoped_counts(workspace, user_id="owner", project="personal")


def test_read_counts_does_not_upgrade_v1(workspace):
    journal = workspace.journal()
    journal.create_many([request()])
    journal.close()
    path = workspace.path / ".tasks.db"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version=1")
    assert scoped_counts(workspace, user_id="owner", project="personal") == {"pending": 1}
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1


async def test_final_observation_failure_cannot_replay_completed_effect(workspace, monkeypatch):
    approve(workspace, [request()], ttl_seconds=60)
    original = NoteWorkspace.publish

    def fail_finish(self, name, text, *, replace=False):
        if name == ".run.json" and json.loads(text)["state"] == "finished":
            raise OSError("diagnostic disk failure")
        return original(self, name, text, replace=replace)

    monkeypatch.setattr(NoteWorkspace, "publish", fail_finish)
    with pytest.raises(OSError):
        await run_local(workspace)
    assert scoped_counts(workspace, user_id="owner", project="personal") == {"succeeded": 1}
    assert read_observation(workspace, active=False)["observed_state"] == "interrupted"
    monkeypatch.setattr(NoteWorkspace, "publish", original)
    assert (await run_local(workspace)).actions == 0
    assert len(list(workspace.path.glob("note-*.md"))) == 1


async def test_diagnostic_error_does_not_mask_primary_error(workspace, monkeypatch):
    original = NoteWorkspace.publish

    def fail_record(self, name, text, *, replace=False):
        if name == ".run.json" and json.loads(text)["state"] == "failed":
            raise OSError("secondary failure")
        return original(self, name, text, replace=replace)

    def fail_journal(self):
        raise sqlite3.OperationalError("primary failure")

    monkeypatch.setattr(NoteWorkspace, "publish", fail_record)
    monkeypatch.setattr(NoteWorkspace, "journal", fail_journal)
    with pytest.raises(sqlite3.OperationalError, match="primary failure"):
        await run_local(workspace)
