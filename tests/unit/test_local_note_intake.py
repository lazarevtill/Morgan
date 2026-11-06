"""Concurrent-safe explicit task intake and lifecycle control."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from morgan_brain.local_notes import (
    ACTION,
    FileApproval,
    LocalNoteAdapter,
    NoteWorkspace,
    RunLimits,
    approve,
    enqueue,
    local_status,
    revoke,
    run_local,
)
from morgan_brain.task_batches import TaskPermissionDenied
from morgan_brain.tasks import TaskConflict, TaskKey, TaskRequest


@pytest.fixture
def workspace(tmp_path):
    workspace = NoteWorkspace(tmp_path / "agent")
    yield workspace
    workspace.close()


def note(task_id="first", text="# Morning\n"):
    return TaskRequest(TaskKey("owner", "personal", task_id), ACTION, text)


def test_enqueue_preserves_old_work_and_does_not_extend_expiry(workspace):
    enqueue(workspace, note(), ttl_seconds=30)
    original, expiry = FileApproval(workspace).snapshot()
    enqueue(workspace, note("second"), ttl_seconds=600)
    requests, after = FileApproval(workspace).snapshot()
    assert requests == (*original, note("second")) and after == expiry
    enqueue(workspace, note(), ttl_seconds=1)
    assert FileApproval(workspace).snapshot() == (requests, after)


def test_shorter_new_grant_shortens_whole_worklist(workspace):
    enqueue(workspace, note(), ttl_seconds=600)
    _, old_expiry = FileApproval(workspace).snapshot()
    enqueue(workspace, note("second"), ttl_seconds=30)
    assert FileApproval(workspace).snapshot()[1] < old_expiry


def test_changed_duplicate_leaves_policy_intact(workspace):
    enqueue(workspace, note(), ttl_seconds=60)
    before = workspace.read(".approval.json", 300000)
    with pytest.raises(TaskConflict):
        enqueue(workspace, note(text="changed"), ttl_seconds=60)
    assert workspace.read(".approval.json", 300000) == before


def test_capacity_and_scope_fail_without_losing_requests(workspace):
    requests = [note(str(i)) for i in range(16)]
    approve(workspace, requests, ttl_seconds=60)
    before = FileApproval(workspace).snapshot()
    with pytest.raises(ValueError):
        enqueue(workspace, note("extra"), ttl_seconds=60)
    with pytest.raises(ValueError):
        enqueue(
            workspace, TaskRequest(TaskKey("other", "personal", "x"), ACTION, ""), ttl_seconds=60
        )
    assert FileApproval(workspace).snapshot() == before


def test_expired_policy_is_not_implicitly_reauthorized(workspace):
    enqueue(workspace, note(), ttl_seconds=60)
    raw = json.loads(workspace.read(".approval.json", 300000))
    raw["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    workspace.publish(".approval.json", json.dumps(raw), replace=True)
    with pytest.raises(TaskPermissionDenied):
        enqueue(workspace, note("new"), ttl_seconds=60)
    assert local_status(workspace)["approval"] == "expired"


def test_status_reports_activity_and_stop_without_journal(workspace):
    assert local_status(workspace) == {
        "worker_active": False,
        "stop_requested": False,
        "approval": "missing",
        "approved_tasks": 0,
        "expires_at": None,
    }
    with workspace.locked():
        workspace.stop()
        state = local_status(workspace)
        assert state["worker_active"] is True and state["stop_requested"] is True
    assert not (workspace.path / ".tasks.db").exists()


async def test_idle_worker_observes_later_task_and_stop(workspace):
    async def submit():
        await asyncio.sleep(0.01)
        enqueue(workspace, note(), ttl_seconds=60)
        await asyncio.sleep(0.08)
        workspace.stop()

    producer = asyncio.create_task(submit())
    result = await run_local(workspace, RunLimits(max_cycles=10, poll_seconds=0.05))
    await producer
    assert result.actions == 1 and result.reason == "stopped"
    assert [p.read_text() for p in workspace.path.glob("note-*.md")] == [note().payload]


async def test_revoke_keeps_history_but_removes_authority(workspace, monkeypatch):
    enqueue(workspace, note(), ttl_seconds=60)
    original = LocalNoteAdapter.execute

    async def revoked(self, task, control):
        revoke(workspace)
        return await original(self, task, control)

    monkeypatch.setattr(LocalNoteAdapter, "execute", revoked)
    result = await run_local(workspace)
    assert result.reason == "permission_denied"
    assert local_status(workspace)["approved_tasks"] == 0
    journal = workspace.journal()
    try:
        assert journal.get(note().key).status == "outcome_unknown"
    finally:
        journal.close()
    assert not list(workspace.path.glob("note-*.md"))


async def test_idle_worker_honors_deadline(workspace):
    result = await run_local(workspace, RunLimits(max_cycles=60, seconds=0.02, poll_seconds=0.05))
    assert result.reason == "deadline" and result.actions == 0


def test_enqueue_does_not_clear_stop(workspace):
    workspace.stop()
    enqueue(workspace, note(), ttl_seconds=60)
    assert local_status(workspace)["stop_requested"] is True
