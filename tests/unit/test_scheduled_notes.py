"""Delivery windows persist across invocations without widening authority."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from morgan_brain import local_notes, task_batches
from morgan_brain.local_notes import (
    ACTION,
    FileApproval,
    NoteWorkspace,
    approve,
    enqueue,
    local_status,
    revoke,
    run_local,
)
from morgan_brain.task_batches import TaskPermissionDenied
from morgan_brain.tasks import TaskKey, TaskRequest


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Freeze wall time while keeping real SQLite and filesystem effects."""

    class Clock(datetime):
        current = datetime(2026, 11, 6, 8, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.current.astimezone(tz)

    monkeypatch.setattr(local_notes, "datetime", Clock)
    monkeypatch.setattr(task_batches, "datetime", Clock)
    workspace = NoteWorkspace(tmp_path / "agent")
    request = TaskRequest(TaskKey("owner", "personal", "morning"), ACTION, "# План дня\n")
    yield workspace, request, Clock
    workspace.close()


async def test_due_restart_and_repeat(setup):
    """A fresh invocation catches up inside the window, once, with the same task identity."""
    workspace, request, clock = setup
    due = clock.current + timedelta(hours=1)
    approve(workspace, [request], ttl_seconds=7200, not_before=due)
    assert local_status(workspace)["approval"] == "scheduled"
    assert (await run_local(workspace)).actions == 0
    assert not list(workspace.path.glob("note-*.md"))
    clock.current = due
    reopened = NoteWorkspace(workspace.path, create=False)
    try:
        assert (await run_local(reopened)).actions == 1
        assert (await run_local(reopened)).actions == 0
    finally:
        reopened.close()
    assert [p.read_text() for p in workspace.path.glob("note-*.md")] == [request.payload]


async def test_stop_resume_and_expired_misfire(setup):
    """STOP survives the due time; resuming after expiry cannot execute missed work."""
    workspace, request, clock = setup
    approve(workspace, [request], ttl_seconds=60, not_before=clock.current + timedelta(seconds=10))
    workspace.stop()
    clock.current += timedelta(seconds=20)
    assert (await run_local(workspace)).reason == "stopped"
    clock.current += timedelta(seconds=40)
    workspace.resume()
    assert (await run_local(workspace)).reason == "permission_denied"
    assert local_status(workspace)["approval"] == "expired"
    assert not list(workspace.path.glob("note-*.md"))


@pytest.mark.parametrize("revoked", [True, False])
async def test_resume_within_window_and_revoke(setup, revoked):
    """Resuming is explicit; revocation retains history but removes future authority."""
    workspace, request, clock = setup
    approve(workspace, [request], ttl_seconds=60, not_before=clock.current + timedelta(seconds=10))
    workspace.stop()
    clock.current += timedelta(seconds=20)
    workspace.resume()
    if revoked:
        revoke(workspace)
    assert (await run_local(workspace)).actions == (0 if revoked else 1)
    assert len(list(workspace.path.glob("note-*.md"))) == (0 if revoked else 1)


@pytest.mark.parametrize("offset", [None, 60, 61])
def test_invalid_window_leaves_old_approval(setup, offset):
    """Naive timestamps and empty or reversed windows cannot replace active grants."""
    workspace, request, clock = setup
    approve(workspace, [request], ttl_seconds=60)
    before = workspace.read(".approval.json", 300000)
    due = (
        clock.current.replace(tzinfo=None)
        if offset is None
        else clock.current + timedelta(seconds=offset)
    )
    with pytest.raises(ValueError):
        approve(workspace, [request], ttl_seconds=60, not_before=due)
    assert workspace.read(".approval.json", 300000) == before


def test_append_cannot_change_scheduled_window(setup):
    """Immediate enqueue cannot strip or silently broaden a scheduled batch."""
    workspace, request, clock = setup
    approve(workspace, [request], ttl_seconds=60, not_before=clock.current + timedelta(seconds=10))
    before = workspace.read(".approval.json", 300000)
    with pytest.raises(TaskPermissionDenied):
        enqueue(workspace, request, ttl_seconds=600)
    assert workspace.read(".approval.json", 300000) == before


@pytest.mark.parametrize("shift", [-1, 60])
async def test_clock_rollback_and_forward_at_effect_boundary(setup, monkeypatch, shift):
    """A final authorization check refuses an effect if wall time leaves its window."""
    workspace, request, clock = setup
    approve(workspace, [request], ttl_seconds=60, not_before=clock.current)
    original = local_notes.LocalNoteAdapter.execute

    async def rollback(self, task, control):
        clock.current += timedelta(seconds=shift)
        return await original(self, task, control)

    monkeypatch.setattr(local_notes.LocalNoteAdapter, "execute", rollback)
    assert (await run_local(workspace)).reason == "permission_denied"
    assert not list(workspace.path.glob("note-*.md"))


def test_offset_is_canonical_and_legacy_remains_readable(setup):
    """An explicit offset identifies one instant; old immediate grants still load."""
    workspace, request, _clock = setup
    due = datetime.fromisoformat("2026-11-06T12:00:00+03:00")
    approve(workspace, [request], ttl_seconds=7200, not_before=due)
    assert local_status(workspace)["not_before"] == "2026-11-06T09:00:00+00:00"
    assert FileApproval(workspace).load()[0] == ()
    approve(workspace, [request], ttl_seconds=60)
    assert FileApproval(workspace).load()[0] == (request,)


@pytest.mark.parametrize("change", ["naive", "after_expiry", "missing", "legacy_with_time"])
def test_malformed_persisted_window_fails_closed(setup, change):
    """On-disk corruption cannot turn a scheduled approval into immediate authority."""
    workspace, request, clock = setup
    approve(workspace, [request], ttl_seconds=60, not_before=clock.current)
    raw = json.loads(workspace.read(".approval.json", 300000))
    if change == "naive":
        raw["not_before"] = "2026-11-06T08:00:00"
    elif change == "after_expiry":
        raw["not_before"] = raw["expires_at"]
    elif change == "missing":
        del raw["not_before"]
    else:
        raw["version"] = "morgan.local-approval.v1"
    workspace.publish(".approval.json", json.dumps(raw), replace=True)
    with pytest.raises(TaskPermissionDenied):
        FileApproval(workspace).load()
