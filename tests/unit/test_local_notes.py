"""Account policy, actual filesystem effects, and bounded worker safety contracts."""

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
    run_local,
)
from morgan_brain.task_receipts import ReceiptMismatch, receipt_request
from morgan_brain.tasks import TaskConflict, TaskKey, TaskRequest


@pytest.fixture
def workspace(tmp_path):
    workspace = NoteWorkspace(tmp_path / "private")
    yield workspace
    workspace.close()


def note(identity="one", payload="# Daily plan\n\nTake a walk.\n"):
    return TaskRequest(TaskKey("owner", "personal", identity), ACTION, payload)


def allow(workspace, requests=None):
    approve(workspace, requests or [note()], ttl_seconds=60)


async def test_real_delivery_and_restart_does_not_rewrite(workspace):
    allow(workspace)
    first = await run_local(workspace)
    files = list(workspace.path.glob("note-*.md"))
    assert first.actions == 1 and len(files) == 1
    assert files[0].read_text() == note().payload
    inode = files[0].stat().st_ino
    second = await run_local(workspace)
    assert second.actions == second.recovered == 0
    assert files[0].stat().st_ino == inode
    assert files[0].stat().st_mode & 0o777 == 0o600


async def test_action_budget_preserves_pending_task(workspace):
    allow(workspace, [note(), note("two")])
    assert (await run_local(workspace)).reason == "action_limit"
    assert len(list(workspace.path.glob("note-*.md"))) == 1
    assert (await run_local(workspace)).actions == 1
    assert len(list(workspace.path.glob("note-*.md"))) == 2


async def test_stop_and_explicit_resume(workspace):
    allow(workspace)
    workspace.stop()
    assert (await run_local(workspace)).reason == "stopped"
    assert not list(workspace.path.glob("note-*.md"))
    workspace.resume()
    assert (await run_local(workspace)).actions == 1


async def test_second_worker_and_resume_refused_with_lock(workspace):
    allow(workspace)
    other = NoteWorkspace(workspace.path)
    try:
        with workspace.locked():
            with pytest.raises(BlockingIOError):
                await run_local(other)
            with pytest.raises(BlockingIOError):
                other.resume()
    finally:
        other.close()


async def test_missing_approval_fails_closed(workspace):
    with pytest.raises(FileNotFoundError):
        await run_local(workspace)
    assert not list(workspace.path.glob("note-*.md"))


async def test_expired_policy_denies(workspace):
    allow(workspace)
    raw = json.loads(workspace.read(".approval.json", 300000))
    raw["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    workspace.publish(".approval.json", json.dumps(raw), replace=True)
    assert (await run_local(workspace)).reason == "permission_denied"


async def test_policy_copy_to_another_workspace_is_not_authority(workspace, tmp_path):
    allow(workspace)
    other = NoteWorkspace(tmp_path / "other")
    try:
        other.publish(".approval.json", workspace.read(".approval.json", 300000))
        assert (await run_local(other)).reason == "permission_denied"
    finally:
        other.close()


async def test_changed_payload_cannot_reuse_id(workspace):
    allow(workspace)
    await run_local(workspace)
    allow(workspace, [note(payload="Changed")])
    with pytest.raises(TaskConflict):
        await run_local(workspace)
    assert len(list(workspace.path.glob("note-*.md"))) == 1


async def test_revocation_is_reloaded_before_effect(workspace, monkeypatch):
    allow(workspace)
    original = LocalNoteAdapter.execute

    async def revoked(self, task, control):
        allow(workspace, [note("different")])
        return await original(self, task, control)

    monkeypatch.setattr(LocalNoteAdapter, "execute", revoked)
    assert (await run_local(workspace)).reason == "permission_denied"
    assert not list(workspace.path.glob("note-*.md"))
    journal = workspace.journal()
    try:
        assert journal.get(note().key).status == "outcome_unknown"
    finally:
        journal.close()


async def test_running_without_file_blocks_without_replay(workspace):
    allow(workspace)
    journal = workspace.journal()
    try:
        journal.create_many([note()])
        journal.claim(note().key, action=ACTION)
    finally:
        journal.close()
    result = await run_local(workspace)
    assert result.reason == "needs_reconciliation" and result.actions == 0
    assert not list(workspace.path.glob("note-*.md"))


async def test_changed_note_cannot_confirm_receipt(workspace):
    allow(workspace)
    journal = workspace.journal()
    try:
        journal.create_many([note()])
        task = journal.claim(note().key, action=ACTION)
        adapter = LocalNoteAdapter(workspace)
        name = adapter.filename(receipt_request(task, provider_id=adapter.provider_id))
        workspace.publish(name, "Different text")
    finally:
        journal.close()
    with pytest.raises(ReceiptMismatch):
        await run_local(workspace)
    assert (workspace.path / name).read_text() == "Different text"


@pytest.mark.parametrize("name", [".approval.json", ".tasks.db", ".tasks.db-wal", ".worker.lock"])
async def test_symlink_is_refused(workspace, tmp_path, name):
    allow(workspace)
    target = tmp_path / "untouched"
    target.write_text("do not change")
    path = workspace.path / name
    path.unlink(missing_ok=True)
    path.symlink_to(target)
    with pytest.raises(OSError):
        await run_local(workspace)
    assert target.read_text() == "do not change"


def test_shared_directory_is_refused(tmp_path):
    path = tmp_path / "shared"
    path.mkdir(mode=0o755)
    path.chmod(0o755)
    with pytest.raises(PermissionError):
        NoteWorkspace(path)


def test_shared_policy_is_refused(workspace):
    allow(workspace)
    (workspace.path / ".approval.json").chmod(0o644)
    with pytest.raises(PermissionError):
        FileApproval(workspace).load()


def test_no_path_escape(workspace, tmp_path):
    with pytest.raises(ValueError):
        workspace.publish("../outside", "no")
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize(
    "kwargs",
    [{"max_actions": 0}, {"max_cycles": 61}, {"seconds": float("nan")}, {"poll_seconds": 0}],
)
def test_limits_reject_unbounded_values(kwargs):
    with pytest.raises(ValueError):
        RunLimits(**kwargs)


async def test_live_stop_during_poll(workspace):
    allow(workspace)

    async def stop():
        await asyncio.sleep(0.01)
        workspace.stop()

    stopper = asyncio.create_task(stop())
    result = await run_local(workspace, RunLimits(max_cycles=3, poll_seconds=0.05))
    await stopper
    assert result.reason == "stopped" and result.actions == 1


async def test_deadline_prevents_next_cycle_effect(workspace):
    allow(workspace)
    result = await run_local(workspace, RunLimits(max_cycles=3, seconds=0.01, poll_seconds=0.05))
    assert result.reason == "deadline"


def test_only_registered_action_allowed(workspace):
    with pytest.raises(ValueError):
        approve(workspace, [TaskRequest(note().key, "shell", "echo no")], ttl_seconds=60)


def test_broken_stop_symlink_still_stops(workspace):
    (workspace.path / "STOP").symlink_to(workspace.path / "absent")
    from morgan_brain.local_notes import StopRequested

    with pytest.raises(StopRequested):
        workspace.check_stop()


async def test_note_symlink_is_not_followed_or_replaced(workspace, tmp_path):
    allow(workspace)
    journal = workspace.journal()
    try:
        journal.create_many([note()])
        task = journal.claim(note().key, action=ACTION)
        adapter = LocalNoteAdapter(workspace)
        name = adapter.filename(receipt_request(task, provider_id=adapter.provider_id))
        target = tmp_path / "external.md"
        target.write_text("untouched")
        (workspace.path / name).symlink_to(target)
    finally:
        journal.close()
    with pytest.raises(OSError):
        await run_local(workspace)
    assert target.read_text() == "untouched"
    assert (workspace.path / name).is_symlink()


async def test_deleted_completed_note_is_not_recreated(workspace):
    allow(workspace)
    await run_local(workspace)
    next(workspace.path.glob("note-*.md")).unlink()
    assert (await run_local(workspace)).actions == 0
    assert not list(workspace.path.glob("note-*.md"))
