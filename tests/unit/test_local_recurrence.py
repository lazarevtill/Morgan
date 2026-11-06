"""Finite repeat authority survives restart without catch-up storms or shifted identities."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from morgan_brain import local_notes, task_batches
from morgan_brain.local_notes import (
    ACTION,
    FileApproval,
    NoteWorkspace,
    approve_occurrences,
    enqueue,
    local_status,
    run_local,
)
from morgan_brain.local_recurrence import RepeatLimits, repeat_plan
from morgan_brain.task_batches import TaskPermissionDenied
from morgan_brain.tasks import TaskKey, TaskRequest


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    """Use one controlled wall clock for time windows and exact grants."""

    class Clock(datetime):
        current = datetime(2026, 11, 6, 8, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.current.astimezone(tz)

    monkeypatch.setattr(local_notes, "datetime", Clock)
    monkeypatch.setattr(task_batches, "datetime", Clock)
    workspace = NoteWorkspace(tmp_path / "agent")
    request = TaskRequest(TaskKey("owner", "personal", "water"), ACTION, "# Сделать перерыв\n")
    yield workspace, request, Clock
    workspace.close()


def plan(request, clock, **overrides):
    """Build a short explicit schedule with deliberately separated delivery windows."""
    args = {
        "start": clock.current,
        "interval_seconds": 60,
        "count": 3,
        "window_seconds": 20,
        "now": clock.current,
    }
    args.update(overrides)
    limits = RepeatLimits(
        args.pop("interval_seconds"), args.pop("count"), args.pop("window_seconds")
    )
    return repeat_plan(request, limits=limits, start=args["start"], now=args["now"])


async def test_restart_skips_missed_windows_and_delivers_each_once(fixture):
    """A late restart skips missed occurrences and cannot repeat already delivered ones."""
    workspace, request, clock = fixture
    entries = plan(request, clock)
    approve_occurrences(workspace, entries)
    assert (await run_local(workspace)).actions == 1
    clock.current += timedelta(seconds=120)
    reopened = NoteWorkspace(workspace.path, create=False)
    try:
        assert (await run_local(reopened)).actions == 1
        assert (await run_local(reopened)).actions == 0
    finally:
        reopened.close()
    journal = workspace.journal()
    try:
        assert journal.get(entries[1].request.key) is None
        assert journal.get(entries[2].request.key).status == "succeeded"
    finally:
        journal.close()
    assert len(list(workspace.path.glob("note-*.md"))) == 2
    assert local_status(workspace)["elapsed_windows"] == 2


async def test_stop_resume_does_not_reauthorize_elapsed_occurrences(fixture):
    """Persistent STOP pauses all occurrences; resume does not create a backlog."""
    workspace, request, clock = fixture
    approve_occurrences(workspace, plan(request, clock))
    workspace.stop()
    clock.current += timedelta(seconds=65)
    assert (await run_local(workspace)).reason == "stopped"
    workspace.resume()
    assert (await run_local(workspace)).actions == 1
    clock.current += timedelta(seconds=75)
    assert (await run_local(workspace)).reason == "permission_denied"
    assert len(list(workspace.path.glob("note-*.md"))) == 1


async def test_boundary_gap_and_clock_rollback(fixture):
    """Half-open windows exclude their end, and rolling back cannot duplicate a prior slot."""
    workspace, request, clock = fixture
    approve_occurrences(workspace, plan(request, clock))
    assert (await run_local(workspace)).actions == 1
    clock.current += timedelta(seconds=20)
    assert (await run_local(workspace)).actions == 0
    clock.current -= timedelta(seconds=1)
    assert (await run_local(workspace)).actions == 0
    clock.current += timedelta(seconds=41)
    assert (await run_local(workspace)).actions == 1
    assert len(list(workspace.path.glob("note-*.md"))) == 2


def test_identity_survives_extension_offsets_and_content_change(fixture):
    """Occurrence IDs bind scope, series ID and UTC instant, not mutable content or count."""
    _workspace, request, clock = fixture
    original = plan(request, clock)
    expanded = plan(
        request, clock, count=4, start=datetime.fromisoformat("2026-11-06T11:00:00+03:00")
    )
    assert [x.request.key for x in expanded[:3]] == [x.request.key for x in original]
    changed = TaskRequest(request.key, ACTION, "Changed content")
    assert plan(changed, clock)[0].request.key == original[0].request.key


@pytest.mark.parametrize(
    "overrides",
    [
        {"count": 1},
        {"count": 17},
        {"count": True},
        {"interval_seconds": 59},
        {"window_seconds": 61},
        {"window_seconds": 0},
        {"start": datetime(2026, 11, 6, 8, tzinfo=UTC).replace(tzinfo=None)},
        {"start": datetime(2027, 1, 1, tzinfo=UTC)},
        {"start": datetime(2026, 1, 1, tzinfo=UTC)},
    ],
)
def test_invalid_plan_is_rejected(fixture, overrides):
    """Bound count, frequency, horizon, window, and timezone before persistence."""
    _workspace, request, clock = fixture
    with pytest.raises(ValueError):
        plan(request, clock, **overrides)


def test_dst_is_elapsed_utc_time_not_wall_clock_recurrence(fixture):
    """An offset is one instant; daily intervals stay 86400 seconds across a DST boundary."""
    _workspace, request, _clock = fixture
    start = datetime.fromisoformat("2026-10-31T09:00:00-04:00")
    entries = repeat_plan(request, start=start, limits=RepeatLimits(86400, 3, 60), now=start)
    assert entries[1].not_before.isoformat() == "2026-11-01T13:00:00+00:00"


@pytest.mark.parametrize("change", ["overlap", "duplicate", "foreign", "extra"])
def test_corrupt_policy_fails_closed(fixture, change):
    """Invalid windows, repeated keys and altered workspace cannot grant an effect."""
    workspace, request, clock = fixture
    approve_occurrences(workspace, plan(request, clock))
    raw = json.loads(workspace.read(".approval.json", 300000))
    if change == "overlap":
        raw["occurrences"][1]["not_before"] = raw["occurrences"][0]["not_before"]
    elif change == "duplicate":
        raw["occurrences"][1]["request"]["key"] = raw["occurrences"][0]["request"]["key"]
    elif change == "foreign":
        raw["workspace"] = "elsewhere"
    else:
        raw["extra"] = True
    workspace.publish(".approval.json", json.dumps(raw), replace=True)
    with pytest.raises((TaskPermissionDenied, ValueError)):
        FileApproval(workspace).load()


def test_enqueue_cannot_widen_recurring_authority(fixture):
    """Adding immediate work cannot silently rewrite an approved repeating plan."""
    workspace, request, clock = fixture
    approve_occurrences(workspace, plan(request, clock))
    before = workspace.read(".approval.json", 300000)
    with pytest.raises(TaskPermissionDenied):
        enqueue(workspace, request, ttl_seconds=600)
    assert workspace.read(".approval.json", 300000) == before


async def test_expired_unknown_attempt_blocks_later_occurrence(fixture):
    """An interrupted prior occurrence cannot be bypassed by advancing to another window."""
    workspace, request, clock = fixture
    entries = plan(request, clock)
    approve_occurrences(workspace, entries)
    journal = workspace.journal()
    try:
        journal.create_many([entries[0].request])
        journal.claim(entries[0].request.key, action=ACTION)
    finally:
        journal.close()
    clock.current += timedelta(seconds=60)
    assert (await run_local(workspace)).reason == "needs_reconciliation"
    assert not list(workspace.path.glob("note-*.md"))


async def test_editing_payload_with_same_series_and_instant_conflicts(fixture):
    """Changing content cannot evade idempotency by generating another occurrence ID."""
    workspace, request, clock = fixture
    approve_occurrences(workspace, plan(request, clock))
    assert (await run_local(workspace)).actions == 1
    changed = TaskRequest(request.key, ACTION, "Changed")
    approve_occurrences(workspace, plan(changed, clock))
    from morgan_brain.tasks import TaskConflict

    with pytest.raises(TaskConflict):
        await run_local(workspace)
    assert len(list(workspace.path.glob("note-*.md"))) == 1
