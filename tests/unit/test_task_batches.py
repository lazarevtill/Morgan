"""Bounded admission, exact current permission, and stop-on-uncertainty semantics."""

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from morgan_brain.task_batches import TaskPermissions, TaskPermit, run_batch
from morgan_brain.tasks import TaskConflict, TaskJournal, TaskKey, TaskRequest

NOW = datetime(2026, 11, 6, tzinfo=UTC)


def request(identity, **kwargs):
    return TaskRequest(
        TaskKey("owner", "project", identity), "echo-v1", kwargs.get("payload", identity)
    )


@pytest.fixture
def journal(tmp_path):
    ledger = TaskJournal(tmp_path / "tasks.db")
    yield ledger
    ledger.close()


def allow(*requests, clock=lambda: NOW):
    permissions = TaskPermissions(clock=clock)
    for item in requests:
        permissions.grant(TaskPermit(item, "synthetic-approval", NOW + timedelta(hours=1)))
    return permissions


async def echo(task, control):
    control.check_ready()
    return task.payload


async def process(journal, requests, permissions, handler=echo, seconds=1):
    return await run_batch(
        journal,
        [item.key for item in requests],
        permissions=permissions,
        handlers={"echo-v1": handler},
        timeout_seconds=seconds,
    )


def test_atomic_submission_rejects_late_conflict_and_preserves_existing(journal):
    a, b = request("a"), request("b")
    journal.create(b.key, action=b.action, payload="old")
    with pytest.raises(TaskConflict):
        journal.create_many([a, b])
    assert journal.get(a.key) is None
    assert journal.get(b.key).payload == "old"
    assert journal.create_many([a]) == journal.create_many([a])


@pytest.mark.parametrize("kind", ["empty", "duplicate", "too-many", "cross-owner", "cross-project"])
async def test_invalid_worklist_fails_before_any_write_or_run(journal, kind):
    a = request("a")
    requests = {
        "empty": [],
        "duplicate": [a, a],
        "too-many": [request(str(i)) for i in range(17)],
        "cross-owner": [a, TaskRequest(TaskKey("other", "project", "b"), "echo-v1", "b")],
        "cross-project": [a, TaskRequest(TaskKey("owner", "other", "b"), "echo-v1", "b")],
    }[kind]
    with pytest.raises(ValueError):
        journal.create_many(requests)
    with pytest.raises(ValueError):
        await process(journal, requests, allow(*requests))
    assert all(journal.get(item.key) is None for item in requests)


async def test_multiple_tasks_succeed_in_order_and_replay_only_reads(journal):
    requests = [request(str(i)) for i in range(16)]
    journal.create_many(requests)
    calls = []

    async def handler(task, control):
        control.check_ready()
        calls.append(task.payload)
        return task.payload

    first = await process(journal, requests, allow(*requests), handler)
    assert first.reason == "complete" and first.stopped_at is None
    assert calls == [item.payload for item in requests]
    second = await process(journal, requests, TaskPermissions(), handler)
    assert second == first  # observing completed work requires no new execution grant
    assert len(calls) == 16


@pytest.mark.parametrize(
    "mismatch", ["missing", "owner", "project", "task", "action", "payload", "expired"]
)
async def test_permission_is_exact_and_denial_leaves_attempt_unclaimed(journal, mismatch):
    a, b = request("a"), request("b")
    journal.create_many([a, b])
    replacement = {
        "owner": TaskRequest(TaskKey("other", "project", "a"), a.action, a.payload),
        "project": TaskRequest(TaskKey("owner", "other", "a"), a.action, a.payload),
        "task": request("other"),
        "action": TaskRequest(a.key, "other-v1", a.payload),
        "payload": request("a", payload="different"),
    }.get(mismatch, a)
    permissions = allow(b)
    if mismatch != "missing":
        expiry = NOW if mismatch == "expired" else NOW + timedelta(hours=1)
        permissions.grant(TaskPermit(replacement, "synthetic-approval", expiry))
    outcome = await process(journal, [a, b], permissions)
    assert outcome.reason == "permission_denied" and outcome.stopped_at == a.key
    assert journal.get(a.key).attempt_id is None
    assert journal.get(b.key).attempt_id is None


async def test_revocation_between_tasks_stops_before_second_claim(journal):
    a, b = request("a"), request("b")
    journal.create_many([a, b])
    permissions = allow(a, b)

    async def handler(task, control):
        control.check_ready()
        permissions.revoke(b.key)
        return "done"

    outcome = await process(journal, [a, b], permissions, handler)
    assert outcome.reason == "permission_denied"
    assert journal.get(a.key).status == "succeeded"
    assert journal.get(b.key).status == "pending"


@pytest.mark.parametrize("cause", ["revoke", "expire", "cancel", "error", "conflict", "timeout"])
async def test_uncertainty_stops_remaining_tasks_and_cannot_replay(journal, cause):
    a, b = request("a"), request("b")
    journal.create_many([a, b])
    clock = [NOW]
    permissions = allow(a, b, clock=lambda: clock[0])
    calls = []

    async def handler(task, control):
        calls.append(task.key)
        if cause == "revoke":
            permissions.revoke(task.key)
        elif cause == "expire":
            clock[0] += timedelta(hours=1)
        elif cause == "cancel":
            journal.cancel(task.key)
        elif cause == "error":
            raise RuntimeError("synthetic failure after an effect")
        elif cause == "conflict":
            raise TaskConflict("synthetic handler conflict after an effect")
        else:
            await asyncio.sleep(5)
        control.check_ready()
        pytest.fail("revoked, expired or cancelled permission allowed another effect")

    outcome = await process(journal, [a, b], permissions, handler, seconds=0.02)
    assert outcome.reason == "needs_reconciliation" and outcome.stopped_at == a.key
    assert journal.get(a.key).status == "outcome_unknown"
    assert journal.get(b.key).attempt_id is None
    replay = await process(journal, [a, b], allow(a, b), handler)
    assert replay.reason == "needs_reconciliation" and calls == [a.key]


@pytest.mark.parametrize("boundary", ["running", "cancelled", "missing", "missing-handler"])
async def test_existing_unsafe_boundaries_stop_without_invoking(journal, boundary):
    a, b = request("a"), request("b")
    if boundary != "missing":
        journal.create_many([a, b])
    if boundary == "running":
        journal.claim(a.key, action=a.action)
    elif boundary == "cancelled":
        journal.cancel(a.key)
    outcome = await run_batch(
        journal,
        [a.key, b.key],
        permissions=allow(a, b),
        handlers={} if boundary == "missing-handler" else {"echo-v1": echo},
        timeout_seconds=1,
    )
    assert (
        outcome.reason
        == {
            "running": "needs_reconciliation",
            "cancelled": "cancelled",
            "missing": "missing_task",
            "missing-handler": "missing_handler",
        }[boundary]
    )
    assert outcome.stopped_at == a.key


async def test_fresh_permission_registry_after_reopen_denies_pending_work(tmp_path):
    path = tmp_path / "tasks.db"
    journal = TaskJournal(path)
    a = request("a")
    journal.create_many([a])
    journal.close()
    reopened = TaskJournal(path)
    try:
        outcome = await process(reopened, [a], TaskPermissions())
        assert outcome.reason == "permission_denied"
        assert reopened.get(a.key).attempt_id is None
    finally:
        reopened.close()


async def test_storage_errors_propagate_and_do_not_start_remaining_tasks(journal, monkeypatch):
    a, b = request("a"), request("b")
    journal.create_many([a, b])

    def broken_finish(*args, **kwargs):
        raise sqlite3.OperationalError("synthetic storage failure")

    monkeypatch.setattr(journal, "finish", broken_finish)
    with pytest.raises(sqlite3.OperationalError):
        await process(journal, [a, b], allow(a, b))
    assert journal.get(b.key).attempt_id is None


async def test_naive_permission_clock_fails_closed(journal):
    a = request("a")
    journal.create_many([a])
    permissions = allow(a, clock=lambda: NOW.replace(tzinfo=None))
    assert (await process(journal, [a], permissions)).reason == "permission_denied"


def test_naive_expiration_is_refused():
    with pytest.raises(ValueError, match="timezone-aware"):
        TaskPermit(request("a"), "approval", NOW.replace(tzinfo=None))


async def test_permission_rechecked_after_claim_before_handler(journal):
    a, b = request("a"), request("b")
    journal.create_many([a, b])
    times = iter([NOW, NOW + timedelta(hours=1)])
    permissions = allow(a, b, clock=lambda: next(times))

    async def forbidden(task, control):
        pytest.fail("permission expired after claim but handler was invoked")

    outcome = await process(journal, [a, b], permissions, forbidden)
    assert outcome.reason == "needs_reconciliation"
    assert journal.get(a.key).status == "outcome_unknown"
    assert journal.get(b.key).attempt_id is None


async def test_timeout_is_shared_across_handlers_not_reset_for_each(journal, monkeypatch):
    requests = [request(str(i)) for i in range(3)]
    journal.create_many(requests)
    loop = asyncio.get_running_loop()
    elapsed = [loop.time()]
    calls = []

    async def slow(task, control):
        calls.append(task.key)
        elapsed[0] += 0.03
        if len(calls) == 2:
            # Yield until the real asyncio timeout fires at the controlled clock time.
            # A reset per-handler budget would incorrectly let this handler finish.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        control.check_ready()
        return "done"

    # Wall-clock scheduling and SQLite latency must not decide which handler times out.
    with monkeypatch.context() as patch:
        patch.setattr(loop, "time", lambda: elapsed[0])
        outcome = await process(journal, requests, allow(*requests), slow, seconds=0.05)
    assert outcome.reason == "needs_reconciliation"
    assert calls == [item.key for item in requests[:2]]
    assert journal.get(requests[0].key).status == "succeeded"
    assert journal.get(requests[1].key).status == "outcome_unknown"
    assert journal.get(requests[-1].key).attempt_id is None


async def test_caller_cancellation_propagates_without_starting_next_task(journal):
    a, b = request("a"), request("b")
    journal.create_many([a, b])
    started = asyncio.Event()

    async def blocked(task, control):
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    runner = asyncio.create_task(process(journal, [a, b], allow(a, b), blocked))
    try:
        await asyncio.wait_for(started.wait(), 2)
    finally:
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner
    assert journal.get(a.key).status == "outcome_unknown"
    assert journal.get(b.key).attempt_id is None
