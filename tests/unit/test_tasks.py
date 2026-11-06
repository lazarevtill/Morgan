"""Explicit task execution never turns an uncertain outcome into an automatic retry."""

import asyncio
import sqlite3
from pathlib import Path

import pytest

from morgan_brain.tasks import (
    TaskCancellationRequested,
    TaskConflict,
    TaskJournal,
    TaskKey,
    run_task,
)

KEY = TaskKey("owner", "synthetic", "one")


@pytest.mark.parametrize("path", [Path(":memory:"), "", ":memory:"])
def test_ephemeral_journal_is_refused(path):
    with pytest.raises(ValueError, match="on-disk"):
        TaskJournal(path)


@pytest.fixture
def journal(tmp_path):
    ledger = TaskJournal(tmp_path / "tasks.db")
    yield ledger
    ledger.close()


async def echo(task, control):
    control.check_cancelled()
    return task.payload


def test_exact_create_replay_is_scoped_and_conflicting_spec_is_refused(journal):
    first = journal.create(KEY, action="echo-v1", payload="hello")
    assert journal.create(KEY, action="echo-v1", payload="hello") == first
    for action, payload in [("other", "hello"), ("echo-v1", "changed")]:
        with pytest.raises(TaskConflict):
            journal.create(KEY, action=action, payload=payload)
    assert journal.get(TaskKey("other", KEY.project, KEY.task_id)) is None
    assert journal.get(TaskKey(KEY.user_id, "other", KEY.task_id)) is None
    with pytest.raises(TaskConflict):
        journal.cancel(TaskKey("other", KEY.project, KEY.task_id))


async def test_success_survives_reopen_and_cannot_run_again(tmp_path):
    path = tmp_path / "tasks.db"
    journal = TaskJournal(path)
    try:
        journal.create(KEY, action="echo-v1", payload="hello")
        result = await run_task(journal, KEY, handlers={"echo-v1": echo}, timeout_seconds=1)
        assert result.status == "succeeded" and result.result == "hello"
        assert result.attempt_id
    finally:
        journal.close()
    reopened = TaskJournal(path)
    try:
        assert reopened.get(KEY) == result
        with pytest.raises(TaskConflict):
            await run_task(reopened, KEY, handlers={"echo-v1": echo}, timeout_seconds=1)
    finally:
        reopened.close()


async def test_pending_cancellation_or_missing_handler_never_invokes(journal):
    journal.create(KEY, action="echo-v1", payload="hello")
    with pytest.raises(ValueError, match="registered"):
        await run_task(journal, KEY, handlers={}, timeout_seconds=1)
    assert journal.get(KEY).status == "pending"
    assert journal.cancel(KEY).status == "cancelled"

    async def forbidden(task, control):
        pytest.fail("cancelled task was invoked")

    with pytest.raises(TaskConflict):
        await run_task(journal, KEY, handlers={"echo-v1": forbidden}, timeout_seconds=1)
    assert journal.get(KEY).attempt_id is None


async def test_two_connections_cannot_invoke_one_task_twice(tmp_path):
    first = TaskJournal(tmp_path / "tasks.db")
    second = TaskJournal(tmp_path / "tasks.db")
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def blocked(task, control):
        calls.append(task.attempt_id)
        started.set()
        await release.wait()
        return "done"

    first.create(KEY, action="blocked-v1", payload="")
    runner = asyncio.create_task(
        run_task(first, KEY, handlers={"blocked-v1": blocked}, timeout_seconds=5)
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        with pytest.raises(TaskConflict):
            await run_task(second, KEY, handlers={"blocked-v1": blocked}, timeout_seconds=1)
        release.set()
        assert (await runner).status == "succeeded"
        assert len(calls) == 1
    finally:
        release.set()
        await runner
        first.close()
        second.close()


@pytest.mark.parametrize("failure", ["exception", "bad-result", "timeout", "cancel"])
async def test_failure_after_claim_is_unknown_and_never_replayed(journal, failure):
    journal.create(KEY, action="action-v1", payload="")
    calls = []

    async def handler(task, control):
        calls.append(task.attempt_id)  # stands in for an already-submitted external action
        if failure == "exception":
            raise RuntimeError("synthetic error with private details")
        if failure == "bad-result":
            return "Ж" * 9000
        if failure == "cancel":
            raise asyncio.CancelledError
        await asyncio.sleep(10)
        return "too late"

    error = {
        "exception": RuntimeError,
        "bad-result": ValueError,
        "timeout": TimeoutError,
        "cancel": asyncio.CancelledError,
    }[failure]
    with pytest.raises(error):
        await run_task(journal, KEY, handlers={"action-v1": handler}, timeout_seconds=0.01)
    unknown = journal.get(KEY)
    assert unknown.status == "outcome_unknown" and unknown.result is None
    with pytest.raises(TaskConflict):
        await run_task(journal, KEY, handlers={"action-v1": handler}, timeout_seconds=1)
    assert len(calls) == 1


@pytest.mark.parametrize("cooperate", [False, True])
async def test_running_cancellation_is_not_false_proof_of_no_effect(journal, cooperate):
    journal.create(KEY, action="action-v1", payload="")

    async def handler(task, control):
        cancelled = journal.cancel(KEY)
        assert cancelled.status == "running" and cancelled.cancel_requested
        if cooperate:
            control.check_cancelled()
        return "already completed"

    if cooperate:
        with pytest.raises(TaskCancellationRequested):
            await run_task(journal, KEY, handlers={"action-v1": handler}, timeout_seconds=1)
        assert journal.get(KEY).status == "outcome_unknown"
    else:
        result = await run_task(journal, KEY, handlers={"action-v1": handler}, timeout_seconds=1)
        assert result.status == "succeeded" and result.cancel_requested


def test_recovery_fences_late_completion_but_does_not_reclaim(journal):
    journal.create(KEY, action="echo-v1", payload="")
    claimed = journal.claim(KEY, action="echo-v1")
    with pytest.raises(TaskConflict):
        journal.mark_unknown(KEY, attempt_id="wrong")
    assert journal.get(KEY).status == "running"
    journal.mark_unknown(KEY, attempt_id=claimed.attempt_id)
    with pytest.raises(TaskConflict):
        journal.finish(KEY, attempt_id=claimed.attempt_id, result="late")
    with pytest.raises(TaskConflict):
        journal.claim(KEY, action="echo-v1")


@pytest.mark.parametrize("content", ["Ж" * 8193, None])
def test_input_bounds_leave_no_task(journal, content):
    with pytest.raises(ValueError):
        journal.create(KEY, action="echo-v1", payload=content)
    assert journal.get(KEY) is None


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), 3601])
async def test_invalid_timeout_never_claims(journal, seconds):
    journal.create(KEY, action="echo-v1", payload="")
    with pytest.raises(ValueError, match="timeout"):
        await run_task(journal, KEY, handlers={"echo-v1": echo}, timeout_seconds=seconds)
    assert journal.get(KEY).status == "pending"


def test_foreign_database_is_refused_without_modification(tmp_path):
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE memories(content TEXT)")
    conn.commit()
    conn.close()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="supported task journal"):
        TaskJournal(path)
    assert path.read_bytes() == before


def test_future_journal_version_is_refused(tmp_path):
    """Refuse a journal version newer than the supported receipt schema."""
    path = tmp_path / "tasks.db"
    TaskJournal(path).close()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version=3")
    conn.close()
    with pytest.raises(ValueError, match="supported task journal"):
        TaskJournal(path)


@pytest.mark.parametrize("committed", [False, True])
async def test_result_storage_failure_preserves_committed_outcome(journal, monkeypatch, committed):
    journal.create(KEY, action="echo-v1", payload="done")
    finish = journal.finish

    def broken_finish(*args, **kwargs):
        if committed:
            finish(*args, **kwargs)
        raise sqlite3.OperationalError("synthetic lost acknowledgement")

    monkeypatch.setattr(journal, "finish", broken_finish)
    with pytest.raises(sqlite3.OperationalError):
        await run_task(journal, KEY, handlers={"echo-v1": echo}, timeout_seconds=1)
    task = journal.get(KEY)
    assert task.status == ("succeeded" if committed else "outcome_unknown")
    assert task.result == ("done" if committed else None)


async def test_cancelling_runner_leaves_unknown_and_propagates_cancellation(journal):
    journal.create(KEY, action="blocked-v1", payload="")
    started = asyncio.Event()

    async def handler(task, control):
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    runner = asyncio.create_task(
        run_task(journal, KEY, handlers={"blocked-v1": handler}, timeout_seconds=5)
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
    finally:
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner
    assert journal.get(KEY).status == "outcome_unknown"
