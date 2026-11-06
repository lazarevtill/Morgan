"""A restarted explicit batch stops at unresolved work and needs fresh execution permits."""

import asyncio
import multiprocessing
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from morgan_brain.task_batches import TaskPermissions, TaskPermit, run_batch
from morgan_brain.tasks import TaskJournal, TaskKey, TaskRequest


def requests():
    return [TaskRequest(TaskKey("owner", "synthetic", key), "record-v1", key) for key in ("a", "b")]


def permissions():
    result = TaskPermissions()
    for item in requests():
        result.grant(TaskPermit(item, "synthetic-approval", datetime.now(UTC) + timedelta(hours=1)))
    return result


def effect(path, value):
    with Path(path).open("a") as file:
        file.write(value + "\n")
        file.flush()
        os.fsync(file.fileno())


def crash(path, receipt, phase):
    journal = TaskJournal(Path(path))

    class CrashBetweenTasks(TaskPermissions):
        def check(self, task):
            if phase == "between_tasks" and task.key.task_id == "b":
                os._exit(51)
            super().check(task)

    policy = CrashBetweenTasks()
    for item in requests():
        policy.grant(TaskPermit(item, "synthetic-approval", datetime.now(UTC) + timedelta(hours=1)))

    async def handler(task, control):
        control.check_ready()
        effect(receipt, task.payload)
        if phase == "after_effect":
            os._exit(51)
        return "recorded"

    asyncio.run(
        run_batch(
            journal,
            [item.key for item in requests()],
            permissions=policy,
            handlers={"record-v1": handler},
            timeout_seconds=2,
        )
    )


@pytest.mark.parametrize("phase", ["after_effect", "between_tasks"])
async def test_batch_restart_never_duplicates_or_skips_uncertain_work(tmp_path, phase):
    path, receipt = tmp_path / "tasks.db", tmp_path / "effects.txt"
    journal = TaskJournal(path)
    journal.create_many(requests())
    journal.close()
    child = multiprocessing.get_context("spawn").Process(
        target=crash, args=(str(path), str(receipt), phase)
    )
    child.start()
    try:
        await asyncio.to_thread(child.join, 20)
        assert not child.is_alive() and child.exitcode == 51
    finally:
        if child.is_alive():
            child.kill()
            await asyncio.to_thread(child.join, 5)
        child.close()

    journal = TaskJournal(path)

    async def handler(task, control):
        control.check_ready()
        effect(receipt, task.payload)
        return "recorded"

    try:
        keys = [item.key for item in requests()]
        first = await run_batch(
            journal,
            keys,
            permissions=TaskPermissions(),
            handlers={"record-v1": handler},
            timeout_seconds=1,
        )
        assert first.reason == (
            "needs_reconciliation" if phase == "after_effect" else "permission_denied"
        )
        assert receipt.read_text() == "a\n"
        assert journal.get(keys[1]).attempt_id is None
        second = await run_batch(
            journal,
            keys,
            permissions=permissions(),
            handlers={"record-v1": handler},
            timeout_seconds=1,
        )
        if phase == "after_effect":
            assert second.reason == "needs_reconciliation"
            assert receipt.read_text() == "a\n"
        else:
            assert second.reason == "complete"
            assert receipt.read_text() == "a\nb\n"
    finally:
        journal.close()
