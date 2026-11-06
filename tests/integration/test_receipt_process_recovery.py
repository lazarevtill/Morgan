"""Provider completion survives process death; recovery only looks up, never replays."""

import asyncio
import multiprocessing
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from examples.synthetic_receipts import SyntheticReceiptProvider
from morgan_brain.task_batches import TaskPermissions, TaskPermit
from morgan_brain.task_receipts import ReceiptAccess, reconcile_task
from morgan_brain.tasks import TaskConflict, TaskJournal, TaskKey, TaskRequest

KEY = TaskKey("owner", "synthetic", "crash")


def _crash(directory, phase):
    """Exit without cleanup after provider completion or after local confirmation."""
    directory = Path(directory)
    journal = TaskJournal(directory / "tasks.db")
    provider = SyntheticReceiptProvider(directory / "provider.db")
    claimed = journal.claim(KEY, action="uppercase-v1")
    asyncio.run(provider.execute(claimed))
    if phase == "after_provider":
        os._exit(51)
    journal.mark_unknown(KEY, attempt_id=claimed.attempt_id)
    asyncio.run(
        reconcile_task(
            journal,
            KEY,
            expected_attempt_id=claimed.attempt_id,
            access=ReceiptAccess(provider, lambda task: None),
            timeout_seconds=1,
        )
    )
    os._exit(51)


@pytest.mark.parametrize("phase", ["after_provider", "after_confirmation"])
async def test_receipt_recovery_after_process_exit(tmp_path, phase):
    """Recover one external synthetic effect after abrupt death without replay."""
    journal = TaskJournal(tmp_path / "tasks.db")
    journal.create(KEY, action="uppercase-v1", payload="synthetic")
    journal.close()
    child = multiprocessing.get_context("spawn").Process(target=_crash, args=(str(tmp_path), phase))
    child.start()
    try:
        await asyncio.to_thread(child.join, 20)
        assert not child.is_alive() and child.exitcode == 51
    finally:
        if child.is_alive():
            child.kill()
            await asyncio.to_thread(child.join, 5)
        child.close()
    journal = TaskJournal(tmp_path / "tasks.db")
    provider = SyntheticReceiptProvider(tmp_path / "provider.db")
    try:
        current = journal.get(KEY)
        assert provider.count_effects() == 1
        if phase == "after_provider":
            assert current.status == "running"
            journal.mark_unknown(KEY, attempt_id=current.attempt_id)
        else:
            assert current.status == "succeeded" and current.receipt_json
        permissions = TaskPermissions()
        permissions.grant(
            TaskPermit(
                TaskRequest(KEY, current.action, current.payload),
                "synthetic-reconciliation",
                datetime.now(UTC) + timedelta(minutes=1),
            )
        )
        result = await reconcile_task(
            journal,
            KEY,
            expected_attempt_id=current.attempt_id,
            access=ReceiptAccess(provider, permissions.check),
            timeout_seconds=1,
        )
        assert result.outcome == ("confirmed" if phase == "after_provider" else "already_confirmed")
        assert result.task.result == "SYNTHETIC" and result.task.receipt_json
        assert provider.count_effects() == 1
        with pytest.raises(TaskConflict):
            journal.claim(KEY, action="uppercase-v1")
    finally:
        provider.close()
        journal.close()
