"""An unacknowledged synthetic side effect must never be automatically repeated."""

import asyncio
import multiprocessing
import os
from pathlib import Path

import pytest

from morgan_brain.tasks import TaskConflict, TaskJournal, TaskKey, run_task

KEY = TaskKey("owner", "synthetic", "crash")


def _effect(receipt):
    with Path(receipt).open("a") as output:
        output.write("effect\n")
        output.flush()
        os.fsync(output.fileno())


def _crash(path, receipt, phase):
    journal = TaskJournal(Path(path))
    if phase == "before_claim":
        os._exit(51)

    async def handler(task, control):
        if phase == "after_claim":
            os._exit(51)
        _effect(receipt)
        if phase == "after_effect":
            os._exit(51)
        return "receipt"

    asyncio.run(run_task(journal, KEY, handlers={"synthetic-v1": handler}, timeout_seconds=2))
    os._exit(51)


def _compete(path, receipt, start):
    journal = TaskJournal(Path(path))
    if not start.wait(10):
        os._exit(3)

    async def handler(task, control):
        _effect(receipt)
        return "receipt"

    try:
        asyncio.run(run_task(journal, KEY, handlers={"synthetic-v1": handler}, timeout_seconds=2))
    except TaskConflict:
        os._exit(2)
    finally:
        journal.close()


async def test_competing_processes_invoke_exactly_one_handler(tmp_path):
    path, receipt = tmp_path / "tasks.db", tmp_path / "effect.txt"
    journal = TaskJournal(path)
    journal.create(KEY, action="synthetic-v1", payload="")
    journal.close()
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(target=_compete, args=(str(path), str(receipt), start)) for _ in range(2)
    ]
    for process in processes:
        process.start()
    try:
        start.set()
        for process in processes:
            await asyncio.to_thread(process.join, 20)
            assert not process.is_alive()
        assert sorted(process.exitcode for process in processes) == [0, 2]
        assert receipt.read_text() == "effect\n"
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 5)
            process.close()


@pytest.mark.parametrize("phase", ["before_claim", "after_claim", "after_effect", "after_finish"])
async def test_abrupt_exit_never_replays_an_attempt(tmp_path, phase):
    path, receipt = tmp_path / "tasks.db", tmp_path / "effect.txt"
    journal = TaskJournal(path)
    journal.create(KEY, action="synthetic-v1", payload="")
    journal.close()
    process = multiprocessing.get_context("spawn").Process(
        target=_crash, args=(str(path), str(receipt), phase)
    )
    process.start()
    try:
        await asyncio.to_thread(process.join, 20)
        assert not process.is_alive()
        assert process.exitcode == 51
    finally:
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 5)
        process.close()

    journal = TaskJournal(path)
    try:
        recovered = journal.get(KEY)

        async def handler(task, control):
            _effect(receipt)
            return "receipt"

        if phase == "before_claim":
            assert recovered.status == "pending"
            await run_task(journal, KEY, handlers={"synthetic-v1": handler}, timeout_seconds=1)
        else:
            assert recovered.status == ("succeeded" if phase == "after_finish" else "running")
            if recovered.status == "running":
                journal.mark_unknown(KEY, attempt_id=recovered.attempt_id)
                assert journal.get(KEY).status == "outcome_unknown"
            with pytest.raises(TaskConflict):
                await run_task(journal, KEY, handlers={"synthetic-v1": handler}, timeout_seconds=1)
        if phase == "after_claim":
            assert not receipt.exists()
        else:
            assert receipt.read_text() == "effect\n"
    finally:
        journal.close()
