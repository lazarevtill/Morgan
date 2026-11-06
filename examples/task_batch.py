"""Two synthetic tasks with exact short-lived permissions; no network or credentials.

PYTHONPATH=. python examples/task_batch.py NEW_JOURNAL --owner demo
These example permits authorize only the pure uppercase handler, never real user actions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from morgan_brain.task_batches import TaskPermissions, TaskPermit, run_batch
from morgan_brain.tasks import Task, TaskControl, TaskJournal, TaskKey, TaskRequest


async def uppercase(task: Task, control: TaskControl) -> str:
    control.check_ready()
    return task.payload.upper()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--owner", required=True)
    args = parser.parse_args()
    journal = TaskJournal(args.journal)
    try:
        requests = [
            TaskRequest(TaskKey(args.owner, "synthetic-batch", key), "uppercase-v1", key)
            for key in ("first", "second")
        ]
        journal.create_many(requests)
        policy = TaskPermissions()
        for item in requests:
            policy.grant(
                TaskPermit(item, "synthetic-example", datetime.now(UTC) + timedelta(minutes=1))
            )
        result = await run_batch(
            journal,
            [item.key for item in requests],
            permissions=policy,
            handlers={"uppercase-v1": uppercase},
            timeout_seconds=2,
        )
        print(json.dumps(asdict(result)))
    finally:
        journal.close()


if __name__ == "__main__":
    asyncio.run(main())
