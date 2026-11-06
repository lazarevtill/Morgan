"""Explicit synthetic task; no model, network, credentials or background activity.

PYTHONPATH=. python examples/single_task.py NEW_JOURNAL --owner demo --task-id example
Run the same command again to read the persisted result without invoking the action again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from morgan_brain.tasks import Task, TaskControl, TaskJournal, TaskKey, run_task


async def uppercase(task: Task, control: TaskControl) -> str:
    control.check_cancelled()
    return task.payload.upper()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    journal = TaskJournal(args.journal)
    try:
        key = TaskKey(args.owner, "synthetic-example", args.task_id)
        task = journal.create(key, action="uppercase-v1", payload="synthetic task")
        if task.status == "pending":
            task = await run_task(
                journal, key, handlers={"uppercase-v1": uppercase}, timeout_seconds=1
            )
        print(json.dumps(asdict(task)))
    finally:
        journal.close()


if __name__ == "__main__":
    asyncio.run(main())
