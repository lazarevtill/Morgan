"""Synthetic idempotent provider and lost-acknowledgement recovery, without network or keys.

PYTHONPATH=. python examples/synthetic_receipts.py NEW_DIRECTORY --owner demo
The provider's only effect is one durable row containing uppercase synthetic text.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from morgan_brain.memory.store.transactions import write_transaction
from morgan_brain.task_batches import TaskPermissions, TaskPermit
from morgan_brain.task_receipts import (
    ReceiptAccess,
    ReceiptMismatch,
    ReceiptRequest,
    TaskReceipt,
    receipt_request,
    reconcile_task,
)
from morgan_brain.tasks import Task, TaskControl, TaskJournal, TaskKey, TaskRequest, run_task


class SyntheticReceiptProvider:
    """An isolated local model of provider-side idempotency, not a real service adapter."""

    provider_id = "synthetic-uppercase-v1"

    def __init__(self, path: Path) -> None:
        self.conn = sqlite3.connect(path)
        try:
            self._validate()
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=FULL")
            with write_transaction(self.conn):
                self.conn.execute(
                    "CREATE TABLE IF NOT EXISTS effects ("
                    "idempotency_key TEXT PRIMARY KEY, request_json TEXT NOT NULL, "
                    "receipt_id TEXT NOT NULL, result TEXT NOT NULL)"
                )
                self.conn.execute("PRAGMA application_id=0x53594E54")
                self.conn.execute("PRAGMA user_version=1")
        except BaseException:
            self.conn.close()
            raise

    def _validate(self) -> None:
        marker = self.conn.execute("PRAGMA application_id").fetchone()[0]
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        tables = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if (marker, version) != (0x53594E54, 1) and (marker != 0 or version != 0 or tables):
            raise ValueError("refusing a foreign synthetic-provider database")

    def close(self) -> None:
        self.conn.close()

    def count_effects(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM effects").fetchone()[0])

    @staticmethod
    def _encode(request: ReceiptRequest) -> str:
        return json.dumps(
            asdict(request), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )

    def _lookup(self, request: ReceiptRequest) -> TaskReceipt | None:
        if request.provider_id != self.provider_id:
            raise ReceiptMismatch("receipt request names another provider")
        row = self.conn.execute(
            "SELECT request_json,receipt_id,result FROM effects WHERE idempotency_key=?",
            (request.idempotency_key,),
        ).fetchone()
        if row is None:
            return None
        if row[0] != self._encode(request):
            raise ReceiptMismatch("idempotency key reused for a different request")
        return TaskReceipt(request, row[1], row[2])

    async def lookup(self, request: ReceiptRequest) -> TaskReceipt | None:
        return self._lookup(request)

    async def execute(self, task: Task) -> TaskReceipt:
        """Synthetic effect and receipt commit together; repeated identical keys return it."""
        if task.action != "uppercase-v1":
            raise ValueError("synthetic provider supports only uppercase-v1")
        request = receipt_request(task, provider_id=self.provider_id)
        with write_transaction(self.conn):
            old = self._lookup(request)
            if old is not None:
                return old
            receipt = TaskReceipt(request, uuid4().hex, task.payload.upper())
            self.conn.execute(
                "INSERT INTO effects VALUES (?,?,?,?)",
                (
                    request.idempotency_key,
                    self._encode(request),
                    receipt.receipt_id,
                    receipt.result,
                ),
            )
            return receipt


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--owner", required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    journal = TaskJournal(args.directory / "tasks.db")
    provider = SyntheticReceiptProvider(args.directory / "synthetic-provider.db")
    try:
        spec = TaskRequest(
            TaskKey(args.owner, "synthetic-receipts", "one"), "uppercase-v1", "hello"
        )
        task = journal.create(spec.key, action=spec.action, payload=spec.payload)
        policy = TaskPermissions()
        policy.grant(
            TaskPermit(spec, "synthetic-example", datetime.now(UTC) + timedelta(minutes=1))
        )

        async def lost_reply(current: Task, control: TaskControl) -> str:
            control.check_ready()
            await provider.execute(current)
            raise RuntimeError("synthetic reply lost after provider commit")

        if task.status == "pending":
            try:
                await run_task(
                    journal,
                    spec.key,
                    handlers={spec.action: lost_reply},
                    timeout_seconds=2,
                    authorize=policy.check,
                )
            except RuntimeError:
                task = journal.get(spec.key) or task
        if task.status == "outcome_unknown" and task.attempt_id is not None:
            recovered = await reconcile_task(
                journal,
                spec.key,
                expected_attempt_id=task.attempt_id,
                access=ReceiptAccess(provider, policy.check),
                timeout_seconds=2,
            )
            task = recovered.task
        print(json.dumps({"task": asdict(task), "synthetic_effects": provider.count_effects()}))
    finally:
        provider.close()
        journal.close()


if __name__ == "__main__":
    asyncio.run(main())
