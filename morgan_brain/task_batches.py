"""Bounded explicit worklists with exact, revocable, caller-issued permissions.

This is an in-process trusted-caller policy, not identity authentication. Never construct
permits from model output or remembered instructions. No permission survives a process restart
unless the application explicitly supplies it again from its current authority source.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from morgan_brain.tasks import (
    Task,
    TaskConflict,
    TaskHandler,
    TaskJournal,
    TaskKey,
    TaskRequest,
    bounded_task_keys,
    run_task,
)


class TaskPermissionDenied(PermissionError):
    """No current permission for this exact owner/project/task/action/input."""


@dataclass(frozen=True)
class TaskPermit:
    request: TaskRequest
    approval_id: str
    expires_at: datetime

    def __post_init__(self) -> None:
        if not self.approval_id.strip() or len(self.approval_id) > 128:
            raise ValueError("approval ID requires 1 to 128 nonblank characters")
        if self.expires_at.utcoffset() is None:
            raise ValueError("permit expiration requires a timezone-aware datetime")


def _utc_now() -> datetime:
    return datetime.now(UTC)


class TaskPermissions:
    """Live, exact grants owned by the trusted application; initially deny everything.

    The approval ID is a caller-supplied audit label, not proof of authorization. No wildcard,
    inference, disk persistence or network policy lookup is provided here.
    """

    def __init__(self, *, clock: Callable[[], datetime] = _utc_now) -> None:
        self._clock = clock
        self._permits: dict[TaskKey, TaskPermit] = {}

    def grant(self, permit: TaskPermit) -> None:
        self._permits[permit.request.key] = permit

    def revoke(self, key: TaskKey) -> None:
        self._permits.pop(key, None)

    def check(self, task: Task) -> None:
        permit = self._permits.get(task.key)
        now = self._clock()
        if now.utcoffset() is None:
            raise TaskPermissionDenied("permission clock is not timezone-aware")
        if (
            permit is None
            or permit.expires_at <= now
            or permit.request.action != task.action
            or permit.request.payload != task.payload
        ):
            raise TaskPermissionDenied("no current permission for the exact task specification")


StopReason = Literal[
    "complete",
    "permission_denied",
    "needs_reconciliation",
    "cancelled",
    "missing_task",
    "missing_handler",
    "deadline",
    "state_changed",
]


@dataclass(frozen=True)
class BatchResult:
    observed: tuple[Task, ...]
    stopped_at: TaskKey | None
    reason: StopReason


def _stop(journal: TaskJournal, key: TaskKey, done: list[Task], reason: StopReason) -> BatchResult:
    current = journal.get(key)
    return BatchResult(tuple([*done, current] if current is not None else done), key, reason)


async def run_batch(
    journal: TaskJournal,
    keys: Sequence[TaskKey],
    *,
    permissions: TaskPermissions,
    handlers: Mapping[str, TaskHandler],
    timeout_seconds: float,
) -> BatchResult:
    """Process at most 16 explicit tasks in order, stopping at the first unsafe boundary.

    Completed tasks are read, never rerun. The timeout is one cooperative whole-batch budget.
    Handlers must call control.check_ready() immediately before each external effect.
    Storage failures and caller cancellation propagate; no scheduler or retries are hidden here.
    """
    ordered = bounded_task_keys(keys)
    if not 0 < timeout_seconds <= 3600:
        raise ValueError("batch timeout must be positive and at most 3600 seconds")
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    done: list[Task] = []
    for key in ordered:
        task = journal.get(key)
        if task is None:
            return _stop(journal, key, done, "missing_task")
        if task.status in ("running", "outcome_unknown"):
            return _stop(journal, key, done, "needs_reconciliation")
        if task.status == "cancelled":
            return _stop(journal, key, done, "cancelled")
        if task.status == "succeeded":
            done.append(task)
            continue
        if task.action not in handlers:
            return _stop(journal, key, done, "missing_handler")
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return _stop(journal, key, done, "deadline")
        try:
            async with asyncio.timeout_at(deadline):
                result = await run_task(
                    journal,
                    key,
                    handlers=handlers,
                    timeout_seconds=remaining,
                    authorize=permissions.check,
                )
        except TaskPermissionDenied:
            current = journal.get(key)
            reason: StopReason = (
                "permission_denied"
                if current is not None and current.status == "pending"
                else "needs_reconciliation"
            )
            return _stop(journal, key, done, reason)
        except TaskConflict:
            current = journal.get(key)
            conflict_reason: StopReason = (
                "needs_reconciliation"
                if current is not None and current.status in ("running", "outcome_unknown")
                else "state_changed"
            )
            return _stop(journal, key, done, conflict_reason)
        except sqlite3.Error:
            raise
        except Exception:
            current = journal.get(key)
            if current is not None and current.status == "outcome_unknown":
                return _stop(journal, key, done, "needs_reconciliation")
            raise
        done.append(result)
    return BatchResult(tuple(done), None, "complete")
