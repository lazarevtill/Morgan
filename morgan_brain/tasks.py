"""Opt-in, single-attempt task execution; no scheduler or implicit model/tool access.

The caller supplies an isolated journal path and trusted handlers. Stored text is data,
never code or authorization. Started tasks are never automatically retried.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from morgan_brain.memory.store.transactions import write_transaction

_APPLICATION_ID = 0x4D54534B  # MTSK, distinct from the memory database.
_VERSION = 1
_MAX_TEXT_BYTES = 16_384
TaskStatus = Literal["pending", "running", "succeeded", "cancelled", "outcome_unknown"]


class TaskConflict(ValueError):
    """The identity/specification or expected execution state no longer matches."""


class TaskCancellationRequested(Exception):
    """Cooperative cancellation; does not prove that an external effect was prevented."""


def _identity(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError("task identities require 1 to 128 nonblank characters")


def _text(value: str) -> None:
    if not isinstance(value, str) or len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise ValueError("task payload/result must be text of at most 16384 UTF-8 bytes")


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class TaskKey:
    user_id: str
    project: str
    task_id: str

    def __post_init__(self) -> None:
        for value in (self.user_id, self.project, self.task_id):
            _identity(value)

    def sql(self) -> tuple[str, str, str]:
        return self.user_id, self.project, self.task_id


@dataclass(frozen=True)
class Task:
    key: TaskKey
    action: str
    payload: str
    status: TaskStatus
    attempt_id: str | None
    cancel_requested: bool
    result: str | None


@dataclass(frozen=True)
class TaskRequest:
    key: TaskKey
    action: str
    payload: str

    def __post_init__(self) -> None:
        _identity(self.action)
        _text(self.payload)


def bounded_task_keys(keys: Sequence[TaskKey]) -> tuple[TaskKey, ...]:
    """One explicit, ordered scope; no unbounded scan or duplicate invocation requests."""
    frozen = tuple(keys)
    if not 1 <= len(frozen) <= 16 or len(set(frozen)) != len(frozen):
        raise ValueError("a task batch requires 1 to 16 distinct task keys")
    if len({(key.user_id, key.project) for key in frozen}) != 1:
        raise ValueError("a task batch must belong to one owner and project")
    return frozen


class TaskJournal:
    """An explicitly located execution ledger, separate from inferred memory.

    Owns one connection. Do not share it across threads. Separate connections/processes
    coordinate through SQLite. Never pass the Morgan memory database here.
    """

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path) or str(path) == ":memory:":
            raise ValueError("task journal requires an explicit on-disk Path")
        self._conn = sqlite3.connect(path, timeout=5)
        self._conn.row_factory = sqlite3.Row
        try:
            self._validate_format()
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            with write_transaction(self._conn):
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS tasks ("
                    "user_id TEXT NOT NULL, project TEXT NOT NULL, task_id TEXT NOT NULL, "
                    "action TEXT NOT NULL, payload TEXT NOT NULL, "
                    "status TEXT NOT NULL CHECK(status IN "
                    "('pending','running','succeeded','cancelled','outcome_unknown')), "
                    "attempt_id TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0, result TEXT, "
                    "created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, "
                    "PRIMARY KEY(user_id, project, task_id))"
                )
                self._conn.execute("PRAGMA application_id=0x4D54534B")
                self._conn.execute("PRAGMA user_version=1")
        except BaseException:
            self._conn.close()
            raise

    def _validate_format(self) -> None:
        application = self._conn.execute("PRAGMA application_id").fetchone()[0]
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        tables = self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if (application, version) != (_APPLICATION_ID, _VERSION) and (
            application != 0 or version != 0 or tables
        ):
            raise ValueError("not a supported task journal; refusing to alter this database")

    def close(self) -> None:
        self._conn.close()

    def get(self, key: TaskKey) -> Task | None:
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE user_id=? AND project=? AND task_id=?", key.sql()
        ).fetchone()
        if row is None:
            return None
        return Task(
            key,
            row["action"],
            row["payload"],
            cast(TaskStatus, row["status"]),
            row["attempt_id"],
            bool(row["cancel_requested"]),
            row["result"],
        )

    def _require(self, key: TaskKey) -> Task:
        task = self.get(key)
        if task is None:
            raise TaskConflict("task does not exist in this owner/project scope")
        return task

    def create(self, key: TaskKey, *, action: str, payload: str) -> Task:
        """Exact create replay is safe; a reused ID cannot change an action or its input."""
        _identity(action)
        _text(payload)
        with write_transaction(self._conn):
            old = self.get(key)
            if old is not None:
                if (old.action, old.payload) != (action, payload):
                    raise TaskConflict("task ID already belongs to a different specification")
                return old
            self._conn.execute(
                "INSERT INTO tasks(user_id,project,task_id,action,payload,status,created_at) "
                "VALUES (?,?,?,?,?,'pending',?)",
                (*key.sql(), action, payload, _now()),
            )
            return self._require(key)

    def claim(self, key: TaskKey, *, action: str) -> Task:
        """Commit ownership before invoking a handler; never reclaim a started task."""
        with write_transaction(self._conn):
            task = self._require(key)
            if task.status != "pending" or task.action != action:
                raise TaskConflict("only a pending task with the expected action can be claimed")
            self._conn.execute(
                "UPDATE tasks SET status='running',attempt_id=?,started_at=? "
                "WHERE user_id=? AND project=? AND task_id=?",
                (uuid4().hex, _now(), *key.sql()),
            )
            return self._require(key)

    def create_many(self, requests: Sequence[TaskRequest]) -> tuple[Task, ...]:
        """Atomically submit a bounded set; a conflicting existing ID rolls back all new rows."""
        frozen = tuple(requests)
        bounded_task_keys([request.key for request in frozen])
        with write_transaction(self._conn):
            return tuple(
                self.create(request.key, action=request.action, payload=request.payload)
                for request in frozen
            )

    def cancel(self, key: TaskKey) -> Task:
        """Pending cancellation prevents execution. Running cancellation is only a request."""
        with write_transaction(self._conn):
            task = self._require(key)
            if task.status in ("pending", "running"):
                self._conn.execute(
                    "UPDATE tasks SET cancel_requested=1, status=?, finished_at=? "
                    "WHERE user_id=? AND project=? AND task_id=?",
                    (
                        "cancelled" if task.status == "pending" else "running",
                        _now() if task.status == "pending" else None,
                        *key.sql(),
                    ),
                )
            return self._require(key)

    def finish(self, key: TaskKey, *, attempt_id: str, result: str) -> Task:
        """Persist a handler's reported success, not independently verified external truth."""
        _text(result)
        return self._end(key, attempt_id=attempt_id, status="succeeded", result=result)

    def mark_unknown(self, key: TaskKey, *, attempt_id: str) -> Task:
        """Fence a specific running attempt; never schedule or authorize another attempt.

        This cannot stop an external operation or a still-running handler. A late completion
        is refused. The caller must reconcile the external outcome separately.
        """
        return self._end(key, attempt_id=attempt_id, status="outcome_unknown", result=None)

    def _end(
        self, key: TaskKey, *, attempt_id: str, status: TaskStatus, result: str | None
    ) -> Task:
        with write_transaction(self._conn):
            task = self._require(key)
            if task.status != "running" or task.attempt_id != attempt_id:
                raise TaskConflict("task is not running with this attempt ID")
            self._conn.execute(
                "UPDATE tasks SET status=?,result=?,finished_at=? "
                "WHERE user_id=? AND project=? AND task_id=?",
                (status, result, _now(), *key.sql()),
            )
            return self._require(key)


@dataclass(frozen=True)
class TaskControl:
    journal: TaskJournal
    task: Task
    authorize: Callable[[Task], None] | None = None

    def check_cancelled(self) -> None:
        current = self.journal.get(self.task.key)
        if (
            current is None
            or current.status != "running"
            or current.attempt_id != self.task.attempt_id
            or current.cancel_requested
        ):
            raise TaskCancellationRequested("task cancelled or attempt no longer active")

    def check_ready(self) -> None:
        """Cooperative cancellation and current permission check, immediately before an effect."""
        self.check_cancelled()
        if self.authorize is not None:
            self.authorize(self.task)


TaskHandler = Callable[[Task, TaskControl], Awaitable[str]]


async def run_task(
    journal: TaskJournal,
    key: TaskKey,
    *,
    handlers: Mapping[str, TaskHandler],
    timeout_seconds: float,
    authorize: Callable[[Task], None] | None = None,
) -> Task:
    """Explicitly run one pending task using a caller-selected, trusted handler.

    No scan, retry, import, shell or model call. Timeout/cancellation is cooperative, not a
    sandbox or hard spending limit. A handler must not suppress cancellation or block the loop.
    """
    if not 0 < timeout_seconds <= 3600:
        raise ValueError("task timeout must be positive and at most 3600 seconds")
    task = journal.get(key)
    if task is None:
        raise TaskConflict("task does not exist in this owner/project scope")
    handler = handlers.get(task.action)
    if handler is None:
        raise ValueError("no explicitly registered handler for this task action")
    if authorize is not None:
        authorize(task)
    claimed = journal.claim(key, action=task.action)
    if claimed.attempt_id is None:
        raise RuntimeError("claimed task is missing its attempt ID")
    try:
        control = TaskControl(journal, claimed, authorize)
        control.check_ready()
        async with asyncio.timeout(timeout_seconds):
            result = await handler(claimed, control)
        return journal.finish(key, attempt_id=claimed.attempt_id, result=result)
    except BaseException:
        # A completed commit whose acknowledgement failed must not be overwritten. A
        # broken database may prevent this write too; 'running' then stays unresolved.
        current = journal.get(key)
        if current is not None and current.status == "running":
            journal.mark_unknown(key, attempt_id=claimed.attempt_id)
        raise
