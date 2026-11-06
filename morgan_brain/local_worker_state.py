"""Bounded, payload-free worker observations; never execution authority or replay evidence."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from morgan_brain.task_receipts import ReceiptMismatch
from morgan_brain.tasks import TaskJournal, TaskKey

if TYPE_CHECKING:
    from morgan_brain.local_notes import NoteWorkspace


def _now() -> str:
    """Use explicit UTC for human-readable observation timestamps."""
    return datetime.now(UTC).isoformat()


class RunObservation:
    """Keep only the latest invocation, using atomic private files at execution boundaries."""

    def __init__(self, workspace: NoteWorkspace, limits: dict[str, int | float]) -> None:
        """Persist start before opening the journal or dispatching any work."""
        self.workspace = workspace
        self.started_at = _now()
        self.run_id = uuid4().hex
        self.limits = limits
        self.actions = self.recovered = 0
        self.phase = "starting"
        self.save("running", None)

    def save(self, state: str, reason: str | None) -> None:
        """Atomically publish metadata only; callers treat failures as visible storage errors."""
        self.workspace.publish(
            ".run.json",
            json.dumps(
                {
                    "version": "morgan.local-run.v1",
                    "workspace": self.workspace.provider_id,
                    "run_id": self.run_id,
                    "started_at": self.started_at,
                    "updated_at": _now(),
                    "state": state,
                    "phase": self.phase,
                    "reason": reason,
                    "actions": self.actions,
                    "recovered": self.recovered,
                    "limits": self.limits,
                }
            ),
            replace=True,
        )

    def progress(self, phase: str, actions: int, recovered: int) -> None:
        """Record a cooperative boundary, not an independent liveness guarantee."""
        self.phase, self.actions, self.recovered = phase, actions, recovered
        self.save("running", None)

    def finished(self, reason: str, actions: int, recovered: int) -> None:
        """Record a normal bounded exit without exposing the task list or note text."""
        self.actions, self.recovered = actions, recovered
        self.phase = "finished"
        self.save("finished", reason)

    def failed(self, error: BaseException) -> None:
        """Classify failure without storing exception strings, paths, SQL, or payloads."""
        if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt)):
            reason = "cancelled"
        elif isinstance(error, ReceiptMismatch):
            reason = "receipt_mismatch"
        elif isinstance(error, (OSError, sqlite3.Error)):
            reason = "storage_error"
        elif isinstance(error, ValueError):
            reason = "invalid_input"
        else:
            reason = "execution_error"
        self.save("failed", reason)


def _validate_metadata(raw: dict[str, object]) -> None:
    """Allow only bounded fixed-vocabulary metadata, including data read from disk."""
    identity = raw["run_id"]
    if not isinstance(identity, str) or re.fullmatch("[0-9a-f]{32}", identity) is None:
        raise ValueError("invalid observation run ID")
    for field in ("started_at", "updated_at"):
        value = raw[field]
        if not isinstance(value, str) or len(value) > 64:
            raise ValueError("invalid observation timestamp")
        if datetime.fromisoformat(value).utcoffset() is None:
            raise ValueError("observation timestamp requires a timezone")
    if raw["phase"] not in ("starting", "polling", "dispatching", "reconciling", "finished"):
        raise ValueError("invalid observation phase")
    reasons = (
        None,
        "deadline",
        "needs_reconciliation",
        "blocked",
        "action_limit",
        "cycle_limit",
        "stopped",
        "permission_denied",
        "cancelled",
        "receipt_mismatch",
        "storage_error",
        "invalid_input",
        "execution_error",
    )
    if raw["reason"] not in reasons:
        raise ValueError("invalid observation reason")
    for field in ("actions", "recovered"):
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 960:
            raise ValueError("invalid observation count")
    limits = raw["limits"]
    bounds = {
        "max_actions": (1, 16),
        "max_cycles": (1, 60),
        "seconds": (0, 3600),
        "poll_seconds": (0.05, 60),
    }
    if not isinstance(limits, dict) or set(limits) != set(bounds):
        raise ValueError("invalid observation limits")
    for name, (lower, upper) in bounds.items():
        value = limits[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not lower <= value <= upper
        ):
            raise ValueError("invalid observation limit value")
        if name in ("max_actions", "max_cycles") and not isinstance(value, int):
            raise ValueError("invalid observation integer limit")
        if name == "seconds" and value == 0:
            raise ValueError("invalid observation duration")


def read_observation(workspace: NoteWorkspace, *, active: bool) -> dict[str, object] | None:
    """Validate local metadata before display; inactive running records imply interruption.

    Lock ownership and this record are separate snapshots. Never use this diagnosis to fence
    a task, authorize an effect, or claim that a particular process is healthy.
    """
    try:
        raw = json.loads(workspace.read(".run.json", 16384))
    except FileNotFoundError:
        return None
    expected = {
        "version",
        "workspace",
        "run_id",
        "started_at",
        "updated_at",
        "state",
        "phase",
        "reason",
        "actions",
        "recovered",
        "limits",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("invalid worker observation fields")
    if raw["version"] != "morgan.local-run.v1" or raw["workspace"] != workspace.provider_id:
        raise ValueError("worker observation belongs to another workspace or version")
    if raw["state"] not in ("running", "finished", "failed"):
        raise ValueError("invalid worker observation state")
    _validate_metadata(raw)
    # Select only known fields, even though all local files belong to the trusted account.
    return {
        "run_id": raw["run_id"],
        "started_at": raw["started_at"],
        "updated_at": raw["updated_at"],
        "recorded_state": raw["state"],
        "observed_state": "interrupted"
        if not active and raw["state"] == "running"
        else raw["state"],
        "phase": raw["phase"],
        "reason": raw["reason"],
        "actions": raw["actions"],
        "recovered": raw["recovered"],
        "limits": raw["limits"],
    }


def scoped_counts(workspace: NoteWorkspace, *, user_id: str, project: str) -> dict[str, int]:
    """Read counts for one explicit scope; refuse unsafe journal files without opening them."""
    TaskKey(user_id, project, "diagnostics")
    try:
        os.close(workspace.open_file(".tasks.db", os.O_RDONLY))
    except FileNotFoundError:
        return {}
    for suffix in ("-wal", "-shm", "-journal"):
        try:
            os.close(workspace.open_file(f".tasks.db{suffix}", os.O_RDONLY))
        except FileNotFoundError:
            continue
    return TaskJournal.status_counts(workspace.path / ".tasks.db", user_id=user_id, project=project)
