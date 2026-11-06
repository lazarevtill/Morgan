"""Private, account-authorized Markdown delivery; no model, shell, or network access.

The owning OS account and workspace ancestors are trusted. Mode checks are not a sandbox
against code running as that account. All cooperating runners must use the workspace lock.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from morgan_brain.local_worker_state import RunObservation
from morgan_brain.task_batches import TaskPermissionDenied, TaskPermissions, TaskPermit
from morgan_brain.task_receipts import (
    ReceiptAccess,
    ReceiptMismatch,
    ReceiptRequest,
    TaskReceipt,
    receipt_request,
    reconcile_task,
)
from morgan_brain.tasks import (
    Task,
    TaskConflict,
    TaskControl,
    TaskJournal,
    TaskKey,
    TaskRequest,
    bounded_task_keys,
    run_task,
)

ACTION = "write-markdown-v1"
_MAX_APPROVAL = 300_000


class StopRequested(Exception):
    """The owner has placed the persistent STOP marker."""


def _private(info: os.stat_result, *, directory: bool = False) -> None:
    """Refuse other owners, shared permissions, and unexpected filesystem object types."""
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise PermissionError("workspace objects must be private and owned by this OS account")


class NoteWorkspace:
    """An explicitly selected private directory, pinned by descriptor on POSIX systems."""

    def __init__(self, path: Path) -> None:
        """Create a private workspace or validate an existing one without repairing its modes."""
        path.mkdir(mode=0o700, exist_ok=True)
        self.path = path.absolute()
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(self.fd)
            _private(info, directory=True)
            self.provider_id = f"local-markdown-v1:{info.st_dev}:{info.st_ino}"
        except BaseException:
            os.close(self.fd)
            raise

    def close(self) -> None:
        """Release the pinned directory descriptor."""
        os.close(self.fd)

    def open_file(self, name: str, flags: int) -> int:
        """Open a private regular child without following symlinks or blocking on a FIFO."""
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("workspace children require a single filename")
        fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=self.fd)
        try:
            _private(os.fstat(fd))
        except BaseException:
            os.close(fd)
            raise
        return fd

    def read(self, name: str, limit: int) -> str:
        """Read bounded UTF-8 data from one checked private file."""
        with os.fdopen(self.open_file(name, os.O_RDONLY), "rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("workspace file exceeds its size limit")
        return data.decode("utf-8")

    def publish(self, name: str, text: str, *, replace: bool = False) -> None:
        """Fsync a temporary file and atomically publish; notes never overwrite any entry."""
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("workspace children require a single filename")
        temporary = f".staging-{uuid4().hex}"
        try:
            with os.fdopen(
                self.open_file(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL), "wb"
            ) as f:
                f.write(text.encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())
            if replace:
                os.replace(temporary, name, src_dir_fd=self.fd, dst_dir_fd=self.fd)
            else:
                os.link(temporary, name, src_dir_fd=self.fd, dst_dir_fd=self.fd)
            os.fsync(self.fd)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=self.fd)

    @contextmanager
    def locked(self, *, approval: bool = False, supervisor: bool = False) -> Iterator[None]:
        """Exclude workers/supervisors immediately; serialize short approval writes separately."""
        if approval and supervisor:
            raise ValueError("select one lock role")
        name = (
            ".approval.lock" if approval else ".supervisor.lock" if supervisor else ".worker.lock"
        )
        fd = self.open_file(name, os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if approval else fcntl.LOCK_NB))
            yield
        finally:
            os.close(fd)

    def check_stop(self) -> None:
        """Treat any STOP directory entry, including a broken symlink, as a stop request."""
        try:
            os.stat("STOP", dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise StopRequested("STOP marker is present")

    def stop(self) -> None:
        """Persist a stop request without touching an existing marker."""
        with suppress(FileExistsError):
            self.publish("STOP", "Stop requested by the local account.\n")

    def resume(self) -> None:
        """Explicitly remove STOP only while no cooperating worker holds the lock."""
        with self.locked(supervisor=True), self.locked():
            with suppress(FileNotFoundError):
                os.unlink("STOP", dir_fd=self.fd)
            os.fsync(self.fd)

    def journal(self) -> TaskJournal:
        """Validate the private journal and sidecars before opening SQLite by its trusted path."""
        os.close(self.open_file(".tasks.db", os.O_RDWR | os.O_CREAT))
        for suffix in ("-wal", "-shm", "-journal"):
            with suppress(FileNotFoundError):
                os.close(self.open_file(f".tasks.db{suffix}", os.O_RDONLY))
        return TaskJournal(self.path / ".tasks.db")


def _requests(raw: object) -> tuple[TaskRequest, ...]:
    """Parse only exact local-note requests; reject extra fields and mixed scopes."""
    if not isinstance(raw, list) or not 1 <= len(raw) <= 16:
        raise ValueError("approval requires 1 to 16 notes")
    requests = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"key", "action", "payload"}:
            raise ValueError("invalid note request fields")
        key = item["key"]
        if not isinstance(key, dict) or set(key) != {"user_id", "project", "task_id"}:
            raise ValueError("invalid task key fields")
        request = TaskRequest(TaskKey(**key), item["action"], item["payload"])
        if request.action != ACTION:
            raise ValueError("only write-markdown-v1 is supported")
        requests.append(request)
    bounded_task_keys([r.key for r in requests])
    return tuple(requests)


def _write_approval(
    workspace: NoteWorkspace, requests: Sequence[TaskRequest], expiry: datetime
) -> None:
    """Publish one validated policy atomically while the caller holds the approval lock."""
    checked = _requests([asdict(r) for r in requests])
    document = json.dumps(
        {
            "version": "morgan.local-approval.v1",
            "workspace": workspace.provider_id,
            "expires_at": expiry.isoformat(),
            "requests": [asdict(r) for r in checked],
        },
        ensure_ascii=False,
    )
    if len(document.encode("utf-8")) > _MAX_APPROVAL:
        raise ValueError("approval exceeds size limit")
    workspace.publish(".approval.json", document, replace=True)


def _expiry(ttl_seconds: int) -> datetime:
    """Bound new grants without silently extending previously approved work."""
    if not 1 <= ttl_seconds <= 86400:
        raise ValueError("approval lifetime must be 1 to 86400 seconds")
    return datetime.now(UTC) + timedelta(seconds=ttl_seconds)


def approve(workspace: NoteWorkspace, requests: Sequence[TaskRequest], *, ttl_seconds: int) -> None:
    """Explicitly replace all active grants; never derive this operation from planner output."""
    with workspace.locked(approval=True):
        _write_approval(workspace, requests, _expiry(ttl_seconds))


def enqueue(workspace: NoteWorkspace, request: TaskRequest, *, ttl_seconds: int) -> None:
    """Append an exact owner-approved note, preserving prior grants and their earliest expiry.

    Identical resubmission is a no-op; changed input for an existing ID is refused. Capacity
    remains 16 notes in one scope. Expired/malformed policies require explicit replacement.
    """
    with workspace.locked(approval=True):
        expiry = _expiry(ttl_seconds)
        try:
            requests, previous_expiry = FileApproval(workspace).snapshot()
        except FileNotFoundError:
            requests = ()
        else:
            if previous_expiry <= datetime.now(UTC):
                raise TaskPermissionDenied("replace the expired approval explicitly")
            for previous in requests:
                if previous.key == request.key:
                    if previous != request:
                        raise TaskConflict("queued task ID already has different input")
                    return
            expiry = min(expiry, previous_expiry)
        _write_approval(workspace, [*requests, request], expiry)


def revoke(workspace: NoteWorkspace) -> None:
    """Remove active authority without deleting journal history, notes, or STOP."""
    with workspace.locked(approval=True):
        with suppress(FileNotFoundError):
            os.unlink(".approval.json", dir_fd=workspace.fd)
        os.fsync(workspace.fd)


def local_status(workspace: NoteWorkspace) -> dict[str, str | bool | int | None]:
    """Observe lifecycle controls without reading note contents or opening the task journal.

    This is a momentary snapshot, not a reservation or proof of completed task delivery.
    """
    active = stopped = False
    try:
        with workspace.locked():
            pass
    except BlockingIOError:
        active = True
    try:
        workspace.check_stop()
    except StopRequested:
        stopped = True
    try:
        requests, expiry = FileApproval(workspace).snapshot()
    except FileNotFoundError:
        return {
            "worker_active": active,
            "stop_requested": stopped,
            "approval": "missing",
            "approved_tasks": 0,
            "expires_at": None,
        }
    return {
        "worker_active": active,
        "stop_requested": stopped,
        "approval": "valid" if expiry > datetime.now(UTC) else "expired",
        "approved_tasks": len(requests),
        "expires_at": expiry.isoformat(),
    }


class FileApproval:
    """Reload exact grants from the private account-owned policy before every effect."""

    def __init__(self, workspace: NoteWorkspace) -> None:
        """Bind authorization to one explicitly selected workspace."""
        self.workspace = workspace

    def snapshot(self) -> tuple[tuple[TaskRequest, ...], datetime]:
        """Read and validate a policy without granting authority or requiring a running state."""
        raw = json.loads(self.workspace.read(".approval.json", _MAX_APPROVAL))
        if (
            not isinstance(raw, dict)
            or set(raw) != {"version", "workspace", "expires_at", "requests"}
            or raw["version"] != "morgan.local-approval.v1"
            or raw["workspace"] != self.workspace.provider_id
        ):
            raise TaskPermissionDenied("approval does not belong to this workspace")
        expiry = datetime.fromisoformat(raw["expires_at"])
        if expiry.utcoffset() is None:
            raise TaskPermissionDenied("approval lacks a timezone")
        return _requests(raw["requests"]), expiry

    def load(self) -> tuple[tuple[TaskRequest, ...], TaskPermissions]:
        """Fail closed on STOP, expiry, malformed policy, or another workspace's approval."""
        self.workspace.check_stop()
        try:
            requests, expiry = self.snapshot()
        except FileNotFoundError:
            return (), TaskPermissions()
        if expiry <= datetime.now(UTC):
            raise TaskPermissionDenied("approval has expired")
        permissions = TaskPermissions()
        for request in requests:
            permissions.grant(TaskPermit(request, "local-account-file", expiry))
        return requests, permissions

    def check(self, task: Task) -> None:
        """Re-read the policy so removal or replacement revokes future effects."""
        _, permissions = self.load()
        permissions.check(task)


class LocalNoteAdapter:
    """Immutable Markdown files are the actual effect and the lookup evidence."""

    def __init__(self, workspace: NoteWorkspace) -> None:
        """Bind all effects to the pinned private directory."""
        self.workspace = workspace
        self.provider_id = workspace.provider_id

    def filename(self, request: ReceiptRequest) -> str:
        """Reject foreign namespaces and unsafe provider keys before filesystem access."""
        if request.provider_id != self.provider_id or not re.fullmatch(
            "[0-9a-f]{64}", request.idempotency_key
        ):
            raise ReceiptMismatch("invalid local note identity")
        return f"note-{request.idempotency_key}.md"

    async def lookup(self, request: ReceiptRequest) -> TaskReceipt | None:
        """Read the actual note and bind its contents to the complete expected specification."""
        name = self.filename(request)
        try:
            text = self.workspace.read(name, 16384)
        except FileNotFoundError:
            return None
        candidate = Task(request.key, ACTION, text, "running", request.attempt_id, False, None)
        if receipt_request(candidate, provider_id=self.provider_id) != request:
            raise ReceiptMismatch("existing note does not match the approved input")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return TaskReceipt(request, digest, json.dumps({"file": name, "sha256": digest}))

    async def execute(self, task: Task, control: TaskControl) -> str:
        """Check authority immediately before atomic publication; never overwrite a note."""
        if task.action != ACTION:
            raise ValueError("unsupported local action")
        request = receipt_request(task, provider_id=self.provider_id)
        control.check_ready()
        with suppress(FileExistsError):
            self.workspace.publish(self.filename(request), task.payload)
        receipt = await self.lookup(request)
        if receipt is None:
            raise ReceiptMismatch("published note disappeared")
        return receipt.result


@dataclass(frozen=True)
class RunLimits:
    """Finite invocation limits; filesystem calls and timeouts remain cooperative."""

    max_actions: int = 1
    max_cycles: int = 1
    seconds: float = 10
    poll_seconds: float = 1

    def __post_init__(self) -> None:
        """Reject unbounded, nonfinite, and excessively aggressive polling requests."""
        if not 1 <= self.max_actions <= 16 or not 1 <= self.max_cycles <= 60:
            raise ValueError("limits require 1..16 actions and 1..60 cycles")
        if not 0 < self.seconds <= 3600 or not 0.05 <= self.poll_seconds <= 60:
            raise ValueError("runtime must be <=3600s; polling interval must be 0.05..60s")


@dataclass(frozen=True)
class LocalRun:
    """Counts for this invocation, without exposing note contents."""

    reason: str
    actions: int
    recovered: int


async def _recover_local_attempt(
    journal: TaskJournal, task: Task, access: ReceiptAccess, timeout_seconds: float
) -> bool:
    """Fence under the caller's worker lock and look up once, without repeating the effect."""
    if task.attempt_id is None:
        raise TaskConflict("interrupted local task lacks an attempt ID")
    if task.status == "running":
        journal.mark_unknown(task.key, attempt_id=task.attempt_id)
    result = await reconcile_task(
        journal,
        task.key,
        expected_attempt_id=task.attempt_id,
        access=access,
        timeout_seconds=min(60, timeout_seconds),
    )
    return result.outcome != "unresolved"


async def _work(
    workspace: NoteWorkspace, journal: TaskJournal, limits: RunLimits, observation: RunObservation
) -> LocalRun:
    """Execute bounded current approvals and reconcile interrupted attempts without replay."""
    policy = FileApproval(workspace)
    adapter = LocalNoteAdapter(workspace)
    deadline = asyncio.get_running_loop().time() + limits.seconds
    actions = recovered = 0
    try:
        for cycle in range(limits.max_cycles):
            observation.progress("polling", actions, recovered)
            workspace.check_stop()
            if asyncio.get_running_loop().time() >= deadline:
                return LocalRun("deadline", actions, recovered)
            requests, _ = policy.load()
            tasks = journal.create_many(requests) if requests else ()
            for task in tasks:
                workspace.check_stop()
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return LocalRun("deadline", actions, recovered)
                if task.status in ("running", "outcome_unknown"):
                    observation.progress("reconciling", actions, recovered)
                    if not await _recover_local_attempt(
                        journal, task, ReceiptAccess(adapter, policy.check), remaining
                    ):
                        return LocalRun("needs_reconciliation", actions, recovered)
                    recovered += 1
                    continue
                if task.status == "succeeded":
                    continue
                if task.status != "pending":
                    return LocalRun("blocked", actions, recovered)
                if actions >= limits.max_actions:
                    return LocalRun("action_limit", actions, recovered)
                policy.check(task)
                actions += 1
                observation.progress("dispatching", actions, recovered)
                await run_task(
                    journal,
                    task.key,
                    handlers={ACTION: adapter.execute},
                    timeout_seconds=remaining,
                    authorize=policy.check,
                )
            if cycle + 1 < limits.max_cycles:
                await asyncio.sleep(
                    min(limits.poll_seconds, max(0, deadline - asyncio.get_running_loop().time()))
                )
        return LocalRun("cycle_limit", actions, recovered)
    except StopRequested:
        return LocalRun("stopped", actions, recovered)
    except TaskPermissionDenied:
        return LocalRun("permission_denied", actions, recovered)


async def run_local(workspace: NoteWorkspace, limits: RunLimits = RunLimits()) -> LocalRun:
    """Hold exclusive ownership through execution/recovery, then close the isolated journal."""
    with workspace.locked():
        observation = RunObservation(workspace, asdict(limits))
        try:
            journal = workspace.journal()
            try:
                result = await _work(workspace, journal, limits, observation)
            finally:
                journal.close()
        except BaseException as exc:
            # A diagnostic write failure must not mask the original execution/storage error.
            with suppress(OSError):
                observation.failed(exc)
            raise
        observation.finished(result.reason, result.actions, result.recovered)
        return result
