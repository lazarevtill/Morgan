"""Explicit receipt lookup and binding checks, without re-executing an uncertain action.

Only a trusted caller selects the provider adapter and authorizes reconciliation. Matching
labels and hashes detect mix-ups; they are not signatures or independent proof of an effect.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

from morgan_brain.tasks import Task, TaskConflict, TaskJournal, TaskKey


def _json(value: object) -> str:
    """Serialize bindings deterministically without escaping non-ASCII text unnecessarily."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    """Hash a canonical structured value so field boundaries remain unambiguous."""
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


class ReceiptMismatch(ValueError):
    """A returned receipt is not bound to the exact expected provider/task/attempt/input."""


@dataclass(frozen=True)
class ReceiptRequest:
    provider_id: str
    key: TaskKey
    attempt_id: str
    idempotency_key: str
    specification_sha256: str


def receipt_request(task: Task, *, provider_id: str) -> ReceiptRequest:
    """Stable provider key for an attempt; input binding is separate to detect key misuse."""
    if not isinstance(provider_id, str) or not provider_id.strip() or len(provider_id) > 128:
        raise ValueError("receipt provider requires 1 to 128 nonblank characters")
    if task.attempt_id is None:
        raise TaskConflict("receipt lookup requires an existing attempt")
    return ReceiptRequest(
        provider_id,
        task.key,
        task.attempt_id,
        _digest(["morgan.receipt.v1", provider_id, *task.key.sql(), task.attempt_id]),
        _digest([task.action, task.payload]),
    )


@dataclass(frozen=True)
class TaskReceipt:
    request: ReceiptRequest
    receipt_id: str
    result: str

    def __post_init__(self) -> None:
        """Reject empty receipt identities and oversized results before persistence."""
        if (
            not isinstance(self.receipt_id, str)
            or not self.receipt_id.strip()
            or len(self.receipt_id) > 256
        ):
            raise ValueError("receipt ID requires 1 to 256 nonblank characters")
        if not isinstance(self.result, str) or len(self.result.encode("utf-8")) > 16384:
            raise ValueError("receipt result must be text of at most 16384 UTF-8 bytes")


class ReceiptReader(Protocol):
    @property
    def provider_id(self) -> str:
        """Return the stable namespace chosen with this provider adapter."""
        raise NotImplementedError

    async def lookup(self, request: ReceiptRequest) -> TaskReceipt | None:
        """Return affirmative completion evidence, or None for no definitive answer.

        Must not submit, retry or mutate the external action. Exceptions propagate.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class ReceiptAccess:
    """Caller-selected adapter and current authorization check; not proof of authority."""

    reader: ReceiptReader
    authorize: Callable[[Task], None]


@dataclass(frozen=True)
class Reconciliation:
    task: Task
    outcome: Literal["confirmed", "unresolved", "already_confirmed"]


def _check_saved_receipt(task: Task, request: ReceiptRequest) -> None:
    """Reject cached evidence that does not bind this provider, attempt and saved result."""
    if task.receipt_json is None or len(task.receipt_json.encode("utf-8")) > 32768:
        raise ReceiptMismatch("missing or oversized saved receipt")
    try:
        saved = json.loads(task.receipt_json)
    except (ValueError, RecursionError) as exc:
        raise ReceiptMismatch("malformed saved receipt") from exc
    if (
        not isinstance(saved, dict)
        or saved.get("version") != "morgan.receipt.v1"
        or saved.get("basis") != "trusted_adapter_report"
        or saved.get("request") != asdict(request)
        or saved.get("result") != task.result
    ):
        raise ReceiptMismatch("saved receipt does not match this task and provider")
    identity, result = saved.get("receipt_id"), saved.get("result")
    if not isinstance(identity, str) or not isinstance(result, str):
        raise ReceiptMismatch("saved receipt has invalid identity/result fields")
    TaskReceipt(request, identity, result)


async def reconcile_task(
    journal: TaskJournal,
    key: TaskKey,
    *,
    expected_attempt_id: str,
    access: ReceiptAccess,
    timeout_seconds: float,
) -> Reconciliation:
    """Look up once, recheck authority, then atomically persist matched completion evidence.

    A caller must fence an interrupted running attempt as unknown before reconciliation.
    Absence, timeout or a mismatch never makes the action eligible for replay.
    """
    if not 0 < timeout_seconds <= 60:
        raise ValueError("receipt lookup timeout must be positive and at most 60 seconds")
    task = journal.get(key)
    if task is None or task.attempt_id != expected_attempt_id:
        raise TaskConflict("receipt reconciliation requires the exact existing attempt")
    access.authorize(task)
    request = receipt_request(task, provider_id=access.reader.provider_id)
    if task.status == "succeeded" and task.receipt_json is not None:
        _check_saved_receipt(task, request)
        return Reconciliation(task, "already_confirmed")
    if task.status != "outcome_unknown":
        raise TaskConflict("fence the interrupted attempt as unknown before receipt lookup")
    async with asyncio.timeout(timeout_seconds):
        receipt = await access.reader.lookup(request)
    if receipt is None:
        return Reconciliation(task, "unresolved")
    if (
        not isinstance(receipt, TaskReceipt)
        or not isinstance(receipt.request, ReceiptRequest)
        or receipt.request != request
    ):
        raise ReceiptMismatch("receipt does not match the expected provider/task/attempt/input")
    # Validate nested returned data again; a trusted adapter still may return malformed data.
    checked = TaskReceipt(receipt.request, receipt.receipt_id, receipt.result)
    access.authorize(task)
    evidence = _json(
        {"version": "morgan.receipt.v1", "basis": "trusted_adapter_report", **asdict(checked)}
    )
    confirmed = journal.confirm_receipt(
        key,
        attempt_id=expected_attempt_id,
        result=checked.result,
        receipt_json=evidence,
    )
    return Reconciliation(confirmed, "confirmed")
