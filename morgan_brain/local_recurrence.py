"""Finite, explicit UTC interval plans; no clock loop, model, or persisted cursor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

from morgan_brain.tasks import TaskKey, TaskRequest


@dataclass(frozen=True)
class Occurrence:
    """One exact request and its independent half-open authorization window."""

    request: TaskRequest
    not_before: datetime | None
    expires_at: datetime

    def document(self) -> dict[str, object]:
        """Serialize an explicit timed occurrence for account review and persistence."""
        if self.not_before is None:
            raise ValueError("recurring occurrence requires an earliest time")
        return {
            "request": asdict(self.request),
            "not_before": self.not_before.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }


@dataclass(frozen=True)
class RepeatLimits:
    """Explicit frequency, total occurrence count and per-occurrence delivery allowance."""

    interval_seconds: int
    count: int
    window_seconds: int

    def __post_init__(self) -> None:
        """Reject boolean/noninteger and out-of-bounds limits before expanding a plan."""
        values = (self.interval_seconds, self.count, self.window_seconds)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError("repeat limits require integers")
        if not 2 <= self.count <= 16 or not 60 <= self.interval_seconds <= 86400:
            raise ValueError("repeat requires 2..16 occurrences and interval 60..86400 seconds")
        if not 1 <= self.window_seconds <= self.interval_seconds:
            raise ValueError("delivery window must be 1..interval seconds")


def repeat_plan(
    request: TaskRequest,
    *,
    start: datetime,
    limits: RepeatLimits,
    now: datetime,
) -> tuple[Occurrence, ...]:
    """Expand 2..16 explicit intervals, ending within seven days, with stable UTC identities."""
    if start.utcoffset() is None or now.utcoffset() is None:
        raise ValueError("repeat timestamps require UTC offsets")
    start = start.astimezone(UTC)
    end = start + timedelta(
        seconds=(limits.count - 1) * limits.interval_seconds + limits.window_seconds
    )
    if end <= now or end > now + timedelta(days=7):
        raise ValueError("repeat must end in the future within seven days")
    result = []
    for index in range(limits.count):
        due = start + timedelta(seconds=index * limits.interval_seconds)
        identity = json.dumps([*request.key.sql(), due.isoformat()], ensure_ascii=False)
        key = TaskKey(
            request.key.user_id,
            request.key.project,
            "repeat-" + hashlib.sha256(identity.encode()).hexdigest(),
        )
        result.append(
            Occurrence(
                TaskRequest(key, request.action, request.payload),
                due,
                due + timedelta(seconds=limits.window_seconds),
            )
        )
    return tuple(result)
