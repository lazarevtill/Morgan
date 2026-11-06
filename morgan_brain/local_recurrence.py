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


def repeat_plan(
    request: TaskRequest,
    *,
    start: datetime,
    interval_seconds: int,
    count: int,
    window_seconds: int,
    now: datetime,
) -> tuple[Occurrence, ...]:
    """Expand 2..16 explicit intervals, ending within seven days, with stable UTC identities."""
    if start.utcoffset() is None or now.utcoffset() is None:
        raise ValueError("repeat timestamps require UTC offsets")
    if any(type(value) is not int for value in (interval_seconds, count, window_seconds)):
        raise ValueError("repeat limits require integers")
    if not 2 <= count <= 16 or not 60 <= interval_seconds <= 86400:
        raise ValueError("repeat requires 2..16 occurrences and interval 60..86400 seconds")
    if not 1 <= window_seconds <= interval_seconds:
        raise ValueError("delivery window must be 1..interval seconds")
    start = start.astimezone(UTC)
    end = start + timedelta(seconds=(count - 1) * interval_seconds + window_seconds)
    if end <= now or end > now + timedelta(days=7):
        raise ValueError("repeat must end in the future within seven days")
    result = []
    for index in range(count):
        due = start + timedelta(seconds=index * interval_seconds)
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
                due + timedelta(seconds=window_seconds),
            )
        )
    return tuple(result)
