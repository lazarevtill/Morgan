"""CurrentTimeTool — returns the current time via an injected clock.

The clock is injectable so unit tests can pass a deterministic callable
instead of relying on the real wall clock.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from morgan_brain.interfaces.tools import ToolResult


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class CurrentTimeTool:
    """Return the current UTC time as an ISO-8601 string.

    Parameters
    ----------
    clock:
        A zero-argument callable that returns a ``datetime``.  Defaults to
        ``datetime.now(tz=timezone.utc)``.  Inject a deterministic callable
        in tests to avoid flakiness.
    """

    name = "current_time"
    description = "Return the current UTC date and time as an ISO-8601 string."

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock: Callable[[], datetime] = clock if clock is not None else _utcnow

    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "description": "No parameters — returns current UTC time.",
        }

    async def run(self, *, user_id: str, **_: Any) -> ToolResult:
        return ToolResult(ok=True, output=self._clock().isoformat())
