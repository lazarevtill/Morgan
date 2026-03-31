"""
Scheduling job dataclasses.

CronJob  — represents a cron-scheduled task (exact timing, optionally isolated).
HeartbeatCheck — represents a lightweight periodic health/status check.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CronJob:
    """A cron-scheduled job definition.

    Attributes:
        job_id:   Unique identifier for the job.
        schedule: Cron expression (e.g. ``"0 9 * * *"``).
        prompt:   The prompt / command to execute.
        channel:  Target channel for output.
        model:    LLM model to use (``"default"`` uses the system default).
        isolated: If True, run in an isolated subprocess.
        metadata: Arbitrary extra data attached to the job.
    """

    job_id: str
    schedule: str
    prompt: str
    channel: str
    model: str = "default"
    isolated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CronJob:
        """Deserialise from a plain dict."""
        return cls(
            job_id=data["job_id"],
            schedule=data["schedule"],
            prompt=data["prompt"],
            channel=data["channel"],
            model=data.get("model", "default"),
            isolated=data.get("isolated", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class HeartbeatCheck:
    """A single check executed during a heartbeat.

    Attributes:
        name:     Human-readable identifier.
        fn:       Callable (sync or async) returning a result.
        priority: Higher value = higher priority.
        last_run: Epoch timestamp of the last execution (0 = never).
    """

    name: str
    fn: Callable[..., Any]
    priority: int = 0
    last_run: float = 0.0
