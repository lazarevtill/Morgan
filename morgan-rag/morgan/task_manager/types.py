"""Task types and state definitions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class TaskType(Enum):
    """Types of tasks the system can manage."""

    AGENT = "agent"
    SHELL = "shell"
    CRON = "cron"
    DREAM = "dream"


class TaskStatus(Enum):
    """Lifecycle status of a task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskState:
    """Represents the full state of a single task."""

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_type: TaskType = TaskType.AGENT
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    result: Optional[Any] = None
    error: Optional[str] = None
    is_backgrounded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
