from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskType(Enum):
    REINDEX = "reindex"
    RERANK = "rerank"
    CACHE_WARM = "cache_warm"
    OPTIMIZE = "optimize"
    PRECOMPUTE = "precompute"


class TaskFrequency(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class BackgroundTask:
    """Domain entity representing a background task."""

    task_id: str
    task_type: TaskType
    status: TaskStatus
    collection_name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def start(self):
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def complete(self):
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)

    def fail(self, error: str):
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error_message = error
