"""Task lifecycle manager."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from morgan.task_manager.types import TaskState, TaskStatus, TaskType


class TaskManager:
    """Creates, tracks, updates, and deletes tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskState] = {}

    def create_task(
        self,
        task_type: TaskType,
        description: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create a new task and return its task_id."""
        task = TaskState(
            task_type=task_type,
            description=description,
            metadata=metadata or {},
        )
        self._tasks[task.task_id] = task
        return task.task_id

    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Retrieve a task by id, or None if not found."""
        return self._tasks.get(task_id)

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update a task's status (and optionally result/error). Returns True on success."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        task.status = status
        task.updated_at = datetime.now(timezone.utc)
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        return True

    def list_tasks(self, status: Optional[TaskStatus] = None) -> list[TaskState]:
        """List all tasks, optionally filtered by status."""
        if status is None:
            return list(self._tasks.values())
        return [t for t in self._tasks.values() if t.status == status]

    def delete_task(self, task_id: str) -> bool:
        """Delete a task by id. Returns True if it existed."""
        return self._tasks.pop(task_id, None) is not None
