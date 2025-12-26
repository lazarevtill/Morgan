"""
Simple Task Scheduler

Basic task scheduling (daily/weekly) following KISS principles.
Single responsibility: task scheduling only.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from morgan.background.domain.entities import TaskFrequency
from morgan.background.domain.interfaces import TaskScheduler

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """Simple scheduled task definition."""

    task_id: str
    task_type: str  # reindex, rerank
    collection_name: str
    frequency: TaskFrequency
    next_run: datetime
    last_run: Optional[datetime] = None
    enabled: bool = True
    metadata: Dict = field(default_factory=dict)


class SimpleTaskScheduler(TaskScheduler):
    """
    Simple task scheduler without over-engineering.

    Single responsibility: basic task scheduling (daily/weekly).
    No complex algorithms - just straightforward scheduling.
    """

    def __init__(self):
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.logger = logging.getLogger(__name__)

    def schedule_task(
        self,
        task_type: str,
        collection_name: str,
        frequency: str = "daily",
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Schedule a simple task.

        Args:
            task_type: Type of task (reindex, rerank)
            collection_name: Collection to process
            frequency: How often to run (daily, weekly, hourly)
            metadata: Optional task metadata

        Returns:
            Task ID for tracking
        """
        task_id = (
            f"{task_type}_{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        try:
            # Handle string frequency conversion
            if isinstance(frequency, str):
                freq = TaskFrequency(frequency.lower())
            else:
                freq = frequency
        except ValueError:
            self.logger.warning(f"Invalid frequency '{frequency}', defaulting to daily")
            freq = TaskFrequency.DAILY

        # Calculate next run time
        next_run = self._calculate_next_run(freq)

        task = ScheduledTask(
            task_id=task_id,
            task_type=task_type,
            collection_name=collection_name,
            frequency=freq,
            next_run=next_run,
            metadata=metadata or {},
        )

        self.scheduled_tasks[task_id] = task

        self.logger.info(
            f"Scheduled {task_type} task for {collection_name}, next run: {next_run}"
        )
        return task_id

    def get_pending_tasks(self) -> List[ScheduledTask]:
        """
        Get tasks that are ready to run.

        Returns:
            List of tasks ready for execution
        """
        now = datetime.now()
        pending = []

        for task in self.scheduled_tasks.values():
            if task.enabled and task.next_run <= now:
                pending.append(task)

        return pending

    def mark_task_completed(self, task_id: str) -> bool:
        """
        Mark a task as completed and schedule next run.

        Args:
            task_id: ID of completed task

        Returns:
            True if task was found and updated
        """
        if task_id not in self.scheduled_tasks:
            self.logger.warning(f"Task {task_id} not found")
            return False

        task = self.scheduled_tasks[task_id]
        task.last_run = datetime.now()
        task.next_run = self._calculate_next_run(task.frequency)

        self.logger.info(f"Task {task_id} completed, next run: {task.next_run}")
        return True

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a scheduled task.

        Args:
            task_id: ID of task to cancel

        Returns:
            True if task was found and cancelled
        """
        if task_id in self.scheduled_tasks:
            del self.scheduled_tasks[task_id]
            self.logger.info(f"Cancelled task {task_id}")
            return True

        self.logger.warning(f"Task {task_id} not found for cancellation")
        return False

    def list_tasks(self) -> List[ScheduledTask]:
        """
        List all scheduled tasks.

        Returns:
            List of all scheduled tasks
        """
        return list(self.scheduled_tasks.values())

    def _calculate_next_run(self, frequency: TaskFrequency) -> datetime:
        """Calculate next run time based on frequency."""
        now = datetime.now()

        if frequency == TaskFrequency.HOURLY:
            return now + timedelta(hours=1)
        elif frequency == TaskFrequency.DAILY:
            # Schedule for next day at 2 AM (quiet hours)
            next_day = now.replace(hour=2, minute=0, second=0, microsecond=0)
            if next_day <= now:
                next_day += timedelta(days=1)
            return next_day
        elif frequency == TaskFrequency.WEEKLY:
            # Schedule for next Sunday at 3 AM
            days_until_sunday = (6 - now.weekday()) % 7
            if days_until_sunday == 0:  # Today is Sunday
                days_until_sunday = 7
            next_sunday = now + timedelta(days=days_until_sunday)
            return next_sunday.replace(hour=3, minute=0, second=0, microsecond=0)

        # Default to daily
        return now + timedelta(days=1)
