"""
Resource Monitor

Simple CPU/memory checking following KISS principles.
Single responsibility: resource monitoring only.
"""

import logging
from dataclasses import dataclass
from typing import Dict

import psutil

logger = logging.getLogger(__name__)


@dataclass
class ResourceStatus:
    """Simple resource status information."""

    cpu_usage: float  # 0.0 to 1.0
    memory_usage: float  # 0.0 to 1.0
    can_run_task: bool
    active_hours: bool  # True if during active hours (9 AM - 6 PM)


class ResourceMonitor:
    """
    Simple resource monitoring without over-engineering.

    Single responsibility: check CPU and memory usage only.
    Simple thresholds: 30% CPU active hours, 70% quiet hours.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Simple thresholds
        self.active_hours_cpu_limit = 0.30  # 30% during active hours
        self.quiet_hours_cpu_limit = 0.70  # 70% during quiet hours
        self.memory_limit = 0.80  # 80% memory limit always

    def check_resources(self) -> ResourceStatus:
        """
        Check current system resources.

        Returns:
            Current resource status with usage and availability
        """
        try:
            # Get CPU usage (1 second average)
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_usage = cpu_percent / 100.0

            # Get memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0

            # Check if we're in active hours (9 AM - 6 PM)
            from datetime import datetime

            current_hour = datetime.now().hour
            active_hours = 9 <= current_hour < 18

            # Determine if we can run a task
            cpu_limit = (
                self.active_hours_cpu_limit
                if active_hours
                else self.quiet_hours_cpu_limit
            )
            can_run_task = (cpu_usage < cpu_limit) and (
                memory_usage < self.memory_limit
            )

            status = ResourceStatus(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                can_run_task=can_run_task,
                active_hours=active_hours,
            )

            self.logger.debug(
                f"Resources: CPU {cpu_usage:.1%}, Memory {memory_usage:.1%}, "
                f"Active hours: {active_hours}, Can run: {can_run_task}"
            )

            return status

        except Exception as e:
            self.logger.error(f"Error checking resources: {e}")
            # Conservative fallback - don't run tasks if we can't check resources
            return ResourceStatus(
                cpu_usage=1.0, memory_usage=1.0, can_run_task=False, active_hours=True
            )

    def can_run_task(self) -> bool:
        """
        Simple check if a task can run now.

        Returns:
            True if resources are available for task execution
        """
        status = self.check_resources()
        return status.can_run_task

    def get_resource_summary(self) -> Dict[str, float]:
        """
        Get simple resource summary.

        Returns:
            Dictionary with current resource usage
        """
        status = self.check_resources()
        return {"cpu": status.cpu_usage, "memory": status.memory_usage}
