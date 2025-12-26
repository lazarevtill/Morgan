"""
Background Processing Module for Morgan Core.

Organized using Domain-Driven Design (DDD) principles:
- application/: Service facades and orchestrators
- infrastructure/: Implementation of scheduling, monitoring, and caching
- domain/: (Planned) Core business rules for background tasks
- tasks/: Concrete task implementations
"""

from morgan.background.infrastructure.executor import BackgroundTaskExecutor
from morgan.background.infrastructure.monitor import ResourceMonitor
from morgan.background.infrastructure.precomputed_cache import PrecomputedSearchCache
from morgan.background.infrastructure.scheduler import SimpleTaskScheduler
from morgan.background.application.service import BackgroundProcessingService
from morgan.background.tasks.reindexing import ReindexingTask
from morgan.background.tasks.reranking import RerankingTask
from morgan.background.domain.entities import TaskStatus, TaskType, TaskFrequency

__all__ = [
    "BackgroundTaskExecutor",
    "ResourceMonitor",
    "PrecomputedSearchCache",
    "SimpleTaskScheduler",
    "BackgroundProcessingService",
    "ReindexingTask",
    "RerankingTask",
    "TaskStatus",
    "TaskType",
    "TaskFrequency",
]
