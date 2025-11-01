"""
Background Processing Module

Simple background processing system following KISS principles.
Each component has single responsibility and minimal interface.
"""

from .scheduler import SimpleTaskScheduler
from .monitor import ResourceMonitor
from .executor import BackgroundTaskExecutor
from .precomputed_cache import PrecomputedSearchCache
from .service import BackgroundProcessingService
from .tasks.reindexing import ReindexingTask
from .tasks.reranking import RerankingTask

__all__ = [
    'SimpleTaskScheduler',
    'ResourceMonitor',
    'BackgroundTaskExecutor',
    'PrecomputedSearchCache',
    'BackgroundProcessingService',
    'ReindexingTask',
    'RerankingTask'
]