"""
Domain interfaces for Background Service.
Enforces dependency inversion principle.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
from .entities import BackgroundTask, TaskFrequency

class TaskScheduler(ABC):
    """Abstract base class for task scheduling."""
    
    @abstractmethod
    def schedule_task(
        self,
        task_type: str,
        collection_name: str,
        frequency: str = "daily",
        metadata: Optional[Dict] = None,
    ) -> str:
        """Schedule a task."""
        pass

    @abstractmethod
    def get_pending_tasks(self) -> List[Any]:
        """Get tasks ready for execution."""
        pass
        
    @abstractmethod
    def mark_task_completed(self, task_id: str) -> bool:
        """Mark task as completed."""
        pass

    @abstractmethod
    def list_tasks(self) -> List[Any]:
        """List all scheduled tasks."""
        pass

class TaskExecutor(ABC):
    """Abstract base class for task execution."""
    
    @abstractmethod
    def run_pending_tasks(self) -> List[BackgroundTask]:
        """Run all pending tasks."""
        pass
        
    @abstractmethod
    def get_execution_history(self, limit: int = 50) -> List[BackgroundTask]:
        """Get history of executed tasks."""
        pass

    @abstractmethod
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        pass

class ResourceMonitor(ABC):
    """Abstract base class for resource monitoring."""
    
    @abstractmethod
    def check_resources(self) -> Any:
        """Check system resources."""
        pass
        
    @abstractmethod
    def is_safe_to_run(self) -> bool:
        """Check if safe to run background tasks."""
        pass

class SearchCache(ABC):
    """Abstract base class for search caching."""
    
    @abstractmethod
    def warm_cache(self, collection_name: str) -> int:
        """Warm up cache for a collection."""
        pass
        
    @abstractmethod
    def track_query(self, query: str, collection_name: str, response_time: float) -> str:
        """Track a search query."""
        pass

    @abstractmethod
    def get_cached_results(self, query: str, collection_name: str) -> Optional[Any]:
        """Get cached results."""
        pass
