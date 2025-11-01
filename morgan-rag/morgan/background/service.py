"""
Background Processing Service

Main service that coordinates all background processing components.
Simple orchestration without over-engineering.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from .scheduler import SimpleTaskScheduler
from .monitor import ResourceMonitor
from .executor import BackgroundTaskExecutor
from .precomputed_cache import PrecomputedSearchCache

logger = logging.getLogger(__name__)


class BackgroundProcessingService:
    """
    Main background processing service.
    
    Coordinates:
    - Task scheduling and execution
    - Resource monitoring
    - Precomputed result caching
    - Continuous optimization
    
    Simple design without over-engineering.
    """
    
    def __init__(
        self,
        vector_db_client=None,
        reranking_service=None,
        check_interval_seconds: int = 300,  # 5 minutes
        enable_auto_scheduling: bool = True
    ):
        """
        Initialize background processing service.
        
        Args:
            vector_db_client: Vector database client
            reranking_service: Reranking service
            check_interval_seconds: How often to check for tasks
            enable_auto_scheduling: Whether to auto-schedule tasks
        """
        self.vector_db_client = vector_db_client
        self.reranking_service = reranking_service
        self.check_interval = check_interval_seconds
        self.enable_auto_scheduling = enable_auto_scheduling
        
        # Initialize components
        self.scheduler = SimpleTaskScheduler()
        self.monitor = ResourceMonitor()
        self.executor = BackgroundTaskExecutor(
            scheduler=self.scheduler,
            monitor=self.monitor,
            vector_db_client=vector_db_client,
            reranking_service=reranking_service
        )
        self.cache = PrecomputedSearchCache(
            vector_db_client=vector_db_client,
            reranking_service=reranking_service
        )
        
        # Service state
        self.running = False
        self.worker_thread = None
        self.last_check = None
        
        # Default collections to process
        self.default_collections = [
            "morgan_knowledge",
            "morgan_memories", 
            "morgan_web_content",
            "morgan_code"
        ]
        
        self.logger = logging.getLogger(__name__)
    
    def start(self) -> bool:
        """
        Start the background processing service.
        
        Returns:
            True if started successfully
        """
        if self.running:
            self.logger.warning("Background service is already running")
            return False
        
        try:
            self.logger.info("Starting background processing service...")
            
            # Schedule default tasks if auto-scheduling is enabled
            if self.enable_auto_scheduling:
                self._schedule_default_tasks()
            
            # Start worker thread
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            
            self.logger.info("Background processing service started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start background service: {e}")
            self.running = False
            return False
    
    def stop(self) -> bool:
        """
        Stop the background processing service.
        
        Returns:
            True if stopped successfully
        """
        if not self.running:
            self.logger.warning("Background service is not running")
            return False
        
        try:
            self.logger.info("Stopping background processing service...")
            
            self.running = False
            
            # Wait for worker thread to finish
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=10)
            
            self.logger.info("Background processing service stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping background service: {e}")
            return False
    
    def schedule_reindexing(
        self,
        collection_name: str,
        frequency: str = "weekly"
    ) -> str:
        """
        Schedule regular reindexing for a collection.
        
        Args:
            collection_name: Collection to reindex
            frequency: How often to reindex (daily, weekly)
            
        Returns:
            Task ID
        """
        task_id = self.scheduler.schedule_task(
            task_type="reindex",
            collection_name=collection_name,
            frequency=frequency
        )
        
        self.logger.info(f"Scheduled reindexing for {collection_name} ({frequency})")
        return task_id
    
    def schedule_reranking(
        self,
        collection_name: str,
        frequency: str = "daily"
    ) -> str:
        """
        Schedule regular reranking for a collection.
        
        Args:
            collection_name: Collection to rerank
            frequency: How often to rerank (hourly, daily)
            
        Returns:
            Task ID
        """
        task_id = self.scheduler.schedule_task(
            task_type="rerank",
            collection_name=collection_name,
            frequency=frequency
        )
        
        self.logger.info(f"Scheduled reranking for {collection_name} ({frequency})")
        return task_id
    
    def warm_cache_for_collection(self, collection_name: str) -> int:
        """
        Warm precomputed cache for a collection.
        
        Args:
            collection_name: Collection to warm cache for
            
        Returns:
            Number of queries precomputed
        """
        return self.cache.warm_cache(collection_name)
    
    def track_search_query(
        self,
        query: str,
        collection_name: str,
        response_time: float = 0.0
    ) -> str:
        """
        Track a search query for popularity analysis.
        
        Args:
            query: Query text
            collection_name: Collection searched
            response_time: Query response time
            
        Returns:
            Query hash
        """
        return self.cache.track_query(query, collection_name, response_time)
    
    def get_cached_results(
        self,
        query: str,
        collection_name: str
    ) -> Optional[Any]:
        """
        Get precomputed results for a query.
        
        Args:
            query: Query text
            collection_name: Collection name
            
        Returns:
            Cached results or None
        """
        return self.cache.get_cached_results(query, collection_name)
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get comprehensive service status.
        
        Returns:
            Dictionary with service status information
        """
        # Get component statuses
        resource_status = self.monitor.check_resources()
        execution_stats = self.executor.get_execution_stats()
        cache_stats = self.cache.get_cache_stats()
        scheduled_tasks = self.scheduler.list_tasks()
        
        return {
            "service_running": self.running,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "check_interval_seconds": self.check_interval,
            "resources": {
                "cpu_usage": resource_status.cpu_usage,
                "memory_usage": resource_status.memory_usage,
                "can_run_task": resource_status.can_run_task,
                "active_hours": resource_status.active_hours
            },
            "execution": execution_stats,
            "cache": cache_stats,
            "scheduled_tasks": len(scheduled_tasks),
            "collections": self.default_collections
        }
    
    def get_recent_activity(self, limit: int = 20) -> Dict[str, List]:
        """
        Get recent background activity.
        
        Args:
            limit: Maximum items per category
            
        Returns:
            Dictionary with recent activity
        """
        return {
            "executions": [
                {
                    "task_id": e.task_id,
                    "task_type": e.task_type,
                    "collection": e.collection_name,
                    "status": e.status.value,
                    "started_at": e.started_at.isoformat() if e.started_at else None,
                    "completed_at": e.completed_at.isoformat() if e.completed_at else None
                }
                for e in self.executor.get_execution_history(limit)
            ],
            "quality_trends": [
                {
                    "before_score": m.before_score,
                    "after_score": m.after_score,
                    "improvement": m.improvement,
                    "measurement_time": m.measurement_time.isoformat()
                }
                for m in self.executor.get_quality_trends(limit)
            ],
            "popular_queries": [
                {
                    "query": q.query_text,
                    "access_count": q.access_count,
                    "collections": q.collections,
                    "avg_response_time": q.average_response_time
                }
                for q in self.cache.get_popular_queries(limit=limit)
            ]
        }
    
    def _worker_loop(self):
        """Main worker loop for background processing."""
        self.logger.info("Background worker loop started")
        
        while self.running:
            try:
                self.last_check = datetime.now()
                
                # Check and execute pending tasks
                executions = self.executor.run_pending_tasks()
                
                if executions:
                    self.logger.info(f"Executed {len(executions)} background tasks")
                
                # Perform cache warming for popular queries (once per hour)
                if self._should_warm_cache():
                    self._perform_cache_warming()
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in background worker loop: {e}", exc_info=True)
                time.sleep(60)  # Wait 1 minute before retrying
        
        self.logger.info("Background worker loop stopped")
    
    def _schedule_default_tasks(self):
        """Schedule default tasks for all collections."""
        for collection in self.default_collections:
            try:
                # Schedule weekly reindexing
                self.schedule_reindexing(collection, "weekly")
                
                # Schedule daily reranking
                self.schedule_reranking(collection, "daily")
                
            except Exception as e:
                self.logger.warning(f"Failed to schedule tasks for {collection}: {e}")
    
    def _should_warm_cache(self) -> bool:
        """Check if cache warming should be performed."""
        if not hasattr(self, '_last_cache_warm'):
            self._last_cache_warm = datetime.now() - timedelta(hours=2)  # Force first run
        
        # Warm cache every hour
        return datetime.now() - self._last_cache_warm > timedelta(hours=1)
    
    def _perform_cache_warming(self):
        """Perform cache warming for all collections."""
        try:
            self.logger.info("Starting cache warming...")
            
            total_precomputed = 0
            
            for collection in self.default_collections:
                try:
                    count = self.cache.warm_cache(collection, max_queries=10)
                    total_precomputed += count
                    
                except Exception as e:
                    self.logger.warning(f"Failed to warm cache for {collection}: {e}")
            
            self._last_cache_warm = datetime.now()
            
            self.logger.info(f"Cache warming completed: {total_precomputed} queries precomputed")
            
        except Exception as e:
            self.logger.error(f"Error during cache warming: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()