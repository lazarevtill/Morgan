"""
Background Processing Service - Thin Facade
Delegates to BackgroundOrchestrator for DDD compliance.
"""

from typing import Any, Dict, List, Optional
from morgan.background.application.orchestrators import BackgroundOrchestrator


class BackgroundProcessingService:
    """
    Facade for background processing.
    Maintains legacy interface while delegating to the new DDD-compliant orchestrator.
    """

    def __init__(
        self,
        vector_db_client=None,
        reranking_service=None,
        check_interval_seconds: int = 300,
        enable_auto_scheduling: bool = True,
    ):
        self.orchestrator = BackgroundOrchestrator(
            vector_db_client=vector_db_client,
            reranking_service=reranking_service,
            check_interval_seconds=check_interval_seconds,
            enable_auto_scheduling=enable_auto_scheduling,
        )

    def start(self) -> bool:
        return self.orchestrator.start()

    def stop(self) -> bool:
        return self.orchestrator.stop()

    def schedule_reindexing(
        self, collection_name: str, frequency: str = "weekly"
    ) -> str:
        return self.orchestrator.scheduler.schedule_task(
            task_type="reindex", collection_name=collection_name, frequency=frequency
        )

    def schedule_reranking(self, collection_name: str, frequency: str = "daily") -> str:
        return self.orchestrator.scheduler.schedule_task(
            task_type="rerank", collection_name=collection_name, frequency=frequency
        )

    def warm_cache_for_collection(self, collection_name: str) -> int:
        return self.orchestrator.cache.warm_cache(collection_name)

    def track_search_query(
        self, query: str, collection_name: str, response_time: float = 0.0
    ) -> str:
        return self.orchestrator.cache.track_query(
            query, collection_name, response_time
        )

    def get_cached_results(self, query: str, collection_name: str) -> Optional[Any]:
        return self.orchestrator.cache.get_cached_results(query, collection_name)

    def get_service_status(self) -> Dict[str, Any]:
        return self.orchestrator.get_status()

    def get_recent_activity(self, limit: int = 20) -> Dict[str, List]:
        # Simplified for now to maintain interface
        return {
            "executions": [
                {
                    "task_id": e.task_id,
                    "task_type": e.task_type,
                    "status": e.status.value,
                    "started_at": e.started_at.isoformat() if e.started_at else None,
                }
                for e in self.orchestrator.executor.get_execution_history(limit)
            ],
            "popular_queries": [
                {
                    "query": q.query_text,
                    "count": q.access_count,
                    "last_accessed": (
                        q.last_accessed.isoformat() if q.last_accessed else None
                    ),
                }
                for q in self.orchestrator.cache.get_popular_queries(limit=limit)
            ],
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
