"""
Background Orchestration Service for Morgan Core.
Coordinates background tasks following DDD patterns.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from morgan.utils.logger import get_logger
from morgan.background.domain.interfaces import (
    TaskScheduler,
    TaskExecutor,
    ResourceMonitor as IResourceMonitor,
    SearchCache,
)
from morgan.background.infrastructure.scheduler import SimpleTaskScheduler
from morgan.background.infrastructure.executor import BackgroundTaskExecutor
from morgan.background.infrastructure.monitor import ResourceMonitor
from morgan.background.infrastructure.precomputed_cache import PrecomputedSearchCache

logger = get_logger(__name__)


class BackgroundOrchestrator:
    """
    Application service that orchestrates background operations.
    A clean facade that coordinates infrastructure components.
    """

    def __init__(
        self,
        vector_db_client=None,
        reranking_service=None,
        check_interval_seconds: int = 300,
        enable_auto_scheduling: bool = True,
        scheduler: Optional[TaskScheduler] = None,
        monitor: Optional[IResourceMonitor] = None,
        executor: Optional[TaskExecutor] = None,
        cache: Optional[SearchCache] = None,
    ):
        self.check_interval = check_interval_seconds
        self.enable_auto_scheduling = enable_auto_scheduling

        # Initialize components (use provided or defaults)
        self.scheduler = scheduler or SimpleTaskScheduler()
        self.monitor = monitor or ResourceMonitor()

        self.executor = executor or BackgroundTaskExecutor(
            scheduler=self.scheduler,
            monitor=self.monitor,
            vector_db_client=vector_db_client,
            reranking_service=reranking_service,
        )

        self.cache = cache or PrecomputedSearchCache(
            vector_db_client=vector_db_client, reranking_service=reranking_service
        )

        self.default_collections = [
            "morgan_knowledge",
            "morgan_memories",
            "morgan_web_content",
            "morgan_code",
        ]

        # Service state
        self.running = False
        self.worker_thread = None
        self.last_check = None

    def start(self) -> bool:
        """Start the background orchestrator."""
        if self.running:
            logger.warning("Background orchestrator is already running")
            return False

        try:
            logger.info("Starting Background Orchestrator...")
            if self.enable_auto_scheduling:
                self._schedule_default_tasks()

            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start Background Orchestrator: {e}")
            self.running = False
            return False

    def stop(self) -> bool:
        """Stop the background orchestrator."""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10)
        logger.info("Background Orchestrator stopped")
        return True

    def _worker_loop(self):
        """Main worker loop for background tasks."""
        while self.running:
            try:
                self.last_check = datetime.now()
                executions = self.executor.run_pending_tasks()
                if executions:
                    logger.info(f"Executed {len(executions)} background tasks")

                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(
                    f"Error in background orchestrator loop: {e}", exc_info=True
                )
                time.sleep(60)

    def _schedule_default_tasks(self):
        """Schedule routine maintenance tasks."""
        for collection in self.default_collections:
            self.scheduler.schedule_task(
                task_type="reindex", collection_name=collection, frequency="weekly"
            )
            self.scheduler.schedule_task(
                task_type="rerank", collection_name=collection, frequency="daily"
            )

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of background operations."""
        return {
            "running": self.running,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "scheduler": len(self.scheduler.list_tasks()),
            "resources": self.monitor.check_resources().cpu_usage,
            "execution_stats": self.executor.get_execution_stats(),
        }
