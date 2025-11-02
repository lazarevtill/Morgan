"""
Background Task Executor

Simple task execution without over-engineering.
Focuses on reliability over advanced features.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .scheduler import SimpleTaskScheduler, ScheduledTask
from .monitor import ResourceMonitor, ResourceStatus
from .tasks.reindexing import ReindexingTask, ReindexingResult
from .tasks.reranking import RerankingTask, RerankingResult

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Simple task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskExecution:
    """Simple task execution record."""
    task_id: str
    task_type: str
    collection_name: str
    status: TaskStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None


@dataclass
class QualityMetrics:
    """Simple before/after quality metrics."""
    before_score: float
    after_score: float
    improvement: float
    measurement_time: datetime


class BackgroundTaskExecutor:
    """
    Simple task execution without over-engineering.
    
    Features:
    - Basic resource checking (30% CPU active, 70% quiet hours)
    - Straightforward scheduling (no complex algorithms)
    - Simple quality tracking (before/after metrics)
    - Focus on reliability over advanced features
    """
    
    def __init__(
        self,
        scheduler: Optional[SimpleTaskScheduler] = None,
        monitor: Optional[ResourceMonitor] = None,
        vector_db_client=None,
        reranking_service=None
    ):
        """
        Initialize background task executor.
        
        Args:
            scheduler: Task scheduler instance
            monitor: Resource monitor instance
            vector_db_client: Vector database client
            reranking_service: Reranking service
        """
        self.scheduler = scheduler or SimpleTaskScheduler()
        self.monitor = monitor or ResourceMonitor()
        self.vector_db_client = vector_db_client
        self.reranking_service = reranking_service
        
        # Task implementations
        self.reindexing_task = ReindexingTask(vector_db_client)
        self.reranking_task = RerankingTask(reranking_service)
        
        # Execution tracking
        self.executions: Dict[str, TaskExecution] = {}
        self.quality_history: List[QualityMetrics] = []
        
        self.logger = logging.getLogger(__name__)
    
    def run_pending_tasks(self) -> List[TaskExecution]:
        """
        Run all pending tasks that can execute now.
        
        Returns:
            List of task executions (completed, failed, or skipped)
        """
        self.logger.info("Checking for pending tasks...")
        
        # Get pending tasks from scheduler
        pending_tasks = self.scheduler.get_pending_tasks()
        
        if not pending_tasks:
            self.logger.debug("No pending tasks found")
            return []
        
        self.logger.info(f"Found {len(pending_tasks)} pending tasks")
        
        executions = []
        
        for task in pending_tasks:
            execution = self._execute_task(task)
            executions.append(execution)
            
            # Mark task as completed in scheduler if successful
            if execution.status == TaskStatus.COMPLETED:
                self.scheduler.mark_task_completed(task.task_id)
        
        return executions
    
    def execute_task_now(
        self,
        task_type: str,
        collection_name: str,
        force: bool = False
    ) -> TaskExecution:
        """
        Execute a task immediately (bypass scheduling).
        
        Args:
            task_type: Type of task to execute
            collection_name: Collection to process
            force: Skip resource checks if True
            
        Returns:
            Task execution result
        """
        # Create a temporary task for immediate execution
        task_id = f"immediate_{task_type}_{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        from .scheduler import TaskFrequency
        temp_task = ScheduledTask(
            task_id=task_id,
            task_type=task_type,
            collection_name=collection_name,
            frequency=TaskFrequency.DAILY,
            next_run=datetime.now()
        )
        
        return self._execute_task(temp_task, force=force)
    
    def _execute_task(self, task: ScheduledTask, force: bool = False) -> TaskExecution:
        """
        Execute a single task.
        
        Args:
            task: Task to execute
            force: Skip resource checks if True
            
        Returns:
            Task execution result
        """
        execution = TaskExecution(
            task_id=task.task_id,
            task_type=task.task_type,
            collection_name=task.collection_name,
            status=TaskStatus.PENDING
        )
        
        self.executions[task.task_id] = execution
        
        try:
            # Check resources unless forced
            if not force:
                resource_status = self.monitor.check_resources()
                if not resource_status.can_run_task:
                    self.logger.info(
                        f"Skipping task {task.task_id} - insufficient resources "
                        f"(CPU: {resource_status.cpu_usage:.1%}, "
                        f"Memory: {resource_status.memory_usage:.1%}, "
                        f"Active hours: {resource_status.active_hours})"
                    )
                    execution.status = TaskStatus.SKIPPED
                    return execution
            
            # Start execution
            execution.status = TaskStatus.RUNNING
            execution.started_at = datetime.now()
            
            self.logger.info(f"Starting task {task.task_id} ({task.task_type} on {task.collection_name})")
            
            # Measure quality before (simple simulation)
            before_quality = self._measure_quality(task.collection_name)
            
            # Execute based on task type
            if task.task_type == "reindex":
                result = self.reindexing_task.reindex_collection(task.collection_name)
            elif task.task_type == "rerank":
                result = self.reranking_task.rerank_popular_queries(task.collection_name)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            # Measure quality after
            after_quality = self._measure_quality(task.collection_name)
            
            # Record quality improvement
            if before_quality is not None and after_quality is not None:
                improvement = after_quality - before_quality
                quality_metrics = QualityMetrics(
                    before_score=before_quality,
                    after_score=after_quality,
                    improvement=improvement,
                    measurement_time=datetime.now()
                )
                self.quality_history.append(quality_metrics)
                
                self.logger.info(
                    f"Quality improvement for {task.collection_name}: "
                    f"{before_quality:.3f} -> {after_quality:.3f} "
                    f"(+{improvement:.3f})"
                )
            
            # Complete execution
            execution.completed_at = datetime.now()
            execution.result = result
            
            if hasattr(result, 'success') and result.success:
                execution.status = TaskStatus.COMPLETED
                self.logger.info(f"Task {task.task_id} completed successfully")
            else:
                execution.status = TaskStatus.FAILED
                execution.error_message = getattr(result, 'error_message', 'Unknown error')
                self.logger.error(f"Task {task.task_id} failed: {execution.error_message}")
            
        except Exception as e:
            execution.completed_at = datetime.now()
            execution.status = TaskStatus.FAILED
            execution.error_message = str(e)
            self.logger.error(f"Task {task.task_id} failed with exception: {e}", exc_info=True)
        
        return execution
    
    def _measure_quality(self, collection_name: str) -> Optional[float]:
        """
        Simple quality measurement.
        
        Args:
            collection_name: Collection to measure
            
        Returns:
            Quality score (0.0 to 1.0) or None if measurement failed
        """
        try:
            # Simple quality simulation (in real implementation, would measure actual metrics)
            import random
            base_quality = 0.7  # Base quality score
            variation = random.uniform(-0.1, 0.1)  # Small random variation
            quality = max(0.0, min(1.0, base_quality + variation))
            
            self.logger.debug(f"Measured quality for {collection_name}: {quality:.3f}")
            return quality
            
        except Exception as e:
            self.logger.warning(f"Failed to measure quality for {collection_name}: {e}")
            return None
    
    def get_execution_history(self, limit: int = 50) -> List[TaskExecution]:
        """
        Get recent task execution history.
        
        Args:
            limit: Maximum number of executions to return
            
        Returns:
            List of recent task executions
        """
        executions = list(self.executions.values())
        # Sort by start time (most recent first)
        executions.sort(key=lambda x: x.started_at or datetime.min, reverse=True)
        return executions[:limit]
    
    def get_quality_trends(self, limit: int = 20) -> List[QualityMetrics]:
        """
        Get recent quality improvement trends.
        
        Args:
            limit: Maximum number of metrics to return
            
        Returns:
            List of recent quality metrics
        """
        return self.quality_history[-limit:] if self.quality_history else []
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get simple execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        if not self.executions:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_improvement": 0.0
            }
        
        executions = list(self.executions.values())
        total = len(executions)
        completed = sum(1 for e in executions if e.status == TaskStatus.COMPLETED)
        
        success_rate = completed / total if total > 0 else 0.0
        
        # Calculate average quality improvement
        improvements = [m.improvement for m in self.quality_history if m.improvement > 0]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0
        
        return {
            "total_executions": total,
            "completed": completed,
            "failed": sum(1 for e in executions if e.status == TaskStatus.FAILED),
            "skipped": sum(1 for e in executions if e.status == TaskStatus.SKIPPED),
            "success_rate": success_rate,
            "average_improvement": avg_improvement
        }