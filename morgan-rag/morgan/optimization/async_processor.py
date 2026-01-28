"""
Async Processing Engine for Morgan RAG.

Provides asynchronous processing capabilities for:
- Non-blocking I/O operations
- Concurrent task execution
- Background processing workflows
- Real-time companion interactions

Key Features:
- Asyncio-based task management
- Configurable concurrency limits
- Task queuing and prioritization
- Error handling and retry logic
- Performance monitoring
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from queue import Empty, PriorityQueue
from typing import Any, Callable, Dict, List, Optional

from morgan.config import get_settings
from morgan.utils.error_handling import ErrorSeverity, VectorizationError
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class TaskPriority(Enum):
    """Task priority levels for async processing."""

    CRITICAL = 1  # Real-time companion interactions
    HIGH = 2  # User-facing operations
    NORMAL = 3  # Background processing
    LOW = 4  # Maintenance tasks


@dataclass
class AsyncConfig:
    """Configuration for async processing."""

    max_concurrent_tasks: int = 10
    max_queue_size: int = 1000
    task_timeout: float = 300.0  # 5 minutes
    enable_task_monitoring: bool = True
    thread_pool_size: int = 4
    heartbeat_interval: float = 30.0


@dataclass
class TaskResult:
    """Result of an async task execution."""

    task_id: str
    success: bool
    result: Any
    error: Optional[str]
    execution_time: float
    priority: TaskPriority
    retry_count: int


@dataclass
class AsyncTask:
    """Async task wrapper with metadata."""

    task_id: str
    priority: TaskPriority
    func: Callable
    args: tuple
    kwargs: dict
    created_at: float
    timeout: Optional[float] = None
    max_retries: int = 3
    retry_count: int = 0

    def __lt__(self, other):
        """Compare tasks by priority for queue ordering."""
        return self.priority.value < other.priority.value


class AsyncProcessor:
    """
    High-performance async processor for Morgan RAG operations.

    Provides:
    - Concurrent task execution with priority queuing
    - Background processing for non-blocking operations
    - Real-time processing for companion interactions
    - Resource management and monitoring
    """

    def __init__(self, config: Optional[AsyncConfig] = None):
        """Initialize async processor with configuration."""
        self.settings = get_settings()
        self.config = config or AsyncConfig()

        # Task management
        self.task_queue = PriorityQueue(maxsize=self.config.max_queue_size)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}

        # Async event loop and executor
        self.loop = None
        self.executor = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)

        # Monitoring and statistics
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_execution_time": 0.0,
            "queue_size": 0,
            "active_tasks_count": 0,
        }
        self.stats_lock = threading.Lock()

        # Control flags
        self.is_running = False
        self.shutdown_event = threading.Event()

        # Background threads
        self.processor_thread = None
        self.monitor_thread = None

        logger.info(
            f"AsyncProcessor initialized: max_concurrent={self.config.max_concurrent_tasks}, "
            f"queue_size={self.config.max_queue_size}"
        )

    def start(self):
        """Start the async processor."""
        if self.is_running:
            logger.warning("AsyncProcessor is already running")
            return

        self.is_running = True
        self.shutdown_event.clear()

        # Start processor thread
        self.processor_thread = threading.Thread(
            target=self._run_processor, name="AsyncProcessor", daemon=True
        )
        self.processor_thread.start()

        # Start monitoring thread
        if self.config.enable_task_monitoring:
            self.monitor_thread = threading.Thread(
                target=self._run_monitor, name="AsyncMonitor", daemon=True
            )
            self.monitor_thread.start()

        logger.info("AsyncProcessor started")

    def stop(self, timeout: float = 30.0):
        """Stop the async processor."""
        if not self.is_running:
            return

        logger.info("Stopping AsyncProcessor...")

        self.is_running = False
        self.shutdown_event.set()

        # Cancel all active tasks
        if self.loop and self.loop.is_running():
            for task_id, task in self.active_tasks.items():
                if not task.done():
                    task.cancel()
                    logger.debug(f"Cancelled task {task_id}")

        # Wait for threads to finish
        if self.processor_thread:
            self.processor_thread.join(timeout=timeout)

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logger.info("AsyncProcessor stopped")

    def submit_task(
        self,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        task_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Submit a task for async execution.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            priority: Task priority level
            timeout: Task timeout (uses default if None)
            max_retries: Maximum retry attempts
            task_id: Optional custom task ID
            **kwargs: Keyword arguments for function

        Returns:
            Task ID for tracking
        """
        if not self.is_running:
            raise VectorizationError(
                "AsyncProcessor is not running",
                operation="submit_task",
                component="async_processor",
                severity=ErrorSeverity.HIGH,
            )

        # Generate task ID if not provided
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}"

        # Create async task
        async_task = AsyncTask(
            task_id=task_id,
            priority=priority,
            func=func,
            args=args,
            kwargs=kwargs,
            created_at=time.time(),
            timeout=timeout or self.config.task_timeout,
            max_retries=max_retries,
        )

        try:
            self.task_queue.put(async_task, block=False)

            with self.stats_lock:
                self.stats["total_tasks"] += 1
                self.stats["queue_size"] = self.task_queue.qsize()

            logger.debug(f"Submitted task {task_id} with priority {priority.name}")
            return task_id

        except Exception as e:
            raise VectorizationError(
                f"Failed to submit task: {e}",
                operation="submit_task",
                component="async_processor",
                severity=ErrorSeverity.HIGH,
                metadata={
                    "task_id": task_id,
                    "priority": priority.name,
                    "error_type": type(e).__name__,
                },
            ) from e

    def submit_companion_task(
        self,
        func: Callable,
        *args,
        timeout: float = 5.0,  # Short timeout for real-time interactions
        **kwargs,
    ) -> str:
        """
        Submit a high-priority task for companion interactions.

        Args:
            func: Function to execute
            *args: Positional arguments
            timeout: Task timeout (short for real-time)
            **kwargs: Keyword arguments

        Returns:
            Task ID for tracking
        """
        return self.submit_task(
            func,
            *args,
            priority=TaskPriority.CRITICAL,
            timeout=timeout,
            max_retries=1,  # Don't retry real-time tasks
            **kwargs,
        )

    def submit_batch_task(
        self,
        func: Callable,
        items: List[Any],
        batch_size: int = 100,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs,
    ) -> List[str]:
        """
        Submit a batch of items for processing.

        Args:
            func: Function to process each batch
            items: Items to process
            batch_size: Size of each batch
            priority: Task priority
            **kwargs: Additional arguments for function

        Returns:
            List of task IDs for tracking
        """
        task_ids = []

        # Split items into batches
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            task_id = self.submit_task(func, batch, priority=priority, **kwargs)
            task_ids.append(task_id)

        logger.info(f"Submitted {len(task_ids)} batch tasks for {len(items)} items")
        return task_ids

    def get_task_result(
        self, task_id: str, timeout: Optional[float] = None
    ) -> Optional[TaskResult]:
        """
        Get result of a completed task.

        Args:
            task_id: Task ID to check
            timeout: Timeout for waiting (None = don't wait)

        Returns:
            TaskResult if completed, None if not found or still running
        """
        # Check completed tasks first
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]

        # If timeout specified, wait for completion
        if timeout is not None:
            start_time = time.time()
            while time.time() - start_time < timeout:
                if task_id in self.completed_tasks:
                    return self.completed_tasks[task_id]
                time.sleep(0.1)

        return None

    def wait_for_tasks(
        self, task_ids: List[str], timeout: Optional[float] = None
    ) -> List[TaskResult]:
        """
        Wait for multiple tasks to complete.

        Args:
            task_ids: List of task IDs to wait for
            timeout: Timeout for waiting

        Returns:
            List of TaskResults (may be incomplete if timeout)
        """
        results = []
        start_time = time.time()

        while task_ids and (timeout is None or time.time() - start_time < timeout):
            completed_ids = []

            for task_id in task_ids:
                if task_id in self.completed_tasks:
                    results.append(self.completed_tasks[task_id])
                    completed_ids.append(task_id)

            # Remove completed tasks from waiting list
            for task_id in completed_ids:
                task_ids.remove(task_id)

            if task_ids:
                time.sleep(0.1)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get async processor statistics."""
        with self.stats_lock:
            stats = self.stats.copy()

        # Add current queue and active task counts
        stats["queue_size"] = self.task_queue.qsize()
        stats["active_tasks_count"] = len(self.active_tasks)

        return stats

    def _run_processor(self):
        """Main processor loop (runs in separate thread)."""
        # Create new event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self._process_tasks())
        except Exception as e:
            logger.error(f"Async processor loop failed: {e}")
        finally:
            self.loop.close()

    async def _process_tasks(self):
        """Process tasks from the queue."""
        logger.info("Async task processor started")

        while self.is_running:
            try:
                # Check if we can accept more tasks
                if len(self.active_tasks) >= self.config.max_concurrent_tasks:
                    await asyncio.sleep(0.1)
                    continue

                # Get next task from queue (non-blocking to let event loop run)
                try:
                    async_task = self.task_queue.get(timeout=0)
                except Empty:
                    await asyncio.sleep(0.05)
                    continue

                # Create and start asyncio task
                task = asyncio.create_task(self._execute_task(async_task))
                self.active_tasks[async_task.task_id] = task

                # Yield to let tasks execute
                await asyncio.sleep(0)

                # Clean up completed tasks
                await self._cleanup_completed_tasks()

            except Exception as e:
                logger.error(f"Error in task processor: {e}")
                await asyncio.sleep(1.0)

        # Wait for remaining tasks to complete
        if self.active_tasks:
            logger.info(
                f"Waiting for {len(self.active_tasks)} active tasks to complete..."
            )
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)

        logger.info("Async task processor stopped")

    async def _execute_task(self, async_task: AsyncTask) -> TaskResult:
        """Execute a single async task."""
        start_time = time.time()

        try:
            # Execute function (handle both sync and async functions)
            if asyncio.iscoroutinefunction(async_task.func):
                # Async function
                result = await asyncio.wait_for(
                    async_task.func(*async_task.args, **async_task.kwargs),
                    timeout=async_task.timeout,
                )
            else:
                # Sync function - run in thread pool
                result = await asyncio.wait_for(
                    self.loop.run_in_executor(
                        self.executor,
                        lambda: async_task.func(*async_task.args, **async_task.kwargs),
                    ),
                    timeout=async_task.timeout,
                )

            execution_time = time.time() - start_time

            # Create success result
            task_result = TaskResult(
                task_id=async_task.task_id,
                success=True,
                result=result,
                error=None,
                execution_time=execution_time,
                priority=async_task.priority,
                retry_count=async_task.retry_count,
            )

            # Update statistics
            with self.stats_lock:
                self.stats["completed_tasks"] += 1
                # Update average execution time
                total_completed = self.stats["completed_tasks"]
                current_avg = self.stats["avg_execution_time"]
                self.stats["avg_execution_time"] = (
                    current_avg * (total_completed - 1) + execution_time
                ) / total_completed

            logger.debug(
                f"Task {async_task.task_id} completed successfully in {execution_time:.3f}s"
            )

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Task timeout after {execution_time:.1f}s"

            task_result = TaskResult(
                task_id=async_task.task_id,
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time,
                priority=async_task.priority,
                retry_count=async_task.retry_count,
            )

            logger.warning(
                f"Task {async_task.task_id} timed out after {execution_time:.1f}s"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)

            # Check if we should retry
            if async_task.retry_count < async_task.max_retries:
                async_task.retry_count += 1
                logger.warning(
                    f"Task {async_task.task_id} failed (attempt {async_task.retry_count}), "
                    f"retrying: {error_msg}"
                )

                # Re-queue for retry with exponential backoff
                await asyncio.sleep(2**async_task.retry_count)
                try:
                    self.task_queue.put(async_task, block=False)
                except Exception:
                    logger.error(
                        f"Failed to re-queue task {async_task.task_id} for retry"
                    )

                # Return without storing result (will retry)
                return None

            task_result = TaskResult(
                task_id=async_task.task_id,
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time,
                priority=async_task.priority,
                retry_count=async_task.retry_count,
            )

            # Update failure statistics
            with self.stats_lock:
                self.stats["failed_tasks"] += 1

            logger.error(
                f"Task {async_task.task_id} failed after {async_task.retry_count} attempts: {error_msg}"
            )

        # Store completed task result
        self.completed_tasks[async_task.task_id] = task_result

        # Clean up old completed tasks (keep last 1000)
        if len(self.completed_tasks) > 1000:
            oldest_tasks = sorted(
                self.completed_tasks.items(), key=lambda x: x[1].execution_time
            )[:100]
            for task_id, _ in oldest_tasks:
                del self.completed_tasks[task_id]

        return task_result

    async def _cleanup_completed_tasks(self):
        """Clean up completed asyncio tasks."""
        completed_task_ids = []

        for task_id, task in self.active_tasks.items():
            if task.done():
                completed_task_ids.append(task_id)

        for task_id in completed_task_ids:
            del self.active_tasks[task_id]

    def _run_monitor(self):
        """Background monitoring thread."""
        logger.info("Async processor monitoring started")

        while not self.shutdown_event.wait(self.config.heartbeat_interval):
            try:
                stats = self.get_stats()

                logger.debug(
                    f"AsyncProcessor stats: queue={stats['queue_size']}, "
                    f"active={stats['active_tasks_count']}, "
                    f"completed={stats['completed_tasks']}, "
                    f"failed={stats['failed_tasks']}"
                )

                # Log warnings for high queue size or failure rate
                if stats["queue_size"] > self.config.max_queue_size * 0.8:
                    logger.warning(
                        f"Task queue is {stats['queue_size']}/{self.config.max_queue_size} (80%+ full)"
                    )

                if stats["total_tasks"] > 0:
                    failure_rate = (stats["failed_tasks"] / stats["total_tasks"]) * 100
                    if failure_rate > 10:  # More than 10% failure rate
                        logger.warning(f"High task failure rate: {failure_rate:.1f}%")

            except Exception as e:
                logger.error(f"Monitoring error: {e}")

        logger.info("Async processor monitoring stopped")


# Singleton instance
_async_processor_instance = None
_async_processor_lock = threading.Lock()


def get_async_processor() -> AsyncProcessor:
    """
    Get singleton async processor instance (thread-safe).

    Returns:
        Shared AsyncProcessor instance
    """
    global _async_processor_instance

    if _async_processor_instance is None:
        with _async_processor_lock:
            if _async_processor_instance is None:
                _async_processor_instance = AsyncProcessor()
                # Auto-start the processor
                _async_processor_instance.start()

    return _async_processor_instance
