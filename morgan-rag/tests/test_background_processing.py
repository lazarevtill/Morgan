"""
Tests for Background Processing System

Tests the simple background processing components following KISS principles.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from morgan.background import (
    SimpleTaskScheduler,
    ResourceMonitor,
    BackgroundTaskExecutor,
    PrecomputedSearchCache,
    BackgroundProcessingService,
)
from morgan.background.scheduler import TaskFrequency, ScheduledTask
from morgan.background.executor import TaskStatus


class TestSimpleTaskScheduler(unittest.TestCase):
    """Test the simple task scheduler."""

    def setUp(self):
        self.scheduler = SimpleTaskScheduler()

    def test_schedule_task(self):
        """Test basic task scheduling."""
        task_id = self.scheduler.schedule_task("reindex", "test_collection", "daily")

        self.assertIsInstance(task_id, str)
        self.assertIn("reindex_test_collection", task_id)

        # Check task was added
        tasks = self.scheduler.list_tasks()
        self.assertEqual(len(tasks), 1)

        task = tasks[0]
        self.assertEqual(task.task_type, "reindex")
        self.assertEqual(task.collection_name, "test_collection")
        self.assertEqual(task.frequency, TaskFrequency.DAILY)

    def test_get_pending_tasks(self):
        """Test getting pending tasks."""
        # Schedule a task for the past (should be pending)
        task_id = self.scheduler.schedule_task("reindex", "test_collection", "daily")
        task = self.scheduler.scheduled_tasks[task_id]
        task.next_run = datetime.now() - timedelta(hours=1)  # Make it pending

        pending = self.scheduler.get_pending_tasks()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].task_id, task_id)

    def test_mark_task_completed(self):
        """Test marking task as completed."""
        task_id = self.scheduler.schedule_task("reindex", "test_collection", "daily")

        # Mark as completed
        result = self.scheduler.mark_task_completed(task_id)
        self.assertTrue(result)

        # Check task was updated
        task = self.scheduler.scheduled_tasks[task_id]
        self.assertIsNotNone(task.last_run)
        self.assertGreater(task.next_run, datetime.now())

    def test_cancel_task(self):
        """Test cancelling a task."""
        task_id = self.scheduler.schedule_task("reindex", "test_collection", "daily")

        # Cancel task
        result = self.scheduler.cancel_task(task_id)
        self.assertTrue(result)

        # Check task was removed
        self.assertNotIn(task_id, self.scheduler.scheduled_tasks)


class TestResourceMonitor(unittest.TestCase):
    """Test the resource monitor."""

    def setUp(self):
        self.monitor = ResourceMonitor()

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_check_resources(self, mock_memory, mock_cpu):
        """Test resource checking."""
        # Mock system resources
        mock_cpu.return_value = 25.0  # 25% CPU
        mock_memory.return_value = Mock(percent=60.0)  # 60% memory

        status = self.monitor.check_resources()

        self.assertAlmostEqual(status.cpu_usage, 0.25)
        self.assertAlmostEqual(status.memory_usage, 0.60)
        self.assertIsInstance(status.can_run_task, bool)
        self.assertIsInstance(status.active_hours, bool)

    def test_can_run_task(self):
        """Test simple task execution check."""
        result = self.monitor.can_run_task()
        self.assertIsInstance(result, bool)

    def test_get_resource_summary(self):
        """Test resource summary."""
        summary = self.monitor.get_resource_summary()

        self.assertIn("cpu", summary)
        self.assertIn("memory", summary)
        self.assertIsInstance(summary["cpu"], float)
        self.assertIsInstance(summary["memory"], float)


class TestBackgroundTaskExecutor(unittest.TestCase):
    """Test the background task executor."""

    def setUp(self):
        self.scheduler = SimpleTaskScheduler()
        self.monitor = ResourceMonitor()
        self.executor = BackgroundTaskExecutor(
            scheduler=self.scheduler, monitor=self.monitor
        )

    def test_execute_task_now(self):
        """Test immediate task execution."""
        execution = self.executor.execute_task_now(
            "reindex", "test_collection", force=True
        )

        self.assertIsNotNone(execution)
        self.assertEqual(execution.task_type, "reindex")
        self.assertEqual(execution.collection_name, "test_collection")
        self.assertIn(execution.status, [TaskStatus.COMPLETED, TaskStatus.FAILED])

    def test_get_execution_stats(self):
        """Test execution statistics."""
        # Execute a task first
        self.executor.execute_task_now("reindex", "test_collection", force=True)

        stats = self.executor.get_execution_stats()

        self.assertIn("total_executions", stats)
        self.assertIn("success_rate", stats)
        self.assertGreaterEqual(stats["total_executions"], 1)


class TestPrecomputedSearchCache(unittest.TestCase):
    """Test the precomputed search cache."""

    def setUp(self):
        self.cache = PrecomputedSearchCache()

    def test_track_query(self):
        """Test query tracking."""
        query_hash = self.cache.track_query("test query", "test_collection", 0.15)

        self.assertIsInstance(query_hash, str)
        self.assertEqual(len(query_hash), 32)  # MD5 hash length

        # Check analytics were created
        self.assertIn(query_hash, self.cache.query_analytics)
        analytics = self.cache.query_analytics[query_hash]
        self.assertEqual(analytics.query_text, "test query")
        self.assertEqual(analytics.access_count, 1)

    def test_get_popular_queries(self):
        """Test getting popular queries."""
        # Track some queries
        self.cache.track_query("popular query", "test_collection")
        self.cache.track_query("popular query", "test_collection")  # Duplicate
        self.cache.track_query("unpopular query", "test_collection")

        popular = self.cache.get_popular_queries(min_frequency=2, limit=5)

        self.assertEqual(len(popular), 1)
        self.assertEqual(popular[0].query_text, "popular query")
        self.assertEqual(popular[0].access_count, 2)

    def test_precompute_query_results(self):
        """Test precomputing query results."""
        result = self.cache.precompute_query_results("test query", "test_collection")

        self.assertIsNotNone(result)
        self.assertEqual(result.query_text, "test query")
        self.assertEqual(result.collection_name, "test_collection")
        self.assertGreater(len(result.results), 0)
        self.assertGreaterEqual(result.quality_score, 0.0)
        self.assertLessEqual(result.quality_score, 1.0)

    def test_get_cached_results(self):
        """Test retrieving cached results."""
        # Precompute first
        self.cache.precompute_query_results("test query", "test_collection")

        # Retrieve from cache
        cached = self.cache.get_cached_results("test query", "test_collection")

        self.assertIsNotNone(cached)
        self.assertEqual(cached.query_text, "test query")
        self.assertEqual(cached.access_count, 1)  # Should increment access count

    def test_cache_stats(self):
        """Test cache statistics."""
        # Add some data
        self.cache.track_query("query1", "collection1")
        self.cache.precompute_query_results("query1", "collection1")

        stats = self.cache.get_cache_stats()

        self.assertIn("total_cached_queries", stats)
        self.assertIn("valid_entries", stats)
        self.assertIn("cache_hit_rate", stats)
        self.assertGreaterEqual(stats["total_cached_queries"], 1)


class TestBackgroundProcessingService(unittest.TestCase):
    """Test the integrated background processing service."""

    def setUp(self):
        self.service = BackgroundProcessingService(
            check_interval_seconds=1,  # Fast for testing
            enable_auto_scheduling=False,  # Manual control for tests
        )

    def test_schedule_reindexing(self):
        """Test scheduling reindexing."""
        task_id = self.service.schedule_reindexing("test_collection", "daily")

        self.assertIsInstance(task_id, str)

        # Check task was scheduled
        tasks = self.service.scheduler.list_tasks()
        self.assertGreater(len(tasks), 0)

        # Find our task
        our_task = next((t for t in tasks if t.task_id == task_id), None)
        self.assertIsNotNone(our_task)
        self.assertEqual(our_task.task_type, "reindex")

    def test_schedule_reranking(self):
        """Test scheduling reranking."""
        task_id = self.service.schedule_reranking("test_collection", "hourly")

        self.assertIsInstance(task_id, str)

        # Check task was scheduled
        tasks = self.service.scheduler.list_tasks()
        our_task = next((t for t in tasks if t.task_id == task_id), None)
        self.assertIsNotNone(our_task)
        self.assertEqual(our_task.task_type, "rerank")

    def test_track_search_query(self):
        """Test search query tracking."""
        query_hash = self.service.track_search_query(
            "test query", "test_collection", 0.12
        )

        self.assertIsInstance(query_hash, str)

        # Check query was tracked in cache
        analytics = self.service.cache.query_analytics.get(query_hash)
        self.assertIsNotNone(analytics)
        self.assertEqual(analytics.query_text, "test query")

    def test_get_service_status(self):
        """Test service status reporting."""
        status = self.service.get_service_status()

        self.assertIn("service_running", status)
        self.assertIn("resources", status)
        self.assertIn("execution", status)
        self.assertIn("cache", status)
        self.assertIsInstance(status["service_running"], bool)

    def test_start_stop_service(self):
        """Test starting and stopping the service."""
        # Start service
        result = self.service.start()
        self.assertTrue(result)
        self.assertTrue(self.service.running)

        # Stop service
        result = self.service.stop()
        self.assertTrue(result)
        self.assertFalse(self.service.running)


if __name__ == "__main__":
    unittest.main()
