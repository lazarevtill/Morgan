#!/usr/bin/env python3
"""
Background Processing Demo

Demonstrates the simple background processing system following KISS principles.
Shows task scheduling, execution, and precomputed caching.
"""

import sys
import os
import time
import logging
from datetime import datetime

# Add the parent directory to the path so we can import morgan
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morgan.background import (
    SimpleTaskScheduler,
    ResourceMonitor,
    BackgroundTaskExecutor,
    PrecomputedSearchCache,
    BackgroundProcessingService,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_task_scheduler():
    """Demonstrate simple task scheduling."""
    print("\n=== Task Scheduler Demo ===")

    scheduler = SimpleTaskScheduler()

    # Schedule some tasks
    task1 = scheduler.schedule_task("reindex", "morgan_knowledge", "daily")
    task2 = scheduler.schedule_task("rerank", "morgan_memories", "weekly")
    task3 = scheduler.schedule_task("reindex", "morgan_web_content", "hourly")

    print(f"Scheduled tasks:")
    for task in scheduler.list_tasks():
        print(
            f"  - {task.task_id}: {task.task_type} on {task.collection_name} ({task.frequency.value})"
        )
        print(f"    Next run: {task.next_run}")

    # Check for pending tasks
    pending = scheduler.get_pending_tasks()
    print(f"\nPending tasks: {len(pending)}")

    return scheduler


def demo_resource_monitor():
    """Demonstrate resource monitoring."""
    print("\n=== Resource Monitor Demo ===")

    monitor = ResourceMonitor()

    # Check current resources
    status = monitor.check_resources()
    print(f"Current resources:")
    print(f"  - CPU usage: {status.cpu_usage:.1%}")
    print(f"  - Memory usage: {status.memory_usage:.1%}")
    print(f"  - Active hours: {status.active_hours}")
    print(f"  - Can run task: {status.can_run_task}")

    # Get simple summary
    summary = monitor.get_resource_summary()
    print(f"\nResource summary: {summary}")

    return monitor


def demo_task_execution():
    """Demonstrate task execution."""
    print("\n=== Task Execution Demo ===")

    # Create components
    scheduler = SimpleTaskScheduler()
    monitor = ResourceMonitor()
    executor = BackgroundTaskExecutor(scheduler=scheduler, monitor=monitor)

    # Schedule a task
    task_id = scheduler.schedule_task("reindex", "demo_collection", "daily")
    print(f"Scheduled task: {task_id}")

    # Execute immediately (force execution)
    execution = executor.execute_task_now("reindex", "demo_collection", force=True)
    print(f"Execution result:")
    print(f"  - Status: {execution.status.value}")
    print(f"  - Started: {execution.started_at}")
    print(f"  - Completed: {execution.completed_at}")

    if execution.result:
        result = execution.result
        print(f"  - Success: {result.success}")
        print(f"  - Documents processed: {result.documents_processed}")
        print(f"  - Processing time: {result.processing_time_seconds:.2f}s")

    # Get execution stats
    stats = executor.get_execution_stats()
    print(f"\nExecution stats: {stats}")

    return executor


def demo_precomputed_cache():
    """Demonstrate precomputed search cache."""
    print("\n=== Precomputed Cache Demo ===")

    cache = PrecomputedSearchCache()

    # Track some queries
    queries = [
        "how to implement authentication",
        "database connection setup",
        "error handling best practices",
        "how to implement authentication",  # Duplicate to show popularity
        "API documentation",
    ]

    print("Tracking queries...")
    for query in queries:
        query_hash = cache.track_query(query, "demo_collection", response_time=0.15)
        print(f"  - Tracked: '{query}' (hash: {query_hash[:8]}...)")

    # Get popular queries
    popular = cache.get_popular_queries(min_frequency=1, limit=5)
    print(f"\nPopular queries ({len(popular)}):")
    for analytics in popular:
        print(f"  - '{analytics.query_text}' (accessed {analytics.access_count} times)")

    # Precompute results for popular query
    if popular:
        query = popular[0].query_text
        print(f"\nPrecomputing results for: '{query}'")

        result = cache.precompute_query_results(query, "demo_collection")
        if result:
            print(f"  - Precomputed {len(result.results)} results")
            print(f"  - Quality score: {result.quality_score:.3f}")
            print(f"  - Computed at: {result.computed_at}")

        # Try to get cached results
        cached = cache.get_cached_results(query, "demo_collection")
        if cached:
            print(f"  - Retrieved from cache (accessed {cached.access_count} times)")

    # Warm cache
    print(f"\nWarming cache...")
    precomputed_count = cache.warm_cache("demo_collection", max_queries=3)
    print(f"  - Precomputed {precomputed_count} queries")

    # Get cache stats
    stats = cache.get_cache_stats()
    print(f"\nCache stats: {stats}")

    return cache


def demo_background_service():
    """Demonstrate the complete background service."""
    print("\n=== Background Service Demo ===")

    # Create service (don't auto-start for demo)
    service = BackgroundProcessingService(
        check_interval_seconds=10,  # Check every 10 seconds for demo
        enable_auto_scheduling=False,  # Manual scheduling for demo
    )

    # Schedule some tasks manually
    print("Scheduling tasks...")
    task1 = service.schedule_reindexing("demo_collection", "daily")
    task2 = service.schedule_reranking("demo_collection", "hourly")
    print(f"  - Scheduled reindexing: {task1}")
    print(f"  - Scheduled reranking: {task2}")

    # Track some queries
    print("\nTracking search queries...")
    queries = [
        "demo query 1",
        "demo query 2",
        "demo query 1",
    ]  # Duplicate for popularity
    for query in queries:
        hash_id = service.track_search_query(query, "demo_collection", 0.12)
        print(f"  - Tracked: '{query}'")

    # Get service status
    status = service.get_service_status()
    print(f"\nService status:")
    print(f"  - Running: {status['service_running']}")
    print(f"  - Scheduled tasks: {status['scheduled_tasks']}")
    print(f"  - CPU usage: {status['resources']['cpu_usage']:.1%}")
    print(f"  - Memory usage: {status['resources']['memory_usage']:.1%}")
    print(f"  - Can run task: {status['resources']['can_run_task']}")

    # Start service briefly to show it works
    print(f"\nStarting service for 5 seconds...")
    service.start()
    time.sleep(5)
    service.stop()
    print("Service stopped")

    # Get recent activity
    activity = service.get_recent_activity(limit=5)
    print(f"\nRecent activity:")
    print(f"  - Executions: {len(activity['executions'])}")
    print(f"  - Quality trends: {len(activity['quality_trends'])}")
    print(f"  - Popular queries: {len(activity['popular_queries'])}")

    return service


def main():
    """Run all demos."""
    print("Background Processing System Demo")
    print("=" * 50)

    try:
        # Run individual component demos
        scheduler = demo_task_scheduler()
        monitor = demo_resource_monitor()
        executor = demo_task_execution()
        cache = demo_precomputed_cache()

        # Run integrated service demo
        service = demo_background_service()

        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nKey features demonstrated:")
        print("✓ Simple task scheduling (daily/weekly/hourly)")
        print("✓ Resource monitoring (CPU/memory)")
        print("✓ Background task execution with quality tracking")
        print("✓ Popular query identification and caching")
        print("✓ Precomputed search results")
        print("✓ Integrated background service")
        print("\nAll components follow KISS principles:")
        print("- Single responsibility per component")
        print("- Simple, focused interfaces")
        print("- Reliability over advanced features")
        print("- Straightforward resource management")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
