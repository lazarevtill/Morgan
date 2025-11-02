# Background Processing System

The background processing system provides continuous optimization and maintenance for Morgan RAG following KISS (Keep It Simple, Stupid) principles. Each component has a single responsibility and minimal interface.

## Overview

The system consists of five main components:

1. **SimpleTaskScheduler** - Basic task scheduling (daily/weekly/hourly)
2. **ResourceMonitor** - Simple CPU/memory checking
3. **BackgroundTaskExecutor** - Task execution with quality tracking
4. **PrecomputedSearchCache** - Popular query caching and precomputation
5. **BackgroundProcessingService** - Integrated service orchestration

## Key Features

### Task Scheduling
- Simple frequency options: hourly, daily, weekly
- Automatic next-run calculation
- Task completion tracking
- No complex algorithms - just straightforward scheduling

### Resource Management
- CPU usage monitoring with different thresholds:
  - 30% limit during active hours (9 AM - 6 PM)
  - 70% limit during quiet hours (10 PM - 6 AM)
- Memory usage monitoring (80% limit)
- Simple resource availability checks

### Task Execution
- Reindexing tasks for collection maintenance
- Reranking tasks for query optimization
- Quality tracking with before/after metrics
- Graceful error handling and recovery

### Precomputed Caching
- Popular query identification and tracking
- Background result precomputation
- Cache warming for frequently accessed content
- Quality assessment and cache invalidation

## Usage Examples

### Basic Task Scheduling

```python
from morgan.background import SimpleTaskScheduler

scheduler = SimpleTaskScheduler()

# Schedule daily reindexing
task_id = scheduler.schedule_task("reindex", "morgan_knowledge", "daily")

# Schedule hourly reranking
task_id = scheduler.schedule_task("rerank", "morgan_memories", "hourly")

# Check pending tasks
pending = scheduler.get_pending_tasks()
```

### Resource Monitoring

```python
from morgan.background import ResourceMonitor

monitor = ResourceMonitor()

# Check current resources
status = monitor.check_resources()
print(f"CPU: {status.cpu_usage:.1%}, Memory: {status.memory_usage:.1%}")
print(f"Can run task: {status.can_run_task}")
```

### Task Execution

```python
from morgan.background import BackgroundTaskExecutor

executor = BackgroundTaskExecutor(
    vector_db_client=your_db_client,
    reranking_service=your_reranking_service
)

# Execute task immediately
execution = executor.execute_task_now("reindex", "collection_name", force=True)
print(f"Status: {execution.status.value}")

# Get execution statistics
stats = executor.get_execution_stats()
print(f"Success rate: {stats['success_rate']:.1%}")
```

### Precomputed Caching

```python
from morgan.background import PrecomputedSearchCache

cache = PrecomputedSearchCache()

# Track search queries
query_hash = cache.track_query("how to implement auth", "docs_collection", 0.15)

# Get popular queries
popular = cache.get_popular_queries(min_frequency=3, limit=10)

# Precompute results
result = cache.precompute_query_results("popular query", "collection_name")

# Warm cache
precomputed_count = cache.warm_cache("collection_name", max_queries=20)
```

### Integrated Service

```python
from morgan.background import BackgroundProcessingService

# Create service
service = BackgroundProcessingService(
    vector_db_client=your_db_client,
    reranking_service=your_reranking_service,
    check_interval_seconds=300,  # 5 minutes
    enable_auto_scheduling=True
)

# Schedule tasks
reindex_task = service.schedule_reindexing("morgan_knowledge", "weekly")
rerank_task = service.schedule_reranking("morgan_memories", "daily")

# Track queries for popularity analysis
service.track_search_query("user query", "collection_name", response_time=0.12)

# Start background processing
service.start()

# Get service status
status = service.get_service_status()
print(f"Running: {status['service_running']}")
print(f"CPU: {status['resources']['cpu_usage']:.1%}")

# Stop when done
service.stop()
```

## Architecture Principles

### KISS Design Philosophy

1. **Single Responsibility**: Each component has one clear purpose
   - `SimpleTaskScheduler` only schedules tasks
   - `ResourceMonitor` only checks resources
   - `ReindexingTask` only reindexes collections
   - `RerankingTask` only reranks queries

2. **Simple Interfaces**: Minimal, focused APIs
   - Clear method names and parameters
   - Obvious return types
   - No complex configuration

3. **Composition over Inheritance**: Build complex behavior from simple parts
   - `BackgroundTaskExecutor` composes scheduler + monitor + tasks
   - `BackgroundProcessingService` composes all components
   - No deep inheritance hierarchies

4. **Reliability over Advanced Features**: Focus on what works
   - Simple resource thresholds instead of complex algorithms
   - Straightforward scheduling without advanced patterns
   - Basic quality tracking without over-engineering

### Resource Management Strategy

**CPU Throttling**:
- Maximum 30% CPU usage during active hours (9 AM - 6 PM)
- Maximum 70% CPU usage during quiet hours (10 PM - 6 AM)
- Dynamic adjustment based on user activity

**Memory Management**:
- 80% memory usage limit
- Conservative fallback if resource checking fails
- Simple garbage collection for caches

**Task Scheduling**:
- Daily tasks scheduled for 2 AM (quiet hours)
- Weekly tasks scheduled for Sunday 3 AM
- Hourly tasks scheduled for next hour

## Configuration

### Default Collections

The service processes these collections by default:
- `morgan_knowledge` - Main document knowledge
- `morgan_memories` - Conversation memories
- `morgan_web_content` - Web scraped content
- `morgan_code` - Code repositories

### Default Schedules

- **Reindexing**: Weekly for main collections, daily for active collections
- **Reranking**: Hourly for popular queries, daily for all queries
- **Cache Warming**: Every hour during background processing

### Resource Limits

```python
# CPU limits
active_hours_cpu_limit = 0.30    # 30% during 9 AM - 6 PM
quiet_hours_cpu_limit = 0.70     # 70% during 10 PM - 6 AM

# Memory limit
memory_limit = 0.80              # 80% always

# Cache limits
cache_ttl_hours = 24             # 24 hour cache TTL
max_cached_queries = 1000        # Maximum cached queries
```

## Monitoring and Observability

### Service Status

```python
status = service.get_service_status()
# Returns:
# {
#   "service_running": bool,
#   "last_check": datetime,
#   "resources": {"cpu_usage": float, "memory_usage": float, ...},
#   "execution": {"total_executions": int, "success_rate": float, ...},
#   "cache": {"total_cached_queries": int, "cache_hit_rate": float, ...},
#   "scheduled_tasks": int
# }
```

### Recent Activity

```python
activity = service.get_recent_activity(limit=20)
# Returns:
# {
#   "executions": [{"task_id": str, "status": str, ...}],
#   "quality_trends": [{"improvement": float, ...}],
#   "popular_queries": [{"query": str, "access_count": int, ...}]
# }
```

### Quality Metrics

The system tracks simple quality improvements:
- Before/after scores for reindexing operations
- Query response time improvements from reranking
- Cache hit rates and performance gains
- User satisfaction trends (when available)

## Error Handling

### Graceful Degradation

- **Resource Exhaustion**: Skip tasks until resources available
- **Database Unavailable**: Queue operations for retry
- **Service Failures**: Continue with available components
- **Cache Corruption**: Rebuild cache in background

### Retry Logic

- Exponential backoff for transient failures
- Maximum retry attempts to prevent infinite loops
- Detailed error logging for troubleshooting
- Fallback to basic operations when advanced features fail

## Performance Considerations

### Optimization Strategies

1. **Batch Processing**: Process multiple items together
2. **Connection Pooling**: Reuse database connections
3. **Async Operations**: Non-blocking background tasks
4. **Memory Management**: Efficient cache eviction

### Scalability

- **Horizontal**: Multiple service instances for different collections
- **Vertical**: Resource-aware task scheduling
- **Storage**: Efficient vector database operations
- **Network**: Optimized batch operations

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/test_background_processing.py -v
```

Run the interactive demo:

```bash
python examples/background_processing_demo.py
```

## Integration

The background processing system integrates seamlessly with existing Morgan components:

- **Vector Database**: Uses existing Qdrant client
- **Search System**: Enhances multi-stage search with precomputed results
- **Memory System**: Processes conversation memories automatically
- **Monitoring**: Extends existing monitoring infrastructure

## Future Enhancements

Potential improvements while maintaining KISS principles:

1. **Adaptive Scheduling**: Adjust frequency based on usage patterns
2. **Quality Feedback**: Learn from user interactions
3. **Resource Prediction**: Anticipate resource needs
4. **Cross-Collection Optimization**: Optimize across multiple collections

All enhancements will maintain the core KISS philosophy of simplicity, reliability, and single responsibility.