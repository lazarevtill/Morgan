"""
Optimization module for Morgan RAG.

Provides performance optimizations including:
- Batch processing for embeddings and vector operations
- Connection pooling for database operations
- Async processing for scalability
- Emotional processing optimizations for real-time companion interactions
- Comprehensive batch optimization for all components
- Unified performance service with caching, deduplication, and monitoring
"""

from .async_processor import AsyncProcessor, get_async_processor
from .batch_processor import BatchProcessor, get_batch_processor
from .comprehensive_batch_optimizer import (
    ComprehensiveBatchOptimizer,
    get_comprehensive_batch_optimizer,
)
from .connection_pool import ConnectionPoolManager, get_connection_pool_manager
from .emotional_optimizer import EmotionalProcessingOptimizer, get_emotional_optimizer
from .performance_service import (
    PerformanceOptimizer,
    ResponseCache,
    CacheConfig,
    PerformanceStats,
    get_performance_optimizer,
)

__all__ = [
    # Batch Processing
    "BatchProcessor",
    "get_batch_processor",
    # Connection Pooling
    "ConnectionPoolManager",
    "get_connection_pool_manager",
    # Async Processing
    "AsyncProcessor",
    "get_async_processor",
    # Emotional Optimization
    "EmotionalProcessingOptimizer",
    "get_emotional_optimizer",
    # Comprehensive Batch Optimization
    "ComprehensiveBatchOptimizer",
    "get_comprehensive_batch_optimizer",
    # Performance Service
    "PerformanceOptimizer",
    "ResponseCache",
    "CacheConfig",
    "PerformanceStats",
    "get_performance_optimizer",
]
