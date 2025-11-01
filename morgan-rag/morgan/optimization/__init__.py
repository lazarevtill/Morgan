"""
Optimization module for Morgan RAG.

Provides performance optimizations including:
- Batch processing for embeddings and vector operations
- Connection pooling for database operations
- Async processing for scalability
- Emotional processing optimizations for real-time companion interactions
- Comprehensive batch optimization for all components
"""

from .batch_processor import BatchProcessor, get_batch_processor
from .connection_pool import ConnectionPoolManager, get_connection_pool_manager
from .async_processor import AsyncProcessor, get_async_processor
from .emotional_optimizer import EmotionalProcessingOptimizer, get_emotional_optimizer
from .comprehensive_batch_optimizer import ComprehensiveBatchOptimizer, get_comprehensive_batch_optimizer

__all__ = [
    "BatchProcessor",
    "get_batch_processor", 
    "ConnectionPoolManager",
    "get_connection_pool_manager",
    "AsyncProcessor", 
    "get_async_processor",
    "EmotionalProcessingOptimizer",
    "get_emotional_optimizer",
    "ComprehensiveBatchOptimizer",
    "get_comprehensive_batch_optimizer"
]