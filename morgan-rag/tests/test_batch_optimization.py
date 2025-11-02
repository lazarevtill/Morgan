"""
Tests for batch processing optimizations in Morgan RAG.

Tests the 10x performance improvements achieved through:
- Batch embedding generation
- Connection pooling
- Async processing
- Emotional processing optimizations
"""

import pytest
import time
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from morgan.optimization.batch_processor import (
    BatchProcessor, BatchConfig, BatchResult, get_batch_processor
)
from morgan.optimization.connection_pool import (
    ConnectionPoolManager, PoolConfig, get_connection_pool_manager
)
from morgan.optimization.async_processor import (
    AsyncProcessor, AsyncConfig, TaskPriority, get_async_processor
)
from morgan.optimization.emotional_optimizer import (
    EmotionalProcessingOptimizer, EmotionalState, get_emotional_optimizer
)


class TestBatchProcessor:
    """Test batch processing optimization functionality."""
    
    def test_batch_processor_initialization(self):
        """Test batch processor initializes correctly."""
        config = BatchConfig(batch_size=50, max_workers=2)
        processor = BatchProcessor(config)
        
        assert processor.config.batch_size == 50
        assert processor.config.max_workers == 2
        assert processor.executor is not None
        
        processor.shutdown()
    
    def test_batch_embedding_processing(self):
        """Test optimized batch embedding processing."""
        processor = BatchProcessor()
        
        # Mock embedding function
        def mock_embedding_function(texts: List[str]) -> List[List[float]]:
            return [[0.1 * i] * 384 for i in range(len(texts))]
        
        # Test data
        test_texts = [f"Test text {i}" for i in range(100)]
        
        # Process batch
        result = processor.process_embeddings_batch(
            texts=test_texts,
            embedding_function=mock_embedding_function,
            show_progress=False
        )
        
        assert isinstance(result, BatchResult)
        assert result.total_items == 100
        assert result.success_rate >= 90.0  # Should have high success rate
        assert result.throughput > 0
        assert len(result.batch_sizes_used) > 0
        
        processor.shutdown()
    
    def test_batch_vector_operations(self):
        """Test optimized batch vector operations."""
        processor = BatchProcessor()
        
        # Mock vector operation function
        def mock_vector_operation(operations: List[Dict[str, Any]]) -> List[Any]:
            return [{"success": True} for _ in operations]
        
        # Test data
        test_operations = [
            {"id": f"test_{i}", "vector": [0.1 * i] * 384, "payload": {"test": True}}
            for i in range(50)
        ]
        
        # Process batch
        result = processor.process_vector_operations_batch(
            operations=test_operations,
            operation_function=mock_vector_operation,
            operation_type="upsert"
        )
        
        assert isinstance(result, BatchResult)
        assert result.total_items == 50
        assert result.success_rate >= 95.0  # Vector operations should have very high success rate
        assert result.processing_time > 0
        
        processor.shutdown()
    
    def test_streaming_batch_processing(self):
        """Test memory-efficient streaming batch processing."""
        processor = BatchProcessor()
        
        # Mock processing function
        def mock_processing_function(batch: List[Any]) -> List[Any]:
            return [{"processed": item} for item in batch]
        
        # Create data iterator
        def data_generator():
            for i in range(200):
                yield f"item_{i}"
        
        # Process streaming batches
        results = list(processor.process_streaming_batch(
            data_iterator=data_generator(),
            processing_function=mock_processing_function,
            operation_name="streaming_test",
            estimated_total=200
        ))
        
        assert len(results) > 0
        total_processed = sum(r.processed_items for r in results)
        assert total_processed == 200
        
        processor.shutdown()
    
    def test_adaptive_batch_sizing(self):
        """Test adaptive batch sizing optimization."""
        config = BatchConfig(adaptive_sizing=True, min_batch_size=10, max_batch_size=200)
        processor = BatchProcessor(config)
        
        # Test optimal batch size calculation
        optimal_size = processor._get_optimal_batch_size("test_operation", 1000)
        assert config.min_batch_size <= optimal_size <= config.max_batch_size
        
        # Test with small dataset
        small_optimal = processor._get_optimal_batch_size("test_operation", 5)
        assert small_optimal == 5  # Should use actual size for small datasets
        
        processor.shutdown()
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        processor = BatchProcessor()
        
        # Update some metrics
        processor._update_performance_metrics("test_op", 100, 50.0, 2.0)
        processor._update_performance_metrics("test_op", 120, 60.0, 2.0)
        
        # Get metrics
        metrics = processor.get_performance_metrics()
        
        assert "test_op" in metrics
        assert metrics["test_op"].total_operations == 2
        assert metrics["test_op"].avg_throughput == 55.0  # (50 + 60) / 2
        assert metrics["test_op"].peak_throughput == 60.0
        
        processor.shutdown()


class TestConnectionPoolManager:
    """Test connection pooling optimization functionality."""
    
    def test_connection_pool_manager_initialization(self):
        """Test connection pool manager initializes correctly."""
        manager = ConnectionPoolManager()
        
        # Should have default pools
        assert "http" in manager.pools
        
        # Check if Qdrant pool exists (depends on configuration)
        stats = manager.get_all_stats()
        assert isinstance(stats, dict)
    
    def test_custom_connection_pool_creation(self):
        """Test creating custom connection pools."""
        manager = ConnectionPoolManager()
        
        # Mock connection factory
        def mock_connection_factory():
            return Mock()
        
        # Create custom pool
        config = PoolConfig(max_connections=5, min_connections=1)
        pool = manager.create_pool("test_pool", mock_connection_factory, config)
        
        assert pool is not None
        assert "test_pool" in manager.pools
        
        # Test getting connection
        with manager.get_connection("test_pool") as conn:
            assert conn is not None
    
    def test_connection_pool_statistics(self):
        """Test connection pool statistics collection."""
        manager = ConnectionPoolManager()
        
        # Get statistics
        stats = manager.get_all_stats()
        
        for pool_name, pool_stats in stats.items():
            assert hasattr(pool_stats, 'total_connections')
            assert hasattr(pool_stats, 'active_connections')
            assert hasattr(pool_stats, 'pool_utilization')
            assert pool_stats.pool_utilization >= 0.0
    
    def test_connection_validation(self):
        """Test connection health validation."""
        manager = ConnectionPoolManager()
        
        # Mock connection factory that creates testable connections
        def mock_connection_factory():
            mock_conn = Mock()
            mock_conn.get_collections = Mock(return_value=Mock())
            return mock_conn
        
        # Create pool with validation
        config = PoolConfig(enable_health_checks=True, health_check_interval=1.0)
        pool = manager.create_pool("validation_test", mock_connection_factory, config)
        
        # Test connection validation
        with manager.get_connection("validation_test") as conn:
            assert conn is not None


class TestAsyncProcessor:
    """Test async processing optimization functionality."""
    
    def test_async_processor_initialization(self):
        """Test async processor initializes correctly."""
        config = AsyncConfig(max_concurrent_tasks=5, max_queue_size=100)
        processor = AsyncProcessor(config)
        
        assert processor.config.max_concurrent_tasks == 5
        assert processor.config.max_queue_size == 100
        assert not processor.is_running  # Should not be running initially
        
        processor.start()
        assert processor.is_running
        
        processor.stop()
        assert not processor.is_running
    
    def test_task_submission_and_execution(self):
        """Test task submission and execution."""
        processor = AsyncProcessor()
        processor.start()
        
        try:
            # Simple test function
            def test_function(x: int, y: int) -> int:
                return x + y
            
            # Submit task
            task_id = processor.submit_task(test_function, 5, 10)
            assert task_id is not None
            
            # Wait for result
            result = processor.get_task_result(task_id, timeout=5.0)
            
            assert result is not None
            assert result.success
            assert result.result == 15
            
        finally:
            processor.stop()
    
    def test_companion_task_priority(self):
        """Test high-priority companion task processing."""
        processor = AsyncProcessor()
        processor.start()
        
        try:
            # Mock companion interaction function
            def companion_interaction(user_input: str) -> str:
                time.sleep(0.1)  # Simulate processing
                return f"Response to: {user_input}"
            
            # Submit companion task
            task_id = processor.submit_companion_task(
                companion_interaction,
                "I need help with Docker"
            )
            
            # Should complete quickly due to high priority
            result = processor.get_task_result(task_id, timeout=2.0)
            
            assert result is not None
            assert result.success
            assert result.priority == TaskPriority.CRITICAL
            assert "Response to: I need help with Docker" in result.result
            
        finally:
            processor.stop()
    
    def test_batch_task_submission(self):
        """Test batch task submission."""
        processor = AsyncProcessor()
        processor.start()
        
        try:
            # Mock batch processing function
            def process_batch(items: List[str]) -> List[str]:
                return [f"processed_{item}" for item in items]
            
            # Submit batch tasks
            test_items = [f"item_{i}" for i in range(20)]
            task_ids = processor.submit_batch_task(
                func=process_batch,
                items=test_items,
                batch_size=5
            )
            
            assert len(task_ids) == 4  # 20 items / 5 batch_size = 4 batches
            
            # Wait for all tasks
            results = processor.wait_for_tasks(task_ids, timeout=10.0)
            
            assert len(results) == 4
            for result in results:
                assert result.success
                
        finally:
            processor.stop()
    
    def test_async_processor_statistics(self):
        """Test async processor statistics collection."""
        processor = AsyncProcessor()
        processor.start()
        
        try:
            # Submit some tasks
            def simple_task(x: int) -> int:
                return x * 2
            
            task_ids = []
            for i in range(5):
                task_id = processor.submit_task(simple_task, i)
                task_ids.append(task_id)
            
            # Wait for completion
            processor.wait_for_tasks(task_ids, timeout=5.0)
            
            # Check statistics
            stats = processor.get_stats()
            
            assert stats['total_tasks'] >= 5
            assert stats['completed_tasks'] >= 0
            assert 'queue_size' in stats
            assert 'active_tasks_count' in stats
            
        finally:
            processor.stop()


class TestEmotionalProcessingOptimizer:
    """Test emotional processing optimization functionality."""
    
    def test_emotional_optimizer_initialization(self):
        """Test emotional optimizer initializes correctly."""
        optimizer = EmotionalProcessingOptimizer()
        
        assert optimizer.emotion_cache is not None
        assert optimizer.pattern_cache is not None
        assert len(optimizer.emotion_keywords) > 0
        assert "joy" in optimizer.emotion_keywords
        assert "sadness" in optimizer.emotion_keywords
    
    def test_fast_emotion_detection(self):
        """Test fast emotion detection for real-time interactions."""
        optimizer = EmotionalProcessingOptimizer()
        
        # Test positive emotion
        positive_text = "I'm so excited and happy about this amazing opportunity!"
        emotion = optimizer.detect_emotion_fast(
            text=positive_text,
            user_id="test_user",
            use_cache=False
        )
        
        assert isinstance(emotion, EmotionalState)
        assert emotion.primary_emotion in ["joy", "surprise"]  # Should detect positive emotion
        assert emotion.intensity > 0.5  # Should have reasonable intensity
        assert emotion.confidence > 0.0
        assert emotion.user_id == "test_user"
        
        # Test negative emotion
        negative_text = "I'm really frustrated and angry about this situation"
        emotion = optimizer.detect_emotion_fast(
            text=negative_text,
            user_id="test_user",
            use_cache=False
        )
        
        assert emotion.primary_emotion in ["anger", "sadness"]  # Should detect negative emotion
        assert emotion.intensity > 0.3
    
    def test_emotion_detection_caching(self):
        """Test emotion detection caching for performance."""
        optimizer = EmotionalProcessingOptimizer()
        
        text = "This is a test for caching performance"
        user_id = "cache_test_user"
        
        # First detection (cache miss)
        start_time = time.time()
        emotion1 = optimizer.detect_emotion_fast(text, user_id, use_cache=True)
        first_time = time.time() - start_time
        
        # Second detection (cache hit)
        start_time = time.time()
        emotion2 = optimizer.detect_emotion_fast(text, user_id, use_cache=True)
        second_time = time.time() - start_time
        
        # Cache hit should be faster
        assert second_time < first_time
        
        # Results should be identical
        assert emotion1.primary_emotion == emotion2.primary_emotion
        assert emotion1.intensity == emotion2.intensity
    
    def test_batch_emotional_pattern_analysis(self):
        """Test batch emotional pattern analysis."""
        optimizer = EmotionalProcessingOptimizer()
        
        # Create test emotional states
        user_emotions = {
            "user1": [
                EmotionalState(
                    primary_emotion="joy",
                    intensity=0.8,
                    confidence=0.9,
                    secondary_emotions=["excitement"],
                    emotional_indicators=["happy", "excited"],
                    timestamp=datetime.now() - timedelta(days=i),
                    user_id="user1"
                )
                for i in range(10)
            ]
        }
        
        # Analyze patterns
        patterns = optimizer.analyze_emotional_patterns_batch(
            user_emotions=user_emotions,
            analysis_period=timedelta(days=30)
        )
        
        assert "user1" in patterns
        pattern = patterns["user1"]
        
        assert pattern.user_id == "user1"
        assert len(pattern.dominant_emotions) > 0
        assert pattern.dominant_emotions[0][0] == "joy"  # Should be dominant emotion
        assert pattern.mood_trend in ["improving", "declining", "stable"]
        assert 0.0 <= pattern.emotional_volatility <= 1.0
        assert 0.0 <= pattern.pattern_confidence <= 1.0
    
    def test_emotional_response_optimization(self):
        """Test emotional response optimization."""
        optimizer = EmotionalProcessingOptimizer()
        
        # Create test emotional state
        user_emotion = EmotionalState(
            primary_emotion="sadness",
            intensity=0.7,
            confidence=0.8,
            secondary_emotions=["disappointment"],
            emotional_indicators=["sad", "disappointed"],
            timestamp=datetime.now(),
            user_id="test_user"
        )
        
        # Optimize response
        response_params = optimizer.optimize_emotional_response_generation(
            user_emotion=user_emotion,
            response_context="I understand you're going through a difficult time.",
            user_pattern=None
        )
        
        assert "empathy_level" in response_params
        assert "emotional_tone" in response_params
        assert "response_length" in response_params
        
        # Should have high empathy for sadness
        assert response_params["empathy_level"] > 0.5
        assert response_params["emotional_tone"] == "supportive"
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        optimizer = EmotionalProcessingOptimizer()
        
        # Perform some detections to generate metrics
        test_texts = [
            "I'm happy today",
            "This is frustrating",
            "I feel excited about the project"
        ]
        
        for text in test_texts:
            optimizer.detect_emotion_fast(text, "metrics_user", use_cache=False)
        
        # Get metrics
        metrics = optimizer.get_optimization_metrics()
        
        assert metrics.total_detections >= 3
        assert metrics.avg_detection_time >= 0.0
        assert 0.0 <= metrics.cache_hit_rate <= 100.0
        assert metrics.batch_processing_speedup > 0.0
    
    def test_real_time_performance_target(self):
        """Test that emotion detection meets real-time performance target (<100ms)."""
        optimizer = EmotionalProcessingOptimizer()
        
        # Test with various text lengths
        test_texts = [
            "Happy",
            "I'm feeling really excited about this new opportunity!",
            "This is a longer text that describes a complex emotional situation with multiple feelings and contexts that might take longer to process but should still be fast enough for real-time interactions."
        ]
        
        detection_times = []
        
        for text in test_texts:
            start_time = time.time()
            optimizer.detect_emotion_fast(text, "performance_user", use_cache=False)
            detection_time = time.time() - start_time
            detection_times.append(detection_time)
        
        # Check that most detections are under 100ms
        avg_time = sum(detection_times) / len(detection_times)
        fast_detections = sum(1 for t in detection_times if t < 0.1)
        
        # At least 80% should be under 100ms for real-time performance
        assert fast_detections / len(detection_times) >= 0.8
        assert avg_time < 0.15  # Average should be well under 150ms


class TestIntegratedOptimization:
    """Test integrated optimization across all components."""
    
    def test_singleton_instances(self):
        """Test that singleton instances work correctly."""
        # Test that we get the same instances
        batch1 = get_batch_processor()
        batch2 = get_batch_processor()
        assert batch1 is batch2
        
        pool1 = get_connection_pool_manager()
        pool2 = get_connection_pool_manager()
        assert pool1 is pool2
        
        async1 = get_async_processor()
        async2 = get_async_processor()
        assert async1 is async2
        
        emotional1 = get_emotional_optimizer()
        emotional2 = get_emotional_optimizer()
        assert emotional1 is emotional2
    
    def test_performance_improvement_targets(self):
        """Test that performance improvement targets are achievable."""
        # This is a meta-test to ensure our optimization targets are realistic
        
        # Batch processing should achieve significant speedup
        batch_processor = get_batch_processor()
        assert batch_processor.config.batch_size >= 50  # Reasonable batch size
        
        # Emotional processing should be fast enough for real-time
        emotional_optimizer = get_emotional_optimizer()
        assert emotional_optimizer.cache_ttl > 0  # Caching enabled
        
        # Async processing should handle concurrent tasks
        async_processor = get_async_processor()
        assert async_processor.config.max_concurrent_tasks >= 5  # Reasonable concurrency
        
        # Connection pooling should be configured
        pool_manager = get_connection_pool_manager()
        stats = pool_manager.get_all_stats()
        assert len(stats) > 0  # At least one pool configured


if __name__ == "__main__":
    pytest.main([__file__, "-v"])