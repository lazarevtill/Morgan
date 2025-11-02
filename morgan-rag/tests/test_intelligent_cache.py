"""
Tests for intelligent cache manager.
"""

import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pytest

from morgan.caching.intelligent_cache import IntelligentCacheManager


class TestIntelligentCacheManager:
    """Test intelligent cache manager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "cache"
        self.cache_manager = IntelligentCacheManager(self.cache_dir, enable_metrics=True)
        
        # Create test source directory
        self.source_dir = self.temp_dir / "source"
        self.source_dir.mkdir()
        
        # Create test files
        (self.source_dir / "doc1.txt").write_text("Document 1 content")
        (self.source_dir / "doc2.txt").write_text("Document 2 content")
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_validity_check(self):
        """Test cache validity checking."""
        collection_name = "test_collection"
        source_path = str(self.source_dir)
        
        # Initially invalid - no cache
        status = self.cache_manager.check_cache_validity(source_path, collection_name)
        assert not status.is_valid
        assert not status.collection_exists
        
        # Store collection
        collection_data = {
            'documents': [
                {'id': '1', 'content': 'Doc 1'},
                {'id': '2', 'content': 'Doc 2'}
            ]
        }
        
        success = self.cache_manager.store_collection(
            collection_name=collection_name,
            source_path=source_path,
            collection_data=collection_data
        )
        assert success
        
        # Now should be valid
        status = self.cache_manager.check_cache_validity(source_path, collection_name)
        assert status.is_valid
        assert status.collection_exists
    
    def test_store_and_retrieve_collection(self):
        """Test storing and retrieving collections."""
        collection_name = "test_collection"
        source_path = str(self.source_dir)
        
        collection_data = {
            'documents': [
                {'id': '1', 'content': 'Document 1'},
                {'id': '2', 'content': 'Document 2'},
                {'id': '3', 'content': 'Document 3'}
            ],
            'metadata': {'total_docs': 3}
        }
        
        # Store collection
        success = self.cache_manager.store_collection(
            collection_name=collection_name,
            source_path=source_path,
            collection_data=collection_data
        )
        assert success
        
        # Retrieve collection
        retrieved_data = self.cache_manager.get_cached_collection(collection_name)
        assert retrieved_data is not None
        assert retrieved_data['collection_name'] == collection_name
        assert retrieved_data['source_path'] == source_path
        assert retrieved_data['document_count'] == 3
        assert len(retrieved_data['data']['documents']) == 3
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        collection_name = "test_collection"
        source_path = str(self.source_dir)
        
        collection_data = {'documents': [{'id': '1', 'content': 'Test'}]}
        
        # Store collection
        self.cache_manager.store_collection(collection_name, source_path, collection_data)
        
        # Verify cached
        assert self.cache_manager.get_cached_collection(collection_name) is not None
        
        # Invalidate
        success = self.cache_manager.invalidate_cache(collection_name)
        assert success
        
        # Verify removed
        assert self.cache_manager.get_cached_collection(collection_name) is None
        
        # Cache validity should be false
        status = self.cache_manager.check_cache_validity(source_path, collection_name)
        assert not status.is_valid
    
    def test_cache_speedup_calculation(self):
        """Test cache speedup calculation."""
        collection_name = "test_collection"
        source_path = str(self.source_dir)
        
        # No cache - no speedup
        speedup = self.cache_manager.get_cache_speedup(source_path, collection_name)
        assert speedup is None
        
        # Store collection with many documents
        collection_data = {
            'documents': [{'id': str(i), 'content': f'Doc {i}'} for i in range(100)]
        }
        
        self.cache_manager.store_collection(collection_name, source_path, collection_data)
        
        # Should have speedup
        speedup = self.cache_manager.get_cache_speedup(source_path, collection_name)
        assert speedup is not None
        assert speedup >= 6.0  # Minimum speedup
    
    def test_metrics_tracking(self):
        """Test performance metrics tracking."""
        collection_name = "test_collection"
        source_path = str(self.source_dir)
        
        # Initial metrics
        initial_requests = self.cache_manager.metrics.total_requests
        
        # Check cache (miss)
        self.cache_manager.check_cache_validity(source_path, collection_name)
        assert self.cache_manager.metrics.total_requests == initial_requests + 1
        assert self.cache_manager.metrics.cache_misses >= 1
        
        # Store and check again (hit)
        collection_data = {'documents': [{'id': '1', 'content': 'Test'}]}
        self.cache_manager.store_collection(collection_name, source_path, collection_data)
        
        self.cache_manager.check_cache_validity(source_path, collection_name)
        assert self.cache_manager.metrics.total_requests == initial_requests + 2
        assert self.cache_manager.metrics.cache_hits >= 1
        
        # Hit rate should be reasonable
        assert 0.0 <= self.cache_manager.metrics.hit_rate <= 1.0
    
    def test_cache_statistics(self):
        """Test cache statistics generation."""
        # Get initial stats
        stats = self.cache_manager.get_cache_statistics()
        assert 'metrics' in stats
        assert 'collections' in stats
        
        # Add some collections
        for i in range(3):
            collection_data = {
                'documents': [{'id': str(j), 'content': f'Doc {j}'} for j in range(i + 1)]
            }
            self.cache_manager.store_collection(
                collection_name=f"collection_{i}",
                source_path=str(self.source_dir),
                collection_data=collection_data
            )
        
        # Get updated stats
        stats = self.cache_manager.get_cache_statistics()
        assert stats['collections']['total_collections'] == 3
        assert stats['collections']['total_documents'] == 6  # 1 + 2 + 3
        assert stats['collections']['avg_documents_per_collection'] == 2.0
    
    def test_list_cached_collections(self):
        """Test listing cached collections."""
        # Initially empty
        collections = self.cache_manager.list_cached_collections()
        assert len(collections) == 0
        
        # Add collections
        for i in range(2):
            collection_data = {'documents': [{'id': '1', 'content': 'Test'}]}
            self.cache_manager.store_collection(
                collection_name=f"collection_{i}",
                source_path=str(self.source_dir),
                collection_data=collection_data
            )
        
        # List collections
        collections = self.cache_manager.list_cached_collections()
        assert len(collections) == 2
        
        collection_names = [c.collection_name for c in collections]
        assert "collection_0" in collection_names
        assert "collection_1" in collection_names
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with IntelligentCacheManager(self.cache_dir) as cache_manager:
            # Use cache manager
            collection_data = {'documents': [{'id': '1', 'content': 'Test'}]}
            cache_manager.store_collection(
                collection_name="test_collection",
                source_path=str(self.source_dir),
                collection_data=collection_data
            )
            
            # Verify it works
            retrieved = cache_manager.get_cached_collection("test_collection")
            assert retrieved is not None
        
        # Context manager should have saved metrics
    
    def test_cache_optimization(self):
        """Test cache performance optimization."""
        # Add some collections
        for i in range(3):
            collection_data = {
                'documents': [{'id': str(j), 'content': f'Doc {j}'} 
                             for j in range(i + 1)]
            }
            self.cache_manager.store_collection(
                collection_name=f"collection_{i}",
                source_path=str(self.source_dir),
                collection_data=collection_data
            )
        
        # Run optimization
        results = self.cache_manager.optimize_cache_performance()
        
        assert 'actions_taken' in results
        assert 'recommendations' in results
        assert 'performance_impact' in results
    
    def test_cache_health_monitoring(self):
        """Test cache health monitoring."""
        # Add a collection
        collection_data = {'documents': [{'id': '1', 'content': 'Test'}]}
        self.cache_manager.store_collection(
            collection_name="test_collection",
            source_path=str(self.source_dir),
            collection_data=collection_data
        )
        
        # Monitor health
        health = self.cache_manager.monitor_cache_health()
        
        assert 'overall_health' in health
        assert 'performance_score' in health
        assert 'metrics' in health
        assert health['performance_score'] >= 0
        assert health['performance_score'] <= 100