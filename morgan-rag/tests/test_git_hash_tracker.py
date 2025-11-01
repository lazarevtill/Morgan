"""
Tests for Git hash tracking system.
"""

import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import pytest

from morgan.caching.git_hash_tracker import GitHashTracker
from morgan.caching.cache_models import CacheStatus


class TestGitHashTracker:
    """Test Git hash tracking functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "cache"
        self.tracker = GitHashTracker(self.cache_dir)
        
        # Create test source directory
        self.source_dir = self.temp_dir / "source"
        self.source_dir.mkdir()
        
        # Create test files
        (self.source_dir / "test1.txt").write_text("Test content 1")
        (self.source_dir / "test2.txt").write_text("Test content 2")
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_file_hash_calculation(self):
        """Test file hash calculation for non-Git directories."""
        # Calculate hash for directory
        hash1 = self.tracker.calculate_git_hash(str(self.source_dir))
        assert hash1 is not None
        assert len(hash1) == 64  # SHA256 hash length
        
        # Hash should be consistent
        hash2 = self.tracker.calculate_git_hash(str(self.source_dir))
        assert hash1 == hash2
        
        # Hash should change when content changes
        (self.source_dir / "test3.txt").write_text("New content")
        hash3 = self.tracker.calculate_git_hash(str(self.source_dir))
        assert hash3 != hash1
    
    def test_store_and_retrieve_hash(self):
        """Test storing and retrieving Git hashes."""
        collection_name = "test_collection"
        source_path = str(self.source_dir)
        test_hash = "abc123def456"
        
        # Store hash
        success = self.tracker.store_git_hash(
            source_path=source_path,
            collection_name=collection_name,
            git_hash=test_hash,
            document_count=5,
            size_bytes=1024
        )
        assert success
        
        # Retrieve hash
        stored_hash = self.tracker.get_stored_hash(collection_name)
        assert stored_hash == test_hash
    
    def test_cache_validity_check(self):
        """Test cache validity checking."""
        collection_name = "test_collection"
        source_path = str(self.source_dir)
        
        # Initially no cache - should be invalid
        status = self.tracker.check_cache_validity(source_path, collection_name)
        assert not status.is_valid
        assert status.stored_hash is None
        assert status.current_hash != ""
        
        # Store current hash
        current_hash = self.tracker.calculate_git_hash(source_path)
        self.tracker.store_git_hash(source_path, collection_name, current_hash)
        
        # Now should be valid
        status = self.tracker.check_cache_validity(source_path, collection_name)
        assert status.is_valid
        assert status.stored_hash == current_hash
        assert status.current_hash == current_hash
        
        # Change content - should become invalid
        (self.source_dir / "new_file.txt").write_text("New content")
        status = self.tracker.check_cache_validity(source_path, collection_name)
        assert not status.is_valid
        assert status.stored_hash == current_hash
        assert status.current_hash != current_hash
    
    def test_collection_info_storage(self):
        """Test collection information storage and retrieval."""
        collection_name = "test_collection"
        source_path = str(self.source_dir)
        test_hash = "abc123def456"
        
        # Store collection info
        success = self.tracker.store_git_hash(
            source_path=source_path,
            collection_name=collection_name,
            git_hash=test_hash,
            document_count=10,
            size_bytes=2048,
            metadata={"test_key": "test_value"}
        )
        assert success
        
        # Retrieve collection info
        info = self.tracker.get_collection_info(collection_name)
        assert info is not None
        assert info.collection_name == collection_name
        assert info.source_path == source_path
        assert info.git_hash == test_hash
        assert info.document_count == 10
        assert info.size_bytes == 2048
        assert info.metadata["test_key"] == "test_value"
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        collection_name = "test_collection"
        source_path = str(self.source_dir)
        test_hash = "abc123def456"
        
        # Store hash
        self.tracker.store_git_hash(source_path, collection_name, test_hash)
        
        # Verify stored
        assert self.tracker.get_stored_hash(collection_name) == test_hash
        assert self.tracker.get_collection_info(collection_name) is not None
        
        # Invalidate
        success = self.tracker.invalidate_cache(collection_name)
        assert success
        
        # Verify removed
        assert self.tracker.get_stored_hash(collection_name) is None
        assert self.tracker.get_collection_info(collection_name) is None
    
    def test_list_collections(self):
        """Test listing collections."""
        # Initially empty
        collections = self.tracker.list_collections()
        assert len(collections) == 0
        
        # Add some collections
        for i in range(3):
            self.tracker.store_git_hash(
                source_path=str(self.source_dir),
                collection_name=f"collection_{i}",
                git_hash=f"hash_{i}",
                document_count=i * 5,
                size_bytes=i * 1024
            )
        
        # List collections
        collections = self.tracker.list_collections()
        assert len(collections) == 3
        
        # Verify collection data
        collection_names = [c.collection_name for c in collections]
        assert "collection_0" in collection_names
        assert "collection_1" in collection_names
        assert "collection_2" in collection_names
    
    def test_cleanup_orphaned_entries(self):
        """Test cleanup of orphaned entries."""
        # Create inconsistent state by manually editing files
        collection_name = "test_collection"
        
        # Add entry to hashes but not collections
        with open(self.tracker.hash_file, 'w') as f:
            json.dump({
                collection_name: {
                    'source_path': str(self.source_dir),
                    'git_hash': 'test_hash',
                    'last_updated': datetime.now().isoformat()
                }
            }, f)
        
        # Cleanup should remove the orphaned entry
        cleaned_count = self.tracker.cleanup_orphaned_entries()
        assert cleaned_count == 1
        
        # Verify cleanup
        with open(self.tracker.hash_file, 'r') as f:
            hashes = json.load(f)
        assert len(hashes) == 0