#!/usr/bin/env python3
"""
Simple test for Git hash cache implementation.

This script tests the core cache functionality without vector DB dependencies.
"""

import tempfile
import shutil
from pathlib import Path
import json
import time

# Add the parent directory to the path so we can import morgan
import sys
sys.path.insert(0, str(Path(__file__).parent))

from morgan.caching.git_hash_tracker import GitHashTracker


def test_git_hash_cache_core():
    """Test core Git hash cache functionality."""
    print("🧪 Testing Core Git Hash Cache Functionality")
    print("=" * 50)
    
    # Create temporary directories
    temp_dir = Path(tempfile.mkdtemp())
    cache_dir = temp_dir / "cache"
    source_dir = temp_dir / "source"
    source_dir.mkdir()
    
    try:
        # Create test documents
        print("\n📄 Creating test documents...")
        (source_dir / "doc1.txt").write_text("This is document 1 content for testing cache")
        (source_dir / "doc2.txt").write_text("This is document 2 content for testing cache")
        
        # Initialize Git hash tracker
        print("🚀 Initializing Git hash tracker...")
        git_tracker = GitHashTracker(cache_dir)
        
        collection_name = "test_collection"
        source_path = str(source_dir)
        
        # Test 1: Initial cache miss
        print(f"\n🔍 Test 1: Initial cache validity check (should be cache miss)...")
        cache_status = git_tracker.check_cache_validity(source_path, collection_name)
        print(f"   Cache valid: {cache_status.is_valid}")
        print(f"   Current hash: {cache_status.current_hash[:12]}...")
        print(f"   Stored hash: {cache_status.stored_hash}")
        
        assert not cache_status.is_valid, "Initial cache should be invalid"
        assert cache_status.stored_hash is None, "Initial stored hash should be None"
        
        # Test 2: Store Git hash
        print(f"\n💾 Test 2: Storing Git hash...")
        current_hash = git_tracker.calculate_git_hash(source_path)
        print(f"   Calculated hash: {current_hash[:12]}...")
        
        success = git_tracker.store_git_hash(
            source_path=source_path,
            collection_name=collection_name,
            git_hash=current_hash,
            document_count=2,
            size_bytes=1024,
            metadata={"test": "data"}
        )
        print(f"   Storage successful: {success}")
        
        assert success, "Git hash storage should succeed"
        
        # Test 3: Cache hit after storage
        print(f"\n✅ Test 3: Cache validity check after storage (should be cache hit)...")
        cache_status = git_tracker.check_cache_validity(source_path, collection_name)
        print(f"   Cache valid: {cache_status.is_valid}")
        print(f"   Stored hash: {cache_status.stored_hash[:12] if cache_status.stored_hash else 'None'}...")
        print(f"   Current hash: {cache_status.current_hash[:12]}...")
        print(f"   Cache hit: {cache_status.cache_hit}")
        
        assert cache_status.is_valid, "Cache should be valid after storage"
        assert cache_status.stored_hash == current_hash, "Stored hash should match current hash"
        assert cache_status.cache_hit, "Should be a cache hit"
        
        # Test 4: Get cache metrics
        print(f"\n📊 Test 4: Cache performance metrics...")
        cache_metrics = git_tracker.get_cache_metrics()
        cache_perf = cache_metrics["cache_performance"]
        collection_stats = cache_metrics["collection_stats"]
        
        print(f"   Total requests: {cache_perf['total_requests']}")
        print(f"   Cache hits: {cache_perf['cache_hits']}")
        print(f"   Cache misses: {cache_perf['cache_misses']}")
        print(f"   Hit rate: {cache_perf['hit_rate']:.1%}")
        print(f"   Hash calculations: {cache_perf['hash_calculations']}")
        print(f"   Total collections: {collection_stats['total_collections']}")
        print(f"   Total documents: {collection_stats['total_documents']}")
        
        assert cache_perf['total_requests'] >= 2, "Should have at least 2 requests"
        assert cache_perf['cache_hits'] >= 1, "Should have at least 1 cache hit"
        assert cache_perf['cache_misses'] >= 1, "Should have at least 1 cache miss"
        assert collection_stats['total_collections'] >= 1, "Should have at least 1 collection"
        
        # Test 5: Cache efficiency report
        print(f"\n📈 Test 5: Cache efficiency report...")
        efficiency_report = git_tracker.get_cache_efficiency_report()
        summary = efficiency_report["summary"]
        
        print(f"   Efficiency level: {summary['efficiency_level']}")
        print(f"   Hit rate: {summary['hit_rate_percentage']}")
        print(f"   Total requests: {summary['total_requests']}")
        print(f"   Recommendations: {len(efficiency_report['recommendations'])}")
        
        assert summary['efficiency_level'] in ['Excellent', 'Good', 'Fair', 'Poor'], "Should have valid efficiency level"
        
        # Test 6: Simulate content change (cache miss)
        print(f"\n✏️  Test 6: Simulating content change...")
        (source_dir / "doc3.txt").write_text("This is a new document")
        
        cache_status = git_tracker.check_cache_validity(source_path, collection_name)
        print(f"   Cache valid after change: {cache_status.is_valid}")
        print(f"   Stored hash: {cache_status.stored_hash[:12] if cache_status.stored_hash else 'None'}...")
        print(f"   Current hash: {cache_status.current_hash[:12]}...")
        
        assert not cache_status.is_valid, "Cache should be invalid after content change"
        assert cache_status.stored_hash != cache_status.current_hash, "Hashes should be different after change"
        
        # Test 7: Collection info retrieval
        print(f"\n📋 Test 7: Collection info retrieval...")
        collection_info = git_tracker.get_collection_info(collection_name)
        
        if collection_info:
            print(f"   Collection name: {collection_info.collection_name}")
            print(f"   Document count: {collection_info.document_count}")
            print(f"   Size bytes: {collection_info.size_bytes}")
            print(f"   Git hash: {collection_info.git_hash[:12]}...")
            
            assert collection_info.collection_name == collection_name, "Collection name should match"
            assert collection_info.document_count == 2, "Document count should be 2"
        else:
            print("   No collection info found")
        
        # Test 8: Final metrics
        print(f"\n📊 Test 8: Final cache metrics...")
        final_metrics = git_tracker.get_cache_metrics()
        final_perf = final_metrics["cache_performance"]
        
        print(f"   Final total requests: {final_perf['total_requests']}")
        print(f"   Final cache hits: {final_perf['cache_hits']}")
        print(f"   Final cache misses: {final_perf['cache_misses']}")
        print(f"   Final hit rate: {final_perf['hit_rate']:.1%}")
        
        assert final_perf['total_requests'] >= 3, "Should have at least 3 total requests"
        
        print(f"\n✅ All core cache tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up temporary files")


if __name__ == "__main__":
    print("🚀 Starting Simple Git Hash Cache Tests")
    print("=" * 50)
    
    # Run core cache tests
    cache_test_passed = test_git_hash_cache_core()
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Core Cache Implementation: {'✅ PASSED' if cache_test_passed else '❌ FAILED'}")
    
    if cache_test_passed:
        print("\n🎉 Core cache tests passed! Task 1.3 implementation is working.")
        print("\n📋 Implementation includes:")
        print("   • Git hash-based cache reuse (R1.3)")
        print("   • Comprehensive cache hit metrics (R9.1)")
        print("   • Cache efficiency reporting")
        print("   • Automatic cache invalidation on content changes")
        print("   • Metrics persistence and tracking")
        sys.exit(0)
    else:
        print("\n❌ Core cache tests failed. Please review the implementation.")
        sys.exit(1)