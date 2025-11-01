#!/usr/bin/env python3
"""
Demo of Git hash tracking system for intelligent caching.

This example demonstrates how the Git hash tracking system works
to provide intelligent caching with automatic invalidation.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the path so we can import morgan
sys.path.insert(0, str(Path(__file__).parent.parent))

from morgan.caching import GitHashTracker, IntelligentCacheManager


def main():
    """Demonstrate Git hash tracking functionality."""
    print("🔍 Git Hash Tracking System Demo")
    print("=" * 50)
    
    # Create temporary directories
    temp_dir = Path(tempfile.mkdtemp())
    cache_dir = temp_dir / "cache"
    source_dir = temp_dir / "source"
    source_dir.mkdir()
    
    try:
        # Create some test documents
        print("\n📄 Creating test documents...")
        (source_dir / "doc1.txt").write_text("This is document 1 content")
        (source_dir / "doc2.txt").write_text("This is document 2 content")
        (source_dir / "doc3.txt").write_text("This is document 3 content")
        
        # Initialize cache manager
        print("🚀 Initializing intelligent cache manager...")
        cache_manager = IntelligentCacheManager(cache_dir)
        
        collection_name = "demo_collection"
        source_path = str(source_dir)
        
        # Check initial cache status
        print(f"\n🔍 Checking initial cache status for '{collection_name}'...")
        status = cache_manager.check_cache_validity(source_path, collection_name)
        print(f"   Cache valid: {status.is_valid}")
        print(f"   Collection exists: {status.collection_exists}")
        print(f"   Current hash: {status.current_hash[:12]}...")
        
        # Create sample collection data
        collection_data = {
            'documents': [
                {'id': '1', 'content': 'Document 1 content', 'source': 'doc1.txt'},
                {'id': '2', 'content': 'Document 2 content', 'source': 'doc2.txt'},
                {'id': '3', 'content': 'Document 3 content', 'source': 'doc3.txt'}
            ],
            'metadata': {
                'total_docs': 3,
                'processed_at': '2024-01-01T12:00:00'
            }
        }
        
        # Store collection in cache
        print(f"\n💾 Storing collection '{collection_name}' in cache...")
        success = cache_manager.store_collection(
            collection_name=collection_name,
            source_path=source_path,
            collection_data=collection_data
        )
        print(f"   Storage successful: {success}")
        
        # Check cache status after storage
        print(f"\n🔍 Checking cache status after storage...")
        status = cache_manager.check_cache_validity(source_path, collection_name)
        print(f"   Cache valid: {status.is_valid}")
        print(f"   Collection exists: {status.collection_exists}")
        print(f"   Stored hash: {status.stored_hash[:12]}...")
        print(f"   Current hash: {status.current_hash[:12]}...")
        print(f"   Cache hit: {status.cache_hit}")
        
        # Calculate potential speedup
        speedup = cache_manager.get_cache_speedup(source_path, collection_name)
        print(f"   Potential speedup: {speedup}x")
        
        # Retrieve cached collection
        print(f"\n📥 Retrieving cached collection...")
        cached_data = cache_manager.get_cached_collection(collection_name)
        if cached_data:
            print(f"   Retrieved collection: {cached_data['collection_name']}")
            print(f"   Document count: {cached_data['document_count']}")
            print(f"   Source path: {cached_data['source_path']}")
            print(f"   Git hash: {cached_data['git_hash'][:12]}...")
        
        # Simulate content change
        print(f"\n✏️  Simulating content change...")
        (source_dir / "doc4.txt").write_text("This is a new document")
        
        # Check cache status after change
        print(f"\n🔍 Checking cache status after content change...")
        status = cache_manager.check_cache_validity(source_path, collection_name)
        print(f"   Cache valid: {status.is_valid}")
        print(f"   Stored hash: {status.stored_hash[:12]}...")
        print(f"   Current hash: {status.current_hash[:12]}...")
        print(f"   Hashes match: {status.stored_hash == status.current_hash}")
        
        # Get incremental changes
        print(f"\n📋 Getting incremental changes...")
        changes = cache_manager.get_incremental_changes(source_path, collection_name)
        print(f"   Changed files: {changes}")
        
        # Show cache statistics
        print(f"\n📊 Cache Statistics:")
        stats = cache_manager.get_cache_statistics()
        metrics = stats['metrics']
        collections = stats['collections']
        
        print(f"   Total requests: {metrics['total_requests']}")
        print(f"   Cache hits: {metrics['cache_hits']}")
        print(f"   Cache misses: {metrics['cache_misses']}")
        print(f"   Hit rate: {metrics['hit_rate']:.2%}")
        print(f"   Total collections: {collections['total_collections']}")
        print(f"   Total documents: {collections['total_documents']}")
        
        # List all cached collections
        print(f"\n📚 Cached Collections:")
        cached_collections = cache_manager.list_cached_collections()
        for collection in cached_collections:
            print(f"   - {collection.collection_name}: {collection.document_count} docs, "
                  f"{collection.size_bytes} bytes")
        
        print(f"\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up temporary files")


if __name__ == "__main__":
    main()