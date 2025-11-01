#!/usr/bin/env python3
"""
Demo of enhanced intelligent cache manager functionality.

This demo shows the new optimization and monitoring features
added to the intelligent cache manager.
"""

import tempfile
import shutil
from pathlib import Path
from morgan.caching.intelligent_cache import IntelligentCacheManager


def main():
    """Demonstrate intelligent cache manager enhancements."""
    print("=== Intelligent Cache Manager Demo ===\n")
    
    # Create temporary directory for demo
    temp_dir = Path(tempfile.mkdtemp())
    cache_dir = temp_dir / "cache"
    
    try:
        # Initialize cache manager
        print("1. Initializing cache manager...")
        cache_manager = IntelligentCacheManager(
            cache_dir=cache_dir,
            enable_metrics=True
        )
        
        # Create some test collections
        print("2. Creating test collections...")
        for i in range(5):
            collection_data = {
                'documents': [
                    {'id': str(j), 'content': f'Document {j} in collection {i}'}
                    for j in range((i + 1) * 10)  # Varying sizes
                ],
                'metadata': {'collection_id': i, 'created_by': 'demo'}
            }
            
            success = cache_manager.store_collection(
                collection_name=f"demo_collection_{i}",
                source_path=str(temp_dir / f"source_{i}"),
                collection_data=collection_data
            )
            print(f"   - Collection {i}: {'✓' if success else '✗'}")
        
        # Demonstrate cache statistics
        print("\n3. Cache Statistics:")
        stats = cache_manager.get_cache_statistics()
        
        print(f"   - Total collections: {stats['collections']['total_collections']}")
        print(f"   - Total documents: {stats['collections']['total_documents']}")
        print(f"   - Cache size: {stats['collections']['total_size_mb']:.2f} MB")
        print(f"   - Hit rate: {stats['metrics']['hit_rate']:.1%}")
        
        # Demonstrate performance insights
        if 'performance_insights' in stats:
            insights = stats['performance_insights']
            print(f"   - Cache efficiency: {insights['cache_efficiency']}")
            print(f"   - Storage utilization: {insights['storage_utilization']}")
            
            if insights['recommendations']:
                print("   - Recommendations:")
                for rec in insights['recommendations']:
                    print(f"     • {rec}")
        
        # Demonstrate cache health monitoring
        print("\n4. Cache Health Monitoring:")
        health = cache_manager.monitor_cache_health()
        
        print(f"   - Overall health: {health['overall_health']}")
        print(f"   - Performance score: {health['performance_score']}/100")
        print(f"   - Cache hit rate: {health['metrics']['hit_rate']:.1%}")
        print(f"   - Cache size: {health['metrics']['cache_size_mb']:.2f} MB")
        
        if health['issues']:
            print("   - Issues detected:")
            for issue in health['issues']:
                print(f"     • {issue}")
        
        if health['recommendations']:
            print("   - Recommendations:")
            for rec in health['recommendations']:
                print(f"     • {rec}")
        
        # Demonstrate cache optimization
        print("\n5. Cache Performance Optimization:")
        optimization = cache_manager.optimize_cache_performance()
        
        print(f"   - Actions taken: {len(optimization['actions_taken'])}")
        for action in optimization['actions_taken']:
            print(f"     • {action}")
        
        print(f"   - Recommendations: {len(optimization['recommendations'])}")
        for rec in optimization['recommendations']:
            print(f"     • {rec}")
        
        impact = optimization['performance_impact']
        print(f"   - Collections cleaned: {impact['collections_cleaned']}")
        print(f"   - Storage freed: {impact['storage_freed_mb']:.2f} MB")
        print(f"   - Estimated speedup: {impact['estimated_speedup']:.1f}x")
        
        # Demonstrate cache validity checking
        print("\n6. Cache Validity Checking:")
        for i in range(3):
            collection_name = f"demo_collection_{i}"
            source_path = str(temp_dir / f"source_{i}")
            
            status = cache_manager.check_cache_validity(source_path, collection_name)
            speedup = cache_manager.get_cache_speedup(source_path, collection_name)
            
            print(f"   - {collection_name}:")
            print(f"     Valid: {'✓' if status.is_valid else '✗'}")
            print(f"     Speedup: {speedup:.1f}x" if speedup else "     Speedup: N/A")
        
        # List all cached collections
        print("\n7. Cached Collections:")
        collections = cache_manager.list_cached_collections()
        
        for collection in collections:
            print(f"   - {collection.collection_name}:")
            print(f"     Documents: {collection.document_count}")
            print(f"     Size: {collection.size_bytes / 1024:.1f} KB")
            print(f"     Last accessed: {collection.last_accessed.strftime('%Y-%m-%d %H:%M')}")
        
        print(f"\n=== Demo completed successfully! ===")
        print(f"Cache directory: {cache_dir}")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("Cleanup completed.")
        except Exception as e:
            print(f"Cleanup error: {e}")


if __name__ == "__main__":
    main()