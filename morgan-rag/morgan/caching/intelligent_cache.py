"""
Intelligent cache manager with Git hash tracking.

This module provides collection-level caching with performance metrics
and automatic invalidation based on Git hash changes.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from .git_hash_tracker import GitHashTracker
from .cache_models import CacheStatus, CacheMetrics, CollectionCacheInfo

logger = logging.getLogger(__name__)


class IntelligentCacheManager:
    """
    Intelligent cache manager with Git hash tracking.
    
    Provides collection-level caching with automatic invalidation
    based on Git hash changes and performance optimization.
    """
    
    def __init__(self, cache_dir: Path, enable_metrics: bool = True):
        """
        Initialize intelligent cache manager.
        
        Args:
            cache_dir: Directory to store cache data
            enable_metrics: Whether to track performance metrics
        """
        self.cache_dir = Path(cache_dir)
        self.git_tracker = GitHashTracker(self.cache_dir / "git_tracking")
        self.enable_metrics = enable_metrics
        
        # Create cache directory structure
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.collections_dir = self.cache_dir / "collections"
        self.collections_dir.mkdir(exist_ok=True)
        
        # Initialize metrics
        self.metrics = CacheMetrics()
        self.metrics_file = self.cache_dir / "metrics.json"
        
        if enable_metrics:
            self._load_metrics()
    
    def _load_metrics(self) -> None:
        """Load metrics from storage."""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Only load fields that exist in CacheMetrics
                    metrics_fields = {
                        'total_requests': data.get('total_requests', 0),
                        'cache_hits': data.get('cache_hits', 0),
                        'cache_misses': data.get('cache_misses', 0),
                        'hash_calculations': data.get('hash_calculations', 0),
                        'invalidations': data.get('invalidations', 0)
                    }
                    self.metrics = CacheMetrics(**metrics_fields)
        except Exception as e:
            logger.warning("Error loading metrics: %s", e)
            self.metrics = CacheMetrics()
    
    def _save_metrics(self) -> None:
        """Save metrics to storage."""
        if not self.enable_metrics:
            return
            
        try:
            metrics_data = {
                'total_requests': self.metrics.total_requests,
                'cache_hits': self.metrics.cache_hits,
                'cache_misses': self.metrics.cache_misses,
                'hash_calculations': self.metrics.hash_calculations,
                'invalidations': self.metrics.invalidations,
                'last_updated': datetime.now().isoformat(),
                'hit_rate': self.metrics.hit_rate
            }
            
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2)
        except Exception as e:
            logger.error("Error saving metrics: %s", e)
    
    def _update_metrics(self, cache_hit: bool, hash_calculated: bool = False) -> None:
        """Update performance metrics."""
        if not self.enable_metrics:
            return
            
        self.metrics.total_requests += 1
        
        if cache_hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1
        
        if hash_calculated:
            self.metrics.hash_calculations += 1
        
        # Save metrics periodically
        if self.metrics.total_requests % 10 == 0:
            self._save_metrics()
    
    def check_cache_validity(
        self, 
        source_path: str, 
        collection_name: str
    ) -> CacheStatus:
        """
        Check if cache is valid for a collection.
        
        Args:
            source_path: Source path of the collection
            collection_name: Name of the collection
            
        Returns:
            CacheStatus with validity information
        """
        start_time = time.time()
        
        try:
            # Check Git hash validity
            cache_status = self.git_tracker.check_cache_validity(source_path, collection_name)
            
            # Check if collection files exist
            if cache_status.is_valid:
                collection_path = self.collections_dir / f"{collection_name}.json"
                if not collection_path.exists():
                    cache_status.is_valid = False
                    cache_status.collection_exists = False
                    cache_status.cache_hit = False
            
            # Update metrics
            self._update_metrics(
                cache_hit=cache_status.is_valid,
                hash_calculated=True
            )
            
            processing_time = time.time() - start_time
            logger.debug(
                "Cache validity check for %s: valid=%s, time=%.3fs",
                collection_name, cache_status.is_valid, processing_time
            )
            
            return cache_status
            
        except Exception as e:
            logger.error(
                "Error checking cache validity for %s: %s", collection_name, e
            )
            self._update_metrics(cache_hit=False, hash_calculated=True)
            
            return CacheStatus(
                is_valid=False,
                stored_hash=None,
                current_hash="",
                last_updated=datetime.now(),
                collection_exists=False
            )
    
    def get_cached_collection(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get cached collection data.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection data or None if not found
        """
        try:
            collection_path = self.collections_dir / f"{collection_name}.json"
            
            if not collection_path.exists():
                return None
            
            with open(collection_path, 'r', encoding='utf-8') as f:
                collection_data = json.load(f)
            
            # Update last accessed time
            collection_data['last_accessed'] = datetime.now().isoformat()
            
            with open(collection_path, 'w', encoding='utf-8') as f:
                json.dump(collection_data, f, indent=2)
            
            logger.debug("Retrieved cached collection %s", collection_name)
            return collection_data
            
        except Exception as e:
            logger.error(
                "Error getting cached collection %s: %s", collection_name, e
            )
            return None
    
    def store_collection(
        self,
        collection_name: str,
        source_path: str,
        collection_data: Dict[str, Any],
        git_hash: Optional[str] = None
    ) -> bool:
        """
        Store collection in cache.
        
        Args:
            collection_name: Name of the collection
            source_path: Source path of the collection
            collection_data: Collection data to store
            git_hash: Git hash (calculated if not provided)
            
        Returns:
            True if successful
        """
        try:
            # Calculate Git hash if not provided
            if git_hash is None:
                git_hash = self.git_tracker.calculate_git_hash(source_path)
                if not git_hash:
                    logger.error(
                        "Failed to calculate Git hash for %s", source_path
                    )
                    return False
            
            # Prepare collection metadata
            now = datetime.now().isoformat()
            document_count = len(collection_data.get('documents', []))
            
            # Store collection data
            collection_path = self.collections_dir / f"{collection_name}.json"
            
            storage_data = {
                'collection_name': collection_name,
                'source_path': source_path,
                'git_hash': git_hash,
                'created_at': now,
                'last_accessed': now,
                'document_count': document_count,
                'data': collection_data
            }
            
            with open(collection_path, 'w', encoding='utf-8') as f:
                json.dump(storage_data, f, indent=2)
            
            # Calculate size
            size_bytes = collection_path.stat().st_size
            
            # Store Git hash
            success = self.git_tracker.store_git_hash(
                source_path=source_path,
                collection_name=collection_name,
                git_hash=git_hash,
                document_count=document_count,
                size_bytes=size_bytes,
                metadata={'cached_at': now}
            )
            
            if success:
                logger.info(
                    "Stored collection %s in cache (%d documents, %d bytes)",
                    collection_name, document_count, size_bytes
                )
            
            return success
            
        except Exception as e:
            logger.error("Error storing collection %s: %s", collection_name, e)
            return False
    
    def invalidate_cache(self, collection_name: str) -> bool:
        """
        Invalidate cache for a collection.
        
        Args:
            collection_name: Name of the collection to invalidate
            
        Returns:
            True if successful
        """
        try:
            # Remove collection file
            collection_path = self.collections_dir / f"{collection_name}.json"
            if collection_path.exists():
                collection_path.unlink()
            
            # Invalidate Git hash tracking
            success = self.git_tracker.invalidate_cache(collection_name)
            
            if success:
                self.metrics.invalidations += 1
                self._save_metrics()
                logger.info("Invalidated cache for collection %s", collection_name)
            
            return success
            
        except Exception as e:
            logger.error(
                "Error invalidating cache for %s: %s", collection_name, e
            )
            return False
    
    def get_cache_speedup(
        self, 
        source_path: str, 
        collection_name: str
    ) -> Optional[float]:
        """
        Calculate potential cache speedup for a collection.
        
        Args:
            source_path: Source path of the collection
            collection_name: Name of the collection
            
        Returns:
            Speedup multiplier or None if not cached
        """
        try:
            cache_status = self.check_cache_validity(source_path, collection_name)
            
            if not cache_status.is_valid:
                return None
            
            # Get collection info
            collection_info = self.git_tracker.get_collection_info(collection_name)
            if not collection_info:
                return None
            
            # Estimate speedup based on document count and size
            # Base speedup of 6x, scaling up to 180x for larger collections
            base_speedup = 6.0
            max_speedup = 180.0
            
            # Scale based on document count (more documents = higher speedup)
            doc_factor = min(collection_info.document_count / 1000.0, 1.0)
            
            # Scale based on size (larger collections benefit more from caching)
            size_mb = collection_info.size_bytes / (1024 * 1024)
            size_factor = min(size_mb / 100.0, 1.0)
            
            # Calculate final speedup
            speedup = base_speedup + (max_speedup - base_speedup) * max(doc_factor, size_factor)
            
            return round(speedup, 1)
            
        except Exception as e:
            logger.error(
                "Error calculating cache speedup for %s: %s", collection_name, e
            )
            return None
    
    def get_incremental_changes(
        self, 
        source_path: str, 
        collection_name: str
    ) -> List[str]:
        """
        Get list of files that changed since last cache.
        
        Args:
            source_path: Source path to check
            collection_name: Name of the collection
            
        Returns:
            List of changed file paths
        """
        try:
            stored_hash = self.git_tracker.get_stored_hash(collection_name)
            if not stored_hash:
                return []
            
            return self.git_tracker.get_changed_files(source_path, stored_hash)
            
        except Exception as e:
            logger.error(
                "Error getting incremental changes for %s: %s", 
                collection_name, e
            )
            return []
    
    def cleanup_expired_cache(self, max_age_days: int = 30) -> int:
        """
        Clean up expired cache entries.
        
        Args:
            max_age_days: Maximum age in days before cache expires
            
        Returns:
            Number of entries cleaned up
        """
        try:
            cleaned_count = 0
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            # Clean up collection files
            for collection_path in self.collections_dir.glob("*.json"):
                try:
                    with open(collection_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    last_accessed = datetime.fromisoformat(
                        data.get('last_accessed', '1970-01-01')
                    )
                    
                    if last_accessed < cutoff_date:
                        collection_name = data.get(
                            'collection_name', collection_path.stem
                        )
                        
                        # Remove collection file
                        collection_path.unlink()
                        
                        # Invalidate Git tracking
                        self.git_tracker.invalidate_cache(collection_name)
                        
                        cleaned_count += 1
                        logger.debug(
                            "Cleaned up expired cache for %s", collection_name
                        )
                        
                except Exception as e:
                    logger.warning("Error cleaning up %s: %s", collection_path, e)
                    continue
            
            # Clean up orphaned Git tracking entries
            orphaned_count = self.git_tracker.cleanup_orphaned_entries()
            cleaned_count += orphaned_count
            
            if cleaned_count > 0:
                logger.info("Cleaned up %d expired cache entries", cleaned_count)
            
            return cleaned_count
            
        except Exception as e:
            logger.error("Error cleaning up expired cache: %s", e)
            return 0
    
    def optimize_cache_performance(self) -> Dict[str, Any]:
        """
        Optimize cache performance by analyzing usage patterns.
        
        Returns:
            Dictionary with optimization results and recommendations
        """
        try:
            optimization_results = {
                'actions_taken': [],
                'recommendations': [],
                'performance_impact': {}
            }
            
            # Get current statistics
            collections = self.git_tracker.list_collections()
            
            if not collections:
                return optimization_results
            
            # Analyze access patterns
            now = datetime.now()
            access_analysis = []
            
            for collection in collections:
                days_since_access = (now - collection.last_accessed).days
                access_analysis.append({
                    'collection': collection.collection_name,
                    'days_since_access': days_since_access,
                    'document_count': collection.document_count,
                    'size_mb': collection.size_bytes / (1024 * 1024)
                })
            
            # Sort by access recency
            access_analysis.sort(key=lambda x: x['days_since_access'])
            
            # Identify optimization opportunities
            stale_collections = [
                c for c in access_analysis if c['days_since_access'] > 7
            ]
            large_unused_collections = [
                c for c in stale_collections 
                if c['size_mb'] > 10 and c['days_since_access'] > 30
            ]
            
            # Auto-cleanup very old collections
            cleaned_count = 0
            for collection in large_unused_collections:
                if collection['days_since_access'] > 60:
                    success = self.invalidate_cache(collection['collection'])
                    if success:
                        cleaned_count += 1
                        optimization_results['actions_taken'].append(
                            f"Removed stale collection: {collection['collection']}"
                        )
            
            # Generate recommendations
            if stale_collections:
                optimization_results['recommendations'].append(
                    f"Consider reviewing {len(stale_collections)} collections "
                    f"not accessed in over 7 days"
                )
            
            if self.metrics.hit_rate < 0.7:
                optimization_results['recommendations'].append(
                    "Cache hit rate is below 70% - consider increasing "
                    "cache retention or reviewing access patterns"
                )
            
            # Calculate performance impact
            total_size_cleaned = sum(
                c['size_mb'] for c in large_unused_collections 
                if c['days_since_access'] > 60
            )
            
            optimization_results['performance_impact'] = {
                'collections_cleaned': cleaned_count,
                'storage_freed_mb': round(total_size_cleaned, 2),
                'estimated_speedup': min(1.2, 1.0 + (cleaned_count * 0.05))
            }
            
            logger.info(
                "Cache optimization completed: %d collections cleaned, "
                "%.2f MB freed", cleaned_count, total_size_cleaned
            )
            
            return optimization_results
            
        except Exception as e:
            logger.error("Error optimizing cache performance: %s", e)
            return {'error': str(e)}
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            # Get basic metrics
            stats = {
                'metrics': {
                    'total_requests': self.metrics.total_requests,
                    'cache_hits': self.metrics.cache_hits,
                    'cache_misses': self.metrics.cache_misses,
                    'hit_rate': self.metrics.hit_rate,
                    'hash_calculations': self.metrics.hash_calculations,
                    'invalidations': self.metrics.invalidations
                }
            }
            
            # Get collection statistics
            collections = self.git_tracker.list_collections()
            
            total_documents = sum(c.document_count for c in collections)
            total_size_mb = sum(c.size_bytes for c in collections) / (1024 * 1024)
            
            stats['collections'] = {
                'total_collections': len(collections),
                'total_documents': total_documents,
                'total_size_mb': round(total_size_mb, 2),
                'avg_documents_per_collection': (
                    round(total_documents / len(collections), 1) 
                    if collections else 0
                )
            }
            
            # Get age statistics
            if collections:
                now = datetime.now()
                ages = [(now - c.last_accessed).days for c in collections]
                
                stats['age_statistics'] = {
                    'oldest_days': max(ages),
                    'newest_days': min(ages),
                    'avg_age_days': round(sum(ages) / len(ages), 1)
                }
            
            # Add performance insights
            stats['performance_insights'] = self._generate_performance_insights(
                collections
            )
            
            return stats
            
        except Exception as e:
            logger.error("Error getting cache statistics: %s", e)
            return {'error': str(e)}
    
    def _generate_performance_insights(
        self, 
        collections: List[CollectionCacheInfo]
    ) -> Dict[str, Any]:
        """
        Generate performance insights from cache data.
        
        Args:
            collections: List of cached collections
            
        Returns:
            Dictionary with performance insights
        """
        insights = {
            'cache_efficiency': 'good',
            'storage_utilization': 'optimal',
            'access_patterns': 'normal',
            'recommendations': []
        }
        
        if not collections:
            return insights
        
        # Analyze cache efficiency
        if self.metrics.hit_rate < 0.5:
            insights['cache_efficiency'] = 'poor'
            insights['recommendations'].append(
                "Low cache hit rate - consider reviewing caching strategy"
            )
        elif self.metrics.hit_rate < 0.7:
            insights['cache_efficiency'] = 'fair'
        
        # Analyze storage utilization
        total_size_mb = sum(c.size_bytes for c in collections) / (1024 * 1024)
        if total_size_mb > 1000:  # > 1GB
            insights['storage_utilization'] = 'high'
            insights['recommendations'].append(
                "High storage usage - consider cleanup of old collections"
            )
        
        # Analyze access patterns
        now = datetime.now()
        recent_access_count = sum(
            1 for c in collections 
            if (now - c.last_accessed).days <= 7
        )
        
        if recent_access_count < len(collections) * 0.3:
            insights['access_patterns'] = 'stale'
            insights['recommendations'].append(
                "Many collections haven't been accessed recently"
            )
        
        return insights
    
    def monitor_cache_health(self) -> Dict[str, Any]:
        """
        Monitor cache health and performance metrics.
        
        Returns:
            Dictionary with health status and performance metrics
        """
        try:
            health_status = {
                'overall_health': 'healthy',
                'performance_score': 100,
                'issues': [],
                'metrics': {},
                'recommendations': []
            }
            
            # Check basic metrics
            health_status['metrics'] = {
                'hit_rate': self.metrics.hit_rate,
                'total_requests': self.metrics.total_requests,
                'cache_size_mb': self._get_cache_size_mb(),
                'collection_count': len(self.git_tracker.list_collections())
            }
            
            # Evaluate performance score
            score = 100
            
            # Hit rate impact (40% of score)
            hit_rate_score = min(40, self.metrics.hit_rate * 40)
            score = score - 40 + hit_rate_score
            
            # Storage efficiency (30% of score)
            cache_size_mb = health_status['metrics']['cache_size_mb']
            if cache_size_mb > 1000:  # > 1GB
                score -= 15
                health_status['issues'].append('High storage usage')
            elif cache_size_mb > 500:  # > 500MB
                score -= 5
            
            # Access pattern health (30% of score)
            collections = self.git_tracker.list_collections()
            if collections:
                now = datetime.now()
                stale_count = sum(
                    1 for c in collections 
                    if (now - c.last_accessed).days > 30
                )
                stale_ratio = stale_count / len(collections)
                
                if stale_ratio > 0.5:
                    score -= 20
                    health_status['issues'].append('Many stale collections')
                elif stale_ratio > 0.3:
                    score -= 10
            
            health_status['performance_score'] = max(0, int(score))
            
            # Determine overall health
            if score >= 80:
                health_status['overall_health'] = 'healthy'
            elif score >= 60:
                health_status['overall_health'] = 'warning'
                health_status['recommendations'].append(
                    'Consider cache optimization'
                )
            else:
                health_status['overall_health'] = 'critical'
                health_status['recommendations'].extend([
                    'Immediate cache optimization needed',
                    'Review caching strategy'
                ])
            
            # Add specific recommendations
            if self.metrics.hit_rate < 0.7:
                health_status['recommendations'].append(
                    'Improve cache hit rate through better retention policies'
                )
            
            if cache_size_mb > 500:
                health_status['recommendations'].append(
                    'Consider cleanup of old or unused collections'
                )
            
            return health_status
            
        except Exception as e:
            logger.error("Error monitoring cache health: %s", e)
            return {
                'overall_health': 'error',
                'error': str(e)
            }
    
    def _get_cache_size_mb(self) -> float:
        """
        Calculate total cache size in MB.
        
        Returns:
            Cache size in megabytes
        """
        try:
            total_size = 0
            
            # Calculate collection files size
            for collection_path in self.collections_dir.glob("*.json"):
                total_size += collection_path.stat().st_size
            
            # Add Git tracking size
            git_tracking_dir = self.cache_dir / "git_tracking"
            if git_tracking_dir.exists():
                for file_path in git_tracking_dir.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
            
            return total_size / (1024 * 1024)
            
        except Exception as e:
            logger.error("Error calculating cache size: %s", e)
            return 0.0
    
    def list_cached_collections(self) -> List[CollectionCacheInfo]:
        """
        List all cached collections with their information.
        
        Returns:
            List of CollectionCacheInfo objects
        """
        return self.git_tracker.list_collections()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save metrics."""
        if self.enable_metrics:
            self._save_metrics()