"""
Cache Storage - Performance caching and optimization

Provides caching for performance optimization using existing caching infrastructure.
Follows KISS principles with simple, focused functionality.

Requirements addressed: 23.1, 23.4, 23.5
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CacheStorage:
    """
    Cache storage following KISS principles.
    
    Single responsibility: Manage performance caching using existing infrastructure.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Use existing caching infrastructure
        self._initialize_cache_systems()
        
    def _initialize_cache_systems(self) -> None:
        """Initialize existing cache systems."""
        try:
            # Use existing intelligent cache
            from ..caching.intelligent_cache import IntelligentCacheManager
            self.intelligent_cache = IntelligentCacheManager(
                self.config.get('intelligent_cache', {}))
            
            # Use existing git hash tracker
            from ..caching.git_hash_tracker import GitHashTracker
            self.git_tracker = GitHashTracker(
                self.config.get('git_tracker', {}))
            
            logger.info("Cache storage initialized with existing cache systems")
            
        except Exception as e:
            logger.error("Failed to initialize cache systems: %s", e)
            # Create minimal fallback cache
            self._cache = {}
            self.intelligent_cache = None
            self.git_tracker = None
            
    def check_cache_validity(self, source_path: str,
                            collection_name: str) -> Dict[str, Any]:
        """
        Check if cached data is still valid.
        
        Args:
            source_path: Path to source data
            collection_name: Collection name
            
        Returns:
            Cache validity information
        """
        try:
            if self.intelligent_cache:
                return self.intelligent_cache.check_cache_validity(
                    source_path, collection_name)
            else:
                # Fallback implementation
                return {
                    'is_valid': False,
                    'reason': 'Cache system not available'
                }
                
        except Exception as e:
            logger.error("Error checking cache validity: %s", e)
            return {
                'is_valid': False,
                'reason': f'Error: {e}'
            }
            
    def get_git_hash(self, source_path: str) -> Optional[str]:
        """
        Get Git hash for source path.
        
        Args:
            source_path: Path to check
            
        Returns:
            Git hash if available, None otherwise
        """
        try:
            if self.git_tracker:
                return self.git_tracker.get_current_hash(source_path)
            else:
                return None
                
        except Exception as e:
            logger.error("Error getting git hash: %s", e)
            return None
            
    def store_cache_metadata(self, source_path: str, collection_name: str,
                           metadata: Dict[str, Any]) -> bool:
        """
        Store cache metadata.
        
        Args:
            source_path: Source path
            collection_name: Collection name
            metadata: Metadata to store
            
        Returns:
            True if stored successfully
        """
        try:
            if self.intelligent_cache:
                return self.intelligent_cache.store_cache_metadata(
                    source_path, collection_name, metadata
                )
            else:
                # Fallback storage
                cache_key = f"{source_path}:{collection_name}"
                self._cache[cache_key] = {
                    'metadata': metadata,
                    'timestamp': datetime.now()
                }
                return True
                
        except Exception as e:
            logger.error("Error storing cache metadata: %s", e)
            return False
            
    def get_cache_metadata(self, source_path: str,
                          collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get cache metadata.
        
        Args:
            source_path: Source path
            collection_name: Collection name
            
        Returns:
            Cache metadata if available
        """
        try:
            if self.intelligent_cache:
                return self.intelligent_cache.get_cache_metadata(
                    source_path, collection_name)
            else:
                # Fallback retrieval
                cache_key = f"{source_path}:{collection_name}"
                cached_data = self._cache.get(cache_key)
                return cached_data['metadata'] if cached_data else None
                
        except Exception as e:
            logger.error("Error getting cache metadata: %s", e)
            return None
            
    def invalidate_cache(self, source_path: str, collection_name: str) -> bool:
        """
        Invalidate cache for source path and collection.
        
        Args:
            source_path: Source path
            collection_name: Collection name
            
        Returns:
            True if invalidated successfully
        """
        try:
            if self.intelligent_cache:
                return self.intelligent_cache.invalidate_cache(
                    source_path, collection_name)
            else:
                # Fallback invalidation
                cache_key = f"{source_path}:{collection_name}"
                if cache_key in self._cache:
                    del self._cache[cache_key]
                return True
                
        except Exception as e:
            logger.error("Error invalidating cache: %s", e)
            return False
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        try:
            stats = {}
            
            if self.intelligent_cache:
                stats['intelligent_cache'] = self.intelligent_cache.get_cache_stats()
            else:
                stats['fallback_cache'] = {
                    'entries': len(self._cache),
                    'keys': list(self._cache.keys())
                }
                
            if self.git_tracker:
                stats['git_tracker'] = self.git_tracker.get_stats()
                
            return stats
            
        except Exception as e:
            logger.error("Error getting cache stats: %s", e)
            return {'error': str(e)}
            
    def cleanup_expired_cache(self, max_age_hours: int = 24) -> int:
        """
        Clean up expired cache entries.
        
        Args:
            max_age_hours: Maximum age in hours before expiration
            
        Returns:
            Number of entries cleaned up
        """
        try:
            cleaned_count = 0
            
            if (self.intelligent_cache and
                hasattr(self.intelligent_cache, 'cleanup_expired')):
                cleaned_count += self.intelligent_cache.cleanup_expired(
                    max_age_hours)
            else:
                # Fallback cleanup
                cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
                expired_keys = []
                
                for key, data in self._cache.items():
                    if data.get('timestamp', datetime.now()) < cutoff_time:
                        expired_keys.append(key)
                        
                for key in expired_keys:
                    del self._cache[key]
                    cleaned_count += 1
                    
            logger.info("Cleaned up %d expired cache entries", cleaned_count)
            return cleaned_count
            
        except Exception as e:
            logger.error("Error cleaning up cache: %s", e)
            return 0