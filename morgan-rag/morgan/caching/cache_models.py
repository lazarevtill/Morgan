"""
Data models for intelligent caching system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class CacheStatus:
    """Cache validity information."""
    is_valid: bool
    stored_hash: Optional[str]
    current_hash: str
    last_updated: datetime
    collection_exists: bool
    cache_hit: bool = False
    
    
@dataclass
class CacheMetrics:
    """Performance metrics for caching operations."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hash_calculations: int = 0
    invalidations: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate


@dataclass
class CollectionCacheInfo:
    """Information about a cached collection."""
    collection_name: str
    source_path: str
    git_hash: str
    document_count: int
    created_at: datetime
    last_accessed: datetime
    size_bytes: int
    metadata: Dict[str, Any]