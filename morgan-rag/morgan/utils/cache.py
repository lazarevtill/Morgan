"""
Simple file-based cache for Morgan RAG.

KISS Principle: Simple, reliable caching with file storage.
"""

import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Optional


class FileCache:
    """
    Simple file-based cache with TTL support.
    
    Uses pickle for Python objects and JSON for simple data.
    """
    
    def __init__(self, cache_dir: Path, ttl: int = 3600):
        """
        Initialize file cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Use hash of key as filename to avoid filesystem issues
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _get_meta_path(self, key: str) -> Path:
        """Get metadata file path for key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.meta"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)
        
        if not cache_path.exists() or not meta_path.exists():
            return None
        
        try:
            # Check if expired
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            if time.time() > meta['expires_at']:
                # Expired - clean up
                cache_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return None
            
            # Load cached value
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
                
        except Exception:
            # Corrupted cache - clean up
            cache_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live (uses default if None)
            
        Returns:
            True if successful
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)
        
        try:
            # Save value
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Save metadata
            expires_at = time.time() + (ttl or self.ttl)
            meta = {
                'key': key,
                'created_at': time.time(),
                'expires_at': expires_at
            }
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            
            return True
            
        except Exception:
            # Clean up on failure
            cache_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        cache_path = self._get_cache_path(key)
        meta_path = self._get_meta_path(key)
        
        deleted = False
        
        if cache_path.exists():
            cache_path.unlink()
            deleted = True
        
        if meta_path.exists():
            meta_path.unlink()
            deleted = True
        
        return deleted
    
    def clear(self) -> int:
        """
        Clear all cache files.
        
        Returns:
            Number of files deleted
        """
        count = 0
        
        for file_path in self.cache_dir.glob("*.cache"):
            file_path.unlink()
            count += 1
        
        for file_path in self.cache_dir.glob("*.meta"):
            file_path.unlink()
            count += 1
        
        return count
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of expired entries removed
        """
        count = 0
        current_time = time.time()
        
        for meta_path in self.cache_dir.glob("*.meta"):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                
                if current_time > meta['expires_at']:
                    # Expired - remove both files
                    cache_path = meta_path.with_suffix('.cache')
                    
                    meta_path.unlink(missing_ok=True)
                    cache_path.unlink(missing_ok=True)
                    count += 1
                    
            except Exception:
                # Corrupted meta file - remove it
                meta_path.unlink(missing_ok=True)
                count += 1
        
        return count