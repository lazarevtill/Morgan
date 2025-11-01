"""
Git hash tracking system for intelligent caching.

This module provides Git hash calculation and storage for document collections,
enabling efficient cache invalidation based on content changes.
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

from .cache_models import CacheStatus, CollectionCacheInfo

logger = logging.getLogger(__name__)


class GitHashTracker:
    """
    Git hash tracking system for document collections.
    
    Tracks Git hashes of document collections to enable intelligent caching
    with automatic invalidation when content changes.
    """
    
    def __init__(self, cache_dir: Path):
        """
        Initialize Git hash tracker.
        
        Args:
            cache_dir: Directory to store hash tracking data
        """
        self.cache_dir = Path(cache_dir)
        self.hash_file = self.cache_dir / "git_hashes.json"
        self.collections_file = self.cache_dir / "collections.json"
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage files
        self._init_storage_files()
    
    def _init_storage_files(self) -> None:
        """Initialize storage files if they don't exist."""
        if not self.hash_file.exists():
            with open(self.hash_file, 'w') as f:
                json.dump({}, f)
        
        if not self.collections_file.exists():
            with open(self.collections_file, 'w') as f:
                json.dump({}, f)
    
    def calculate_git_hash(self, source_path: str) -> Optional[str]:
        """
        Calculate Git hash for a source path.
        
        Args:
            source_path: Path to calculate hash for
            
        Returns:
            Git hash string or None if not a Git repository
        """
        try:
            source_path = Path(source_path).resolve()
            
            # Check if path is in a Git repository
            if not self._is_git_repository(source_path):
                logger.debug(f"Path {source_path} is not in a Git repository")
                return self._calculate_file_hash(source_path)
            
            # Get Git hash for the path
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=source_path if source_path.is_dir() else source_path.parent,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                git_hash = result.stdout.strip()
                logger.debug(f"Git hash for {source_path}: {git_hash}")
                return git_hash
            else:
                logger.warning(f"Failed to get Git hash for {source_path}: {result.stderr}")
                return self._calculate_file_hash(source_path)
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Git command timed out for {source_path}")
            return self._calculate_file_hash(source_path)
        except Exception as e:
            logger.warning(f"Error calculating Git hash for {source_path}: {e}")
            return self._calculate_file_hash(source_path)
    
    def _is_git_repository(self, path: Path) -> bool:
        """Check if path is in a Git repository."""
        try:
            current_path = path if path.is_dir() else path.parent
            
            while current_path != current_path.parent:
                if (current_path / '.git').exists():
                    return True
                current_path = current_path.parent
            
            return False
        except Exception:
            return False
    
    def _calculate_file_hash(self, source_path: Path) -> str:
        """
        Calculate hash based on file content and modification times.
        
        Fallback when Git is not available.
        """
        try:
            hasher = hashlib.sha256()
            
            if source_path.is_file():
                # Single file - hash content and mtime
                with open(source_path, 'rb') as f:
                    hasher.update(f.read())
                hasher.update(str(source_path.stat().st_mtime).encode())
            elif source_path.is_dir():
                # Directory - hash all files recursively
                for file_path in sorted(source_path.rglob('*')):
                    if file_path.is_file():
                        try:
                            with open(file_path, 'rb') as f:
                                hasher.update(f.read())
                            hasher.update(str(file_path.stat().st_mtime).encode())
                        except (OSError, IOError):
                            # Skip files that can't be read
                            continue
            else:
                # Path doesn't exist
                hasher.update(b'nonexistent')
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.warning(f"Error calculating file hash for {source_path}: {e}")
            return hashlib.sha256(str(source_path).encode()).hexdigest()
    
    def store_git_hash(
        self, 
        source_path: str, 
        collection_name: str, 
        git_hash: str,
        document_count: int = 0,
        size_bytes: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store Git hash for a collection.
        
        Args:
            source_path: Source path of the collection
            collection_name: Name of the collection
            git_hash: Git hash to store
            document_count: Number of documents in collection
            size_bytes: Size of collection in bytes
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            # Load existing hashes
            with open(self.hash_file, 'r') as f:
                hashes = json.load(f)
            
            # Store hash
            hashes[collection_name] = {
                'source_path': str(source_path),
                'git_hash': git_hash,
                'last_updated': datetime.now().isoformat(),
                'document_count': document_count,
                'size_bytes': size_bytes
            }
            
            # Save updated hashes
            with open(self.hash_file, 'w') as f:
                json.dump(hashes, f, indent=2)
            
            # Store collection info
            self._store_collection_info(
                collection_name, source_path, git_hash, 
                document_count, size_bytes, metadata or {}
            )
            
            logger.debug(f"Stored Git hash for collection {collection_name}: {git_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing Git hash for {collection_name}: {e}")
            return False
    
    def _store_collection_info(
        self,
        collection_name: str,
        source_path: str, 
        git_hash: str,
        document_count: int,
        size_bytes: int,
        metadata: Dict[str, Any]
    ) -> None:
        """Store detailed collection information."""
        try:
            with open(self.collections_file, 'r') as f:
                collections = json.load(f)
            
            now = datetime.now().isoformat()
            
            collections[collection_name] = {
                'source_path': str(source_path),
                'git_hash': git_hash,
                'document_count': document_count,
                'created_at': collections.get(collection_name, {}).get('created_at', now),
                'last_accessed': now,
                'size_bytes': size_bytes,
                'metadata': metadata
            }
            
            with open(self.collections_file, 'w') as f:
                json.dump(collections, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error storing collection info for {collection_name}: {e}")
    
    def get_stored_hash(self, collection_name: str) -> Optional[str]:
        """
        Get stored Git hash for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Stored Git hash or None if not found
        """
        try:
            with open(self.hash_file, 'r') as f:
                hashes = json.load(f)
            
            collection_data = hashes.get(collection_name)
            if collection_data:
                return collection_data['git_hash']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting stored hash for {collection_name}: {e}")
            return None
    
    def check_cache_validity(
        self, 
        source_path: str, 
        collection_name: str
    ) -> CacheStatus:
        """
        Check if cache is valid by comparing Git hashes.
        
        Args:
            source_path: Source path to check
            collection_name: Name of the collection
            
        Returns:
            CacheStatus with validity information
        """
        try:
            # Get current hash
            current_hash = self.calculate_git_hash(source_path)
            if not current_hash:
                return CacheStatus(
                    is_valid=False,
                    stored_hash=None,
                    current_hash="",
                    last_updated=datetime.now(),
                    collection_exists=False
                )
            
            # Get stored hash
            stored_hash = self.get_stored_hash(collection_name)
            
            # Check if collection exists
            collection_exists = self._collection_exists(collection_name)
            
            # Determine validity
            is_valid = (
                stored_hash is not None and 
                stored_hash == current_hash and 
                collection_exists
            )
            
            # Get last updated time
            last_updated = self._get_last_updated(collection_name)
            
            return CacheStatus(
                is_valid=is_valid,
                stored_hash=stored_hash,
                current_hash=current_hash,
                last_updated=last_updated,
                collection_exists=collection_exists,
                cache_hit=is_valid
            )
            
        except Exception as e:
            logger.error(f"Error checking cache validity for {collection_name}: {e}")
            return CacheStatus(
                is_valid=False,
                stored_hash=None,
                current_hash="",
                last_updated=datetime.now(),
                collection_exists=False
            )
    
    def _collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists in storage."""
        try:
            with open(self.collections_file, 'r') as f:
                collections = json.load(f)
            return collection_name in collections
        except Exception:
            return False
    
    def _get_last_updated(self, collection_name: str) -> datetime:
        """Get last updated time for collection."""
        try:
            with open(self.hash_file, 'r') as f:
                hashes = json.load(f)
            
            collection_data = hashes.get(collection_name)
            if collection_data and 'last_updated' in collection_data:
                return datetime.fromisoformat(collection_data['last_updated'])
            
            return datetime.now()
            
        except Exception:
            return datetime.now()
    
    def get_changed_files(self, source_path: str, since_hash: str) -> List[str]:
        """
        Get list of files changed since a specific Git hash.
        
        Args:
            source_path: Source path to check
            since_hash: Git hash to compare against
            
        Returns:
            List of changed file paths
        """
        try:
            source_path = Path(source_path).resolve()
            
            if not self._is_git_repository(source_path):
                logger.debug(f"Path {source_path} is not in a Git repository")
                return []
            
            # Get changed files using git diff
            result = subprocess.run(
                ['git', 'diff', '--name-only', since_hash, 'HEAD'],
                cwd=source_path if source_path.is_dir() else source_path.parent,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                changed_files = [
                    line.strip() for line in result.stdout.split('\n') 
                    if line.strip()
                ]
                logger.debug(f"Changed files since {since_hash}: {changed_files}")
                return changed_files
            else:
                logger.warning(f"Failed to get changed files: {result.stderr}")
                return []
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Git diff command timed out for {source_path}")
            return []
        except Exception as e:
            logger.warning(f"Error getting changed files for {source_path}: {e}")
            return []
    
    def invalidate_cache(self, collection_name: str) -> bool:
        """
        Invalidate cache for a collection.
        
        Args:
            collection_name: Name of the collection to invalidate
            
        Returns:
            True if successful
        """
        try:
            # Remove from hashes
            with open(self.hash_file, 'r') as f:
                hashes = json.load(f)
            
            if collection_name in hashes:
                del hashes[collection_name]
                
                with open(self.hash_file, 'w') as f:
                    json.dump(hashes, f, indent=2)
            
            # Remove from collections
            with open(self.collections_file, 'r') as f:
                collections = json.load(f)
            
            if collection_name in collections:
                del collections[collection_name]
                
                with open(self.collections_file, 'w') as f:
                    json.dump(collections, f, indent=2)
            
            logger.info(f"Invalidated cache for collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating cache for {collection_name}: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[CollectionCacheInfo]:
        """
        Get detailed information about a cached collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            CollectionCacheInfo or None if not found
        """
        try:
            with open(self.collections_file, 'r') as f:
                collections = json.load(f)
            
            collection_data = collections.get(collection_name)
            if not collection_data:
                return None
            
            return CollectionCacheInfo(
                collection_name=collection_name,
                source_path=collection_data['source_path'],
                git_hash=collection_data['git_hash'],
                document_count=collection_data.get('document_count', 0),
                created_at=datetime.fromisoformat(collection_data['created_at']),
                last_accessed=datetime.fromisoformat(collection_data['last_accessed']),
                size_bytes=collection_data.get('size_bytes', 0),
                metadata=collection_data.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {e}")
            return None
    
    def list_collections(self) -> List[CollectionCacheInfo]:
        """
        List all cached collections.
        
        Returns:
            List of CollectionCacheInfo objects
        """
        try:
            with open(self.collections_file, 'r') as f:
                collections = json.load(f)
            
            result = []
            for collection_name, collection_data in collections.items():
                try:
                    info = CollectionCacheInfo(
                        collection_name=collection_name,
                        source_path=collection_data['source_path'],
                        git_hash=collection_data['git_hash'],
                        document_count=collection_data.get('document_count', 0),
                        created_at=datetime.fromisoformat(collection_data['created_at']),
                        last_accessed=datetime.fromisoformat(collection_data['last_accessed']),
                        size_bytes=collection_data.get('size_bytes', 0),
                        metadata=collection_data.get('metadata', {})
                    )
                    result.append(info)
                except Exception as e:
                    logger.warning(f"Error parsing collection {collection_name}: {e}")
                    continue
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def cleanup_orphaned_entries(self) -> int:
        """
        Clean up orphaned cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        try:
            cleaned_count = 0
            
            # Load current data
            with open(self.hash_file, 'r') as f:
                hashes = json.load(f)
            
            with open(self.collections_file, 'r') as f:
                collections = json.load(f)
            
            # Find orphaned entries
            hash_keys = set(hashes.keys())
            collection_keys = set(collections.keys())
            
            # Remove entries that exist in one but not the other
            orphaned_in_hashes = hash_keys - collection_keys
            orphaned_in_collections = collection_keys - hash_keys
            
            for key in orphaned_in_hashes:
                del hashes[key]
                cleaned_count += 1
            
            for key in orphaned_in_collections:
                del collections[key]
                cleaned_count += 1
            
            # Save cleaned data
            if cleaned_count > 0:
                with open(self.hash_file, 'w') as f:
                    json.dump(hashes, f, indent=2)
                
                with open(self.collections_file, 'w') as f:
                    json.dump(collections, f, indent=2)
                
                logger.info(f"Cleaned up {cleaned_count} orphaned cache entries")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up orphaned entries: {e}")
            return 0