"""
Model Cache - Intelligent model caching and optimization

Provides caching for loaded models to improve performance and reduce memory
usage. Follows KISS principles with simple, effective caching strategies.

Requirements addressed: 23.1, 23.4, 23.5
"""

from typing import Dict, Any, Optional
import logging
import time
import threading
from dataclasses import dataclass
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Simple cache entry with metadata."""
    model: Any
    timestamp: float
    access_count: int
    size_mb: float


class ModelCache:
    """
    Model cache following KISS principles.

    Single responsibility: Cache loaded models for performance.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # 2GB default
        self.max_cache_size_mb = config.get('max_cache_size_mb', 2048)
        self.max_entries = config.get('max_entries', 10)
        self.ttl_seconds = config.get('ttl_seconds', 3600)  # 1 hour default

        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._current_size_mb = 0.0

        # Optional persistent cache
        self.cache_dir = Path(config.get('cache_dir', './data/model_cache'))
        self.persistent = config.get('persistent', False)

        if self.persistent:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_persistent_cache()

    def store_model(self, model_name: str, model: Any) -> bool:
        """
        Store a model in cache.

        Args:
            model_name: Unique identifier for the model
            model: The model instance to cache

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            with self._lock:
                # Estimate model size (rough approximation)
                model_size = self._estimate_model_size(model)

                # Check if we need to evict models
                if self._needs_eviction(model_size):
                    self._evict_models(model_size)

                # Store the model
                entry = CacheEntry(
                    model=model,
                    timestamp=time.time(),
                    access_count=1,
                    size_mb=model_size
                )

                self._cache[model_name] = entry
                self._current_size_mb += model_size

                logger.info("Cached model %s (%.1fMB)", model_name, model_size)

                # Optionally persist to disk
                if self.persistent:
                    self._persist_model(model_name, model)

                return True

        except Exception as e:
            logger.error("Failed to cache model %s: %s", model_name, e)
            return False

    def get_model(self, model_name: str) -> Optional[Any]:
        """
        Retrieve a model from cache.

        Args:
            model_name: Unique identifier for the model

        Returns:
            Cached model instance or None if not found
        """
        try:
            with self._lock:
                entry = self._cache.get(model_name)

                if entry is None:
                    # Try loading from persistent cache
                    if self.persistent:
                        model = self._load_persistent_model(model_name)
                        if model:
                            self.store_model(model_name, model)
                            return model
                    return None

                # Check TTL
                if time.time() - entry.timestamp > self.ttl_seconds:
                    self.remove_model(model_name)
                    return None

                # Update access statistics
                entry.access_count += 1
                entry.timestamp = time.time()

                logger.debug("Cache hit for model %s", model_name)
                return entry.model

        except Exception as e:
            logger.error("Failed to retrieve cached model %s: %s",
                         model_name, e)
            return None

    def remove_model(self, model_name: str) -> bool:
        """Remove a model from cache."""
        try:
            with self._lock:
                entry = self._cache.pop(model_name, None)
                if entry:
                    self._current_size_mb -= entry.size_mb
                    logger.info("Removed model %s from cache", model_name)

                    # Remove from persistent cache
                    if self.persistent:
                        self._remove_persistent_model(model_name)

                    return True
                return False

        except Exception as e:
            logger.error("Failed to remove cached model %s: %s",
                         model_name, e)
            return False

    def clear_cache(self) -> bool:
        """Clear all cached models."""
        try:
            with self._lock:
                self._cache.clear()
                self._current_size_mb = 0.0

                # Clear persistent cache
                if self.persistent:
                    for cache_file in self.cache_dir.glob("*.pkl"):
                        cache_file.unlink()

                logger.info("Cleared model cache")
                return True

        except Exception as e:
            logger.error("Failed to clear cache: %s", e)
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            utilization = (self._current_size_mb / self.max_cache_size_mb
                           if self.max_cache_size_mb > 0 else 0)
            return {
                'entries': len(self._cache),
                'size_mb': self._current_size_mb,
                'max_size_mb': self.max_cache_size_mb,
                'utilization': utilization,
                'models': list(self._cache.keys())
            }

    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB (rough approximation)."""
        try:
            # Simple size estimation based on model type
            if isinstance(model, dict):
                if 'provider' in model:
                    # Remote model client - minimal size
                    return 0.1
                else:
                    # Local model - estimate based on parameters
                    return 100.0  # Default estimate
            else:
                # Try to get actual size if possible
                return 100.0  # Default estimate

        except Exception:
            return 100.0  # Default fallback

    def _needs_eviction(self, new_model_size: float) -> bool:
        """Check if we need to evict models to make space."""
        return (
            len(self._cache) >= self.max_entries or
            self._current_size_mb + new_model_size > self.max_cache_size_mb
        )

    def _evict_models(self, required_space: float) -> None:
        """Evict models using LRU strategy."""
        # Sort by last access time (oldest first)
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].timestamp
        )

        freed_space = 0.0
        for model_name, entry in sorted_entries:
            if (freed_space >= required_space and
                    len(self._cache) < self.max_entries):
                break

            self.remove_model(model_name)
            freed_space += entry.size_mb

    def _persist_model(self, model_name: str, model: Any) -> None:
        """Persist model to disk."""
        try:
            cache_file = self.cache_dir / f"{model_name}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            logger.warning("Failed to persist model %s: %s", model_name, e)

    def _load_persistent_model(self, model_name: str) -> Optional[Any]:
        """Load model from persistent cache."""
        try:
            cache_file = self.cache_dir / f"{model_name}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning("Failed to load persistent model %s: %s",
                           model_name, e)
        return None

    def _remove_persistent_model(self, model_name: str) -> None:
        """Remove model from persistent cache."""
        try:
            cache_file = self.cache_dir / f"{model_name}.pkl"
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            logger.warning("Failed to remove persistent model %s: %s",
                           model_name, e)

    def _load_persistent_cache(self) -> None:
        """Load all models from persistent cache on startup."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                model_name = cache_file.stem
                model = self._load_persistent_model(model_name)
                if model:
                    self.store_model(model_name, model)
        except Exception as e:
            logger.warning("Failed to load persistent cache: %s", e)
