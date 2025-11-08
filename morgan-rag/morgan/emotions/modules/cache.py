"""
Emotion Cache Module.

Provides caching for emotion detection results to improve performance.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from morgan.emotions.base import AsyncCache, EmotionModule
from morgan.emotions.exceptions import EmotionCacheError
from morgan.emotions.types import EmotionResult


class EmotionCache(EmotionModule):
    """
    Caches emotion detection results.

    Improves performance by caching results for identical inputs.
    Uses content-based hashing for cache keys.
    """

    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: float = 3600.0,
    ) -> None:
        """
        Initialize emotion cache.

        Args:
            max_size: Maximum cache entries
            default_ttl: Default time-to-live in seconds (1 hour default)
        """
        super().__init__("EmotionCache")
        self._cache: AsyncCache[EmotionResult] = AsyncCache(max_size, default_ttl)

    async def initialize(self) -> None:
        """Initialize cache."""
        pass

    async def cleanup(self) -> None:
        """Cleanup cache."""
        await self._cache.clear()

    async def get(
        self,
        text: str,
        context_key: Optional[str] = None,
    ) -> Optional[EmotionResult]:
        """
        Get cached result.

        Args:
            text: Input text
            context_key: Optional context identifier

        Returns:
            Cached result if available, None otherwise

        Raises:
            EmotionCacheError: If cache retrieval fails
        """
        await self.ensure_initialized()

        try:
            cache_key = self._generate_key(text, context_key)
            return await self._cache.get(cache_key)

        except Exception as e:
            raise EmotionCacheError(f"Failed to get from cache: {str(e)}", cause=e)

    async def set(
        self,
        text: str,
        result: EmotionResult,
        context_key: Optional[str] = None,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Cache result.

        Args:
            text: Input text
            result: Emotion result to cache
            context_key: Optional context identifier
            ttl: Optional time-to-live override

        Raises:
            EmotionCacheError: If caching fails
        """
        await self.ensure_initialized()

        try:
            cache_key = self._generate_key(text, context_key)
            await self._cache.set(cache_key, result, ttl)

        except Exception as e:
            raise EmotionCacheError(f"Failed to set cache: {str(e)}", cause=e)

    async def invalidate(
        self,
        text: str,
        context_key: Optional[str] = None,
    ) -> None:
        """
        Invalidate cached result.

        Args:
            text: Input text
            context_key: Optional context identifier
        """
        await self.ensure_initialized()

        cache_key = self._generate_key(text, context_key)
        await self._cache.invalidate(cache_key)

    async def clear_all(self) -> None:
        """Clear entire cache."""
        await self._cache.clear()

    def _generate_key(self, text: str, context_key: Optional[str]) -> str:
        """
        Generate cache key.

        Uses SHA256 hash of text + context for content-based caching.
        """
        content = text
        if context_key:
            content = f"{text}::{context_key}"

        return hashlib.sha256(content.encode()).hexdigest()
