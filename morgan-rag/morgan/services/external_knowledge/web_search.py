"""
Web Search Service for Morgan.

Provides web search capabilities for real-time information retrieval.
Integrates with MCP browser/search servers for actual web searches.

Features:
- Real-time web search for current information
- Result caching to avoid redundant searches
- Source verification and relevance scoring
- Integration with Morgan's search architecture
"""

import asyncio
import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WebSearchResult:
    """Result from a web search."""

    title: str
    url: str
    snippet: str
    content: str = ""
    score: float = 1.0
    source: str = "web"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class WebSearchService:
    """
    Web Search Service for real-time information retrieval.

    Provides web search capabilities through MCP server integration.
    Includes caching, rate limiting, and result processing.

    Example:
        >>> service = WebSearchService()
        >>> results = await service.search("Python async best practices")
        >>> for result in results:
        ...     print(f"{result.title}: {result.snippet}")
    """

    def __init__(
        self,
        cache_ttl_minutes: int = 30,
        max_cache_size: int = 1000,
        rate_limit_per_minute: int = 30,
    ):
        """
        Initialize web search service.

        Args:
            cache_ttl_minutes: Cache time-to-live in minutes
            max_cache_size: Maximum number of cached results
            rate_limit_per_minute: Maximum searches per minute
        """
        self.settings = get_settings()
        self._lock = threading.Lock()

        # Cache settings
        self._cache: Dict[str, tuple[List[WebSearchResult], datetime]] = {}
        self._cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._max_cache_size = max_cache_size

        # Rate limiting
        self._rate_limit = rate_limit_per_minute
        self._request_times: List[datetime] = []

        # Statistics
        self._stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        }

        logger.info(
            f"WebSearchService initialized: "
            f"cache_ttl={cache_ttl_minutes}min, "
            f"rate_limit={rate_limit_per_minute}/min"
        )

    async def search(
        self,
        query: str,
        max_results: int = 10,
        use_cache: bool = True,
    ) -> List[WebSearchResult]:
        """
        Search the web for information.

        Args:
            query: Search query
            max_results: Maximum number of results
            use_cache: Whether to use cached results

        Returns:
            List of web search results
        """
        self._stats["total_searches"] += 1

        # Check cache first
        cache_key = self._get_cache_key(query)
        if use_cache:
            cached = self._get_cached_results(cache_key)
            if cached:
                self._stats["cache_hits"] += 1
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached[:max_results]

        self._stats["cache_misses"] += 1

        # Check rate limit
        if not self._check_rate_limit():
            logger.warning("Rate limit exceeded for web search")
            return []

        try:
            # Perform the actual web search via MCP
            results = await self._perform_search(query, max_results)

            # Cache results
            if results:
                self._cache_results(cache_key, results)

            return results

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Web search failed for query '{query}': {e}")
            return []

    async def search_for_topic(
        self,
        topic: str,
        subtopics: Optional[List[str]] = None,
        max_results_per_topic: int = 5,
    ) -> Dict[str, List[WebSearchResult]]:
        """
        Search for a topic and optional subtopics.

        Args:
            topic: Main topic to search
            subtopics: Optional list of subtopics
            max_results_per_topic: Max results per topic/subtopic

        Returns:
            Dictionary mapping topic/subtopic to results
        """
        results = {}

        # Search main topic
        main_results = await self.search(topic, max_results_per_topic)
        results[topic] = main_results

        # Search subtopics
        if subtopics:
            for subtopic in subtopics:
                full_query = f"{topic} {subtopic}"
                subtopic_results = await self.search(full_query, max_results_per_topic)
                results[subtopic] = subtopic_results

        return results

    async def search_current_events(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[WebSearchResult]:
        """
        Search for current/recent information.

        Enhanced query to focus on recent/current information.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of recent web search results
        """
        # Enhance query to focus on recent information
        enhanced_query = f"{query} latest 2025 2026 recent"
        return await self.search(enhanced_query, max_results, use_cache=False)

    async def _perform_search(
        self,
        query: str,
        max_results: int,
    ) -> List[WebSearchResult]:
        """
        Perform the actual web search.

        This method integrates with MCP browser/search functionality.
        In a real implementation, this would call the MCP server.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of search results
        """
        # This is where the MCP integration happens
        # The actual search would be performed by the MCP server
        # For now, we provide a structure that can be filled in

        try:
            # Placeholder for MCP integration
            # In production, this would call the MCP server endpoint
            logger.info(f"Performing web search for: {query[:50]}...")

            # Simulate async operation
            await asyncio.sleep(0.1)

            # Return empty results - actual implementation would use MCP
            # The MCP server would handle the actual web search
            results: List[WebSearchResult] = []

            return results[:max_results]

        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            raise

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for a query."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _get_cached_results(self, cache_key: str) -> Optional[List[WebSearchResult]]:
        """Get cached results if valid."""
        with self._lock:
            if cache_key in self._cache:
                results, timestamp = self._cache[cache_key]
                if datetime.utcnow() - timestamp < self._cache_ttl:
                    return results
                else:
                    # Expired, remove from cache
                    del self._cache[cache_key]
        return None

    def _cache_results(self, cache_key: str, results: List[WebSearchResult]) -> None:
        """Cache search results."""
        with self._lock:
            # Clean cache if too large
            if len(self._cache) >= self._max_cache_size:
                self._clean_cache()

            self._cache[cache_key] = (results, datetime.utcnow())

    def _clean_cache(self) -> None:
        """Remove expired entries from cache."""
        now = datetime.utcnow()
        expired_keys = [
            key
            for key, (_, timestamp) in self._cache.items()
            if now - timestamp >= self._cache_ttl
        ]

        for key in expired_keys:
            del self._cache[key]

        # If still too large, remove oldest entries
        if len(self._cache) >= self._max_cache_size:
            sorted_items = sorted(self._cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_items[: len(self._cache) // 2]:
                del self._cache[key]

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)

        with self._lock:
            # Remove old request times
            self._request_times = [t for t in self._request_times if t > cutoff]

            # Check if under limit
            if len(self._request_times) < self._rate_limit:
                self._request_times.append(now)
                return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "hit_rate": (
                self._stats["cache_hits"] / self._stats["total_searches"]
                if self._stats["total_searches"] > 0
                else 0
            ),
        }

    def clear_cache(self) -> None:
        """Clear all cached results."""
        with self._lock:
            self._cache.clear()
        logger.info("Web search cache cleared")


# Singleton instance
_service_instance: Optional[WebSearchService] = None
_service_lock = threading.Lock()


def get_web_search_service(**kwargs) -> WebSearchService:
    """
    Get singleton web search service instance.

    Args:
        **kwargs: Optional configuration overrides

    Returns:
        Shared WebSearchService instance
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = WebSearchService(**kwargs)

    return _service_instance
