"""
Precomputed Search Results System

Popular query identification and caching system.
Background reranking with result precomputation.
"""

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Simple search result structure."""

    content: str
    source: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class PrecomputedResult:
    """Precomputed search result with metadata."""

    query_text: str
    query_hash: str
    collection_name: str
    results: List[SearchResult]
    rerank_scores: List[float]
    computed_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    quality_score: float = 0.0
    language: str = "en"


@dataclass
class QueryAnalytics:
    """Query usage analytics."""

    query_text: str
    query_hash: str
    access_count: int
    last_accessed: datetime
    first_accessed: datetime
    collections: List[str]
    average_response_time: float = 0.0


class PrecomputedSearchCache:
    """
    Simple precomputed search results system.

    Features:
    - Popular query identification and caching
    - Background reranking with result precomputation
    - Cache warming for frequently accessed content
    - Quality assessment for precomputed results
    - Cache invalidation and refresh mechanisms
    """

    def __init__(
        self,
        vector_db_client=None,
        reranking_service=None,
        cache_ttl_hours: int = 24,
        max_cached_queries: int = 1000,
    ):
        """
        Initialize precomputed search cache.

        Args:
            vector_db_client: Vector database client
            reranking_service: Reranking service
            cache_ttl_hours: Cache time-to-live in hours
            max_cached_queries: Maximum number of cached queries
        """
        self.vector_db_client = vector_db_client
        self.reranking_service = reranking_service
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.max_cached_queries = max_cached_queries

        # In-memory cache (in production, would use Redis or similar)
        self.precomputed_cache: Dict[str, PrecomputedResult] = {}
        self.query_analytics: Dict[str, QueryAnalytics] = {}

        # Popular query tracking
        self.query_frequency = Counter()
        self.recent_queries = []  # Rolling window of recent queries

        self.logger = logging.getLogger(__name__)

    def track_query(
        self, query: str, collection_name: str, response_time: float = 0.0
    ) -> str:
        """
        Track a query for popularity analysis.

        Args:
            query: Query text
            collection_name: Collection searched
            response_time: Query response time in seconds

        Returns:
            Query hash for tracking
        """
        query_hash = self._hash_query(query)
        now = datetime.now()

        # Update frequency counter
        self.query_frequency[query_hash] += 1

        # Add to recent queries (keep last 1000)
        self.recent_queries.append(
            {
                "query_hash": query_hash,
                "query_text": query,
                "collection": collection_name,
                "timestamp": now,
                "response_time": response_time,
            }
        )

        if len(self.recent_queries) > 1000:
            self.recent_queries = self.recent_queries[-1000:]

        # Update analytics
        if query_hash in self.query_analytics:
            analytics = self.query_analytics[query_hash]
            analytics.access_count += 1
            analytics.last_accessed = now
            if collection_name not in analytics.collections:
                analytics.collections.append(collection_name)

            # Update average response time
            total_time = analytics.average_response_time * (analytics.access_count - 1)
            analytics.average_response_time = (
                total_time + response_time
            ) / analytics.access_count
        else:
            self.query_analytics[query_hash] = QueryAnalytics(
                query_text=query,
                query_hash=query_hash,
                access_count=1,
                last_accessed=now,
                first_accessed=now,
                collections=[collection_name],
                average_response_time=response_time,
            )

        self.logger.debug(f"Tracked query: '{query}' (hash: {query_hash[:8]}...)")
        return query_hash

    def get_popular_queries(
        self,
        collection_name: Optional[str] = None,
        limit: int = 20,
        min_frequency: int = 3,
    ) -> List[QueryAnalytics]:
        """
        Get popular queries for precomputation.

        Args:
            collection_name: Filter by collection (None for all)
            limit: Maximum number of queries to return
            min_frequency: Minimum access count to be considered popular

        Returns:
            List of popular query analytics
        """
        # Filter analytics by criteria
        popular = []

        for analytics in self.query_analytics.values():
            if analytics.access_count < min_frequency:
                continue

            if collection_name and collection_name not in analytics.collections:
                continue

            popular.append(analytics)

        # Sort by access count (descending)
        popular.sort(key=lambda x: x.access_count, reverse=True)

        self.logger.info(f"Found {len(popular)} popular queries (limit: {limit})")
        return popular[:limit]

    def precompute_query_results(
        self, query: str, collection_name: str, force_refresh: bool = False
    ) -> Optional[PrecomputedResult]:
        """
        Precompute and cache search results for a query.

        Args:
            query: Query text
            collection_name: Collection to search
            force_refresh: Force recomputation even if cached

        Returns:
            Precomputed result or None if failed
        """
        query_hash = self._hash_query(query)
        cache_key = f"{query_hash}_{collection_name}"

        # Check if already cached and not expired
        if not force_refresh and cache_key in self.precomputed_cache:
            cached = self.precomputed_cache[cache_key]
            if datetime.now() - cached.computed_at < self.cache_ttl:
                self.logger.debug(f"Using cached result for query: '{query}'")
                return cached

        try:
            self.logger.info(
                f"Precomputing results for query: '{query}' in {collection_name}"
            )

            # Simulate search execution (in real implementation, would use actual search)
            search_results = self._execute_search(query, collection_name)

            if not search_results:
                self.logger.warning(f"No results found for query: '{query}'")
                return None

            # Apply reranking if available
            rerank_scores = []
            if self.reranking_service:
                reranked_results, rerank_scores = self._apply_reranking(
                    query, search_results
                )
                search_results = reranked_results

            # Assess quality
            quality_score = self._assess_result_quality(search_results, rerank_scores)

            # Create precomputed result
            precomputed = PrecomputedResult(
                query_text=query,
                query_hash=query_hash,
                collection_name=collection_name,
                results=search_results,
                rerank_scores=rerank_scores,
                computed_at=datetime.now(),
                quality_score=quality_score,
            )

            # Cache the result
            self.precomputed_cache[cache_key] = precomputed

            # Manage cache size
            self._manage_cache_size()

            self.logger.info(
                f"Precomputed {len(search_results)} results for '{query}' "
                f"(quality: {quality_score:.3f})"
            )

            return precomputed

        except Exception as e:
            self.logger.error(f"Failed to precompute results for '{query}': {e}")
            return None

    def get_cached_results(
        self, query: str, collection_name: str
    ) -> Optional[PrecomputedResult]:
        """
        Get cached results for a query.

        Args:
            query: Query text
            collection_name: Collection name

        Returns:
            Cached results or None if not found/expired
        """
        query_hash = self._hash_query(query)
        cache_key = f"{query_hash}_{collection_name}"

        if cache_key not in self.precomputed_cache:
            return None

        cached = self.precomputed_cache[cache_key]

        # Check if expired
        if datetime.now() - cached.computed_at > self.cache_ttl:
            self.logger.debug(f"Cached result expired for query: '{query}'")
            del self.precomputed_cache[cache_key]
            return None

        # Update access tracking
        cached.access_count += 1
        cached.last_accessed = datetime.now()

        self.logger.debug(f"Retrieved cached result for query: '{query}'")
        return cached

    def warm_cache(self, collection_name: str, max_queries: int = 50) -> int:
        """
        Warm cache with popular queries.

        Args:
            collection_name: Collection to warm cache for
            max_queries: Maximum queries to precompute

        Returns:
            Number of queries precomputed
        """
        self.logger.info(f"Warming cache for collection: {collection_name}")

        # Get popular queries for this collection
        popular_queries = self.get_popular_queries(
            collection_name=collection_name,
            limit=max_queries,
            min_frequency=2,  # Lower threshold for cache warming
        )

        precomputed_count = 0

        for analytics in popular_queries:
            try:
                result = self.precompute_query_results(
                    analytics.query_text,
                    collection_name,
                    force_refresh=False,  # Don't refresh if already cached
                )

                if result:
                    precomputed_count += 1

            except Exception as e:
                self.logger.warning(
                    f"Failed to precompute '{analytics.query_text}': {e}"
                )
                continue

        self.logger.info(
            f"Cache warming completed: {precomputed_count} queries precomputed"
        )
        return precomputed_count

    def invalidate_cache(
        self, collection_name: Optional[str] = None, query_pattern: Optional[str] = None
    ) -> int:
        """
        Invalidate cached results.

        Args:
            collection_name: Invalidate for specific collection (None for all)
            query_pattern: Invalidate queries matching pattern (None for all)

        Returns:
            Number of cache entries invalidated
        """
        keys_to_remove = []

        for cache_key, cached_result in self.precomputed_cache.items():
            should_remove = True

            if collection_name and cached_result.collection_name != collection_name:
                should_remove = False

            if (
                query_pattern
                and query_pattern.lower() not in cached_result.query_text.lower()
            ):
                should_remove = False

            if should_remove:
                keys_to_remove.append(cache_key)

        # Remove invalidated entries
        for key in keys_to_remove:
            del self.precomputed_cache[key]

        self.logger.info(f"Invalidated {len(keys_to_remove)} cache entries")
        return len(keys_to_remove)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        now = datetime.now()

        # Count valid (non-expired) entries
        valid_entries = 0
        expired_entries = 0
        total_access_count = 0

        for cached in self.precomputed_cache.values():
            if now - cached.computed_at < self.cache_ttl:
                valid_entries += 1
                total_access_count += cached.access_count
            else:
                expired_entries += 1

        # Calculate hit rate (simplified)
        total_queries = len(self.query_analytics)
        cache_hit_rate = valid_entries / total_queries if total_queries > 0 else 0.0

        return {
            "total_cached_queries": len(self.precomputed_cache),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "total_tracked_queries": total_queries,
            "cache_hit_rate": cache_hit_rate,
            "total_access_count": total_access_count,
            "cache_size_mb": self._estimate_cache_size_mb(),
        }

    def _hash_query(self, query: str) -> str:
        """Generate hash for query."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()

    def _execute_search(
        self, query: str, collection_name: str, max_results: int = 10
    ) -> List[SearchResult]:
        """
        Execute search (simulation for now).

        In real implementation, would use actual search service.
        """
        # Simulate search results
        results = []

        for i in range(min(max_results, 5)):  # Simulate 5 results
            result = SearchResult(
                content=f"Sample content {i+1} for query '{query}'",
                source=f"document_{i+1}.md",
                score=0.9 - (i * 0.1),
                metadata={
                    "collection": collection_name,
                    "timestamp": datetime.now().isoformat(),
                    "chunk_id": f"chunk_{i+1}",
                },
            )
            results.append(result)

        return results

    def _apply_reranking(
        self, query: str, results: List[SearchResult]
    ) -> tuple[List[SearchResult], List[float]]:
        """
        Apply reranking to search results.

        Returns:
            Tuple of (reranked_results, rerank_scores)
        """
        # Simulate reranking (in real implementation, would use actual reranking service)
        import random

        rerank_scores = [random.uniform(0.7, 0.95) for _ in results]

        # Sort results by rerank scores
        combined = list(zip(results, rerank_scores))
        combined.sort(key=lambda x: x[1], reverse=True)

        reranked_results = [item[0] for item in combined]
        sorted_scores = [item[1] for item in combined]

        return reranked_results, sorted_scores

    def _assess_result_quality(
        self, results: List[SearchResult], rerank_scores: List[float]
    ) -> float:
        """
        Assess quality of search results.

        Returns:
            Quality score (0.0 to 1.0)
        """
        if not results:
            return 0.0

        # Simple quality assessment based on scores
        avg_search_score = sum(r.score for r in results) / len(results)
        avg_rerank_score = (
            sum(rerank_scores) / len(rerank_scores)
            if rerank_scores
            else avg_search_score
        )

        # Combine scores with slight weight on reranking
        quality = (avg_search_score * 0.4) + (avg_rerank_score * 0.6)

        return min(1.0, max(0.0, quality))

    def _manage_cache_size(self):
        """Manage cache size by removing least recently used entries."""
        if len(self.precomputed_cache) <= self.max_cached_queries:
            return

        # Sort by last accessed (oldest first)
        cache_items = list(self.precomputed_cache.items())
        cache_items.sort(key=lambda x: x[1].last_accessed or datetime.min)

        # Remove oldest entries
        entries_to_remove = len(cache_items) - self.max_cached_queries

        for i in range(entries_to_remove):
            cache_key = cache_items[i][0]
            del self.precomputed_cache[cache_key]

        self.logger.debug(f"Removed {entries_to_remove} old cache entries")

    def _estimate_cache_size_mb(self) -> float:
        """Estimate cache size in MB (rough approximation)."""
        if not self.precomputed_cache:
            return 0.0

        # Rough estimate: 1KB per result, 10 results per query average
        estimated_size_bytes = len(self.precomputed_cache) * 10 * 1024
        return estimated_size_bytes / (1024 * 1024)  # Convert to MB
