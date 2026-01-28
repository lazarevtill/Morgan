"""
Search Service Wrapper for Morgan RAG.

Provides a unified search interface that wraps the multi-stage search engine.
This module exists for backward compatibility and convenient imports.

Example:
    >>> from morgan.search.service import get_search_service
    >>>
    >>> service = get_search_service()
    >>> results = service.search("What is Python?")
"""

import threading
from typing import Any, Dict, List, Optional

from morgan.utils.logger import get_logger

from .multi_stage_search import (
    MultiStageSearchEngine,
    SearchResult,
    SearchResults,
    SearchStrategy,
    get_multi_stage_search_engine,
)

logger = get_logger(__name__)


class SearchService:
    """
    Unified search service wrapping the multi-stage search engine.

    Provides a simplified interface for common search operations while
    delegating to the full-featured MultiStageSearchEngine underneath.

    Features:
        - Simple search interface
        - Automatic strategy selection
        - Result caching
        - Health checking

    Example:
        >>> service = SearchService()
        >>> results = service.search("machine learning basics")
        >>> for result in results.results:
        ...     print(f"{result.score:.2f}: {result.content[:50]}...")
    """

    def __init__(
        self,
        engine: Optional[MultiStageSearchEngine] = None,
        default_strategies: Optional[List[SearchStrategy]] = None,
        default_max_results: int = 10,
    ):
        """
        Initialize search service.

        Args:
            engine: Optional pre-configured search engine
            default_strategies: Default search strategies
            default_max_results: Default number of results to return
        """
        self._engine = engine
        self._engine_lock = threading.Lock()
        self.default_strategies = default_strategies or [SearchStrategy.SEMANTIC]
        self.default_max_results = default_max_results

        logger.info("SearchService initialized")

    def _get_engine(self) -> MultiStageSearchEngine:
        """Get or create the search engine instance."""
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    self._engine = get_multi_stage_search_engine()
        return self._engine

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        strategies: Optional[List[SearchStrategy]] = None,
        min_score: float = 0.7,
        use_hierarchical: bool = True,
        conversation_context: Optional[str] = None,
        emotional_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> SearchResults:
        """
        Perform a search query.

        Args:
            query: Search query text
            max_results: Number of results to return
            strategies: Search strategies to use
            min_score: Minimum relevance score
            use_hierarchical: Whether to use hierarchical search
            conversation_context: Optional conversation context
            emotional_context: Optional emotional context
            user_id: Optional user ID

        Returns:
            SearchResults with matching documents
        """
        engine = self._get_engine()

        max_results = max_results or self.default_max_results
        strategies = strategies or self.default_strategies

        try:
            results = engine.search(
                query=query,
                max_results=max_results,
                strategies=strategies,
                min_score=min_score,
                use_hierarchical=use_hierarchical,
                conversation_context=conversation_context,
                emotional_context=emotional_context,
                user_id=user_id,
            )

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return SearchResults(results=[], total_candidates=0)

    def search_with_reranking(
        self,
        query: str,
        max_results: Optional[int] = None,
        rerank_candidates: Optional[int] = None,
    ) -> SearchResults:
        """
        Perform search with reranking for improved relevance.

        Args:
            query: Search query text
            max_results: Number of final results
            rerank_candidates: Number of candidates to rerank

        Returns:
            Reranked search results
        """
        engine = self._get_engine()

        max_results = max_results or self.default_max_results
        rerank_candidates = rerank_candidates or (max_results * 3)

        try:
            # Get more candidates for reranking using combined strategies
            results = engine.search(
                query=query,
                max_results=rerank_candidates,
                strategies=[SearchStrategy.SEMANTIC, SearchStrategy.KEYWORD],
            )

            if not results or not results.results:
                return SearchResults(results=[], total_candidates=0)

            # Rerank results if engine has reranking capability
            if hasattr(engine, "rerank_results"):
                reranked = engine.rerank_results(
                    query=query,
                    results=results.results,
                    top_k=max_results,
                )
                return SearchResults(
                    results=reranked[:max_results],
                    total_candidates=results.total_candidates,
                    strategies_used=results.strategies_used,
                    search_time=results.search_time,
                )

            # Return top results if no reranking available
            return SearchResults(
                results=results.results[:max_results],
                total_candidates=results.total_candidates,
                strategies_used=results.strategies_used,
                search_time=results.search_time,
            )

        except Exception as e:
            logger.error(f"Search with reranking failed: {e}")
            return SearchResults(results=[], total_candidates=0)

    def semantic_search(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> SearchResults:
        """
        Perform pure semantic (vector) search.

        Args:
            query: Search query text
            max_results: Number of results

        Returns:
            Semantic search results
        """
        return self.search(
            query=query,
            max_results=max_results,
            strategies=[SearchStrategy.SEMANTIC],
        )

    def keyword_search(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> SearchResults:
        """
        Perform keyword-based search.

        Args:
            query: Search query text
            max_results: Number of results

        Returns:
            Keyword search results
        """
        return self.search(
            query=query,
            max_results=max_results,
            strategies=[SearchStrategy.KEYWORD],
        )

    def is_ready(self) -> bool:
        """
        Check if search service is ready.

        Returns:
            True if service is operational
        """
        try:
            engine = self._get_engine()
            return engine is not None
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get search service statistics.

        Returns:
            Dictionary with service statistics
        """
        try:
            engine = self._get_engine()
            if hasattr(engine, "get_stats"):
                return engine.get_stats()
            return {"status": "operational", "engine_loaded": True}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton instance
_service_instance: Optional[SearchService] = None
_service_lock = threading.Lock()


def get_search_service() -> SearchService:
    """
    Get singleton search service instance.

    Returns:
        Shared SearchService instance
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = SearchService()

    return _service_instance


# Re-export for convenience
__all__ = [
    "SearchService",
    "get_search_service",
    "SearchResult",
    "SearchResults",
    "SearchStrategy",
]
