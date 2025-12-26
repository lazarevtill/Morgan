"""
Reranking Integration for Multi-Stage Search.

Provides reranking capabilities to improve search result relevance:
- Integration with LocalRerankingService
- Result score boosting based on reranking
- Configurable reranking strategies
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.infrastructure.local_reranking import (
    LocalRerankingService,
    RerankResult,
    get_local_reranking_service,
)
from morgan.infrastructure.distributed_gpu_manager import (
    HostRole,
    get_distributed_gpu_manager,
)
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RerankedSearchResult:
    """Search result with reranking information."""

    content: str
    original_score: float
    rerank_score: float
    combined_score: float
    original_index: int
    metadata: Dict[str, Any]


class SearchReranker:
    """
    Reranker for search results.
    
    100% Self-Hosted - No API Keys Required.

    Integrates with the distributed reranking service to improve
    search result relevance through cross-encoder scoring.
    
    Self-hosted models (via sentence-transformers CrossEncoder):
    - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, English (recommended)
    - cross-encoder/ms-marco-MiniLM-L-12-v2: Better quality, English
    - BAAI/bge-reranker-base: Multilingual

    Example:
        >>> reranker = SearchReranker()
        >>> 
        >>> # Rerank search results
        >>> results = [
        ...     {"content": "Python is a language", "score": 0.85},
        ...     {"content": "Java is a language", "score": 0.80},
        ... ]
        >>> reranked = await reranker.rerank(
        ...     query="What is Python?",
        ...     results=results,
        ...     top_k=5
        ... )
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        auto_discover: bool = True,
        rerank_weight: float = 0.6,
        original_weight: float = 0.4,
    ):
        """
        Initialize search reranker.

        Args:
            endpoint: Reranking endpoint URL (auto-discovered if None)
            model: CrossEncoder model name
            auto_discover: Auto-discover endpoints from GPU manager
            rerank_weight: Weight for reranking score in combined score
            original_weight: Weight for original score in combined score
        """
        self.settings = get_settings()
        self._endpoint = endpoint
        self._model = model
        self._auto_discover = auto_discover
        self._rerank_weight = rerank_weight
        self._original_weight = original_weight

        # Service instance (lazy initialization)
        self._reranking_service: Optional[LocalRerankingService] = None
        self._initialized = False

        # Statistics
        self._stats = {
            "total_reranks": 0,
            "total_documents": 0,
            "total_time": 0.0,
            "errors": 0,
        }

        logger.info(
            f"SearchReranker created: model={model}, "
            f"weights=({rerank_weight}, {original_weight})"
        )

    async def _ensure_initialized(self):
        """Ensure reranking service is initialized."""
        if self._initialized:
            return

        endpoint = self._endpoint

        # Auto-discover endpoints if needed
        if endpoint is None and self._auto_discover:
            try:
                gpu_manager = get_distributed_gpu_manager()
                endpoints = await gpu_manager.get_endpoints(HostRole.RERANKING)
                if endpoints:
                    # Reranking endpoints might have different paths
                    endpoint = endpoints[0].rstrip("/v1") + "/rerank"
                    logger.info(f"Auto-discovered reranking endpoint: {endpoint}")
            except Exception as e:
                logger.warning(f"Failed to auto-discover reranking endpoint: {e}")

        # Fallback to settings
        if not endpoint:
            endpoint = getattr(
                self.settings, "reranking_endpoint", None
            )

        # Initialize reranking service
        self._reranking_service = LocalRerankingService(
            endpoint=endpoint,
            model=self._model,
        )

        self._initialized = True
        logger.info("SearchReranker initialized")

    async def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        content_key: str = "content",
        score_key: str = "score",
    ) -> List[RerankedSearchResult]:
        """
        Rerank search results.

        Args:
            query: Search query
            results: List of search results with content and scores
            top_k: Return top K results (None = return all)
            content_key: Key for content in result dicts
            score_key: Key for score in result dicts

        Returns:
            List of reranked results sorted by combined score
        """
        if not results:
            return []

        await self._ensure_initialized()

        start_time = time.time()

        try:
            # Extract documents
            documents = [r.get(content_key, "") for r in results]
            original_scores = [r.get(score_key, 0.0) for r in results]

            # Get reranking scores
            rerank_results = await self._reranking_service.rerank(
                query=query,
                documents=documents,
                top_k=None,  # Get all scores first
            )

            # Create mapping of original index to rerank score
            rerank_scores = {r.original_index: r.score for r in rerank_results}

            # Normalize rerank scores to 0-1 range
            if rerank_scores:
                max_score = max(rerank_scores.values())
                min_score = min(rerank_scores.values())
                score_range = max_score - min_score if max_score != min_score else 1.0

                for idx in rerank_scores:
                    rerank_scores[idx] = (rerank_scores[idx] - min_score) / score_range

            # Create reranked results
            reranked = []
            for i, result in enumerate(results):
                rerank_score = rerank_scores.get(i, 0.0)
                original_score = original_scores[i]

                # Calculate combined score
                combined_score = (
                    self._rerank_weight * rerank_score +
                    self._original_weight * original_score
                )

                reranked.append(RerankedSearchResult(
                    content=result.get(content_key, ""),
                    original_score=original_score,
                    rerank_score=rerank_score,
                    combined_score=combined_score,
                    original_index=i,
                    metadata={k: v for k, v in result.items() if k not in [content_key, score_key]},
                ))

            # Sort by combined score
            reranked.sort(key=lambda r: r.combined_score, reverse=True)

            # Apply top_k
            if top_k is not None:
                reranked = reranked[:top_k]

            # Update stats
            elapsed = time.time() - start_time
            self._stats["total_reranks"] += 1
            self._stats["total_documents"] += len(results)
            self._stats["total_time"] += elapsed

            logger.debug(
                f"Reranked {len(results)} results in {elapsed:.3f}s"
            )

            return reranked

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Reranking failed: {e}")
            # Fall back to original order
            return [
                RerankedSearchResult(
                    content=r.get(content_key, ""),
                    original_score=r.get(score_key, 0.0),
                    rerank_score=0.0,
                    combined_score=r.get(score_key, 0.0),
                    original_index=i,
                    metadata={k: v for k, v in r.items() if k not in [content_key, score_key]},
                )
                for i, r in enumerate(results)
            ]

    def is_available(self) -> bool:
        """Check if reranking service is available."""
        if self._reranking_service:
            return self._reranking_service.is_available()
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics."""
        avg_time = (
            self._stats["total_time"] / self._stats["total_reranks"]
            if self._stats["total_reranks"] > 0
            else 0.0
        )
        avg_docs = (
            self._stats["total_documents"] / self._stats["total_reranks"]
            if self._stats["total_reranks"] > 0
            else 0
        )

        return {
            **self._stats,
            "average_time": f"{avg_time:.3f}s",
            "average_documents": avg_docs,
            "service_stats": self._reranking_service.get_stats() if self._reranking_service else {},
        }


# Singleton instance
_reranker: Optional[SearchReranker] = None


def get_search_reranker() -> SearchReranker:
    """Get singleton search reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = SearchReranker()
    return _reranker

