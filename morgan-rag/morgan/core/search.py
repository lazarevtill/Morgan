"""Production-quality multi-stage RAG search with reciprocal rank fusion."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from morgan.jina.reranking.service import RerankingService, RerankingConfig
from morgan.services.embedding_service import EmbeddingService, EmbeddingConfig
from morgan.vector_db.client import QdrantClient, QdrantConfig

logger = logging.getLogger(__name__)


class SearchGranularity(Enum):
    """Search granularity levels."""
    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"


@dataclass
class SearchConfig:
    """Multi-stage search configuration."""
    # Collection names for hierarchical search
    coarse_collection: str = "documents_coarse"
    medium_collection: str = "documents_medium"
    fine_collection: str = "documents_fine"

    # Search parameters per stage
    coarse_top_k: int = 50
    medium_top_k: int = 30
    fine_top_k: int = 20

    # Final result count
    final_top_k: int = 10

    # Score thresholds
    min_score_threshold: float = 0.3

    # Reciprocal Rank Fusion (RRF) parameter
    rrf_k: int = 60

    # Reranking
    enable_reranking: bool = True
    rerank_top_k: int = 50
    rerank_score_weight: float = 0.7  # Weight for reranking vs vector search

    # Performance
    max_concurrent_searches: int = 3
    search_timeout: float = 30.0

    # Circuit breaker
    max_failures: int = 5
    failure_timeout: float = 60.0


@dataclass
class SearchResult:
    """Search result with metadata."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source_stage: str
    rank: int


@dataclass
class SearchMetrics:
    """Search performance metrics."""
    query: str
    total_duration_ms: float
    stages_duration_ms: Dict[str, float] = field(default_factory=dict)
    total_results: int = 0
    reranked: bool = False
    cache_hit: bool = False


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion for combining search results."""

    def __init__(self, k: int = 60):
        """Initialize RRF.

        Args:
            k: RRF constant (typically 60).
        """
        self.k = k

    def fuse(
        self,
        result_lists: List[List[Tuple[str, float]]],
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """Fuse multiple ranked lists using RRF.

        Args:
            result_lists: List of ranked result lists, each containing (id, score) tuples.
            weights: Optional weights for each list.

        Returns:
            Fused and ranked results.
        """
        if not result_lists:
            return []

        if weights is None:
            weights = [1.0] * len(result_lists)

        if len(weights) != len(result_lists):
            raise ValueError("Number of weights must match number of result lists")

        # Calculate RRF scores
        rrf_scores: Dict[str, float] = {}

        for result_list, weight in zip(result_lists, weights):
            for rank, (doc_id, _) in enumerate(result_list, start=1):
                rrf_score = weight / (self.k + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score

        # Sort by RRF score
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        logger.debug(
            "RRF fusion completed",
            extra={
                "input_lists": len(result_lists),
                "unique_docs": len(fused),
            }
        )

        return fused


class MultiStageSearch:
    """Production-quality multi-stage RAG search with hierarchical retrieval.

    Features:
    - Hierarchical search (coarse -> medium -> fine)
    - Reciprocal Rank Fusion for result merging
    - Optional reranking with cross-encoder
    - Async concurrent search across stages
    - Proper timeout handling
    - Circuit breaker for resilience
    - Performance metrics tracking
    - Resource cleanup
    """

    def __init__(
        self,
        vector_db: QdrantClient,
        embedding_service: EmbeddingService,
        reranking_service: Optional[RerankingService] = None,
        config: Optional[SearchConfig] = None,
    ):
        """Initialize multi-stage search.

        Args:
            vector_db: Qdrant vector database client.
            embedding_service: Embedding generation service.
            reranking_service: Optional reranking service.
            config: Search configuration.
        """
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.reranking_service = reranking_service
        self.config = config or SearchConfig()

        self._rrf = ReciprocalRankFusion(k=self.config.rrf_k)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_searches)
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None

        logger.info(
            "Initializing multi-stage search",
            extra={
                "coarse_top_k": self.config.coarse_top_k,
                "medium_top_k": self.config.medium_top_k,
                "fine_top_k": self.config.fine_top_k,
                "final_top_k": self.config.final_top_k,
                "reranking": self.config.enable_reranking,
            }
        )

    async def _check_circuit_breaker(self) -> None:
        """Check circuit breaker status.

        Raises:
            RuntimeError: If circuit is open.
        """
        if self._failure_count >= self.config.max_failures:
            if self._last_failure_time:
                time_since_failure = time.time() - self._last_failure_time
                if time_since_failure < self.config.failure_timeout:
                    raise RuntimeError(
                        f"Circuit breaker open. Too many failures. "
                        f"Retry in {self.config.failure_timeout - time_since_failure:.1f}s"
                    )
                else:
                    # Reset circuit breaker
                    logger.info("Circuit breaker reset")
                    self._failure_count = 0
                    self._last_failure_time = None

    async def _search_collection(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
        stage_name: str,
    ) -> Tuple[List[SearchResult], float]:
        """Search a single collection.

        Args:
            collection_name: Collection to search.
            query_vector: Query embedding.
            top_k: Number of results.
            stage_name: Stage identifier for logging.

        Returns:
            Tuple of (results, duration_ms).
        """
        start_time = time.time()

        try:
            async with self._semaphore:
                points = await asyncio.wait_for(
                    self.vector_db.search(
                        collection_name=collection_name,
                        query_vector=query_vector,
                        limit=top_k,
                        score_threshold=self.config.min_score_threshold,
                    ),
                    timeout=self.config.search_timeout,
                )

            # Convert to SearchResult objects
            results = [
                SearchResult(
                    chunk_id=point.id,
                    document_id=point.payload.get("document_id", ""),
                    content=point.payload.get("content", ""),
                    score=point.score,
                    metadata=point.payload.get("metadata", {}),
                    source_stage=stage_name,
                    rank=idx,
                )
                for idx, point in enumerate(points, start=1)
            ]

            duration_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"{stage_name} search completed",
                extra={
                    "collection": collection_name,
                    "results": len(results),
                    "duration_ms": round(duration_ms, 2),
                }
            )

            return results, duration_ms

        except asyncio.TimeoutError:
            logger.error(f"{stage_name} search timeout")
            self._failure_count += 1
            self._last_failure_time = time.time()
            return [], (time.time() - start_time) * 1000

        except Exception as e:
            logger.error(
                f"{stage_name} search failed",
                extra={"error": str(e)}
            )
            self._failure_count += 1
            self._last_failure_time = time.time()
            return [], (time.time() - start_time) * 1000

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        enable_reranking: Optional[bool] = None,
    ) -> Tuple[List[SearchResult], SearchMetrics]:
        """Execute multi-stage hierarchical search.

        Args:
            query: Search query.
            top_k: Number of final results. Defaults to config.final_top_k.
            enable_reranking: Override config reranking setting.

        Returns:
            Tuple of (results, metrics).
        """
        start_time = time.time()
        await self._check_circuit_breaker()

        top_k = top_k or self.config.final_top_k
        enable_reranking = enable_reranking if enable_reranking is not None else self.config.enable_reranking

        metrics = SearchMetrics(query=query, total_duration_ms=0)

        try:
            # Generate query embedding
            embed_start = time.time()
            query_vector = await self.embedding_service.embed(query)
            embed_duration = (time.time() - embed_start) * 1000
            metrics.stages_duration_ms["embedding"] = embed_duration

            # Execute hierarchical search in parallel
            search_tasks = [
                self._search_collection(
                    self.config.coarse_collection,
                    query_vector.tolist(),
                    self.config.coarse_top_k,
                    "coarse"
                ),
                self._search_collection(
                    self.config.medium_collection,
                    query_vector.tolist(),
                    self.config.medium_top_k,
                    "medium"
                ),
                self._search_collection(
                    self.config.fine_collection,
                    query_vector.tolist(),
                    self.config.fine_top_k,
                    "fine"
                ),
            ]

            search_results = await asyncio.gather(*search_tasks)

            coarse_results, coarse_duration = search_results[0]
            medium_results, medium_duration = search_results[1]
            fine_results, fine_duration = search_results[2]

            metrics.stages_duration_ms["coarse"] = coarse_duration
            metrics.stages_duration_ms["medium"] = medium_duration
            metrics.stages_duration_ms["fine"] = fine_duration

            # Prepare result lists for RRF
            result_lists = [
                [(r.chunk_id, r.score) for r in coarse_results],
                [(r.chunk_id, r.score) for r in medium_results],
                [(r.chunk_id, r.score) for r in fine_results],
            ]

            # Apply different weights to different granularities
            weights = [0.3, 0.4, 0.3]  # Medium gets slightly higher weight

            # Fuse results
            fuse_start = time.time()
            fused_scores = self._rrf.fuse(result_lists, weights)
            fuse_duration = (time.time() - fuse_start) * 1000
            metrics.stages_duration_ms["fusion"] = fuse_duration

            # Create unified result map
            all_results = {r.chunk_id: r for r in coarse_results + medium_results + fine_results}

            # Build fused results
            fused_results = []
            for chunk_id, rrf_score in fused_scores:
                if chunk_id in all_results:
                    result = all_results[chunk_id]
                    result.score = rrf_score  # Replace with RRF score
                    fused_results.append(result)

            # Limit to top candidates for reranking
            if enable_reranking and self.reranking_service and len(fused_results) > 0:
                rerank_start = time.time()

                # Take top N for reranking
                rerank_candidates = fused_results[:self.config.rerank_top_k]

                # Prepare for reranking
                documents_with_scores = [
                    (r.content, r.score) for r in rerank_candidates
                ]

                # Rerank
                reranked = await self.reranking_service.rerank_with_scores(
                    query=query,
                    documents=documents_with_scores,
                    top_k=top_k,
                    score_weight=self.config.rerank_score_weight,
                )

                # Map back to SearchResult objects
                final_results = []
                for rerank_result in reranked:
                    original = rerank_candidates[rerank_result.index]
                    original.score = rerank_result.score
                    original.rank = len(final_results) + 1
                    final_results.append(original)

                rerank_duration = (time.time() - rerank_start) * 1000
                metrics.stages_duration_ms["reranking"] = rerank_duration
                metrics.reranked = True

            else:
                # No reranking, just take top K
                final_results = fused_results[:top_k]
                for idx, result in enumerate(final_results, start=1):
                    result.rank = idx

            # Update metrics
            metrics.total_results = len(final_results)
            metrics.total_duration_ms = (time.time() - start_time) * 1000

            # Reset failure count on success
            self._failure_count = 0

            logger.info(
                "Search completed successfully",
                extra={
                    "query_length": len(query),
                    "results": len(final_results),
                    "duration_ms": round(metrics.total_duration_ms, 2),
                    "reranked": metrics.reranked,
                }
            )

            return final_results, metrics

        except Exception as e:
            self._failure_count += 1
            self._last_failure_time = time.time()

            metrics.total_duration_ms = (time.time() - start_time) * 1000

            logger.error(
                "Search failed",
                extra={
                    "query": query,
                    "error": str(e),
                    "duration_ms": round(metrics.total_duration_ms, 2),
                }
            )

            return [], metrics

    async def search_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[List[SearchResult], SearchMetrics]]:
        """Execute batch search for multiple queries.

        Args:
            queries: List of search queries.
            top_k: Number of results per query.

        Returns:
            List of (results, metrics) tuples.
        """
        if not queries:
            return []

        start_time = time.time()

        # Process all queries concurrently
        tasks = [self.search(query, top_k=top_k) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Batch search failed for query",
                    extra={
                        "query_index": i,
                        "error": str(result),
                    }
                )
                processed_results.append(([], SearchMetrics(query=queries[i], total_duration_ms=0)))
            else:
                processed_results.append(result)

        duration = time.time() - start_time

        logger.info(
            "Batch search completed",
            extra={
                "queries": len(queries),
                "duration_s": round(duration, 2),
            }
        )

        return processed_results

    async def search_with_filter(
        self,
        query: str,
        metadata_filter: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> Tuple[List[SearchResult], SearchMetrics]:
        """Search with metadata filtering.

        Args:
            query: Search query.
            metadata_filter: Metadata filter conditions.
            top_k: Number of results.

        Returns:
            Tuple of (results, metrics).
        """
        # This is a simplified version - in production you'd want to implement
        # proper Qdrant filter syntax
        logger.warning("Metadata filtering not fully implemented in this example")

        # For now, just do regular search and filter in memory
        results, metrics = await self.search(query, top_k=None)

        # Apply filter
        filtered_results = [
            r for r in results
            if all(r.metadata.get(k) == v for k, v in metadata_filter.items())
        ]

        # Limit to top_k
        top_k = top_k or self.config.final_top_k
        filtered_results = filtered_results[:top_k]

        logger.info(
            "Filtered search completed",
            extra={
                "total_results": len(results),
                "filtered_results": len(filtered_results),
                "filter": metadata_filter,
            }
        )

        return filtered_results, metrics

    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "config": {
                "coarse_top_k": self.config.coarse_top_k,
                "medium_top_k": self.config.medium_top_k,
                "fine_top_k": self.config.fine_top_k,
                "final_top_k": self.config.final_top_k,
                "reranking_enabled": self.config.enable_reranking,
            },
            "circuit_breaker": {
                "failure_count": self._failure_count,
                "max_failures": self.config.max_failures,
                "is_open": self._failure_count >= self.config.max_failures,
            },
        }
