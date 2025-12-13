"""
Enhanced Reranking Service for Morgan Knowledge Engine.

This module provides advanced reranking capabilities:
- Local CrossEncoder reranking
- Remote reranking endpoint support (for distributed setups)
- Batch processing for efficiency
- Performance tracking and statistics
- Fallback mechanisms

Migrated and enhanced from morgan-rag/morgan/infrastructure/local_reranking.py
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import structlog

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


logger = structlog.get_logger(__name__)


@dataclass
class RerankResult:
    """Result from reranking operation."""
    text: str
    score: float
    original_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankStats:
    """Statistics for reranking operations."""
    total_requests: int = 0
    total_pairs: int = 0
    total_time: float = 0.0
    errors: int = 0
    remote_requests: int = 0
    local_requests: int = 0

    @property
    def average_time(self) -> float:
        """Calculate average reranking time."""
        return self.total_time / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def throughput(self) -> float:
        """Calculate throughput (pairs/sec)."""
        return self.total_pairs / self.total_time if self.total_time > 0 else 0.0


class RerankingService:
    """
    Advanced reranking service with remote and local support.

    Features:
    - Remote FastAPI reranking endpoint (primary)
    - Local CrossEncoder (fallback)
    - Batch processing with configurable size
    - Performance tracking
    - Automatic fallback on remote failure

    Example:
        >>> service = RerankingService(
        ...     remote_endpoint="http://reranking-server:8080/rerank",
        ...     model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        ... )
        >>>
        >>> # Rerank search results
        >>> results = await service.rerank(
        ...     query="What is Python?",
        ...     documents=["Python is a language", "Java is a language", ...]
        ... )
        >>>
        >>> # Get top 5 results
        >>> top_5 = results[:5]
        >>>
        >>> # Get stats
        >>> stats = service.get_stats()
    """

    def __init__(
        self,
        remote_endpoint: Optional[str] = None,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        timeout: float = 30.0,
        batch_size: int = 100,
        local_device: str = "cpu",
        enable_local_fallback: bool = True,
    ):
        """
        Initialize reranking service.

        Args:
            remote_endpoint: Remote reranking endpoint URL (optional)
            model: Model name for local CrossEncoder
            timeout: Request timeout in seconds
            batch_size: Batch size for processing
            local_device: Device for local model ("cpu", "cuda", "mps")
            enable_local_fallback: Enable local fallback if remote fails
        """
        self.remote_endpoint = remote_endpoint
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self.local_device = local_device
        self.enable_local_fallback = enable_local_fallback

        # Initialize local model (lazy)
        self._local_model = None
        self._local_available = None

        # Statistics
        self.stats = RerankStats()

        # HTTP session for remote requests
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info(
            "reranking_service_initialized",
            remote_endpoint=remote_endpoint,
            model=model,
            local_fallback=enable_local_fallback,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        if self.remote_endpoint and AIOHTTP_AVAILABLE:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None

    async def is_available(self) -> bool:
        """
        Check if reranking service is available.

        Returns:
            True if remote or local reranking available
        """
        # Check remote first
        if self.remote_endpoint and AIOHTTP_AVAILABLE:
            try:
                health_url = f"{self.remote_endpoint.rstrip('/rerank')}/health"
                
                if self._session is None:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                            if response.status == 200:
                                logger.info("remote_reranking_available")
                                return True
                else:
                    async with self._session.get(health_url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                        if response.status == 200:
                            logger.info("remote_reranking_available")
                            return True
            except Exception as e:
                logger.warning("remote_reranking_unavailable", error=str(e))

        # Check local fallback
        if self.enable_local_fallback and self._check_local_available():
            logger.info("local_reranking_available")
            return True

        logger.error("no_reranking_available")
        return False

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RerankResult]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Return top K results (None = return all)
            metadata: Optional metadata for each document

        Returns:
            List of RerankResult sorted by score (descending)

        Raises:
            RuntimeError: If no reranking service is available
        """
        self.stats.total_requests += 1
        self.stats.total_pairs += len(documents)

        if not documents:
            return []

        start_time = time.time()

        try:
            # Try remote first
            if self.remote_endpoint and AIOHTTP_AVAILABLE:
                try:
                    results = await self._rerank_remote(query, documents, top_k, metadata)
                    self.stats.remote_requests += 1
                except Exception as e:
                    logger.warning("remote_reranking_failed", error=str(e))
                    
                    # Fallback to local if enabled
                    if self.enable_local_fallback and self._check_local_available():
                        logger.info("falling_back_to_local_reranking")
                        results = self._rerank_local(query, documents, top_k, metadata)
                        self.stats.local_requests += 1
                    else:
                        raise
            # Use local if no remote endpoint
            elif self.enable_local_fallback and self._check_local_available():
                results = self._rerank_local(query, documents, top_k, metadata)
                self.stats.local_requests += 1
            else:
                self.stats.errors += 1
                raise RuntimeError("No reranking service available")

            # Update stats
            elapsed = time.time() - start_time
            self.stats.total_time += elapsed

            logger.info(
                "reranking_complete",
                num_documents=len(documents),
                num_results=len(results),
                elapsed=f"{elapsed:.3f}s",
                throughput=f"{len(documents)/elapsed:.1f} docs/sec",
            )

            return results

        except Exception as e:
            self.stats.errors += 1
            logger.error("reranking_failed", error=str(e))
            raise

    async def _rerank_remote(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RerankResult]:
        """Rerank using remote endpoint."""
        payload = {
            "query": query,
            "documents": documents,
        }

        if top_k is not None:
            payload["top_k"] = top_k

        try:
            if self._session is None:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.remote_endpoint,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
            else:
                async with self._session.post(
                    self.remote_endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

            # Parse results
            results = []
            for item in data.get("results", []):
                result_metadata = metadata[item["index"]] if metadata else {}
                results.append(
                    RerankResult(
                        text=item["text"],
                        score=item["score"],
                        original_index=item["index"],
                        metadata=result_metadata,
                    )
                )

            return results

        except Exception as e:
            logger.error("remote_reranking_error", error=str(e))
            raise

    def _check_local_available(self) -> bool:
        """Check if local CrossEncoder is available."""
        if self._local_available is not None:
            return self._local_available

        if not CROSS_ENCODER_AVAILABLE:
            logger.warning("sentence_transformers_not_installed")
            self._local_available = False
            return False

        try:
            logger.info("loading_local_reranking_model", model=self.model)
            self._local_model = CrossEncoder(self.model, device=self.local_device)
            logger.info("local_reranking_model_loaded")
            self._local_available = True
            return True
        except Exception as e:
            logger.error("local_model_load_failed", error=str(e))
            self._local_available = False
            return False

    def _rerank_local(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RerankResult]:
        """Rerank using local CrossEncoder."""
        if self._local_model is None:
            raise RuntimeError("Local model not initialized")

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score all pairs
        scores = self._local_model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Create results with original indices
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            result_metadata = metadata[i] if metadata else {}
            results.append(
                RerankResult(
                    text=doc,
                    score=float(score),
                    original_index=i,
                    metadata=result_metadata,
                )
            )

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        # Return top K if specified
        if top_k is not None:
            results = results[:top_k]

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_requests": self.stats.total_requests,
            "total_pairs": self.stats.total_pairs,
            "total_time": f"{self.stats.total_time:.2f}s",
            "average_time": f"{self.stats.average_time:.3f}s",
            "throughput": f"{self.stats.throughput:.1f} pairs/sec",
            "errors": self.stats.errors,
            "remote_requests": self.stats.remote_requests,
            "local_requests": self.stats.local_requests,
            "remote_endpoint": self.remote_endpoint,
            "model": self.model,
            "local_fallback_enabled": self.enable_local_fallback,
        }


# Global instance for singleton pattern
_service: Optional[RerankingService] = None


def get_reranking_service(
    remote_endpoint: Optional[str] = None,
    **kwargs
) -> RerankingService:
    """
    Get global reranking service instance (singleton).

    Args:
        remote_endpoint: Remote endpoint URL (optional)
        **kwargs: Additional arguments for RerankingService

    Returns:
        RerankingService instance
    """
    global _service

    if _service is None:
        _service = RerankingService(remote_endpoint=remote_endpoint, **kwargs)

    return _service
