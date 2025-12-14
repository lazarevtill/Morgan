"""
Local Reranking Service for Distributed Morgan Setup

Provides reranking via:
- Remote reranking endpoint (Host 6 - RTX 2060)
- Local CrossEncoder fallback
- Batch processing for efficiency
- Performance tracking

Designed for 6-host distributed architecture where reranking runs on
dedicated host (Host 6 with RTX 2060).
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

from morgan.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RerankResult:
    """Result from reranking operation."""

    text: str
    score: float
    original_index: int


@dataclass
class RerankStats:
    """Statistics for reranking operations."""

    total_requests: int = 0
    total_pairs: int = 0
    total_time: float = 0.0
    errors: int = 0

    @property
    def average_time(self) -> float:
        """Calculate average reranking time."""
        return self.total_time / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def throughput(self) -> float:
        """Calculate throughput (pairs/sec)."""
        return self.total_pairs / self.total_time if self.total_time > 0 else 0.0


class LocalRerankingService:
    """
    Local reranking service for distributed Morgan setup.

    Supports:
    - Remote FastAPI reranking endpoint (primary)
    - Local CrossEncoder (fallback)
    - Batch processing with configurable size
    - Performance tracking

    Example:
        >>> service = LocalRerankingService(
        ...     endpoint="http://192.168.1.23:8080/rerank"
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
        endpoint: Optional[str] = None,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        timeout: float = 30.0,
        batch_size: int = 100,
        local_device: str = "cpu",
        force_remote: bool = False,
    ):
        """
        Initialize local reranking service.

        Args:
            endpoint: Reranking endpoint URL (e.g., "http://host6:8080/rerank")
            model: Model name for local CrossEncoder
            timeout: Request timeout in seconds
            batch_size: Batch size for processing
            local_device: Device for local model ("cpu" or "cuda")
            force_remote: Require remote reranking; disable local fallback
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests package not available, remote reranking disabled")

        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self.local_device = local_device
        self.force_remote = force_remote

        # Initialize local model (lazy)
        self._local_model = None
        self._local_available = None

        # Statistics
        self.stats = RerankStats()

        logger.info(
            f"LocalRerankingService initialized: " f"endpoint={endpoint}, model={model}"
        )

    def is_available(self) -> bool:
        """
        Check if reranking service is available.

        Returns:
            True if remote or local reranking available
        """
        # Check remote first
        if self.endpoint and REQUESTS_AVAILABLE:
            try:
                response = requests.get(self._health_url(), timeout=5.0)
                if response.status_code == 200:
                    logger.info("Remote reranking service available")
                    return True
            except Exception as e:
                logger.warning(f"Remote reranking service not available: {e}")

        if self.force_remote:
            logger.error("Reranking forced to remote but endpoint unavailable")
            return False

        # Check local fallback (only if allowed)
        if self._check_local_available():
            logger.info("Local reranking service available")
            return True

        logger.error("No reranking service available")
        return False

    async def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Return top K results (None = return all)

        Returns:
            List of RerankResult sorted by score (descending)
        """
        self.stats.total_requests += 1
        self.stats.total_pairs += len(documents)

        if not documents:
            return []

        start_time = time.time()

        try:
            # Try remote first
            if self.endpoint and REQUESTS_AVAILABLE:
                results = await self._rerank_remote(query, documents, top_k)
            # Fallback to local
            elif not self.force_remote and self._check_local_available():
                results = self._rerank_local(query, documents, top_k)
            else:
                self.stats.errors += 1
                raise RuntimeError("No reranking service available (remote unreachable)")

            # Update stats
            elapsed = time.time() - start_time
            self.stats.total_time += elapsed

            logger.info(
                f"Reranked {len(documents)} documents in {elapsed:.3f}s "
                f"({len(documents)/elapsed:.1f} docs/sec)"
            )

            return results

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Reranking failed: {e}")
            raise

    async def _rerank_remote(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Rerank using remote endpoint."""
        payload = {"query": query, "documents": documents}

        if top_k is not None:
            payload["top_k"] = top_k

        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            # Parse results
            results = []
            for item in data.get("results", []):
                results.append(
                    RerankResult(
                        text=item["text"],
                        score=item["score"],
                        original_index=item["index"],
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Remote reranking failed: {e}")
            raise

    def _check_local_available(self) -> bool:
        """Check if local CrossEncoder is available."""
        if self._local_available is not None:
            return self._local_available

        if not CROSS_ENCODER_AVAILABLE:
            logger.warning("sentence-transformers not installed")
            self._local_available = False
            return False

        try:
            logger.info(f"Loading local reranking model ({self.model})...")
            self._local_model = CrossEncoder(self.model, device=self.local_device)
            logger.info("Local reranking model loaded successfully")
            self._local_available = True
            return True
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            self._local_available = False
            return False

    def _rerank_local(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Rerank using local CrossEncoder."""
        if self._local_model is None:
            raise RuntimeError("Local model not initialized")

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score all pairs
        scores = self._local_model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        # Create results with original indices
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append(RerankResult(text=doc, score=float(score), original_index=i))

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
            "endpoint": self.endpoint,
            "model": self.model,
            "force_remote": self.force_remote,
        }

    def _health_url(self) -> str:
        """Build health check URL for remote reranker."""
        if not self.endpoint:
            return ""
        base = self.endpoint.rstrip("/")
        if base.endswith("/rerank"):
            base = base[: -len("/rerank")]
        return f"{base}/health"


# Global instance for singleton pattern
_service: Optional[LocalRerankingService] = None


def get_local_reranking_service(
    endpoint: Optional[str] = None, **kwargs
) -> LocalRerankingService:
    """
    Get global local reranking service instance (singleton).

    Args:
        endpoint: Endpoint URL (optional)
        **kwargs: Additional arguments for LocalRerankingService

    Returns:
        LocalRerankingService instance
    """
    global _service

    if _service is None:
        _service = LocalRerankingService(endpoint=endpoint, **kwargs)

    return _service
