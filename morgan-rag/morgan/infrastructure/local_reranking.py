"""
Local Reranking Service for Distributed Morgan Setup

Provides reranking via:
- Remote reranking endpoint (Host 6 - RTX 2060)
- Local CrossEncoder fallback
- Batch processing for efficiency
- Performance tracking

Designed for 6-host distributed architecture where reranking runs on
dedicated host (Host 6 with RTX 2060).

Model weights are cached locally to avoid re-downloading on each startup.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
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


def setup_reranker_cache(cache_dir: Optional[str] = None):
    """
    Setup model cache directories and environment variables for rerankers.

    This ensures models are downloaded once and reused on subsequent starts.
    Also configures HF_TOKEN for downloading gated models if available.

    Args:
        cache_dir: Base directory for model cache. Defaults to ~/.morgan/models

    Environment variables loaded from .env:
        - HF_TOKEN: Hugging Face API token (for gated models)
        - HUGGING_FACE_HUB_TOKEN: Alternative HF token variable
        - MORGAN_MODEL_CACHE: Override default cache directory
    """
    # Try to load .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv

        load_dotenv()
        logger.debug("Loaded environment from .env file")
    except ImportError:
        pass  # dotenv not installed, use existing env vars

    if cache_dir is None:
        cache_dir = os.environ.get("MORGAN_MODEL_CACHE", "~/.morgan/models")

    cache_path = Path(cache_dir).expanduser()

    # Create subdirectories
    sentence_transformers_path = cache_path / "sentence-transformers"
    hf_path = cache_path / "huggingface"

    for path in [cache_path, sentence_transformers_path, hf_path]:
        path.mkdir(parents=True, exist_ok=True)

    # Set environment variables for model caching
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(sentence_transformers_path)
    os.environ["HF_HOME"] = str(hf_path)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_path)

    # Configure HF_TOKEN for gated model downloads
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        logger.info("HF_TOKEN configured for authenticated model downloads")
    else:
        logger.debug("No HF_TOKEN found - some gated models may not be accessible")

    logger.info("Reranker model cache configured at %s", cache_path)
    return cache_path


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
        if self.total_requests > 0:
            return self.total_time / self.total_requests
        return 0.0

    @property
    def throughput(self) -> float:
        """Calculate throughput (pairs/sec)."""
        if self.total_time > 0:
            return self.total_pairs / self.total_time
        return 0.0


class LocalRerankingService:
    """
    Local reranking service for distributed Morgan setup.

    100% Self-Hosted - No API Keys Required.

    Supports:
    - Remote FastAPI reranking endpoint (primary)
    - Local CrossEncoder via sentence-transformers (fallback)
    - Batch processing with configurable size
    - Performance tracking

    Self-hosted models (via sentence-transformers CrossEncoder):
    - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, English (recommended)
    - cross-encoder/ms-marco-MiniLM-L-12-v2: Better quality, English
    - BAAI/bge-reranker-base: Multilingual

    Example:
        >>> service = LocalRerankingService(
        ...     endpoint="http://192.168.1.23:8080/rerank",
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
        endpoint: Optional[str] = None,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        timeout: float = 30.0,
        batch_size: int = 100,
        local_device: str = "cpu",
        model_cache_dir: Optional[str] = None,
        preload_model: bool = False,
    ):
        """
        Initialize local reranking service.

        Args:
            endpoint: Reranking endpoint URL
            model: Model name for local CrossEncoder
            timeout: Request timeout in seconds
            batch_size: Batch size for processing
            local_device: Device for local model ("cpu" or "cuda")
            model_cache_dir: Directory for model weights cache
            preload_model: If True, load model immediately
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests package not available, remote reranking disabled")

        # Setup model cache directory before any model loading
        self.model_cache_path = setup_reranker_cache(model_cache_dir)

        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self.local_device = local_device

        # Initialize local model (lazy by default)
        self._local_model = None
        self._local_available = None

        # Statistics
        self.stats = RerankStats()

        logger.info(
            "LocalRerankingService initialized: endpoint=%s, model=%s, " "cache_dir=%s",
            endpoint,
            model,
            self.model_cache_path,
        )

        # Preload model if requested (downloads weights on first run)
        if preload_model and CROSS_ENCODER_AVAILABLE:
            logger.info("Preloading reranking model %s...", model)
            self._check_local_available()

    def is_available(self) -> bool:
        """
        Check if reranking service is available.

        Returns:
            True if remote or local reranking available
        """
        # Check remote first
        if self.endpoint and REQUESTS_AVAILABLE:
            try:
                response = requests.get(
                    f"{self.endpoint.rstrip('/rerank')}/health", timeout=5.0
                )
                if response.status_code == 200:
                    logger.info("Remote reranking service available")
                    return True
            except Exception as e:
                logger.warning(f"Remote reranking service not available: {e}")

        # Check local fallback
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
            elif self._check_local_available():
                results = self._rerank_local(query, documents, top_k)
            else:
                self.stats.errors += 1
                raise RuntimeError("No reranking service available")

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
            self._local_model = CrossEncoder(
                self.model,
                device=self.local_device,
            )
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
        }


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
