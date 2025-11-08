"""
Local Embedding Service for Distributed Morgan Setup

Provides embeddings via:
- Remote Ollama/OpenAI-compatible endpoints (Host 5 - RTX 4070)
- Local sentence-transformers fallback
- Batch processing for efficiency
- Caching for performance

Designed for 6-host distributed architecture where embeddings run on
dedicated host (Host 5 with RTX 4070).
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import httpx
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from morgan.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingStats:
    """Statistics for embedding operations."""

    total_requests: int = 0
    total_embeddings: int = 0
    total_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def average_time(self) -> float:
        """Calculate average embedding time."""
        return self.total_time / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def throughput(self) -> float:
        """Calculate throughput (embeddings/sec)."""
        return self.total_embeddings / self.total_time if self.total_time > 0 else 0.0


class LocalEmbeddingService:
    """
    Local embedding service for distributed Morgan setup.

    Supports:
    - Remote Ollama/OpenAI-compatible endpoints (primary)
    - Local sentence-transformers (fallback)
    - Batch processing with configurable size
    - Content-based caching
    - Performance tracking

    Example:
        >>> service = LocalEmbeddingService(
        ...     endpoint="http://192.168.1.22:11434/v1",
        ...     model="nomic-embed-text"
        ... )
        >>>
        >>> # Single embedding
        >>> embedding = await service.embed_text("What is Python?")
        >>>
        >>> # Batch embeddings
        >>> embeddings = await service.embed_batch([
        ...     "Document 1",
        ...     "Document 2",
        ...     "Document 3"
        ... ])
        >>>
        >>> # Get stats
        >>> stats = service.get_stats()
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: str = "nomic-embed-text",
        dimensions: int = 768,
        api_key: str = "ollama",
        timeout: float = 30.0,
        batch_size: int = 100,
        local_model: str = "all-MiniLM-L6-v2",
        local_device: str = "cpu",
        use_cache: bool = True,
        max_cache_size: int = 10000,
    ):
        """
        Initialize local embedding service.

        Args:
            endpoint: OpenAI-compatible endpoint URL (e.g., "http://host5:11434/v1")
            model: Model name for remote embeddings
            dimensions: Embedding dimensions (768 for nomic-embed-text)
            api_key: API key for endpoint (default: "ollama")
            timeout: Request timeout in seconds
            batch_size: Batch size for processing
            local_model: Local fallback model name
            local_device: Device for local model ("cpu" or "cuda")
            use_cache: Enable caching
            max_cache_size: Maximum cache entries
        """
        if not OPENAI_AVAILABLE:
            logger.warning("openai package not available, remote embeddings disabled")

        self.endpoint = endpoint
        self.model = model
        self.dimensions = dimensions
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = batch_size
        self.local_model_name = local_model
        self.local_device = local_device
        self.use_cache = use_cache
        self.max_cache_size = max_cache_size

        # Initialize cache
        self._cache: Dict[str, List[float]] = {}

        # Initialize local model (lazy)
        self._local_model = None
        self._local_available = None

        # Statistics
        self.stats = EmbeddingStats()

        logger.info(
            f"LocalEmbeddingService initialized: "
            f"endpoint={endpoint}, model={model}, dims={dimensions}"
        )

    async def is_available(self) -> bool:
        """
        Check if embedding service is available.

        Returns:
            True if remote or local embeddings available
        """
        # Check remote first
        if self.endpoint and OPENAI_AVAILABLE:
            try:
                client = AsyncOpenAI(
                    base_url=self.endpoint,
                    api_key=self.api_key,
                    timeout=httpx.Timeout(5.0),
                )

                # Test with simple embedding
                response = await client.embeddings.create(
                    model=self.model, input="test"
                )

                if response.data and len(response.data) > 0:
                    logger.info("Remote embedding service available")
                    return True
            except Exception as e:
                logger.warning(f"Remote embedding service not available: {e}")

        # Check local fallback
        if self._check_local_available():
            logger.info("Local embedding service available")
            return True

        logger.error("No embedding service available")
        return False

    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding dimension
        """
        return self.dimensions

    async def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Embed single text.

        Args:
            text: Text to embed
            use_cache: Whether to use cache

        Returns:
            Embedding vector
        """
        self.stats.total_requests += 1

        # Check cache
        if use_cache and self.use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                self.stats.cache_hits += 1
                logger.debug(f"Cache hit for text (length={len(text)})")
                return self._cache[cache_key]
            self.stats.cache_misses += 1

        # Generate embedding
        start_time = time.time()

        try:
            # Try remote first
            if self.endpoint and OPENAI_AVAILABLE:
                embedding = await self._embed_remote(text)
            # Fallback to local
            elif self._check_local_available():
                embedding = self._embed_local(text)
            else:
                self.stats.errors += 1
                raise RuntimeError("No embedding service available")

            # Update stats
            elapsed = time.time() - start_time
            self.stats.total_embeddings += 1
            self.stats.total_time += elapsed

            # Cache result
            if use_cache and self.use_cache:
                self._cache_embedding(cache_key, embedding)

            logger.debug(f"Embedded text (length={len(text)}) in {elapsed:.3f}s")

            return embedding

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Embedding failed: {e}")
            raise

    async def embed_batch(
        self, texts: List[str], use_cache: bool = True, show_progress: bool = False
    ) -> List[List[float]]:
        """
        Embed batch of texts.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use cache
            show_progress: Show progress bar

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self.stats.total_requests += 1

        # Check cache for all texts
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        if use_cache and self.use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    embeddings.append(self._cache[cache_key])
                    self.stats.cache_hits += 1
                else:
                    embeddings.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    self.stats.cache_misses += 1

            if not uncached_texts:
                logger.debug(f"All {len(texts)} texts found in cache")
                return embeddings

            logger.debug(f"Cache hit: {len(texts) - len(uncached_texts)}/{len(texts)}")
        else:
            embeddings = [None] * len(texts)
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # Generate embeddings for uncached texts
        start_time = time.time()

        try:
            # Try remote first
            if self.endpoint and OPENAI_AVAILABLE:
                new_embeddings = await self._embed_batch_remote(uncached_texts)
            # Fallback to local
            elif self._check_local_available():
                new_embeddings = self._embed_batch_local(uncached_texts)
            else:
                self.stats.errors += 1
                raise RuntimeError("No embedding service available")

            # Fill in embeddings and cache
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                if use_cache and self.use_cache:
                    cache_key = self._get_cache_key(texts[idx])
                    self._cache_embedding(cache_key, embedding)

            # Update stats
            elapsed = time.time() - start_time
            self.stats.total_embeddings += len(uncached_texts)
            self.stats.total_time += elapsed

            logger.info(
                f"Embedded {len(uncached_texts)} texts in {elapsed:.3f}s "
                f"({len(uncached_texts)/elapsed:.1f} texts/sec)"
            )

            return embeddings

        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Batch embedding failed: {e}")
            raise

    async def _embed_remote(self, text: str) -> List[float]:
        """Embed text using remote endpoint."""
        client = AsyncOpenAI(
            base_url=self.endpoint,
            api_key=self.api_key,
            timeout=httpx.Timeout(self.timeout),
        )

        response = await client.embeddings.create(model=self.model, input=text)

        if not response.data or len(response.data) == 0:
            raise ValueError("No embedding in response")

        return response.data[0].embedding

    async def _embed_batch_remote(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using remote endpoint."""
        client = AsyncOpenAI(
            base_url=self.endpoint,
            api_key=self.api_key,
            timeout=httpx.Timeout(self.timeout),
        )

        # Process in batches
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            response = await client.embeddings.create(model=self.model, input=batch)

            if not response.data or len(response.data) != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} embeddings, got {len(response.data)}"
                )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _check_local_available(self) -> bool:
        """Check if local sentence-transformers is available."""
        if self._local_available is not None:
            return self._local_available

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not installed")
            self._local_available = False
            return False

        try:
            logger.info(f"Loading local embedding model ({self.local_model_name})...")
            self._local_model = SentenceTransformer(
                self.local_model_name, device=self.local_device
            )
            logger.info("Local embedding model loaded successfully")
            self._local_available = True
            return True
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            self._local_available = False
            return False

    def _embed_local(self, text: str) -> List[float]:
        """Embed text using local model."""
        if self._local_model is None:
            raise RuntimeError("Local model not initialized")

        embedding = self._local_model.encode(
            text, convert_to_numpy=True, show_progress_bar=False
        )

        return embedding.tolist()

    def _embed_batch_local(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using local model."""
        if self._local_model is None:
            raise RuntimeError("Local model not initialized")

        embeddings = self._local_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,  # Internal batch size
        )

        return embeddings.tolist()

    def _get_cache_key(self, text: str) -> str:
        """Get cache key for text."""
        cache_input = f"{self.model}:{text}"
        return hashlib.sha256(cache_input.encode()).hexdigest()

    def _cache_embedding(self, key: str, embedding: List[float]):
        """Cache embedding with size limit."""
        # Simple FIFO eviction if cache too large
        if len(self._cache) >= self.max_cache_size:
            # Remove oldest entry (first item)
            first_key = next(iter(self._cache))
            del self._cache[first_key]

        self._cache[key] = embedding

    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_requests": self.stats.total_requests,
            "total_embeddings": self.stats.total_embeddings,
            "total_time": f"{self.stats.total_time:.2f}s",
            "average_time": f"{self.stats.average_time:.3f}s",
            "throughput": f"{self.stats.throughput:.1f} embeddings/sec",
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "cache_hit_rate": f"{self.stats.cache_hit_rate * 100:.1f}%",
            "cache_size": len(self._cache),
            "errors": self.stats.errors,
            "endpoint": self.endpoint,
            "model": self.model,
            "dimensions": self.dimensions,
        }

    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")


# Global instance for singleton pattern
_service: Optional[LocalEmbeddingService] = None


def get_local_embedding_service(
    endpoint: Optional[str] = None, model: Optional[str] = None, **kwargs
) -> LocalEmbeddingService:
    """
    Get global local embedding service instance (singleton).

    Args:
        endpoint: Endpoint URL (required on first call)
        model: Model name (required on first call)
        **kwargs: Additional arguments for LocalEmbeddingService

    Returns:
        LocalEmbeddingService instance
    """
    global _service

    if _service is None:
        if endpoint is None:
            raise ValueError("endpoint required for first initialization")

        if model is None:
            model = "nomic-embed-text"

        _service = LocalEmbeddingService(endpoint=endpoint, model=model, **kwargs)

    return _service
