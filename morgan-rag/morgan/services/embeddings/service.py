"""
Unified Embedding Service for Morgan AI Assistant.

Consolidates embeddings/service.py, infrastructure/local_embeddings.py,
and services/distributed_embedding_service.py into a single implementation.
"""

import asyncio
import hashlib
import os
import threading
import time
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.services.embeddings.models import EmbeddingStats
from morgan.utils.logger import get_logger
from morgan.utils.model_cache import setup_model_cache  # Use canonical implementation

logger = get_logger(__name__)

# Optional imports
try:
    import httpx
    from openai import AsyncOpenAI, OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingService:
    """
    Unified embedding service supporting remote and local providers.

    This service consolidates the functionality from embeddings/service.py,
    infrastructure/local_embeddings.py, and distributed_embedding_service.py.

    Features:
    - Remote Ollama/OpenAI-compatible endpoints (primary)
    - Local sentence-transformers fallback
    - Automatic failover between providers
    - Batch processing with configurable size
    - Content-based caching
    - Both sync and async interfaces
    - Thread-safe singleton pattern

    Self-hosted models (Qwen3-Embedding via Ollama):
    - qwen3-embedding:0.6b: 896 dims (lightweight)
    - qwen3-embedding:4b: 2048 dims (recommended)
    - qwen3-embedding:8b: 4096 dims (best quality)
    - all-MiniLM-L6-v2: Via sentence-transformers, 384 dims (fallback)

    Example:
        >>> # Get singleton service
        >>> service = get_embedding_service()

        >>> # Single embedding (sync)
        >>> embedding = service.encode("What is Python?")

        >>> # Batch embeddings (sync)
        >>> embeddings = service.encode_batch(["Doc 1", "Doc 2"])

        >>> # Async embedding
        >>> embedding = await service.aencode("Explain Docker")
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        local_model: str = "all-MiniLM-L6-v2",
        local_device: str = "cpu",
        api_key: str = "ollama",
        timeout: float = 30.0,
        batch_size: int = 100,
        use_cache: bool = True,
        max_cache_size: int = 10000,
        force_remote: bool = False,
        model_cache_dir: Optional[str] = None,
    ):
        """
        Initialize embedding service.

        Args:
            endpoint: OpenAI-compatible endpoint URL (e.g., "http://host:11434/v1")
            model: Model name for remote embeddings
            dimensions: Embedding dimensions (auto-detected if not specified)
            local_model: Local fallback model name
            local_device: Device for local model ("cpu" or "cuda")
            api_key: API key for endpoint (default: "ollama")
            timeout: Request timeout in seconds
            batch_size: Batch size for processing
            use_cache: Enable caching
            max_cache_size: Maximum cache entries
            force_remote: Fail if remote is unavailable (no local fallback)
            model_cache_dir: Directory for model weights cache
        """
        self.settings = get_settings()
        self._lock = threading.Lock()
        self._availability_lock = threading.Lock()

        # Setup model cache
        self.model_cache_path = setup_model_cache(model_cache_dir)

        # Configuration from settings or parameters
        self.endpoint = (
            endpoint
            or getattr(self.settings, "embedding_base_url", None)
            or getattr(self.settings, "llm_base_url", "http://localhost:11434/v1")
        )

        self.model = model or getattr(
            self.settings, "embedding_model", "qwen3-embedding:4b"
        )

        self.dimensions = dimensions or getattr(
            self.settings, "embedding_dimensions", 2048
        )

        self.local_model_name = local_model or getattr(
            self.settings, "embedding_local_model", "all-MiniLM-L6-v2"
        )

        self.local_device = local_device or getattr(
            self.settings, "embedding_device", "cpu"
        )

        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = batch_size or getattr(
            self.settings, "embedding_batch_size", 100
        )
        self.use_cache = use_cache
        self.max_cache_size = max_cache_size
        self.force_remote = force_remote or getattr(
            self.settings, "embedding_force_remote", False
        )

        # Initialize cache
        self._cache: Dict[str, List[float]] = {}

        # Lazy-loaded local model
        self._local_model = None
        self._local_available: Optional[bool] = None
        self._remote_available: Optional[bool] = None

        # Statistics
        self.stats = EmbeddingStats()

        logger.info(
            "EmbeddingService initialized: endpoint=%s, model=%s, dims=%d, local=%s",
            self.endpoint,
            self.model,
            self.dimensions,
            self.local_model_name,
        )

    # =========================================================================
    # Availability Checks
    # =========================================================================

    def is_available(self) -> bool:
        """
        Check if embedding service is available (sync).

        Returns:
            True if remote or local embeddings available
        """
        # Check remote
        if self._check_remote_available_sync():
            return True

        if self.force_remote:
            logger.error("Remote embedding forced but not available")
            return False

        # Check local fallback
        return self._check_local_available()

    async def ais_available(self) -> bool:
        """
        Check if embedding service is available (async).

        Returns:
            True if remote or local embeddings available
        """
        # Check remote
        if await self._check_remote_available():
            return True

        if self.force_remote:
            logger.error("Remote embedding forced but not available")
            return False

        # Check local fallback
        return self._check_local_available()

    def _check_remote_available_sync(self) -> bool:
        """Check if remote endpoint is available (sync)."""
        if not self.endpoint or not OPENAI_AVAILABLE:
            return False

        if self._remote_available is not None:
            return self._remote_available

        try:
            client = OpenAI(
                base_url=self.endpoint,
                api_key=self.api_key,
                timeout=5.0,
            )

            response = client.embeddings.create(model=self.model, input="test")

            if response.data and len(response.data) > 0:
                with self._availability_lock:
                    self._remote_available = True
                logger.info("Remote embedding service available at %s", self.endpoint)
                return True

        except Exception as e:
            logger.warning("Remote embedding service not available: %s", e)
            with self._availability_lock:
                self._remote_available = False

        return False

    async def _check_remote_available(self) -> bool:
        """Check if remote endpoint is available (async)."""
        if not self.endpoint or not OPENAI_AVAILABLE:
            return False

        if self._remote_available is not None:
            return self._remote_available

        try:
            client = AsyncOpenAI(
                base_url=self.endpoint,
                api_key=self.api_key,
                timeout=httpx.Timeout(5.0),
            )

            response = await client.embeddings.create(model=self.model, input="test")

            if response.data and len(response.data) > 0:
                with self._availability_lock:
                    self._remote_available = True
                logger.info("Remote embedding service available at %s", self.endpoint)
                return True

        except Exception as e:
            logger.warning("Remote embedding service not available: %s", e)
            with self._availability_lock:
                self._remote_available = False

        return False

    def _check_local_available(self) -> bool:
        """Check if local sentence-transformers is available."""
        if self._local_available is not None:
            return self._local_available

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not installed")
            with self._availability_lock:
                self._local_available = False
            return False

        try:
            logger.info("Loading local embedding model (%s)...", self.local_model_name)
            self._local_model = SentenceTransformer(
                self.local_model_name, device=self.local_device
            )
            logger.info("Local embedding model loaded successfully")
            with self._availability_lock:
                self._local_available = True
            return True
        except Exception as e:
            logger.error("Failed to load local model: %s", e)
            with self._availability_lock:
                self._local_available = False
            return False

    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding dimension
        """
        return self.dimensions

    # =========================================================================
    # Synchronous Methods
    # =========================================================================

    def encode(
        self,
        text: str,
        instruction: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[float]:
        """
        Encode single text (synchronous).

        Args:
            text: Text to encode
            instruction: Optional instruction prefix
            use_cache: Whether to use cache

        Returns:
            Embedding vector
        """
        self.stats.total_requests += 1

        # Apply instruction prefix if provided
        if instruction:
            text = f"{instruction}: {text}"

        # Check cache
        if use_cache and self.use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                self.stats.cache_hits += 1
                return self._cache[cache_key]
            self.stats.cache_misses += 1
        else:
            cache_key = None

        # Generate embedding
        start_time = time.time()

        try:
            # Try remote first
            if self._check_remote_available_sync():
                embedding = self._encode_remote_sync(text)
                self.stats.remote_calls += 1
            elif not self.force_remote and self._check_local_available():
                embedding = self._encode_local(text)
                self.stats.local_calls += 1
            else:
                self.stats.errors += 1
                raise RuntimeError("No embedding service available")

            # Update stats
            elapsed = time.time() - start_time
            self.stats.total_embeddings += 1
            self.stats.total_time += elapsed

            # Cache result
            if cache_key:
                self._cache_embedding(cache_key, embedding)

            return embedding

        except Exception as e:
            self.stats.errors += 1
            logger.error("Embedding failed: %s", e)
            raise

    def encode_batch(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        use_cache: bool = True,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Encode batch of texts (synchronous).

        Args:
            texts: List of texts to encode
            instruction: Optional instruction prefix
            use_cache: Whether to use cache
            show_progress: Show progress bar (not implemented)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self.stats.total_requests += 1

        # Apply instruction prefix if provided
        if instruction:
            texts = [f"{instruction}: {t}" for t in texts]

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
                    embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    self.stats.cache_misses += 1

            if not uncached_texts:
                return embeddings
        else:
            embeddings = [None] * len(texts)
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # Generate embeddings for uncached texts
        start_time = time.time()

        try:
            # Try remote first
            if self._check_remote_available_sync():
                new_embeddings = self._encode_batch_remote_sync(uncached_texts)
                self.stats.remote_calls += 1
            elif not self.force_remote and self._check_local_available():
                new_embeddings = self._encode_batch_local(uncached_texts)
                self.stats.local_calls += 1
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

            logger.debug(
                "Embedded %d texts in %.3fs (%.1f texts/sec)",
                len(uncached_texts),
                elapsed,
                len(uncached_texts) / elapsed if elapsed > 0 else 0,
            )

            return embeddings

        except Exception as e:
            self.stats.errors += 1
            logger.error("Batch embedding failed: %s", e)
            raise

    def _encode_remote_sync(self, text: str) -> List[float]:
        """Encode text using remote endpoint (sync)."""
        client = OpenAI(
            base_url=self.endpoint,
            api_key=self.api_key,
            timeout=self.timeout,
        )

        response = client.embeddings.create(model=self.model, input=text)

        if not response.data or len(response.data) == 0:
            raise ValueError("No embedding in response")

        return response.data[0].embedding

    def _encode_batch_remote_sync(self, texts: List[str]) -> List[List[float]]:
        """Encode batch using remote endpoint (sync)."""
        client = OpenAI(
            base_url=self.endpoint,
            api_key=self.api_key,
            timeout=self.timeout,
        )

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = client.embeddings.create(model=self.model, input=batch)

            if not response.data or len(response.data) != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} embeddings, got {len(response.data)}"
                )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _encode_local(self, text: str) -> List[float]:
        """Encode text using local model."""
        if self._local_model is None:
            raise RuntimeError("Local model not initialized")

        embedding = self._local_model.encode(
            text, convert_to_numpy=True, show_progress_bar=False
        )
        return embedding.tolist()

    def _encode_batch_local(self, texts: List[str]) -> List[List[float]]:
        """Encode batch using local model."""
        if self._local_model is None:
            raise RuntimeError("Local model not initialized")

        embeddings = self._local_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings.tolist()

    # =========================================================================
    # Asynchronous Methods
    # =========================================================================

    async def aencode(
        self,
        text: str,
        instruction: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[float]:
        """
        Encode single text (asynchronous).

        Args:
            text: Text to encode
            instruction: Optional instruction prefix
            use_cache: Whether to use cache

        Returns:
            Embedding vector
        """
        self.stats.total_requests += 1

        # Apply instruction prefix if provided
        if instruction:
            text = f"{instruction}: {text}"

        # Check cache
        if use_cache and self.use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                self.stats.cache_hits += 1
                return self._cache[cache_key]
            self.stats.cache_misses += 1
        else:
            cache_key = None

        # Generate embedding
        start_time = time.time()

        try:
            # Try remote first
            if await self._check_remote_available():
                embedding = await self._encode_remote(text)
                self.stats.remote_calls += 1
            elif not self.force_remote and self._check_local_available():
                # Run local in thread pool
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(None, self._encode_local, text)
                self.stats.local_calls += 1
            else:
                self.stats.errors += 1
                raise RuntimeError("No embedding service available")

            # Update stats
            elapsed = time.time() - start_time
            self.stats.total_embeddings += 1
            self.stats.total_time += elapsed

            # Cache result
            if cache_key:
                self._cache_embedding(cache_key, embedding)

            return embedding

        except Exception as e:
            self.stats.errors += 1
            logger.error("Async embedding failed: %s", e)
            raise

    async def aencode_batch(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[List[float]]:
        """
        Encode batch of texts (asynchronous).

        Args:
            texts: List of texts to encode
            instruction: Optional instruction prefix
            use_cache: Whether to use cache

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self.stats.total_requests += 1

        # Apply instruction prefix if provided
        if instruction:
            texts = [f"{instruction}: {t}" for t in texts]

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
                    embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    self.stats.cache_misses += 1

            if not uncached_texts:
                return embeddings
        else:
            embeddings = [None] * len(texts)
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # Generate embeddings for uncached texts
        start_time = time.time()

        try:
            # Try remote first
            if await self._check_remote_available():
                new_embeddings = await self._encode_batch_remote(uncached_texts)
                self.stats.remote_calls += 1
            elif not self.force_remote and self._check_local_available():
                # Run local in thread pool
                loop = asyncio.get_event_loop()
                new_embeddings = await loop.run_in_executor(
                    None, self._encode_batch_local, uncached_texts
                )
                self.stats.local_calls += 1
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

            return embeddings

        except Exception as e:
            self.stats.errors += 1
            logger.error("Async batch embedding failed: %s", e)
            raise

    async def _encode_remote(self, text: str) -> List[float]:
        """Encode text using remote endpoint (async)."""
        client = AsyncOpenAI(
            base_url=self.endpoint,
            api_key=self.api_key,
            timeout=httpx.Timeout(self.timeout),
        )

        response = await client.embeddings.create(model=self.model, input=text)

        if not response.data or len(response.data) == 0:
            raise ValueError("No embedding in response")

        return response.data[0].embedding

    async def _encode_batch_remote(self, texts: List[str]) -> List[List[float]]:
        """Encode batch using remote endpoint (async)."""
        client = AsyncOpenAI(
            base_url=self.endpoint,
            api_key=self.api_key,
            timeout=httpx.Timeout(self.timeout),
        )

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

    # =========================================================================
    # Cache Management
    # =========================================================================

    def _get_cache_key(self, text: str) -> str:
        """Get cache key for text."""
        cache_input = f"{self.model}:{text}"
        return hashlib.sha256(cache_input.encode()).hexdigest()

    def _cache_embedding(self, key: str, embedding: List[float]):
        """Cache embedding with size limit."""
        with self._lock:
            # Simple FIFO eviction if cache too large
            if len(self._cache) >= self.max_cache_size:
                first_key = next(iter(self._cache))
                del self._cache[first_key]

            self._cache[key] = embedding

    def clear_cache(self):
        """Clear embedding cache."""
        with self._lock:
            self._cache.clear()
        logger.info("Embedding cache cleared")

    # =========================================================================
    # Statistics & Info
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary with statistics
        """
        stats = self.stats.to_dict()
        stats.update(
            {
                "cache_size": len(self._cache),
                "endpoint": self.endpoint,
                "model": self.model,
                "dimensions": self.dimensions,
                "local_model": self.local_model_name,
                "remote_available": self._remote_available,
                "local_available": self._local_available,
            }
        )
        return stats

    def reset_stats(self):
        """Reset statistics."""
        self.stats = EmbeddingStats()
        logger.info("Embedding stats reset")


# =============================================================================
# Singleton Management
# =============================================================================

from morgan.utils.singleton import SingletonFactory


def _cleanup_embedding_service(service: EmbeddingService) -> None:
    """Cleanup function for embedding service."""
    if service:
        service.clear_cache()


# Create singleton factory with cleanup
_embedding_service_factory = SingletonFactory(EmbeddingService, cleanup_method="clear_cache")


def get_embedding_service(
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    force_new: bool = False,
    **kwargs,
) -> EmbeddingService:
    """
    Get singleton embedding service instance.

    Args:
        endpoint: Endpoint URL (uses settings if not specified)
        model: Model name (uses settings if not specified)
        force_new: Force create new instance
        **kwargs: Additional service configuration

    Returns:
        Shared EmbeddingService instance
    """
    return _embedding_service_factory.get_instance(
        force_new=force_new,
        endpoint=endpoint,
        model=model,
        **kwargs,
    )


def reset_embedding_service():
    """Reset singleton instance (useful for testing)."""
    _embedding_service_factory.reset()
