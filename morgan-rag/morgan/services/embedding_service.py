"""Production-quality embedding service with batching and resilience."""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Embedding service configuration."""
    base_url: str = "http://localhost:11434"  # Ollama default
    model: str = "qwen3-embedding:latest"
    batch_size: int = 32
    max_concurrent_requests: int = 10
    timeout: float = 60.0
    connection_pool_size: int = 100
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0


class EmbeddingCache:
    """Simple in-memory LRU cache for embeddings."""

    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        """Initialize cache.

        Args:
            max_size: Maximum cache entries.
            ttl: Time to live in seconds.
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()

    def _get_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        combined = f"{model}:{text}"
        return hashlib.sha256(combined.encode()).hexdigest()

    async def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache.

        Args:
            text: Input text.
            model: Model name.

        Returns:
            Cached embedding or None.
        """
        async with self._lock:
            key = self._get_key(text, model)
            if key in self._cache:
                embedding, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    # Update access order (LRU)
                    self._access_order.remove(key)
                    self._access_order.append(key)
                    logger.debug("Cache hit", extra={"key": key[:16]})
                    return embedding
                else:
                    # Expired
                    del self._cache[key]
                    self._access_order.remove(key)
            return None

    async def set(self, text: str, model: str, embedding: np.ndarray) -> None:
        """Store embedding in cache.

        Args:
            text: Input text.
            model: Model name.
            embedding: Embedding vector.
        """
        async with self._lock:
            key = self._get_key(text, model)

            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
                logger.debug("Cache eviction", extra={"key": oldest_key[:16]})

            self._cache[key] = (embedding, time.time())
            self._access_order.append(key)
            logger.debug("Cache set", extra={"key": key[:16]})

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
        }


class EmbeddingService:
    """Production-quality embedding service with batching and resilience.

    Features:
    - Efficient batching with configurable batch size
    - Async HTTP with connection pooling
    - In-memory LRU cache with TTL
    - Circuit breaker via tenacity retry
    - Proper timeout handling
    - Resource cleanup
    - Structured logging
    - Concurrent request limiting
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize embedding service.

        Args:
            config: Service configuration.
        """
        self.config = config or EmbeddingConfig()
        self._cache = EmbeddingCache(ttl=self.config.cache_ttl) if self.config.cache_enabled else None
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._client: Optional[httpx.AsyncClient] = None

        logger.info(
            "Initializing embedding service",
            extra={
                "model": self.config.model,
                "batch_size": self.config.batch_size,
                "cache_enabled": self.config.cache_enabled,
            }
        )

    async def __aenter__(self):
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.disconnect()
        return False

    async def connect(self) -> None:
        """Initialize HTTP client with connection pooling."""
        if self._client is not None:
            logger.debug("Already connected")
            return

        limits = httpx.Limits(
            max_connections=self.config.connection_pool_size,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        )

        timeout = httpx.Timeout(
            timeout=self.config.timeout,
            connect=10.0,
            read=self.config.timeout,
            write=10.0,
            pool=5.0,
        )

        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            limits=limits,
            timeout=timeout,
            http2=True,  # Enable HTTP/2 for multiplexing
        )

        # Verify connection
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            response.raise_for_status()
            logger.info("Connected to embedding service successfully")
        except Exception as e:
            await self._client.aclose()
            self._client = None
            raise ConnectionError(f"Failed to connect to embedding service: {e}") from e

    async def disconnect(self) -> None:
        """Close HTTP client and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Disconnected from embedding service")

        if self._cache:
            await self._cache.clear()

    async def _embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text.

        Args:
            text: Input text.

        Returns:
            Embedding vector.

        Raises:
            RuntimeError: If not connected.
            httpx.HTTPError: If request fails.
        """
        if not self._client:
            raise RuntimeError("Not connected. Call connect() first.")

        # Check cache first
        if self._cache:
            cached = await self._cache.get(text, self.config.model)
            if cached is not None:
                return cached

        # Make request with retry logic
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait,
            ),
        ):
            with attempt:
                async with self._semaphore:
                    start_time = time.time()

                    response = await self._client.post(
                        "/api/embeddings",
                        json={
                            "model": self.config.model,
                            "prompt": text,
                        },
                    )
                    response.raise_for_status()

                    duration = time.time() - start_time
                    data = response.json()
                    embedding = np.array(data["embedding"], dtype=np.float32)

                    logger.debug(
                        "Generated embedding",
                        extra={
                            "text_length": len(text),
                            "embedding_dim": len(embedding),
                            "duration_ms": round(duration * 1000, 2),
                        }
                    )

                    # Cache result
                    if self._cache:
                        await self._cache.set(text, self.config.model, embedding)

                    return embedding

    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text.

        Args:
            text: Input text.

        Returns:
            Embedding vector.
        """
        return await self._embed_single(text)

    async def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts.
            show_progress: Whether to log progress.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        start_time = time.time()
        embeddings = []

        # Process in batches
        total_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size

        for batch_idx in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[batch_idx:batch_idx + self.config.batch_size]

            # Check cache for batch items
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []

            if self._cache:
                for i, text in enumerate(batch_texts):
                    cached = await self._cache.get(text, self.config.model)
                    if cached is not None:
                        cached_embeddings.append((i, cached))
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
            else:
                uncached_texts = batch_texts
                uncached_indices = list(range(len(batch_texts)))

            # Process uncached texts concurrently
            if uncached_texts:
                tasks = [self._embed_single(text) for text in uncached_texts]
                uncached_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle exceptions
                for i, result in enumerate(uncached_results):
                    if isinstance(result, Exception):
                        logger.error(
                            "Failed to generate embedding",
                            extra={
                                "text_length": len(uncached_texts[i]),
                                "error": str(result),
                            }
                        )
                        # Use zero vector as fallback
                        uncached_results[i] = np.zeros(768, dtype=np.float32)

            # Merge cached and uncached results
            batch_embeddings = [None] * len(batch_texts)
            for i, embedding in cached_embeddings:
                batch_embeddings[i] = embedding
            for i, embedding in zip(uncached_indices, uncached_results if uncached_texts else []):
                batch_embeddings[i] = embedding

            embeddings.extend(batch_embeddings)

            if show_progress:
                batch_num = (batch_idx // self.config.batch_size) + 1
                logger.info(
                    "Batch processed",
                    extra={
                        "batch": f"{batch_num}/{total_batches}",
                        "texts": len(batch_texts),
                        "cached": len(cached_embeddings),
                    }
                )

        duration = time.time() - start_time
        logger.info(
            "Batch embedding completed",
            extra={
                "total_texts": len(texts),
                "batches": total_batches,
                "duration_s": round(duration, 2),
                "texts_per_sec": round(len(texts) / duration, 2),
            }
        )

        return embeddings

    async def embed_stream(
        self,
        texts: List[str],
    ) -> asyncio.Queue:
        """Generate embeddings with streaming output.

        Args:
            texts: List of input texts.

        Returns:
            Queue yielding (index, embedding) tuples.
        """
        queue = asyncio.Queue(maxsize=self.config.batch_size * 2)

        async def producer():
            """Producer coroutine to generate embeddings."""
            try:
                for idx, text in enumerate(texts):
                    embedding = await self._embed_single(text)
                    await queue.put((idx, embedding))

                # Signal completion
                await queue.put(None)
            except Exception as e:
                logger.error("Streaming producer failed", extra={"error": str(e)})
                await queue.put(e)

        # Start producer task
        asyncio.create_task(producer())

        return queue

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics.

        Returns:
            Statistics dictionary.
        """
        stats = {
            "model": self.config.model,
            "batch_size": self.config.batch_size,
            "max_concurrent": self.config.max_concurrent_requests,
            "connected": self._client is not None,
        }

        if self._cache:
            stats["cache"] = self._cache.get_stats()

        return stats
