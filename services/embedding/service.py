"""
Production-quality Embedding Service

Features:
- Async HTTP client with connection pooling
- Circuit breaker pattern for fault tolerance
- Rate limiting for API protection
- Retry with exponential backoff and jitter
- Health monitoring
- Batch processing support
- Caching support
- Comprehensive error handling
"""

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel

from shared.config.base import ServiceConfig
from shared.infrastructure import (
    CircuitBreakerConfig,
    ConnectionPoolConfig,
    EnhancedHTTPClient,
    RateLimitConfig,
    RetryConfig,
    TimeoutConfig,
)
from shared.utils.exceptions import (
    ErrorCategory,
    ModelException,
    ModelInferenceError,
    ModelLoadError,
    MorganException,
)
from shared.utils.logging import Timer, setup_logging

logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseModel):
    """Embedding service configuration"""

    # Service settings
    host: str = "0.0.0.0"
    port: int = 8003
    model: str = "qwen3-embedding:latest"
    api_base: str = "https://gpt.lazarev.cloud/ollama"
    api_key: str = ""
    log_level: str = "INFO"

    # Embedding settings
    embedding_dimension: int = 768
    batch_size: int = 32
    max_text_length: int = 8192

    # Connection pool settings
    max_connections: int = 50
    max_keepalive_connections: int = 10
    keepalive_expiry: float = 5.0

    # Retry settings
    max_retries: int = 3
    base_retry_delay: float = 1.0
    max_retry_delay: float = 30.0

    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: float = 60.0

    # Rate limiting (requests per second)
    rate_limit_rps: float = 5.0
    rate_limit_burst: int = 10

    # Health monitoring
    enable_health_monitoring: bool = True
    health_check_interval: float = 30.0

    # Caching
    enable_caching: bool = True
    cache_size: int = 1000


@dataclass
class EmbeddingResult:
    """Result of embedding operation"""

    embedding: List[float]
    model: str
    dimension: int
    text_length: int
    cached: bool = False
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingCache:
    """Simple LRU cache for embeddings"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, List[float]] = {}
        self.access_order: List[str] = []
        self.hits = 0
        self.misses = 0

    def _get_key(self, text: str, model: str) -> str:
        """Generate cache key"""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding"""
        key = self._get_key(text, model)

        if key in self.cache:
            self.hits += 1
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]

        self.misses += 1
        return None

    def put(self, text: str, model: str, embedding: List[float]):
        """Cache embedding"""
        key = self._get_key(text, model)

        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

        self.cache[key] = embedding
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0


class ProductionEmbeddingService:
    """
    Production-quality Embedding Service

    Provides robust text embedding with enterprise-grade reliability:
    - Connection pooling for efficient resource usage
    - Circuit breaker to prevent cascading failures
    - Rate limiting to protect backend services
    - Retry logic with exponential backoff and jitter
    - Comprehensive health monitoring
    - Batch processing for efficiency
    - LRU caching for repeated queries
    - Structured error handling and logging
    """

    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig("embedding")
        self.logger = setup_logging(
            "production_embedding_service",
            self.config.get("log_level", "INFO"),
            "logs/production_embedding_service.log",
        )

        # Load configuration
        config_dict = self._load_config()
        self.embedding_config = EmbeddingConfig(**config_dict)

        # HTTP client (initialized in start())
        self.http_client: Optional[EnhancedHTTPClient] = None

        # Cache
        self.cache: Optional[EmbeddingCache] = None
        if self.embedding_config.enable_caching:
            self.cache = EmbeddingCache(max_size=self.embedding_config.cache_size)

        # Metrics
        self.embedding_count = 0
        self.batch_count = 0
        self.total_processing_time = 0.0

        self.logger.info(
            f"Production Embedding Service initialized: "
            f"model={self.embedding_config.model}, "
            f"api_base={self.embedding_config.api_base}, "
            f"caching={'enabled' if self.cache else 'disabled'}"
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with environment variable overrides"""
        config_dict = self.config.all()

        # Environment variable overrides
        env_overrides = {
            "OLLAMA_BASE_URL": "api_base",
            "MORGAN_LLM_API_KEY": "api_key",
            "MORGAN_EMBEDDING_MODEL": "model",
        }

        for env_var, config_key in env_overrides.items():
            value = os.getenv(env_var)
            if value:
                # Remove /v1 suffix if present
                if config_key == "api_base" and value.endswith("/v1"):
                    value = value.rsplit("/v1", 1)[0]
                config_dict[config_key] = value
                self.logger.debug(f"Config override from {env_var}: {config_key}")

        return config_dict

    async def start(self):
        """Start the embedding service"""
        try:
            # Initialize enhanced HTTP client
            pool_config = ConnectionPoolConfig(
                max_connections=self.embedding_config.max_connections,
                max_keepalive_connections=self.embedding_config.max_keepalive_connections,
                keepalive_expiry=self.embedding_config.keepalive_expiry,
            )

            timeout_config = TimeoutConfig(connect=5.0, read=30.0, write=10.0, pool=5.0)

            circuit_breaker_config = CircuitBreakerConfig(
                failure_threshold=self.embedding_config.circuit_breaker_failure_threshold,
                timeout=self.embedding_config.circuit_breaker_timeout,
            )

            rate_limit_config = RateLimitConfig(
                requests_per_second=self.embedding_config.rate_limit_rps,
                burst_size=self.embedding_config.rate_limit_burst,
            )

            retry_config = RetryConfig(
                max_retries=self.embedding_config.max_retries,
                base_delay=self.embedding_config.base_retry_delay,
                max_delay=self.embedding_config.max_retry_delay,
                jitter=True,
            )

            self.http_client = EnhancedHTTPClient(
                service_name="ollama_embeddings",
                base_url=self.embedding_config.api_base,
                pool_config=pool_config,
                retry_config=retry_config,
                timeout_config=timeout_config,
                circuit_breaker_config=circuit_breaker_config,
                rate_limit_config=rate_limit_config,
                enable_health_monitoring=self.embedding_config.enable_health_monitoring,
                health_check_interval=self.embedding_config.health_check_interval,
            )

            await self.http_client.start()

            self.logger.info("Production Embedding Service started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start embedding service: {e}")
            raise

    async def stop(self):
        """Stop the embedding service"""
        self.logger.info("Production Embedding Service stopping...")

        if self.http_client:
            await self.http_client.stop()

        self.logger.info("Production Embedding Service stopped")

    async def embed(
        self, text: Union[str, List[str]], model: Optional[str] = None
    ) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """
        Generate embeddings for text(s)

        Args:
            text: Single text or list of texts to embed
            model: Optional model name (defaults to config)

        Returns:
            EmbeddingResult or list of EmbeddingResults

        Raises:
            ModelError: On embedding failure
        """
        # Handle single text vs batch
        if isinstance(text, str):
            result = await self._embed_single(text, model)
            return result
        else:
            results = await self._embed_batch(text, model)
            return results

    async def _embed_single(
        self, text: str, model: Optional[str] = None
    ) -> EmbeddingResult:
        """Generate embedding for single text"""
        embedding_model = model or self.embedding_config.model

        # Check cache first
        if self.cache:
            cached_embedding = self.cache.get(text, embedding_model)
            if cached_embedding is not None:
                self.logger.debug("Embedding cache hit")
                return EmbeddingResult(
                    embedding=cached_embedding,
                    model=embedding_model,
                    dimension=len(cached_embedding),
                    text_length=len(text),
                    cached=True,
                )

        # Generate embedding
        with Timer(self.logger, f"Embedding generation (len={len(text)})"):
            try:
                # Truncate if too long
                if len(text) > self.embedding_config.max_text_length:
                    self.logger.warning(
                        f"Text truncated from {len(text)} to "
                        f"{self.embedding_config.max_text_length} chars"
                    )
                    text = text[: self.embedding_config.max_text_length]

                # Prepare request
                payload = {"model": embedding_model, "input": [text]}

                headers = {"Content-Type": "application/json"}
                if self.embedding_config.api_key:
                    headers["Authorization"] = f"Bearer {self.embedding_config.api_key}"

                # Make request
                response = await self.http_client.post(
                    "/api/embed", json=payload, headers=headers
                )

                result = response.json()

                # Extract embedding
                if "embeddings" in result and len(result["embeddings"]) > 0:
                    embedding = result["embeddings"][0]
                elif "embedding" in result:
                    embedding = result["embedding"]
                else:
                    raise ModelError(
                        "Unexpected embedding response format",
                        ErrorCode.MODEL_INFERENCE_ERROR,
                        {"response": result},
                    )

                # Cache the result
                if self.cache:
                    self.cache.put(text, embedding_model, embedding)

                # Track metrics
                self.embedding_count += 1

                return EmbeddingResult(
                    embedding=embedding,
                    model=embedding_model,
                    dimension=len(embedding),
                    text_length=len(text),
                    cached=False,
                    metadata={"service": "production_embedding"},
                )

            except Exception as e:
                self.logger.error(f"Embedding generation failed: {e}")
                raise ModelError(
                    f"Embedding generation failed: {e}", ErrorCode.MODEL_INFERENCE_ERROR
                )

    async def _embed_batch(
        self, texts: List[str], model: Optional[str] = None
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for batch of texts

        Args:
            texts: List of texts to embed
            model: Optional model name

        Returns:
            List of EmbeddingResults
        """
        embedding_model = model or self.embedding_config.model

        with Timer(self.logger, f"Batch embedding (count={len(texts)})"):
            # Process in batches
            batch_size = self.embedding_config.batch_size
            all_results = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                # Process batch concurrently
                tasks = [self._embed_single(text, embedding_model) for text in batch]

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle results and exceptions
                for result in batch_results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Batch embedding error: {result}")
                        # Create error result
                        all_results.append(
                            EmbeddingResult(
                                embedding=[0.0]
                                * self.embedding_config.embedding_dimension,
                                model=embedding_model,
                                dimension=self.embedding_config.embedding_dimension,
                                text_length=0,
                                metadata={"error": str(result)},
                            )
                        )
                    else:
                        all_results.append(result)

                self.batch_count += 1

            return all_results

    async def compute_similarity(
        self, text1: str, text2: str, model: Optional[str] = None
    ) -> float:
        """
        Compute cosine similarity between two texts

        Args:
            text1: First text
            text2: Second text
            model: Optional model name

        Returns:
            Cosine similarity score (0-1)
        """
        # Generate embeddings
        results = await self._embed_batch([text1, text2], model)

        # Compute cosine similarity
        emb1 = np.array(results[0].embedding)
        emb2 = np.array(results[1].embedding)

        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(similarity)

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            health_status = {
                "service": "production_embedding",
                "model": self.embedding_config.model,
                "api_base": self.embedding_config.api_base,
                "metrics": self.get_metrics(),
            }

            # Test connectivity by generating a test embedding
            test_result = await self._embed_single("health check", None)

            client_status = self.http_client.get_status()

            health_status.update(
                {
                    "status": "healthy",
                    "test_embedding_dimension": test_result.dimension,
                    "http_client": client_status,
                    "circuit_breaker": client_status.get("circuit_breaker"),
                    "rate_limiter": client_status.get("rate_limiter"),
                }
            )

            return health_status

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "service": "production_embedding",
                "status": "unhealthy",
                "error": str(e),
                "api_base": self.embedding_config.api_base,
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        metrics = {
            "embedding_count": self.embedding_count,
            "batch_count": self.batch_count,
            "avg_processing_time": (
                self.total_processing_time / self.embedding_count
                if self.embedding_count > 0
                else 0.0
            ),
        }

        if self.cache:
            metrics["cache"] = self.cache.get_stats()

        if self.http_client:
            metrics["http_client"] = self.http_client.get_status()

        return metrics

    def clear_cache(self):
        """Clear embedding cache"""
        if self.cache:
            self.cache.clear()
            self.logger.info("Embedding cache cleared")
