"""Production-quality Qdrant vector database client with connection pooling and resilience."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0  # seconds
    half_open_max_calls: int = 3


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise Exception(f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}")

            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise Exception("Circuit breaker HALF_OPEN max calls exceeded")
                self.half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.config.timeout

    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            self.failure_count = 0
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    logger.info("Circuit breaker transitioning to CLOSED")
                    self.state = CircuitState.CLOSED
                    self.success_count = 0

    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.config.failure_threshold:
                logger.warning(
                    "Circuit breaker opening",
                    extra={"failure_count": self.failure_count}
                )
                self.state = CircuitState.OPEN
                self.success_count = 0


@dataclass
class QdrantConfig:
    """Qdrant client configuration."""
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = True
    api_key: Optional[str] = None
    timeout: float = 30.0
    connection_pool_size: int = 100
    max_retries: int = 3
    retry_backoff_factor: float = 0.5


class QdrantClient:
    """Production-quality async Qdrant client with connection pooling and resilience.

    Features:
    - Async connection pooling with configurable size
    - Circuit breaker pattern for fault tolerance
    - Automatic retry with exponential backoff
    - Proper timeout handling
    - Resource cleanup with context managers
    - Structured logging
    - Connection health monitoring
    """

    def __init__(self, config: Optional[QdrantConfig] = None):
        """Initialize Qdrant client.

        Args:
            config: Client configuration. If None, uses defaults.
        """
        self.config = config or QdrantConfig()
        self._client: Optional[AsyncQdrantClient] = None
        self._circuit_breaker = CircuitBreaker()
        self._connection_pool_semaphore = asyncio.Semaphore(self.config.connection_pool_size)
        self._is_connected = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown = False

        logger.info(
            "Initializing Qdrant client",
            extra={
                "host": self.config.host,
                "port": self.config.port,
                "prefer_grpc": self.config.prefer_grpc,
                "pool_size": self.config.connection_pool_size,
            }
        )

    async def connect(self) -> None:
        """Establish connection to Qdrant.

        Raises:
            ConnectionError: If connection fails after retries.
        """
        if self._is_connected:
            logger.debug("Already connected to Qdrant")
            return

        for attempt in range(self.config.max_retries):
            try:
                self._client = AsyncQdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                    grpc_port=self.config.grpc_port,
                    prefer_grpc=self.config.prefer_grpc,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                )

                # Verify connection
                await asyncio.wait_for(
                    self._client.get_collections(),
                    timeout=self.config.timeout
                )

                self._is_connected = True
                logger.info("Connected to Qdrant successfully")

                # Start health check background task
                self._health_check_task = asyncio.create_task(self._health_check_loop())
                return

            except Exception as e:
                wait_time = self.config.retry_backoff_factor * (2 ** attempt)
                logger.warning(
                    f"Connection attempt {attempt + 1}/{self.config.max_retries} failed",
                    extra={
                        "error": str(e),
                        "retry_in": wait_time,
                    }
                )

                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(wait_time)
                else:
                    raise ConnectionError(f"Failed to connect to Qdrant after {self.config.max_retries} attempts") from e

    async def disconnect(self) -> None:
        """Close connection to Qdrant and cleanup resources."""
        self._shutdown = True

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.close()
            self._client = None

        self._is_connected = False
        logger.info("Disconnected from Qdrant")

    async def _health_check_loop(self) -> None:
        """Background task to monitor connection health."""
        while not self._shutdown:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                if self._client:
                    await asyncio.wait_for(
                        self._client.get_collections(),
                        timeout=5.0
                    )
                    logger.debug("Health check passed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check failed", extra={"error": str(e)})
                self._is_connected = False

    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[AsyncQdrantClient]:
        """Get client with connection pool management.

        Yields:
            Configured Qdrant client.

        Raises:
            RuntimeError: If not connected.
        """
        if not self._is_connected or not self._client:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        async with self._connection_pool_semaphore:
            yield self._client

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry.

        Args:
            func: Async function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function result.

        Raises:
            Last exception if all retries fail.
        """
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                return await self._circuit_breaker.call(func, *args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_backoff_factor * (2 ** attempt)
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self.config.max_retries}",
                        extra={"error": str(e), "wait_time": wait_time}
                    )
                    await asyncio.sleep(wait_time)

        logger.error(
            "All retry attempts exhausted",
            extra={"attempts": self.config.max_retries}
        )
        raise last_exception

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: models.Distance = models.Distance.COSINE,
        **kwargs
    ) -> bool:
        """Create a new collection with retry logic.

        Args:
            collection_name: Name of collection.
            vector_size: Dimension of vectors.
            distance: Distance metric.
            **kwargs: Additional collection parameters.

        Returns:
            True if successful.
        """
        async def _create():
            async with self._get_client() as client:
                return await client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance,
                    ),
                    **kwargs
                )

        result = await self._retry_with_backoff(_create)
        logger.info(
            "Collection created",
            extra={
                "collection": collection_name,
                "vector_size": vector_size,
                "distance": distance.value,
            }
        )
        return result

    async def upsert_batch(
        self,
        collection_name: str,
        points: List[models.PointStruct],
        batch_size: int = 100,
        wait: bool = True,
    ) -> List[models.UpdateResult]:
        """Upsert points in batches with efficient processing.

        Args:
            collection_name: Target collection.
            points: Points to upsert.
            batch_size: Batch size for processing.
            wait: Wait for operation completion.

        Returns:
            List of update results.
        """
        results = []

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]

            async def _upsert():
                async with self._get_client() as client:
                    return await client.upsert(
                        collection_name=collection_name,
                        points=batch,
                        wait=wait,
                    )

            result = await self._retry_with_backoff(_upsert)
            results.append(result)

            logger.debug(
                "Batch upserted",
                extra={
                    "collection": collection_name,
                    "batch_num": i // batch_size + 1,
                    "points": len(batch),
                }
            )

        logger.info(
            "All batches upserted",
            extra={
                "collection": collection_name,
                "total_points": len(points),
                "batches": len(results),
            }
        )
        return results

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        query_filter: Optional[models.Filter] = None,
        **kwargs
    ) -> List[models.ScoredPoint]:
        """Search for similar vectors with filtering.

        Args:
            collection_name: Collection to search.
            query_vector: Query vector.
            limit: Maximum results.
            score_threshold: Minimum score threshold.
            query_filter: Metadata filter.
            **kwargs: Additional search parameters.

        Returns:
            List of scored points.
        """
        async def _search():
            async with self._get_client() as client:
                return await client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                    **kwargs
                )

        start_time = time.time()
        results = await self._retry_with_backoff(_search)
        duration = time.time() - start_time

        logger.info(
            "Search completed",
            extra={
                "collection": collection_name,
                "results": len(results),
                "duration_ms": round(duration * 1000, 2),
                "limit": limit,
            }
        )
        return results

    async def search_batch(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        limit: int = 10,
        **kwargs
    ) -> List[List[models.ScoredPoint]]:
        """Batch search for multiple query vectors efficiently.

        Args:
            collection_name: Collection to search.
            query_vectors: List of query vectors.
            limit: Maximum results per query.
            **kwargs: Additional search parameters.

        Returns:
            List of result lists.
        """
        async def _search_batch():
            async with self._get_client() as client:
                search_requests = [
                    models.SearchRequest(
                        vector=vector,
                        limit=limit,
                        **kwargs
                    )
                    for vector in query_vectors
                ]
                return await client.search_batch(
                    collection_name=collection_name,
                    requests=search_requests,
                )

        start_time = time.time()
        results = await self._retry_with_backoff(_search_batch)
        duration = time.time() - start_time

        logger.info(
            "Batch search completed",
            extra={
                "collection": collection_name,
                "queries": len(query_vectors),
                "duration_ms": round(duration * 1000, 2),
            }
        )
        return results

    async def delete_points(
        self,
        collection_name: str,
        points_selector: Union[models.PointIdsList, models.Filter],
        wait: bool = True,
    ) -> models.UpdateResult:
        """Delete points from collection.

        Args:
            collection_name: Target collection.
            points_selector: Points to delete.
            wait: Wait for operation completion.

        Returns:
            Update result.
        """
        async def _delete():
            async with self._get_client() as client:
                return await client.delete(
                    collection_name=collection_name,
                    points_selector=points_selector,
                    wait=wait,
                )

        result = await self._retry_with_backoff(_delete)
        logger.info(
            "Points deleted",
            extra={"collection": collection_name}
        )
        return result

    async def get_collection_info(self, collection_name: str) -> models.CollectionInfo:
        """Get collection information.

        Args:
            collection_name: Collection name.

        Returns:
            Collection information.
        """
        async def _get_info():
            async with self._get_client() as client:
                return await client.get_collection(collection_name=collection_name)

        return await self._retry_with_backoff(_get_info)

    async def __aenter__(self):
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        await self.disconnect()
        return False
