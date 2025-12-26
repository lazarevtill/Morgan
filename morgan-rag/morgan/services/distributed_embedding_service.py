"""
Distributed Embedding Service for Morgan.

Integrates the distributed embedding infrastructure with Morgan's service layer,
providing a unified interface for embedding operations across distributed hosts.
"""

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.infrastructure.local_embeddings import (
    LocalEmbeddingService,
    get_local_embedding_service,
)
from morgan.infrastructure.distributed_gpu_manager import (
    DistributedGPUManager,
    HostRole,
    get_distributed_gpu_manager,
)
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Embedding result wrapper."""

    embedding: List[float]
    model: str
    dimensions: int
    latency_ms: float
    endpoint_used: Optional[str] = None


class DistributedEmbeddingService:
    """
    Distributed embedding service integrating multi-host infrastructure.

    Provides:
    - Automatic endpoint discovery from distributed GPU manager
    - Caching for repeated embeddings
    - Fallback to local embedding models
    - Both sync and async interfaces
    - Batch processing support

    Example:
        >>> service = DistributedEmbeddingService()
        >>>
        >>> # Single embedding
        >>> embedding = await service.aembed("What is Python?")
        >>>
        >>> # Batch embedding
        >>> embeddings = await service.aembed_batch([
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
        model: Optional[str] = None,
        dimensions: int = 768,
        auto_discover: bool = True,
        use_cache: bool = True,
    ):
        """
        Initialize distributed embedding service.

        Args:
            endpoint: Embedding endpoint URL (auto-discovered if None)
            model: Embedding model name
            dimensions: Embedding dimensions
            auto_discover: Auto-discover endpoints from GPU manager
            use_cache: Enable caching
        """
        self.settings = get_settings()
        self._lock = threading.Lock()
        self._initialized = False

        # Configuration (Qwen3-Embedding via Ollama by default)
        self.model = model or getattr(
            self.settings, "embedding_model", "qwen3-embedding:4b"
        )
        self.dimensions = dimensions
        self._endpoint = endpoint
        self._auto_discover = auto_discover
        self._use_cache = use_cache

        # Service instances (lazy initialization)
        self._embedding_service: Optional[LocalEmbeddingService] = None
        self._gpu_manager: Optional[DistributedGPUManager] = None

        logger.info(
            f"DistributedEmbeddingService created: "
            f"model={self.model}, dimensions={self.dimensions}"
        )

    async def _ensure_initialized(self):
        """Ensure service is initialized with endpoints."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            endpoint = self._endpoint

            # Auto-discover endpoints if needed
            if endpoint is None and self._auto_discover:
                try:
                    self._gpu_manager = get_distributed_gpu_manager()
                    endpoints = await self._gpu_manager.get_endpoints(
                        HostRole.EMBEDDINGS
                    )
                    if endpoints:
                        endpoint = endpoints[0]
                        logger.info(f"Auto-discovered embedding endpoint: {endpoint}")
                except Exception as e:
                    logger.warning(f"Failed to auto-discover endpoints: {e}")

            # Fallback to settings
            if not endpoint:
                endpoint = getattr(self.settings, "embedding_endpoint", None)

            # Initialize local embedding service
            try:
                self._embedding_service = LocalEmbeddingService(
                    endpoint=endpoint,
                    model=self.model,
                    dimensions=self.dimensions,
                    use_cache=self._use_cache,
                )
            except Exception as e:
                logger.warning(f"Failed to create embedding service: {e}")
                # Create with local fallback only
                self._embedding_service = LocalEmbeddingService(
                    endpoint=None,
                    model=self.model,
                    dimensions=self.dimensions,
                    use_cache=self._use_cache,
                )

            self._initialized = True
            logger.info("DistributedEmbeddingService initialized")

    async def aembed(
        self,
        text: str,
        use_cache: bool = True,
    ) -> List[float]:
        """
        Generate embedding for text asynchronously.

        Args:
            text: Text to embed
            use_cache: Whether to use cache

        Returns:
            Embedding vector
        """
        await self._ensure_initialized()
        return await self._embedding_service.embed_text(text, use_cache=use_cache)

    async def aembed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts asynchronously.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use cache
            show_progress: Show progress bar

        Returns:
            List of embedding vectors
        """
        await self._ensure_initialized()
        return await self._embedding_service.embed_batch(
            texts, use_cache=use_cache, show_progress=show_progress
        )

    # Synchronous wrappers

    def embed(self, text: str, use_cache: bool = True) -> List[float]:
        """Synchronous embed wrapper."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.aembed(text, use_cache=use_cache))
        finally:
            loop.close()

    def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """Synchronous batch embed wrapper."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.aembed_batch(
                    texts, use_cache=use_cache, show_progress=show_progress
                )
            )
        finally:
            loop.close()

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimensions

    async def is_available(self) -> bool:
        """Check if embedding service is available."""
        await self._ensure_initialized()
        return await self._embedding_service.is_available()

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        if self._embedding_service:
            return self._embedding_service.get_stats()
        return {}

    def clear_cache(self):
        """Clear embedding cache."""
        if self._embedding_service:
            self._embedding_service.clear_cache()


# Singleton instance
_service_instance: Optional[DistributedEmbeddingService] = None
_service_lock = threading.Lock()


def get_distributed_embedding_service(
    endpoint: Optional[str] = None,
    **kwargs,
) -> DistributedEmbeddingService:
    """
    Get singleton distributed embedding service instance.

    Args:
        endpoint: Optional endpoint URL
        **kwargs: Additional service configuration

    Returns:
        Shared DistributedEmbeddingService instance
    """
    global _service_instance

    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = DistributedEmbeddingService(
                    endpoint=endpoint,
                    **kwargs,
                )

    return _service_instance
