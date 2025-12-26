"""
Infrastructure Factory for Morgan.

Unified factory for creating and managing all infrastructure services:
- LLM Service (single or distributed)
- Embedding Service
- Reranking Service
- Connection Pools
- Batch Processors

This provides a single point of configuration for Morgan infrastructure.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from morgan.config import get_settings
from morgan.config.distributed_config import (
    DistributedArchitectureConfig,
    get_distributed_config,
)
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InfrastructureServices:
    """Container for all infrastructure services."""

    llm_service: Any
    embedding_service: Optional[Any] = None
    reranking_service: Optional[Any] = None
    connection_pool_manager: Optional[Any] = None
    batch_processor: Optional[Any] = None
    config: Optional[DistributedArchitectureConfig] = None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all services."""
        stats = {}

        if hasattr(self.llm_service, "get_stats"):
            stats["llm"] = self.llm_service.get_stats()

        if self.embedding_service:
            if hasattr(self.embedding_service, "get_stats"):
                stats["embedding"] = self.embedding_service.get_stats()

        if self.reranking_service:
            if hasattr(self.reranking_service, "get_stats"):
                stats["reranking"] = self.reranking_service.get_stats()

        if self.connection_pool_manager:
            if hasattr(self.connection_pool_manager, "get_all_stats"):
                stats["pools"] = self.connection_pool_manager.get_all_stats()

        if self.batch_processor:
            if hasattr(self.batch_processor, "get_performance_metrics"):
                stats["batch"] = self.batch_processor.get_performance_metrics()

        return stats


class InfrastructureFactory:
    """
    Factory for creating Morgan infrastructure services.

    Supports two modes:
    1. Simple mode: Single-host setup with basic configuration
    2. Distributed mode: Multi-host setup with load balancing

    Example:
        # Simple mode
        >>> factory = InfrastructureFactory()
        >>> services = factory.create_services()

        # Distributed mode
        >>> factory = InfrastructureFactory(distributed=True)
        >>> services = await factory.create_services_async()

        # Access services
        >>> response = services.llm_service.generate("Hello")
    """

    def __init__(
        self,
        distributed: bool = False,
        config_path: Optional[str] = None,
    ):
        """
        Initialize factory.

        Args:
            distributed: Enable distributed mode
            config_path: Path to distributed config file
        """
        self.settings = get_settings()
        self.distributed = distributed
        self.config_path = config_path

        # Load distributed config if needed
        self.distributed_config = None
        if distributed:
            self.distributed_config = get_distributed_config(config_path=config_path)
            self.distributed_config.setup_model_cache()

        logger.info("InfrastructureFactory initialized (distributed=%s)", distributed)

    def create_services(self) -> InfrastructureServices:
        """
        Create all infrastructure services (synchronous).

        Returns:
            InfrastructureServices container
        """
        # Create LLM service
        llm_service = self._create_llm_service()

        # Create embedding service
        embedding_service = self._create_embedding_service()

        # Create reranking service
        reranking_service = self._create_reranking_service()

        # Create batch processor
        batch_processor = self._create_batch_processor()

        return InfrastructureServices(
            llm_service=llm_service,
            embedding_service=embedding_service,
            reranking_service=reranking_service,
            batch_processor=batch_processor,
            config=self.distributed_config,
        )

    async def create_services_async(self) -> InfrastructureServices:
        """
        Create all infrastructure services (asynchronous).

        Useful for distributed mode where async initialization is needed.

        Returns:
            InfrastructureServices container
        """
        # Create LLM service
        llm_service = self._create_llm_service()

        # Create embedding service
        embedding_service = await self._create_embedding_service_async()

        # Create reranking service
        reranking_service = await self._create_reranking_service_async()

        # Create connection pool manager
        pool_manager = await self._create_connection_pool_manager()

        # Create batch processor
        batch_processor = self._create_batch_processor()

        return InfrastructureServices(
            llm_service=llm_service,
            embedding_service=embedding_service,
            reranking_service=reranking_service,
            connection_pool_manager=pool_manager,
            batch_processor=batch_processor,
            config=self.distributed_config,
        )

    def _create_llm_service(self):
        """Create LLM service."""
        from morgan.services.llm_service import LLMService

        if self.distributed and self.distributed_config:
            # Get LLM endpoints from config
            llm_hosts = self.distributed_config.get_hosts_by_role("llm")

            if llm_hosts:
                endpoints = [
                    f"http://{h.address}:{h.port}{h.api_path}" for h in llm_hosts
                ]
                strategy = self.distributed_config.settings.load_balancing_strategy

                return LLMService(
                    mode="distributed",
                    endpoints=endpoints,
                    load_balancing_strategy=strategy,
                )

        # Single mode
        return LLMService(mode="single")

    def _create_embedding_service(self):
        """Create embedding service (synchronous)."""
        try:
            from morgan.infrastructure.local_embeddings import LocalEmbeddingService

            if self.distributed and self.distributed_config:
                # Get embedding host
                emb_hosts = self.distributed_config.get_hosts_by_role("embeddings")

                if emb_hosts:
                    host = emb_hosts[0]
                    endpoint = f"http://{host.address}:{host.port}"
                    endpoint += host.api_path

                    return LocalEmbeddingService(
                        endpoint=endpoint,
                        model=self.distributed_config.embeddings.model,
                        dimensions=self.distributed_config.embeddings.dimensions,
                    )

            # Default local embedding service
            return LocalEmbeddingService()

        except ImportError:
            logger.warning("LocalEmbeddingService not available")
            return None

    async def _create_embedding_service_async(self):
        """Create embedding service (asynchronous)."""
        return self._create_embedding_service()

    def _create_reranking_service(self):
        """Create reranking service (synchronous)."""
        try:
            from morgan.infrastructure.local_reranking import LocalRerankingService

            if self.distributed and self.distributed_config:
                # Get reranking host
                rerank_hosts = self.distributed_config.get_hosts_by_role("reranking")

                if rerank_hosts:
                    host = rerank_hosts[0]
                    endpoint = f"http://{host.address}:{host.port}/rerank"

                    return LocalRerankingService(
                        endpoint=endpoint,
                        model=self.distributed_config.reranking.model,
                    )

            # Default local reranking service
            return LocalRerankingService()

        except ImportError:
            logger.warning("LocalRerankingService not available")
            return None

    async def _create_reranking_service_async(self):
        """Create reranking service (asynchronous)."""
        return self._create_reranking_service()

    async def _create_connection_pool_manager(self):
        """Create and start connection pool manager."""
        try:
            from morgan.optimization.connection_pool import (
                ConnectionConfig,
                get_connection_pool_manager,
            )

            manager = get_connection_pool_manager()

            if self.distributed and self.distributed_config:
                # Create pools for each service type
                config = ConnectionConfig(
                    min_connections=2,
                    max_connections=10,
                    initial_connections=3,
                )

                # LLM connection pools
                for host in self.distributed_config.get_hosts_by_role("llm"):
                    url = f"http://{host.address}:{host.port}"
                    try:
                        await manager.create_http_pool(
                            name=f"llm_{host.host_id}",
                            base_url=url,
                            config=config,
                        )
                    except ValueError:
                        pass  # Pool already exists

            # Start the manager
            await manager.start()

            return manager

        except ImportError:
            logger.warning("ConnectionPoolManager not available")
            return None

    def _create_batch_processor(self):
        """Create batch processor."""
        try:
            from morgan.optimization.batch_processor import (
                BatchProcessor,
                BatchConfig,
            )

            config = BatchConfig(
                batch_size=100,
                max_workers=4,
                adaptive_sizing=True,
            )

            return BatchProcessor(config)

        except ImportError:
            logger.warning("BatchProcessor not available")
            return None


# Global instance
_infrastructure_services: Optional[InfrastructureServices] = None


def get_infrastructure_services(
    distributed: bool = False,
    config_path: Optional[str] = None,
    force_new: bool = False,
) -> InfrastructureServices:
    """
    Get global infrastructure services instance.

    Args:
        distributed: Enable distributed mode
        config_path: Path to distributed config
        force_new: Force create new instance

    Returns:
        InfrastructureServices instance
    """
    global _infrastructure_services

    if _infrastructure_services is None or force_new:
        factory = InfrastructureFactory(
            distributed=distributed,
            config_path=config_path,
        )
        _infrastructure_services = factory.create_services()

    return _infrastructure_services


async def get_infrastructure_services_async(
    distributed: bool = False,
    config_path: Optional[str] = None,
    force_new: bool = False,
) -> InfrastructureServices:
    """
    Get global infrastructure services instance (async version).

    Args:
        distributed: Enable distributed mode
        config_path: Path to distributed config
        force_new: Force create new instance

    Returns:
        InfrastructureServices instance
    """
    global _infrastructure_services

    if _infrastructure_services is None or force_new:
        factory = InfrastructureFactory(
            distributed=distributed,
            config_path=config_path,
        )
        _infrastructure_services = await factory.create_services_async()

    return _infrastructure_services
