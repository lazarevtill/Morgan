"""
Infrastructure Layer - Distributed Morgan Implementation

This layer provides infrastructure for fully self-hosted distributed operation:
- Distributed LLM client with load balancing and failover
- Multi-GPU management and model routing (single-host and distributed)
- Infrastructure factory for unified service creation
- No external API dependencies

Designed for 6-host distributed architecture:
- Host 1-2 (CPU): Morgan Core + Background Services
- Host 3-4 (3090): Main LLM load balanced
- Host 5 (4070): Embeddings + Fast LLM
- Host 6 (2060): Reranking + Utilities

Components:
    distributed_llm.py         - Distributed LLM with load balancing
    distributed_gpu_manager.py - Manage distributed GPU hosts
    multi_gpu_manager.py       - Manage models across local GPUs
    factory.py                 - Unified infrastructure factory

Note: Embedding and Reranking services are now in morgan.services
"""

from morgan.infrastructure.distributed_llm import (
    DistributedLLMClient,
    LoadBalancingStrategy,
    get_distributed_llm_client,
)
from morgan.infrastructure.distributed_gpu_manager import (
    DistributedGPUManager,
    DistributedConfig,
    HostRole,
    HostStatus,
    HostConfig,
    HostHealth,
    get_distributed_gpu_manager,
)
from morgan.infrastructure.multi_gpu_manager import MultiGPUManager
from morgan.infrastructure.factory import (
    InfrastructureFactory,
    InfrastructureServices,
    get_infrastructure_services,
    get_infrastructure_services_async,
)

# Re-export from services for backward compatibility
from morgan.services.embeddings import (
    EmbeddingService as LocalEmbeddingService,
    get_embedding_service as get_local_embedding_service,
)
from morgan.services.reranking import (
    RerankingService as LocalRerankingService,
    get_reranking_service as get_local_reranking_service,
)

__all__ = [
    # Distributed LLM
    "DistributedLLMClient",
    "get_distributed_llm_client",
    "LoadBalancingStrategy",
    # Distributed GPU Management
    "DistributedGPUManager",
    "DistributedConfig",
    "get_distributed_gpu_manager",
    "HostRole",
    "HostStatus",
    "HostConfig",
    "HostHealth",
    # Single-host GPU Management
    "MultiGPUManager",
    # Embeddings (re-exported from services)
    "LocalEmbeddingService",
    "get_local_embedding_service",
    # Reranking (re-exported from services)
    "LocalRerankingService",
    "get_local_reranking_service",
    # Infrastructure Factory
    "InfrastructureFactory",
    "InfrastructureServices",
    "get_infrastructure_services",
    "get_infrastructure_services_async",
]
