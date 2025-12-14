"""
Infrastructure Layer - Distributed Morgan Implementation

This layer provides infrastructure for fully self-hosted distributed operation:
- Distributed LLM client with load balancing and failover
- Local embedding service (OpenAI-compatible + local fallback)
- Local reranking service (remote + local fallback)
- Distributed GPU management for multi-host setups
- No external API dependencies

Designed for 6-host distributed architecture:
- Host 1-2 (CPU): Morgan Core + Background Services
- Host 3-4 (3090): Main LLM load balanced
- Host 5 (4070): Embeddings + Fast LLM
- Host 6 (2060): Reranking + Utilities

Components:
    distributed_llm.py       - Distributed LLM with load balancing
    local_embeddings.py      - Local embedding generation
    local_reranking.py       - Local reranking implementation
    distributed_gpu_manager.py - Distributed GPU monitoring (optional)
    distributed_manager.py   - SSH-based host management (optional)
    consul_client.py         - Service discovery (optional)
"""

from morgan.infrastructure.distributed_llm import (
    DistributedLLMClient,
    LoadBalancingStrategy,
    get_distributed_llm_client,
)
from morgan.infrastructure.local_embeddings import (
    LocalEmbeddingService,
    get_local_embedding_service,
)
from morgan.infrastructure.local_reranking import (
    LocalRerankingService,
    get_local_reranking_service,
)

__all__ = [
    # Distributed LLM
    "DistributedLLMClient",
    "get_distributed_llm_client",
    "LoadBalancingStrategy",
    # Embeddings
    "LocalEmbeddingService",
    "get_local_embedding_service",
    # Reranking
    "LocalRerankingService",
    "get_local_reranking_service",
]


# Optional imports - don't fail if dependencies missing
def get_distributed_gpu_manager():
    """Get distributed GPU manager (optional - requires SSH hosts)."""
    from morgan.infrastructure.distributed_gpu_manager import (
        DistributedGPUManager,
        get_distributed_gpu_manager as _get_manager,
    )
    return _get_manager()


def get_distributed_host_manager():
    """Get distributed host manager for SSH operations (optional)."""
    from morgan.infrastructure.distributed_manager import (
        DistributedHostManager,
        get_distributed_manager as _get_manager,
    )
    return _get_manager()


def get_consul_registry():
    """Get Consul service registry (optional)."""
    from morgan.infrastructure.consul_client import (
        ConsulServiceRegistry,
        ConsulConfig,
    )
    return ConsulServiceRegistry(ConsulConfig())
