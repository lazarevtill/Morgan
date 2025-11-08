"""
Infrastructure Layer - Distributed Morgan Implementation

This layer provides infrastructure for fully self-hosted distributed operation:
- Distributed LLM client with load balancing and failover
- Multi-GPU management and model routing
- Local embedding service (OpenAI-compatible + local fallback)
- Local reranking service (remote + local fallback)
- No external API dependencies

Designed for 6-host distributed architecture:
- Host 1-2 (CPU): Morgan Core + Background Services
- Host 3-4 (3090): Main LLM load balanced
- Host 5 (4070): Embeddings + Fast LLM
- Host 6 (2060): Reranking + Utilities

Components:
    distributed_llm.py   - Distributed LLM with load balancing
    multi_gpu_manager.py - Manage models across GPUs
    local_embeddings.py  - Local embedding generation
    local_reranking.py   - Local reranking implementation
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
from morgan.infrastructure.multi_gpu_manager import MultiGPUManager

__all__ = [
    # Distributed LLM
    "DistributedLLMClient",
    "get_distributed_llm_client",
    "LoadBalancingStrategy",
    # GPU Management
    "MultiGPUManager",
    # Embeddings
    "LocalEmbeddingService",
    "get_local_embedding_service",
    # Reranking
    "LocalRerankingService",
    "get_local_reranking_service",
]
