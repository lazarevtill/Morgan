"""
Vectorization module for Morgan RAG.

This module contains advanced vectorization capabilities including:
- Hierarchical embeddings (coarse, medium, fine scales)
- Contrastive clustering for improved search quality
- Multi-stage search with result fusion
- Intelligent caching with Git hash tracking
"""

from .hierarchical_embeddings import (
    HierarchicalEmbeddingService,
    HierarchicalEmbedding,
    get_hierarchical_embedding_service
)
from .contrastive_clustering import (
    ContrastiveClusteringEngine,
    get_contrastive_clustering_engine
)

__all__ = [
    'HierarchicalEmbeddingService',
    'HierarchicalEmbedding',
    'get_hierarchical_embedding_service',
    'ContrastiveClusteringEngine',
    'get_contrastive_clustering_engine'
]