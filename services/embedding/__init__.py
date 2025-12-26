"""
Embedding Service for Morgan AI Assistant.

Production-ready embedding service for Docker deployment.
Uses sentence-transformers for local embedding generation.
"""

from .service import (
    ProductionEmbeddingService,
    get_production_embedding_service,
)

__all__ = [
    "ProductionEmbeddingService",
    "get_production_embedding_service",
]
