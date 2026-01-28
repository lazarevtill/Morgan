"""
Unified Embedding Service for Morgan AI Assistant.

This module provides a single, consolidated embedding service that supports:
- Remote Ollama/OpenAI-compatible endpoints
- Local sentence-transformers fallback
- Automatic failover between providers
- Batch processing with configurable size
- Content-based caching for performance
- Both sync and async interfaces

Usage:
    from morgan.services.embeddings import get_embedding_service

    # Get singleton service
    service = get_embedding_service()

    # Encode single text (sync)
    embedding = service.encode("What is Python?")

    # Encode batch (sync)
    embeddings = service.encode_batch(["Doc 1", "Doc 2", "Doc 3"])

    # Async encoding
    embedding = await service.aencode("Explain Docker")
"""

from morgan.services.embeddings.models import EmbeddingStats
from morgan.services.embeddings.service import (
    EmbeddingService,
    get_embedding_service,
    reset_embedding_service,
)

__all__ = [
    "EmbeddingService",
    "EmbeddingStats",
    "get_embedding_service",
    "reset_embedding_service",
]
