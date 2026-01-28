"""
Unified Reranking Service for Morgan AI Assistant.

This module provides a single, consolidated reranking service that supports:
- Remote reranking endpoints
- Local CrossEncoder via sentence-transformers
- Embedding-based similarity fallback
- BM25-style lexical matching as last resort
- Performance tracking and statistics

Usage:
    from morgan.services.reranking import get_reranking_service, RerankResult

    # Get singleton service
    service = get_reranking_service()

    # Rerank documents (async)
    results = await service.rerank(
        query="What is Python?",
        documents=["Python is a language", "Java is a language"]
    )

    # Get top results
    for result in results[:5]:
        print(f"{result.score:.3f}: {result.text[:50]}...")
"""

from morgan.services.reranking.models import RerankResult, RerankStats
from morgan.services.reranking.service import (
    RerankingService,
    get_reranking_service,
    reset_reranking_service,
)

__all__ = [
    "RerankingService",
    "RerankResult",
    "RerankStats",
    "get_reranking_service",
    "reset_reranking_service",
]
