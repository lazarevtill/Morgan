"""
Morgan Services Layer.

Provides unified access to all Morgan services:
- LLM Service: Text generation with distributed support
- Embedding Service: Text embeddings with fallback
- Reranking Service: Document reranking with multiple strategies

Usage:
    from morgan.services import (
        get_llm_service,
        get_embedding_service,
        get_reranking_service,
    )

    # Get services
    llm = get_llm_service()
    embeddings = get_embedding_service()
    reranking = get_reranking_service()

    # Use services
    response = llm.generate("What is Python?")
    embedding = embeddings.encode("Document text")
    results = await reranking.rerank("query", ["doc1", "doc2"])
"""

from morgan.services.llm import (
    LLMService,
    LLMResponse,
    LLMMode,
    get_llm_service,
    reset_llm_service,
)

from morgan.services.embeddings import (
    EmbeddingService,
    EmbeddingStats,
    get_embedding_service,
    reset_embedding_service,
)

from morgan.services.reranking import (
    RerankingService,
    RerankResult,
    RerankStats,
    get_reranking_service,
    reset_reranking_service,
)

__all__ = [
    # LLM
    "LLMService",
    "LLMResponse",
    "LLMMode",
    "get_llm_service",
    "reset_llm_service",
    # Embeddings
    "EmbeddingService",
    "EmbeddingStats",
    "get_embedding_service",
    "reset_embedding_service",
    # Reranking
    "RerankingService",
    "RerankResult",
    "RerankStats",
    "get_reranking_service",
    "reset_reranking_service",
]


def initialize_services(
    llm_mode: str = "single",
    llm_endpoints: list = None,
    embedding_endpoint: str = None,
    reranking_endpoint: str = None,
) -> dict:
    """
    Initialize all services with optional configuration.

    Args:
        llm_mode: LLM mode ("single" or "distributed")
        llm_endpoints: LLM endpoint URLs for distributed mode
        embedding_endpoint: Embedding service endpoint
        reranking_endpoint: Reranking service endpoint

    Returns:
        Dictionary with initialized services
    """
    services = {}

    # Initialize LLM service
    services["llm"] = get_llm_service(
        mode=llm_mode,
        endpoints=llm_endpoints,
    )

    # Initialize embedding service
    services["embeddings"] = get_embedding_service(
        endpoint=embedding_endpoint,
    )

    # Initialize reranking service
    services["reranking"] = get_reranking_service(
        endpoint=reranking_endpoint,
    )

    return services


def shutdown_services():
    """Shutdown all services and cleanup resources."""
    reset_llm_service()
    reset_embedding_service()
    reset_reranking_service()


def get_service_stats() -> dict:
    """
    Get statistics from all services.

    Returns:
        Dictionary with stats from each service
    """
    stats = {}

    try:
        llm = get_llm_service()
        stats["llm"] = llm.get_stats()
    except Exception:
        stats["llm"] = {"error": "Not initialized"}

    try:
        embeddings = get_embedding_service()
        stats["embeddings"] = embeddings.get_stats()
    except Exception:
        stats["embeddings"] = {"error": "Not initialized"}

    try:
        reranking = get_reranking_service()
        stats["reranking"] = reranking.get_stats()
    except Exception:
        stats["reranking"] = {"error": "Not initialized"}

    return stats
