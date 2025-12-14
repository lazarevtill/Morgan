"""
Services module for Morgan RAG.

Provides unified interfaces for:
- LLM service (single or distributed)
- Embedding service (with separate host support)
- Reranking service (optional)
- Service factory for centralized management
"""

from .embedding_service import EmbeddingService, get_embedding_service
from .llm_service import LLMService, LLMResponse, get_llm_service, reset_llm_service
from .service_factory import (
    ServiceFactory,
    ServiceStatus,
    get_service_factory,
    reset_service_factory,
    get_reranking_service,
)

__all__ = [
    # Embedding service
    "EmbeddingService",
    "get_embedding_service",
    # LLM service
    "LLMService",
    "LLMResponse",
    "get_llm_service",
    "reset_llm_service",
    # Service factory
    "ServiceFactory",
    "ServiceStatus",
    "get_service_factory",
    "reset_service_factory",
    "get_reranking_service",
]
