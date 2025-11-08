"""Morgan RAG - Production-quality Retrieval-Augmented Generation pipeline.

This package provides a complete RAG implementation with:
- Multi-stage hierarchical search
- Async processing throughout
- Connection pooling and resource management
- Circuit breaker pattern for resilience
- Efficient batching and streaming
- Structured logging and metrics
"""

__version__ = "2.0.0"

from morgan.core.search import MultiStageSearch, SearchConfig
from morgan.ingestion.enhanced_processor import EnhancedDocumentProcessor, ProcessingConfig
from morgan.jina.reranking.service import RerankingService, RerankingConfig
from morgan.services.embedding_service import EmbeddingService, EmbeddingConfig
from morgan.vector_db.client import QdrantClient, QdrantConfig

__all__ = [
    "MultiStageSearch",
    "SearchConfig",
    "EnhancedDocumentProcessor",
    "ProcessingConfig",
    "RerankingService",
    "RerankingConfig",
    "EmbeddingService",
    "EmbeddingConfig",
    "QdrantClient",
    "QdrantConfig",
]
