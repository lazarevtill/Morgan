"""
Jina AI Embedding Services

Simple embedding generation services following KISS principles.
"""

from .multimodal_service import (
    ImageContent,
    MultimodalContentProcessor,
    MultimodalDocument,
    MultimodalEmbedding,
    MultimodalSearchResult,
)
from .service import JinaEmbeddingService

__all__ = [
    "JinaEmbeddingService",
    "MultimodalContentProcessor",
    "MultimodalDocument",
    "MultimodalEmbedding",
    "MultimodalSearchResult",
    "ImageContent",
]
