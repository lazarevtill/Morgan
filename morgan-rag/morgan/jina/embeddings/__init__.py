"""
Jina AI Embedding Services

Simple embedding generation services following KISS principles.
"""

from .service import JinaEmbeddingService
from .multimodal_service import (
    MultimodalContentProcessor,
    MultimodalDocument,
    MultimodalEmbedding,
    MultimodalSearchResult,
    ImageContent
)

__all__ = [
    'JinaEmbeddingService',
    'MultimodalContentProcessor',
    'MultimodalDocument',
    'MultimodalEmbedding',
    'MultimodalSearchResult',
    'ImageContent'
]