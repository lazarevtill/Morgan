"""
Jina AI Integration Module

This module provides simple, focused integration with Jina AI models following KISS principles.
Each service has a single responsibility and minimal interface.
"""

from .models.selector import ModelSelector
from .embeddings.service import JinaEmbeddingService
from .reranking.service import JinaRerankingService
from .scraping.service import JinaWebScrapingService

__all__ = [
    'ModelSelector',
    'JinaEmbeddingService', 
    'JinaRerankingService',
    'JinaWebScrapingService'
]