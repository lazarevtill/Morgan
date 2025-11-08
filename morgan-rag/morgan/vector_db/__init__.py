"""
Enhanced vector database integration for Morgan RAG with companion features.
"""

from .client import BatchOperationResult, CollectionInfo, SearchResult, VectorDBClient

__all__ = ["VectorDBClient", "SearchResult", "BatchOperationResult", "CollectionInfo"]
