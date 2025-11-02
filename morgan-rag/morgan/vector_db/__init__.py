"""
Enhanced vector database integration for Morgan RAG with companion features.
"""

from .client import VectorDBClient, SearchResult, BatchOperationResult, CollectionInfo

__all__ = [
    'VectorDBClient',
    'SearchResult', 
    'BatchOperationResult',
    'CollectionInfo'
]