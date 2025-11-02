"""
Memory processing module for Morgan RAG.

Provides enhanced memory processing with emotional awareness, importance scoring,
and personal preference detection for building meaningful companion relationships.
"""

from .memory_processor import (
    MemoryProcessor,
    Memory,
    MemoryExtractionResult,
    get_memory_processor
)

__all__ = [
    'MemoryProcessor',
    'Memory', 
    'MemoryExtractionResult',
    'get_memory_processor'
]