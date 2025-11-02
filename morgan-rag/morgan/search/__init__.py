"""
Multi-Stage Search Engine for Morgan RAG.

Advanced search capabilities with hierarchical filtering and result fusion.
"""

from .multi_stage_search import (
    MultiStageSearchEngine,
    SearchStrategy,
    SearchResult,
    SearchResults,
    get_multi_stage_search_engine
)

__all__ = [
    "MultiStageSearchEngine",
    "SearchStrategy",
    "SearchResult",
    "SearchResults",
    "get_multi_stage_search_engine"
]