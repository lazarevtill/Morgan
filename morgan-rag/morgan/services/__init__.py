"""
Services module for Morgan RAG.

Core services:
- LLM Service: Language model integration
- External Knowledge: Web search and Context7 documentation
- Distributed Services: Multi-host LLM and embedding services
"""

from .llm_service import LLMService, get_llm_service

# External knowledge services
from .external_knowledge import (
    ExternalKnowledgeService,
    ExternalKnowledgeResult,
    KnowledgeSource,
    get_external_knowledge_service,
    WebSearchService,
    WebSearchResult,
    get_web_search_service,
    Context7Service,
    DocumentationMode,
    LibraryInfo,
    DocumentationResult,
    get_context7_service,
    MCPClient,
    get_mcp_client,
    mcp_web_search,
    mcp_resolve_library,
    mcp_get_library_docs,
)

__all__ = [
    # Core LLM
    "LLMService",
    "get_llm_service",
    # External Knowledge - Unified
    "ExternalKnowledgeService",
    "ExternalKnowledgeResult",
    "KnowledgeSource",
    "get_external_knowledge_service",
    # External Knowledge - Web Search
    "WebSearchService",
    "WebSearchResult",
    "get_web_search_service",
    # External Knowledge - Context7
    "Context7Service",
    "DocumentationMode",
    "LibraryInfo",
    "DocumentationResult",
    "get_context7_service",
    # MCP Client
    "MCPClient",
    "get_mcp_client",
    "mcp_web_search",
    "mcp_resolve_library",
    "mcp_get_library_docs",
]
