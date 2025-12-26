"""
External Knowledge Service Module for Morgan.

Provides integration with external knowledge sources via MCP servers:
- Web search for real-time information
- Context7 for library documentation and best practices
- Unified interface for external knowledge retrieval

This module enables Morgan to access up-to-date information beyond
the local knowledge base for better, more accurate responses.

Usage:
    >>> from morgan.services.external_knowledge import (
    ...     get_external_knowledge_service,
    ...     KnowledgeSource,
    ... )
    >>>
    >>> # Get the unified service
    >>> service = get_external_knowledge_service()
    >>>
    >>> # Search all sources
    >>> results = await service.search(
    ...     "How to implement async middleware in FastAPI",
    ...     sources=KnowledgeSource.ALL
    ... )
    >>>
    >>> # Use intelligent routing
    >>> results = await service.intelligent_search(
    ...     "FastAPI best practices for error handling"
    ... )
    >>>
    >>> # Get library documentation directly
    >>> docs = await service.get_library_info("fastapi", topic="middleware")
"""

from morgan.services.external_knowledge.web_search import (
    WebSearchService,
    WebSearchResult,
    get_web_search_service,
)
from morgan.services.external_knowledge.context7 import (
    Context7Service,
    DocumentationMode,
    LibraryInfo,
    DocumentationResult,
    get_context7_service,
)
from morgan.services.external_knowledge.service import (
    ExternalKnowledgeService,
    ExternalKnowledgeResult,
    KnowledgeSource,
    get_external_knowledge_service,
)
from morgan.services.external_knowledge.mcp_client import (
    MCPClient,
    MCPRequest,
    MCPResponse,
    MCPServerType,
    get_mcp_client,
    mcp_web_search,
    mcp_resolve_library,
    mcp_get_library_docs,
)

__all__ = [
    # Web Search
    "WebSearchService",
    "WebSearchResult",
    "get_web_search_service",
    # Context7 Documentation
    "Context7Service",
    "DocumentationMode",
    "LibraryInfo",
    "DocumentationResult",
    "get_context7_service",
    # Unified Service
    "ExternalKnowledgeService",
    "ExternalKnowledgeResult",
    "KnowledgeSource",
    "get_external_knowledge_service",
    # MCP Client
    "MCPClient",
    "MCPRequest",
    "MCPResponse",
    "MCPServerType",
    "get_mcp_client",
    "mcp_web_search",
    "mcp_resolve_library",
    "mcp_get_library_docs",
]
