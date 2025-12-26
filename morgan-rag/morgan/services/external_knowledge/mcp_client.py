"""
MCP Client for External Knowledge Services.

Provides the actual communication layer with MCP servers:
- Web search via MCP browser/search tools
- Context7 documentation retrieval

This module handles the low-level MCP protocol communication
and provides a clean async interface for higher-level services.
"""

import asyncio
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

from morgan.config import get_settings
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class MCPServerType(str, Enum):
    """Types of MCP servers."""

    PLAYWRIGHT = "playwright"  # Browser automation
    CONTEXT7 = "context7"  # Documentation


@dataclass
class MCPRequest:
    """Request to an MCP server."""

    server: MCPServerType
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0


@dataclass
class MCPResponse:
    """Response from an MCP server."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MCPClient:
    """
    MCP Client for communicating with Model Context Protocol servers.

    Handles the actual tool invocations for:
    - Web search via playwright browser tools
    - Context7 documentation retrieval

    This client is designed to be used by the higher-level
    services (WebSearchService, Context7Service).

    Example:
        >>> client = MCPClient()
        >>>
        >>> # Web search
        >>> response = await client.web_search("Python async patterns")
        >>>
        >>> # Context7 library resolution
        >>> response = await client.resolve_library("fastapi")
    """

    def __init__(self):
        """Initialize MCP client."""
        self.settings = get_settings()
        self._lock = threading.Lock()

        # Tool execution callbacks (set by the hosting environment)
        self._tool_executor: Optional[Callable] = None

        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "playwright_requests": 0,
            "context7_requests": 0,
        }

        logger.info("MCPClient initialized")

    def set_tool_executor(self, executor: Callable[[str, Dict[str, Any]], Any]) -> None:
        """
        Set the tool executor callback.

        This is called by the hosting environment (e.g., Cursor IDE)
        to provide the actual MCP tool execution capability.

        Args:
            executor: Callback function (tool_name, params) -> result
        """
        self._tool_executor = executor
        logger.info("MCP tool executor registered")

    async def web_search(
        self,
        search_term: str,
        explanation: str = "Searching for real-time information",
    ) -> MCPResponse:
        """
        Perform a web search.

        Uses the web_search MCP tool for real-time information retrieval.

        Args:
            search_term: Search query
            explanation: Explanation for the search

        Returns:
            MCPResponse with search results
        """
        self._stats["total_requests"] += 1
        self._stats["playwright_requests"] += 1

        start_time = asyncio.get_event_loop().time()

        try:
            request = MCPRequest(
                server=MCPServerType.PLAYWRIGHT,
                tool_name="web_search",
                parameters={
                    "search_term": search_term,
                    "explanation": explanation,
                },
            )

            result = await self._execute_tool(request)

            self._stats["successful_requests"] += 1

            return MCPResponse(
                success=True,
                data=result,
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

        except Exception as e:
            self._stats["failed_requests"] += 1
            logger.error(f"Web search failed: {e}")

            return MCPResponse(
                success=False,
                error=str(e),
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

    async def resolve_library(
        self,
        library_name: str,
    ) -> MCPResponse:
        """
        Resolve a library name to Context7 ID.

        Uses the Context7 MCP resolve-library-id tool.

        Args:
            library_name: Library name to resolve

        Returns:
            MCPResponse with library information
        """
        self._stats["total_requests"] += 1
        self._stats["context7_requests"] += 1

        start_time = asyncio.get_event_loop().time()

        try:
            request = MCPRequest(
                server=MCPServerType.CONTEXT7,
                tool_name="mcp_context7_resolve-library-id",
                parameters={
                    "libraryName": library_name,
                },
            )

            result = await self._execute_tool(request)

            self._stats["successful_requests"] += 1

            return MCPResponse(
                success=True,
                data=result,
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

        except Exception as e:
            self._stats["failed_requests"] += 1
            logger.error(f"Library resolution failed: {e}")

            return MCPResponse(
                success=False,
                error=str(e),
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

    async def get_library_docs(
        self,
        library_id: str,
        topic: Optional[str] = None,
        mode: str = "code",
        page: int = 1,
    ) -> MCPResponse:
        """
        Get documentation for a library.

        Uses the Context7 MCP get-library-docs tool.

        Args:
            library_id: Context7-compatible library ID
            topic: Optional topic to focus on
            mode: "code" for examples, "info" for concepts
            page: Page number for pagination

        Returns:
            MCPResponse with documentation content
        """
        self._stats["total_requests"] += 1
        self._stats["context7_requests"] += 1

        start_time = asyncio.get_event_loop().time()

        try:
            params = {
                "context7CompatibleLibraryID": library_id,
                "mode": mode,
                "page": page,
            }

            if topic:
                params["topic"] = topic

            request = MCPRequest(
                server=MCPServerType.CONTEXT7,
                tool_name="mcp_context7_get-library-docs",
                parameters=params,
            )

            result = await self._execute_tool(request)

            self._stats["successful_requests"] += 1

            return MCPResponse(
                success=True,
                data=result,
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

        except Exception as e:
            self._stats["failed_requests"] += 1
            logger.error(f"Documentation fetch failed: {e}")

            return MCPResponse(
                success=False,
                error=str(e),
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

    async def browser_navigate(
        self,
        url: str,
    ) -> MCPResponse:
        """
        Navigate browser to a URL.

        Args:
            url: URL to navigate to

        Returns:
            MCPResponse with navigation result
        """
        self._stats["total_requests"] += 1
        self._stats["playwright_requests"] += 1

        start_time = asyncio.get_event_loop().time()

        try:
            request = MCPRequest(
                server=MCPServerType.PLAYWRIGHT,
                tool_name="mcp_playwright_browser_navigate",
                parameters={"url": url},
            )

            result = await self._execute_tool(request)

            self._stats["successful_requests"] += 1

            return MCPResponse(
                success=True,
                data=result,
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

        except Exception as e:
            self._stats["failed_requests"] += 1
            logger.error(f"Browser navigation failed: {e}")

            return MCPResponse(
                success=False,
                error=str(e),
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

    async def browser_snapshot(self) -> MCPResponse:
        """
        Get accessibility snapshot of current page.

        Returns:
            MCPResponse with page snapshot
        """
        self._stats["total_requests"] += 1
        self._stats["playwright_requests"] += 1

        start_time = asyncio.get_event_loop().time()

        try:
            request = MCPRequest(
                server=MCPServerType.PLAYWRIGHT,
                tool_name="mcp_playwright_browser_snapshot",
                parameters={},
            )

            result = await self._execute_tool(request)

            self._stats["successful_requests"] += 1

            return MCPResponse(
                success=True,
                data=result,
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

        except Exception as e:
            self._stats["failed_requests"] += 1
            logger.error(f"Browser snapshot failed: {e}")

            return MCPResponse(
                success=False,
                error=str(e),
                execution_time=asyncio.get_event_loop().time() - start_time,
            )

    async def _execute_tool(self, request: MCPRequest) -> Any:
        """
        Execute an MCP tool request.

        This method handles the actual tool execution, either through
        a registered executor or by simulating the request.

        Args:
            request: MCP request to execute

        Returns:
            Tool execution result
        """
        if self._tool_executor:
            # Use registered executor (from hosting environment)
            return await asyncio.get_event_loop().run_in_executor(
                None,
                self._tool_executor,
                request.tool_name,
                request.parameters,
            )
        else:
            # Simulation mode - return empty result
            # In production, this would be replaced by actual MCP calls
            logger.debug(
                f"MCP tool simulation: {request.tool_name} "
                f"with params: {request.parameters}"
            )

            # Simulate async delay
            await asyncio.sleep(0.1)

            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            **self._stats,
            "has_executor": self._tool_executor is not None,
        }


# Singleton instance
_client_instance: Optional[MCPClient] = None
_client_lock = threading.Lock()


def get_mcp_client() -> MCPClient:
    """
    Get singleton MCP client instance.

    Returns:
        Shared MCPClient instance
    """
    global _client_instance

    if _client_instance is None:
        with _client_lock:
            if _client_instance is None:
                _client_instance = MCPClient()

    return _client_instance


# Convenience functions for direct tool access


async def mcp_web_search(
    search_term: str,
    explanation: str = "Searching for information",
) -> Optional[Any]:
    """
    Convenience function for web search.

    Args:
        search_term: Search query
        explanation: Explanation for the search

    Returns:
        Search results or None on error
    """
    client = get_mcp_client()
    response = await client.web_search(search_term, explanation)
    return response.data if response.success else None


async def mcp_resolve_library(library_name: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function for library resolution.

    Args:
        library_name: Library name

    Returns:
        Library info or None on error
    """
    client = get_mcp_client()
    response = await client.resolve_library(library_name)
    return response.data if response.success else None


async def mcp_get_library_docs(
    library_id: str,
    topic: Optional[str] = None,
    mode: str = "code",
) -> Optional[str]:
    """
    Convenience function for documentation retrieval.

    Args:
        library_id: Context7 library ID
        topic: Optional topic
        mode: "code" or "info"

    Returns:
        Documentation content or None on error
    """
    client = get_mcp_client()
    response = await client.get_library_docs(library_id, topic, mode)
    return response.data if response.success else None
