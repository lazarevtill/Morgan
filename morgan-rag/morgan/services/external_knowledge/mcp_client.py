"""
MCP Client for External Knowledge Services.

Provides the actual communication layer with MCP servers using the
official MCP Python SDK:
- Web search via MCP browser/search tools
- Context7 documentation retrieval

Supports multiple transports:
- stdio: For local MCP server processes
- SSE: For Server-Sent Events connections
- streamable-http: For HTTP-based servers

Example:
    >>> from morgan.services.external_knowledge.mcp_client import (
    ...     get_mcp_client,
    ... )
    >>>
    >>> client = get_mcp_client()
    >>> await client.connect_stdio("context7", "npx", ["-y", "@ctx7/mcp"])
    >>> response = await client.call_tool("resolve-library-id", {...})
"""

import asyncio
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import MCP SDK
_MCP_AVAILABLE = False
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    _MCP_AVAILABLE = True
    logger.info("MCP SDK loaded successfully")
except ImportError as e:
    logger.warning("MCP SDK not available: %s. pip install mcp", e)

# Try to import HTTP transport
_MCP_HTTP_AVAILABLE = False
try:
    from mcp.client.streamable_http import streamablehttp_client

    _MCP_HTTP_AVAILABLE = True
except ImportError:
    logger.debug("MCP streamable HTTP transport not available")

# Try to import SSE transport
_MCP_SSE_AVAILABLE = False
try:
    from mcp.client.sse import sse_client

    _MCP_SSE_AVAILABLE = True
except ImportError:
    logger.debug("MCP SSE transport not available")


class MCPServerType(str, Enum):
    """Types of MCP servers."""

    PLAYWRIGHT = "playwright"  # Browser automation
    CONTEXT7 = "context7"  # Documentation
    CUSTOM = "custom"  # Custom servers


class MCPTransport(str, Enum):
    """MCP transport types."""

    STDIO = "stdio"  # Standard I/O (local processes)
    SSE = "sse"  # Server-Sent Events
    HTTP = "streamable-http"  # Streamable HTTP


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""

    name: str
    transport: MCPTransport
    # For stdio
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    # For HTTP/SSE
    url: Optional[str] = None
    # Connection state
    connected: bool = False


@dataclass
class MCPResponse:
    """Response from an MCP server."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MCPClientError(Exception):
    """Raised when MCP client encounters an error."""


class MCPNotConnectedError(MCPClientError):
    """Raised when no MCP server is connected."""


class MCPSDKNotAvailableError(MCPClientError):
    """Raised when MCP SDK is not installed."""


class MCPClient:
    """
    MCP Client using the official Model Context Protocol SDK.

    Handles connections to MCP servers and tool invocations for:
    - Web search via playwright browser tools
    - Context7 documentation retrieval
    - Custom MCP servers

    Example:
        >>> client = MCPClient()
        >>>
        >>> # Connect to a stdio server
        >>> await client.connect_stdio(
        ...     "context7",
        ...     command="npx",
        ...     args=["-y", "@anthropic-ai/context7-mcp"]
        ... )
        >>>
        >>> # Or connect to HTTP server
        >>> await client.connect_http("myserver", "http://localhost:8000/mcp")
        >>>
        >>> # Call tools
        >>> result = await client.call_tool("resolve-library-id", {
        ...     "libraryName": "fastapi"
        ... })
    """

    def __init__(self):
        """Initialize MCP client."""
        self.settings = get_settings()
        self._lock = threading.Lock()

        # Active sessions per server
        self._sessions: Dict[str, ClientSession] = {}
        self._server_configs: Dict[str, MCPServerConfig] = {}

        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "playwright_requests": 0,
            "context7_requests": 0,
        }

        logger.info(
            "MCPClient initialized (SDK: %s, HTTP: %s, SSE: %s)",
            _MCP_AVAILABLE,
            _MCP_HTTP_AVAILABLE,
            _MCP_SSE_AVAILABLE,
        )

    # =========================================================================
    # Connection Methods
    # =========================================================================

    async def connect_stdio(
        self,
        server_name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Connect to an MCP server via stdio transport.

        Args:
            server_name: Name to identify this server
            command: Command to run (e.g., "python", "npx")
            args: Command arguments
            env: Environment variables

        Returns:
            True if connected successfully

        Raises:
            MCPSDKNotAvailableError: If MCP SDK is not installed
        """
        if not _MCP_AVAILABLE:
            raise MCPSDKNotAvailableError(
                "MCP SDK not available. Install with: pip install mcp"
            )

        try:
            self._server_configs[server_name] = MCPServerConfig(
                name=server_name,
                transport=MCPTransport.STDIO,
                command=command,
                args=args,
                env=env,
            )

            logger.info("Configured stdio: %s (%s)", server_name, command)
            self._server_configs[server_name].connected = True

            return True

        except Exception as e:
            logger.error("Failed to configure stdio %s: %s", server_name, e)
            raise MCPClientError(f"Failed to configure: {e}") from e

    async def connect_http(
        self,
        server_name: str,
        url: str,
    ) -> bool:
        """
        Connect to an MCP server via streamable HTTP transport.

        Args:
            server_name: Name to identify this server
            url: HTTP endpoint URL

        Returns:
            True if configured successfully

        Raises:
            MCPSDKNotAvailableError: If HTTP transport not available
        """
        if not _MCP_HTTP_AVAILABLE:
            raise MCPSDKNotAvailableError("MCP HTTP transport not available")

        try:
            self._server_configs[server_name] = MCPServerConfig(
                name=server_name,
                transport=MCPTransport.HTTP,
                url=url,
            )

            logger.info("Configured HTTP: %s (%s)", server_name, url)
            self._server_configs[server_name].connected = True

            return True

        except Exception as e:
            logger.error("Failed to configure HTTP %s: %s", server_name, e)
            raise MCPClientError(f"Failed to configure: {e}") from e

    async def connect_sse(
        self,
        server_name: str,
        url: str,
    ) -> bool:
        """
        Connect to an MCP server via SSE transport.

        Args:
            server_name: Name to identify this server
            url: SSE endpoint URL

        Returns:
            True if configured successfully

        Raises:
            MCPSDKNotAvailableError: If SSE transport not available
        """
        if not _MCP_SSE_AVAILABLE:
            raise MCPSDKNotAvailableError("MCP SSE transport not available")

        try:
            self._server_configs[server_name] = MCPServerConfig(
                name=server_name,
                transport=MCPTransport.SSE,
                url=url,
            )

            logger.info("Configured SSE: %s (%s)", server_name, url)
            self._server_configs[server_name].connected = True

            return True

        except Exception as e:
            logger.error("Failed to configure SSE %s: %s", server_name, e)
            raise MCPClientError(f"Failed to configure: {e}") from e

    def disconnect(self, server_name: str) -> bool:
        """
        Disconnect from an MCP server.

        Args:
            server_name: Server to disconnect from

        Returns:
            True if disconnected
        """
        if server_name in self._server_configs:
            self._server_configs[server_name].connected = False
            if server_name in self._sessions:
                del self._sessions[server_name]
            logger.info("Disconnected from server: %s", server_name)
            return True
        return False

    def is_connected(self) -> bool:
        """Check if any server is connected."""
        return any(cfg.connected for cfg in self._server_configs.values())

    def get_connected_servers(self) -> List[str]:
        """Get list of connected server names."""
        return [name for name, cfg in self._server_configs.items() if cfg.connected]

    # =========================================================================
    # Tool Execution
    # =========================================================================

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        server_name: Optional[str] = None,
    ) -> MCPResponse:
        """
        Call a tool on an MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            server_name: Specific server (or auto-detect)

        Returns:
            MCPResponse with result

        Raises:
            MCPNotConnectedError: If no server is connected
            MCPSDKNotAvailableError: If MCP SDK not installed
        """
        if not _MCP_AVAILABLE:
            raise MCPSDKNotAvailableError(
                "MCP SDK not available. Install with: pip install mcp"
            )

        if not self.is_connected():
            raise MCPNotConnectedError(
                "No MCP server connected. "
                "Call connect_stdio(), connect_http(), or connect_sse() first."
            )

        self._stats["total_requests"] += 1
        start_time = asyncio.get_event_loop().time()

        try:
            result = await self._call_tool_mcp(tool_name, arguments, server_name)

            if result is not None:
                self._stats["successful_requests"] += 1
                elapsed = asyncio.get_event_loop().time() - start_time
                return MCPResponse(
                    success=True,
                    data=result,
                    execution_time=elapsed,
                )

            # Tool returned None - still a valid response
            self._stats["successful_requests"] += 1
            elapsed = asyncio.get_event_loop().time() - start_time
            return MCPResponse(
                success=True,
                data=None,
                execution_time=elapsed,
            )

        except Exception as e:
            self._stats["failed_requests"] += 1
            logger.error("Tool call failed: %s", e)
            elapsed = asyncio.get_event_loop().time() - start_time
            return MCPResponse(
                success=False,
                error=str(e),
                execution_time=elapsed,
            )

    async def _call_tool_mcp(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        server_name: Optional[str] = None,
    ) -> Optional[Any]:
        """Execute tool via MCP SDK."""
        # Find appropriate server
        config = None
        if server_name and server_name in self._server_configs:
            config = self._server_configs[server_name]
        elif self._server_configs:
            # Use first connected server
            for _, cfg in self._server_configs.items():
                if cfg.connected:
                    config = cfg
                    break

        if not config:
            raise MCPNotConnectedError("No connected server found")

        if config.transport == MCPTransport.STDIO:
            return await self._call_stdio_tool(config, tool_name, arguments)
        elif config.transport == MCPTransport.HTTP:
            return await self._call_http_tool(config, tool_name, arguments)
        elif config.transport == MCPTransport.SSE:
            return await self._call_sse_tool(config, tool_name, arguments)

        raise MCPClientError(f"Unknown transport: {config.transport}")

    async def _call_stdio_tool(
        self,
        config: MCPServerConfig,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[Any]:
        """Call tool via stdio transport."""
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args or [],
            env=config.env or dict(os.environ),
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(tool_name, arguments=arguments)

                # Extract result content
                if result and result.content:
                    content = result.content[0]
                    if hasattr(content, "text"):
                        return content.text
                    elif hasattr(result, "structuredContent"):
                        return result.structuredContent

                return result

    async def _call_http_tool(
        self,
        config: MCPServerConfig,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[Any]:
        """Call tool via HTTP transport."""
        if not config.url:
            raise MCPClientError("HTTP config missing URL")

        async with streamablehttp_client(config.url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(tool_name, arguments=arguments)

                if result and result.content:
                    content = result.content[0]
                    if hasattr(content, "text"):
                        return content.text
                    elif hasattr(result, "structuredContent"):
                        return result.structuredContent

                return result

    async def _call_sse_tool(
        self,
        config: MCPServerConfig,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[Any]:
        """Call tool via SSE transport."""
        if not config.url:
            raise MCPClientError("SSE config missing URL")

        async with sse_client(config.url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(tool_name, arguments=arguments)

                if result and result.content:
                    content = result.content[0]
                    if hasattr(content, "text"):
                        return content.text
                    elif hasattr(result, "structuredContent"):
                        return result.structuredContent

                return result

    async def list_tools(
        self, server_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available tools from connected server.

        Args:
            server_name: Specific server or first connected

        Returns:
            List of tool definitions

        Raises:
            MCPNotConnectedError: If no server is connected
        """
        if not _MCP_AVAILABLE:
            raise MCPSDKNotAvailableError("MCP SDK not available")

        config = None
        if server_name and server_name in self._server_configs:
            config = self._server_configs[server_name]
        elif self._server_configs:
            for _, cfg in self._server_configs.items():
                if cfg.connected:
                    config = cfg
                    break

        if not config:
            raise MCPNotConnectedError("No connected server found")

        tools = []
        if config.transport == MCPTransport.STDIO:
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args or [],
                env=config.env or dict(os.environ),
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    for tool in result.tools:
                        tools.append(
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": getattr(tool, "inputSchema", {}),
                            }
                        )
        return tools

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def web_search(
        self,
        search_term: str,
        explanation: str = "Searching for real-time information",
    ) -> MCPResponse:
        """
        Perform a web search.

        Args:
            search_term: Search query
            explanation: Explanation for the search

        Returns:
            MCPResponse with search results
        """
        self._stats["playwright_requests"] += 1
        return await self.call_tool(
            "web_search",
            {"search_term": search_term, "explanation": explanation},
        )

    async def resolve_library(self, library_name: str) -> MCPResponse:
        """
        Resolve a library name to Context7 ID.

        Args:
            library_name: Library name to resolve

        Returns:
            MCPResponse with library information
        """
        self._stats["context7_requests"] += 1
        return await self.call_tool(
            "mcp_context7_resolve-library-id",
            {"libraryName": library_name},
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

        Args:
            library_id: Context7-compatible library ID
            topic: Optional topic to focus on
            mode: "code" for examples, "info" for concepts
            page: Page number for pagination

        Returns:
            MCPResponse with documentation content
        """
        self._stats["context7_requests"] += 1
        params = {
            "context7CompatibleLibraryID": library_id,
            "mode": mode,
            "page": page,
        }
        if topic:
            params["topic"] = topic

        return await self.call_tool("mcp_context7_get-library-docs", params)

    async def browser_navigate(self, url: str) -> MCPResponse:
        """Navigate browser to a URL."""
        self._stats["playwright_requests"] += 1
        return await self.call_tool("mcp_playwright_browser_navigate", {"url": url})

    async def browser_snapshot(self) -> MCPResponse:
        """Get accessibility snapshot of current page."""
        self._stats["playwright_requests"] += 1
        return await self.call_tool("mcp_playwright_browser_snapshot", {})

    # =========================================================================
    # Status Methods
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            **self._stats,
            "mcp_available": _MCP_AVAILABLE,
            "http_transport": _MCP_HTTP_AVAILABLE,
            "sse_transport": _MCP_SSE_AVAILABLE,
            "connected_servers": self.get_connected_servers(),
            "is_connected": self.is_connected(),
        }

    def is_ready(self) -> bool:
        """Check if client is ready for use."""
        return _MCP_AVAILABLE and self.is_connected()


# =============================================================================
# Singleton Instance
# =============================================================================

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


# =============================================================================
# Convenience Functions
# =============================================================================


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


async def setup_context7_server() -> bool:
    """
    Set up Context7 MCP server.

    Uses npx to run the official Context7 server.

    Returns:
        True if configured successfully
    """
    client = get_mcp_client()
    return await client.connect_stdio(
        "context7",
        command="npx",
        args=["-y", "@upstash/context7-mcp"],
    )


async def setup_playwright_server() -> bool:
    """
    Set up Playwright MCP server for browser automation.

    Returns:
        True if configured successfully
    """
    client = get_mcp_client()
    return await client.connect_stdio(
        "playwright",
        command="npx",
        args=["-y", "@anthropic-ai/mcp-server-playwright"],
    )
