"""MCP client seam — injectable Protocol + fake + lazy real-client stub.

Design constraints
------------------
* The ``McpClient`` Protocol is ``@runtime_checkable`` so ``isinstance()`` works
  in the hub for type safety at runtime.
* ``FakeMcpClient`` is the in-process double used by ALL unit tests — no network,
  no ``mcp`` package required.
* ``RealMcpClient`` imports the ``mcp`` SDK lazily (inside ``connect`` / method
  bodies) so the module can be imported even when the optional ``[mcp]`` extra is
  not installed.  The ImportError surfaces at call-time with a clear install hint.
* Live transport wiring (stdio / SSE / Streamable-HTTP) is a follow-on; this
  commit establishes the seam and the lazy-import pattern.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class McpToolInfo(BaseModel):
    """Minimal description of a tool advertised by an MCP server.

    The ``input_schema`` field holds the JSON Schema for the tool's arguments
    (named ``input_schema`` to avoid shadowing ``BaseModel.schema``).
    """

    name: str
    description: str
    input_schema: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Protocol (the seam injected into McpHub)
# ---------------------------------------------------------------------------


@runtime_checkable
class McpClient(Protocol):
    """Injectable MCP client.  Both ``FakeMcpClient`` and ``RealMcpClient``
    implement this Protocol; unit tests always use the fake.
    """

    async def list_tools(self) -> list[McpToolInfo]:
        """Discover tools advertised by the connected MCP server."""
        ...

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Invoke *name* on the MCP server with *args* and return the raw result."""
        ...


# ---------------------------------------------------------------------------
# Fake (test double)
# ---------------------------------------------------------------------------


class FakeMcpClient:
    """In-process MCP client double.  No network, no ``mcp`` package.

    Parameters
    ----------
    tools:
        The list of ``McpToolInfo`` objects returned by ``list_tools``.
    results:
        Mapping of ``tool_name`` → value returned by ``call_tool``.
        If a tool name is not present the call returns ``None``.
    """

    def __init__(
        self,
        tools: list[McpToolInfo],
        results: dict[str, Any] | None = None,
    ) -> None:
        self._tools = list(tools)
        self._results: dict[str, Any] = results or {}

    async def list_tools(self) -> list[McpToolInfo]:
        return list(self._tools)

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        return self._results.get(name)


# ---------------------------------------------------------------------------
# Real client stub (lazy mcp SDK import)
# ---------------------------------------------------------------------------


class RealMcpClient:
    """Thin seam around the ``mcp`` SDK.

    The ``mcp`` package (``morgan-brain[mcp]``) is imported *lazily* inside
    each method body so this module can be imported in environments where the
    optional extra is not installed.  Attempting to call ``connect`` (or any
    live method) without the extra installed raises a clear ``ImportError`` with
    the install command.

    Live transport wiring (stdio / SSE / Streamable-HTTP session management,
    OAuth 2.1 + PKCE token handling, reconnect logic) is deferred to a follow-on
    increment.  This stub provides the correct seam so the hub can reference
    ``RealMcpClient`` today.
    """

    def __init__(self, url_or_cmd: str, transport: str = "stdio") -> None:
        self._url_or_cmd = url_or_cmd
        self._transport = transport
        self._session: Any = None  # will hold the live mcp.ClientSession once connected

    def _require_mcp(self) -> Any:
        """Import and return the ``mcp`` top-level module or raise a friendly error."""
        try:
            import mcp  # type: ignore[import-not-found]  # optional dep

            return mcp
        except ImportError as exc:
            raise ImportError(
                "The 'mcp' package is required for live MCP server connections. "
                "Install it with:  pip install morgan-brain[mcp]"
            ) from exc

    async def connect(self) -> None:
        """Establish a connection to the MCP server.

        Raises
        ------
        ImportError
            If ``morgan-brain[mcp]`` is not installed.

        Note: full transport negotiation (stdio subprocess spawn / SSE / HTTP)
        and OAuth 2.1 + PKCE + RFC 8707 token acquisition are deferred to the
        next increment.
        """
        _mcp = self._require_mcp()
        # TODO (follow-on): spawn transport, negotiate session, store in self._session.
        # Placeholder so the method body executes once mcp is present.
        _ = _mcp  # suppress unused-import; real wiring replaces this line

    async def list_tools(self) -> list[McpToolInfo]:
        """Discover tools from the connected MCP server.

        Raises
        ------
        ImportError
            If ``morgan-brain[mcp]`` is not installed.
        RuntimeError
            If ``connect()`` has not been called yet.
        """
        self._require_mcp()
        if self._session is None:
            raise RuntimeError("RealMcpClient: call connect() before list_tools()")
        # TODO (follow-on): delegate to self._session.list_tools()
        return []  # pragma: no cover

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Call *name* on the connected MCP server.

        Raises
        ------
        ImportError
            If ``morgan-brain[mcp]`` is not installed.
        RuntimeError
            If ``connect()`` has not been called yet.
        """
        self._require_mcp()
        if self._session is None:
            raise RuntimeError("RealMcpClient: call connect() before call_tool()")
        # TODO (follow-on): delegate to self._session.call_tool(name, args)
        return None  # pragma: no cover
