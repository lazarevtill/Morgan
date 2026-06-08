"""Hardened MCP hub.

The hub connects to configured MCP servers, discovers their tools, and exposes
them through the same ``ToolRegistry`` / ``PermissionGate`` machinery as built-in
tools.  All hardening is non-negotiable — applied unconditionally:

Hardening applied (in order, per tool)
---------------------------------------
1. **Sanitization** — ``sanitize_tool_description`` strips prompt-injection
   patterns from the MCP-provided description before it ever reaches the
   model's context.
2. **Fingerprint-pin verification** — if the owner has pinned a fingerprint for
   a tool (in ``McpServerConfig.pinned_fingerprints``), the freshly computed
   hash must match; on mismatch the tool is SKIPPED and added to
   ``skipped_tools`` (rug-pull defence).
3. **Namespace isolation** — every MCP tool is registered under the name
   ``mcp__{server}__{tool}`` so tools from different servers cannot collide and
   grants are scoped per-server.
4. **Untrusted-provenance flagging** — the ``run`` method of every wrapped tool
   returns ``ToolResult(ok=True, output={"untrusted": True, "data": <raw>})``
   so the call-site always knows the result came from an external, unverified
   source.  Callers must provenance-gate before writing to memory.
5. **Default-deny grants** — tools are registered in the ``ToolRegistry`` but
   NOT granted in the ``PermissionGate`` unless ``cfg.auto_grant`` is True.
   The owner must explicitly call ``gate.grant(Grant(tool=...))`` to allow
   execution.
6. **Allowlist enforcement** — if a ``ServerAllowlist`` is configured, servers
   not on it are rejected before any connection attempt.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from morgan_brain.interfaces.tools import ToolResult
from morgan_brain.modules.mcp.client import McpClient, McpToolInfo
from morgan_brain.modules.mcp.security import (
    ServerAllowlist,
    sanitize_tool_description,
    verify_fingerprint,
)
from morgan_brain.modules.tools.executor import ToolRegistry
from morgan_brain.security.permissions import Grant, PermissionGate


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------


class McpServerConfig(BaseModel):
    """Configuration for a single MCP server connection.

    Fields
    ------
    name:
        Unique identifier for this server (used in tool namespace prefix).
    url_or_cmd:
        URL (for SSE / HTTP transports) or command (for stdio transport).
    transport:
        One of ``"stdio"``, ``"sse"``, or ``"http"``.  Default: ``"stdio"``.
    pinned_fingerprints:
        Mapping of ``tool_name`` → SHA-256 hex fingerprint.  When present for
        a tool, ``connect_server`` verifies the fingerprint; on mismatch the
        tool is rejected (rug-pull defence).  An empty dict means no pinning.
    auto_grant:
        When True, each discovered tool is automatically granted ``execute``
        scope after sanitization and pin verification.  When False (the
        default), tools are registered but default-deny until the owner grants
        them explicitly — the safe default.
    """

    name: str
    url_or_cmd: str
    transport: Literal["stdio", "sse", "http"] = "stdio"
    pinned_fingerprints: dict[str, str] = Field(default_factory=dict)
    auto_grant: bool = False


# ---------------------------------------------------------------------------
# Wrapped MCP tool (BaseTool implementation)
# ---------------------------------------------------------------------------


class _McpToolWrapper:
    """Wraps a single MCP tool as a ``BaseTool``.

    The tool name is namespaced as ``mcp__{server}__{original_name}`` to
    provide per-server isolation.  The description is already sanitized before
    this class is instantiated.  All results are tagged with
    ``{"untrusted": True, "data": <raw_result>}`` so callers can provenance-
    gate before writing to memory.
    """

    def __init__(
        self,
        *,
        namespaced_name: str,
        sanitized_description: str,
        tool_schema: dict[str, Any],
        client: McpClient,
        original_name: str,
    ) -> None:
        self.name: str = namespaced_name
        self.description: str = sanitized_description
        self._tool_schema = tool_schema
        self._client = client
        self._original_name = original_name

    def schema(self) -> dict[str, Any]:
        return self._tool_schema

    async def run(self, *, user_id: str, **kwargs: Any) -> ToolResult:
        """Invoke the MCP tool and wrap result as untrusted provenance."""
        raw = await self._client.call_tool(self._original_name, kwargs)
        return ToolResult(
            ok=True,
            output={"untrusted": True, "data": raw},
        )


# ---------------------------------------------------------------------------
# Hub
# ---------------------------------------------------------------------------


class McpHub:
    """Hardened MCP hub that connects to servers and exposes tools safely.

    Parameters
    ----------
    registry:
        The ``ToolRegistry`` into which discovered (and not-rejected) tools are
        registered.
    gate:
        The ``PermissionGate`` consulted/updated per tool (auto-grant or skip).
    allowlist:
        Optional ``ServerAllowlist``.  When provided, any server whose name is
        not on the list is rejected before connection (0 tools registered).
        When ``None``, all servers are accepted.
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        gate: PermissionGate,
        allowlist: ServerAllowlist | None = None,
    ) -> None:
        self._registry = registry
        self._gate = gate
        self._allowlist = allowlist
        self._skipped: list[str] = []  # tool names (namespaced) rejected by pin mismatch

    @property
    def skipped_tools(self) -> list[str]:
        """Names of tools rejected due to fingerprint mismatch (rug-pull defence)."""
        return list(self._skipped)

    async def connect_server(self, cfg: McpServerConfig, client: McpClient) -> int:
        """Discover and register tools from an MCP server.

        Returns the number of tools successfully registered.

        Steps
        -----
        1. Allowlist check — reject the whole server if not allowed.
        2. ``client.list_tools()`` — discover available tools.
        3. Per-tool: sanitize description → pin verify (if configured) → wrap →
           register → conditionally grant.

        Security properties (all enforced unconditionally):
          * Descriptions are sanitized before storage.
          * Pinned tools whose hash mismatches are SKIPPED (not registered).
          * Every registered tool carries the ``mcp__<server>__`` prefix.
          * Results from ``run()`` are always tagged ``untrusted: True``.
          * No grant is issued unless ``cfg.auto_grant`` is True.
        """
        if self._allowlist is not None and not self._allowlist.is_allowed(cfg.name):
            return 0

        tools: list[McpToolInfo] = await client.list_tools()
        registered = 0

        for tool_info in tools:
            namespaced = f"mcp__{cfg.name}__{tool_info.name}"
            sanitized_desc = sanitize_tool_description(tool_info.description)

            # Pin verification — skip tool on mismatch (rug-pull defence).
            if tool_info.name in cfg.pinned_fingerprints:
                pinned = cfg.pinned_fingerprints[tool_info.name]
                if not verify_fingerprint(
                    tool_info.name,
                    tool_info.description,
                    tool_info.input_schema,
                    pinned,
                ):
                    self._skipped.append(namespaced)
                    continue  # do NOT register

            wrapper = _McpToolWrapper(
                namespaced_name=namespaced,
                sanitized_description=sanitized_desc,
                tool_schema=tool_info.input_schema,
                client=client,
                original_name=tool_info.name,
            )
            self._registry.register(wrapper)

            if cfg.auto_grant:
                self._gate.grant(Grant(tool=namespaced, scope="execute"))

            registered += 1

        return registered
