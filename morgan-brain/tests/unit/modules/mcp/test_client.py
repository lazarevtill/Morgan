"""Unit tests for morgan_brain.modules.mcp.client.

All deterministic, in-process, no network, no ``mcp`` package required.
"""
from __future__ import annotations

import importlib
import sys

import pytest

from morgan_brain.modules.mcp.client import FakeMcpClient, McpClient, McpToolInfo, RealMcpClient


# ---------------------------------------------------------------------------
# McpToolInfo
# ---------------------------------------------------------------------------


def test_mcp_tool_info_defaults() -> None:
    info = McpToolInfo(name="my_tool", description="Does X.")
    assert info.name == "my_tool"
    assert info.description == "Does X."
    assert info.input_schema == {}


def test_mcp_tool_info_with_schema() -> None:
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    info = McpToolInfo(name="search", description="Search.", input_schema=schema)
    assert info.input_schema == schema


# ---------------------------------------------------------------------------
# FakeMcpClient — list_tools
# ---------------------------------------------------------------------------


async def test_fake_client_list_tools_returns_configured_tools() -> None:
    tools = [
        McpToolInfo(name="tool_a", description="Tool A"),
        McpToolInfo(name="tool_b", description="Tool B"),
    ]
    client = FakeMcpClient(tools=tools)
    result = await client.list_tools()
    assert len(result) == 2
    assert result[0].name == "tool_a"
    assert result[1].name == "tool_b"


async def test_fake_client_list_tools_empty() -> None:
    client = FakeMcpClient(tools=[])
    assert await client.list_tools() == []


async def test_fake_client_list_tools_returns_copy() -> None:
    """Mutating the returned list must not affect the fake's internal state."""
    tools = [McpToolInfo(name="t", description="d")]
    client = FakeMcpClient(tools=tools)
    first = await client.list_tools()
    first.clear()
    second = await client.list_tools()
    assert len(second) == 1


# ---------------------------------------------------------------------------
# FakeMcpClient — call_tool
# ---------------------------------------------------------------------------


async def test_fake_client_call_tool_returns_configured_result() -> None:
    tools = [McpToolInfo(name="echo", description="echo")]
    client = FakeMcpClient(tools=tools, results={"echo": {"out": "hello"}})
    result = await client.call_tool("echo", {"msg": "hello"})
    assert result == {"out": "hello"}


async def test_fake_client_call_tool_unknown_returns_none() -> None:
    client = FakeMcpClient(tools=[], results={})
    assert await client.call_tool("nonexistent", {}) is None


async def test_fake_client_call_tool_no_results_kwarg_returns_none() -> None:
    client = FakeMcpClient(tools=[McpToolInfo(name="t", description="d")])
    assert await client.call_tool("t", {}) is None


# ---------------------------------------------------------------------------
# FakeMcpClient satisfies McpClient Protocol
# ---------------------------------------------------------------------------


def test_fake_client_satisfies_protocol() -> None:
    tools = [McpToolInfo(name="t", description="d")]
    client = FakeMcpClient(tools=tools)
    assert isinstance(client, McpClient)


# ---------------------------------------------------------------------------
# RealMcpClient — module import does NOT require mcp package
# ---------------------------------------------------------------------------


def test_real_client_module_import_does_not_require_mcp() -> None:
    """The module must be importable even when 'mcp' is not installed.

    We simulate the absence of the mcp package by temporarily removing it
    from sys.modules (if present) and reloading the client module.
    """
    # Remove mcp from sys.modules for the duration of this test.
    mcp_backup = sys.modules.pop("mcp", None)
    client_module_name = "morgan_brain.modules.mcp.client"
    # Force a fresh import of the client module.
    sys.modules.pop(client_module_name, None)
    try:
        imported = importlib.import_module(client_module_name)
        # Must succeed (no ImportError at module level).
        assert hasattr(imported, "RealMcpClient")
    finally:
        # Restore mcp if it was present.
        if mcp_backup is not None:
            sys.modules["mcp"] = mcp_backup
        # Restore the module to its original cached state.
        sys.modules.pop(client_module_name, None)
        importlib.import_module(client_module_name)


def test_real_client_construction_does_not_require_mcp() -> None:
    """Constructing RealMcpClient must not import mcp."""
    mcp_backup = sys.modules.pop("mcp", None)
    try:
        # Should not raise.
        client = RealMcpClient(url_or_cmd="stdio://my_server", transport="stdio")
        assert client._url_or_cmd == "stdio://my_server"
    finally:
        if mcp_backup is not None:
            sys.modules["mcp"] = mcp_backup


async def test_real_client_connect_raises_import_error_when_mcp_absent() -> None:
    """connect() must raise ImportError with an install hint when mcp is absent."""
    mcp_backup = sys.modules.pop("mcp", None)
    try:
        client = RealMcpClient(url_or_cmd="stdio://server", transport="stdio")
        with pytest.raises(ImportError, match="pip install morgan-brain\\[mcp\\]"):
            await client.connect()
    finally:
        if mcp_backup is not None:
            sys.modules["mcp"] = mcp_backup


async def test_real_client_list_tools_raises_import_error_when_mcp_absent() -> None:
    mcp_backup = sys.modules.pop("mcp", None)
    try:
        client = RealMcpClient(url_or_cmd="stdio://server")
        with pytest.raises(ImportError, match="pip install morgan-brain\\[mcp\\]"):
            await client.list_tools()
    finally:
        if mcp_backup is not None:
            sys.modules["mcp"] = mcp_backup


async def test_real_client_call_tool_raises_import_error_when_mcp_absent() -> None:
    mcp_backup = sys.modules.pop("mcp", None)
    try:
        client = RealMcpClient(url_or_cmd="stdio://server")
        with pytest.raises(ImportError, match="pip install morgan-brain\\[mcp\\]"):
            await client.call_tool("tool", {})
    finally:
        if mcp_backup is not None:
            sys.modules["mcp"] = mcp_backup


def test_real_client_satisfies_protocol() -> None:
    """RealMcpClient must satisfy McpClient at runtime (structural subtyping)."""
    client = RealMcpClient(url_or_cmd="stdio://server")
    assert isinstance(client, McpClient)
