"""Opening memory sends nothing to the embedding server.

Without keep-alive on the embedding host, a request at open puts a 43 s model load in front of
every command, including the ones that never embed: facts, forget, doctor, and the MCP tool list
that Codex waits on before it will use the server at all.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from morgan_brain.composition import build_memory_context, build_memory_module, sqlite_path, utcnow
from morgan_brain.config import Settings
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.store import spaces
from morgan_brain.memory.store.db import open_db
from tests.fakes import counting_model_server


async def test_build_memory_context_makes_no_request(tmp_path):
    with counting_model_server() as (url, calls):
        ctx = build_memory_context(_settings(tmp_path, embedding_endpoint=url))
        try:
            assert calls.total == 0
            # Not a server it never knew about: the first real embedding goes to it.
            await ctx.embedder.embed("a real query")
            assert calls.total == 1
        finally:
            ctx.conn.close()


async def test_the_tool_list_makes_no_request(tmp_path):
    with counting_model_server() as (url, calls):
        async with _stdio_mcp(tmp_path, embedding_endpoint=url) as client:
            assert (await client.list_tools()).tools
            assert calls.total == 0
            # The server does address this model once a tool embeds something.
            stored = await client.call_tool("remember", {"text": "x", "project": "p"})
            assert not stored.isError, stored.content
        assert calls.total == 1


async def test_facts_answers_with_the_embedder_stopped(tmp_path):
    # port 1 refuses at once: facts needs no embedding and must not care.
    result = await _facts(tmp_path, embedding_endpoint="http://127.0.0.1:1/v1")
    assert result["facts"] == []


def test_a_width_that_disagrees_with_the_space_refuses_without_embedding(tmp_path):
    _a_database_whose_space_is(tmp_path, dims=4096)
    with counting_model_server() as (url, calls):
        with pytest.raises(RuntimeError, match=r"4096.*1024|1024.*4096"):
            build_memory_context(_settings(tmp_path, embedding_endpoint=url, embedding_dim=1024))
        assert calls.total == 0


# --- helpers ---------------------------------------------------------------------------------


def _settings(tmp_path: Path, *, embedding_endpoint: str, embedding_dim: int = 1024) -> Settings:
    """The live embedding backend, at *embedding_endpoint*: under the hash backend no request
    could be sent at all, and a zero would prove nothing."""
    return Settings(
        data_dir=str(tmp_path),
        llm_endpoint="http://chat.invalid/v1",
        embedding_endpoint=embedding_endpoint,
        embedding_backend="provider",
        embedding_dim=embedding_dim,
    )


@asynccontextmanager
async def _stdio_mcp(tmp_path: Path, *, embedding_endpoint: str) -> AsyncIterator[ClientSession]:
    """``morgan-mcp --transport stdio`` in its own process, initialised, as a client sees it."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MORGAN_")}
    env.update(
        MORGAN_DATA_DIR=str(tmp_path),
        MORGAN_LLM_ENDPOINT="http://chat.invalid/v1",
        MORGAN_EMBEDDING_ENDPOINT=embedding_endpoint,
        MORGAN_EMBEDDING_BACKEND="provider",
        PYTHONUNBUFFERED="1",
    )
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "morgan_brain.surfaces.mcp_server", "--transport", "stdio"],
        env=env,
        cwd=str(tmp_path),
    )
    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session


async def _facts(tmp_path: Path, *, embedding_endpoint: str) -> dict[str, Any]:
    async with _stdio_mcp(tmp_path, embedding_endpoint=embedding_endpoint) as client:
        result = await client.call_tool("facts", {"project": "p"})
    assert not result.isError, result.content
    assert result.structuredContent is not None
    return result.structuredContent


def _a_database_whose_space_is(tmp_path: Path, *, dims: int) -> None:
    """A database at the code's version whose active space is *dims* wide."""
    conn = open_db(sqlite_path(Settings(data_dir=str(tmp_path)).temporal_db_url))
    try:
        build_memory_module(conn, embedder=FakeEmbedder(dim=dims), dim=dims)
        spaces.register(conn, model="a-wide-model", dims=dims, table_name="vec_items", clock=utcnow)
    finally:
        conn.close()
