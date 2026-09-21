"""Where a memory came from is recorded by the surface that wrote it.

``cwd`` is provenance, not the project: the project comes from the git repository, and a row
that learned its project from ``cwd`` would re-home itself whenever a client changed
directory.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import types
from mcp.client.session import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

from morgan_brain.app.chat import Chat
from morgan_brain.composition import build_memory_context, build_memory_module, sqlite_path, utcnow
from morgan_brain.config import Settings
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.knowledge.consolidation import (
    FactOp,
    FactOpBatch,
    FactOpKind,
    MemoryConsolidator,
)
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import MemorySource, OriginKind, Scope
from morgan_brain.surfaces.cli.commands import cmd_remember
from morgan_brain.surfaces.mcp_server import build_server
from tests.fakes import FakeChatClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=str(tmp_path), embedding_backend="hash")


def _memory_row(settings: Settings, memory_id: str) -> sqlite3.Row:
    conn = sqlite3.connect(sqlite_path(settings.temporal_db_url))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    finally:
        conn.close()
    assert row is not None, memory_id
    return row


def _memory_row_by_source(settings: Settings, source: str) -> sqlite3.Row:
    """Select by *source* rather than "the last row written": ``Chat.ask`` writes both
    halves of a turn in one loop, and their insert order is incidental."""
    conn = sqlite3.connect(sqlite_path(settings.temporal_db_url))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM memories WHERE source = ?", (source,)).fetchone()
    finally:
        conn.close()
    assert row is not None, source
    return row


class _RecordingClient:
    """Wraps a real ``ClientSession`` so a test can read a tool's JSON result as a dict, the
    way the CLI and ``MorganMcpServer.call_tool``'s direct dispatch already do."""

    def __init__(self, session: ClientSession) -> None:
        self._session = session

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self._session.call_tool(name, arguments)
        assert not result.isError, result.content
        assert result.structuredContent is not None
        return result.structuredContent


@asynccontextmanager
async def _mcp_client(tmp_path: Path, *, client_name: str) -> AsyncIterator[_RecordingClient]:
    """A client connected to a real ``morgan-mcp`` server over the SDK's in-process
    transport, so ``clientInfo`` reaches the server exactly as it would over stdio or HTTP --
    unlike ``MorganMcpServer.call_tool``'s direct dispatch, which bypasses the protocol (and
    the ``Context`` it injects) entirely.
    """
    server = build_server(_settings(tmp_path))
    async with create_connected_server_and_client_session(
        server.mcp, client_info=types.Implementation(name=client_name, version="0.0.0")
    ) as session:
        yield _RecordingClient(session)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def test_the_cli_marks_its_own_writes(settings_for_tmp: Settings) -> None:
    result = await cmd_remember(argparse.Namespace(text="Harbor"), settings_for_tmp, "p")

    stored = _memory_row(settings_for_tmp, result["id"])
    assert stored["origin_kind"] == OriginKind.REMEMBER.value
    assert stored["client"] == "cli" and stored["session_id"] == ""
    assert stored["cwd"] and stored["project"] == "p"


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------


async def test_the_mcp_server_records_the_client_it_was_called_by(tmp_path: Path) -> None:
    async with _mcp_client(tmp_path, client_name="claude-desktop") as client:
        result = await client.call_tool("remember", {"text": "Harbor", "project": "p"})

    stored = _memory_row(_settings(tmp_path), result["id"])
    assert stored["client"] == "claude-desktop"
    assert stored["session_id"] != ""


async def test_a_tools_input_schema_exposes_no_ctx(tmp_path: Path) -> None:
    """The ``Context`` parameter FastMCP injects into ``remember`` must stay invisible to a
    client: it is dependency injection, not a tool argument the caller supplies."""
    server = build_server(_settings(tmp_path))

    for tool in await server.mcp.list_tools():
        assert "ctx" not in tool.inputSchema.get("properties", {}), tool.name


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


async def test_an_answer_is_stored_as_agent_inferred_and_origin_ask(
    settings_for_tmp: Settings,
) -> None:
    """Attribution does not change: the reply is still agent_inferred, and now says it came
    from ask rather than from the owner typing it.

    ``Chat.ask`` is exercised directly with a ``FakeChatClient`` rather than through
    ``cmd_ask``/``model_server``: ``tests/fakes.py::model_server`` answers only
    ``/v1/embeddings`` (a real llama-server probe double), not chat completions, and the hash
    embedding backend here needs no embedding server either.
    """
    ctx = build_memory_context(settings_for_tmp)
    try:
        chat = Chat(
            gate=ctx.gate,
            history=ctx.history,
            client=FakeChatClient(reply="Harbor is the project you're on."),
            model="test-model",
            clock=utcnow,
        )
        await chat.ask(
            user_id=settings_for_tmp.owner_user_id, project="p", text="what did I decide?"
        )
    finally:
        ctx.conn.close()

    stored = _memory_row_by_source(settings_for_tmp, MemorySource.AGENT_INFERRED.value)
    assert stored["source"] == MemorySource.AGENT_INFERRED.value
    assert stored["origin_kind"] == OriginKind.ASK.value


# ---------------------------------------------------------------------------
# consolidate (facts)
# ---------------------------------------------------------------------------


async def test_consolidated_facts_carry_the_owners_authorship() -> None:
    """Controller ruling on Task 10's review: consolidation's ``upsert_fact`` path must set
    ``author_id`` and ``scope`` too, or every fact written after ``migrate`` counts as
    missing provenance -- the temporal store round-trips whatever the caller gives it."""
    module = build_memory_module(open_db(":memory:"), embedder=FakeEmbedder(dim=16), dim=16)
    gate = MemoryGate(module)
    consolidator = MemoryConsolidator(
        gate=gate, client=FakeChatClient(), model="test-model", clock=utcnow
    )
    batch = FactOpBatch(
        ops=[FactOp(op=FactOpKind.ADD, subject="user", predicate="lives_in", object="Berlin")]
    )

    await consolidator.apply("owner", batch, project="p")

    current = await gate.current_facts(user_id="owner", project="p")
    assert len(current) == 1
    assert current[0].author_id == "owner"
    assert current[0].scope == Scope.PRIVATE
