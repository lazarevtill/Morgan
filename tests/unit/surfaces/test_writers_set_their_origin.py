"""Where a memory came from is recorded by the surface that wrote it.

``cwd`` is provenance, not the project: the project comes from the git repository, and a row
that learned its project from ``cwd`` would re-home itself whenever a client changed
directory.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import types
from mcp.client.session import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

import morgan_brain.composition as composition
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
from morgan_brain.surfaces.cli.commands import cmd_ask, cmd_import, cmd_remember
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


def _chatgpt_export(tmp_path: Path) -> Path:
    """A one-turn ChatGPT export, in the export's own shape -- the same fixture shape
    ``tests/integration/test_read_only_refuses_before_work.py::_export`` uses."""
    message = {
        "id": "m0",
        "author": {"role": "user"},
        "create_time": 1700000000.0,
        "content": {"content_type": "text", "parts": ["The Harbor mirror is back."]},
    }
    conversation = {"conversation_id": "c0", "id": "c0", "mapping": {"m0": {"message": message}}}
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps([conversation]), encoding="utf-8")
    return path


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


async def test_an_ask_through_the_cli_stores_client_cli(
    settings_for_tmp: Settings, monkeypatch: Any
) -> None:
    """``cmd_ask`` -- the CLI's own handler -- must default the two new ``Chat.ask``
    parameters to the CLI's values, the same way ``cmd_remember`` already does.

    ``build_chat_client`` (not ``tests/fakes.py::model_server``) is monkeypatched to return a
    ``FakeChatClient``: a real ``ask`` needs a chat-completions endpoint, which
    ``model_server``'s double does not serve (see the docstring above), and the hash embedding
    backend already needs no embedding server.
    """
    monkeypatch.setattr(composition, "build_chat_client", lambda settings: FakeChatClient())

    await cmd_ask(argparse.Namespace(text="what did I decide?"), settings_for_tmp, "p")

    stored = _memory_row_by_source(settings_for_tmp, MemorySource.AGENT_INFERRED.value)
    assert stored["client"] == "cli"
    assert stored["session_id"] == ""


async def test_ask_morgan_through_the_mcp_server_stores_the_callers_client(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(composition, "build_chat_client", lambda settings: FakeChatClient())

    async with _mcp_client(tmp_path, client_name="claude-desktop") as client:
        await client.call_tool("ask_morgan", {"text": "what did I decide?", "project": "p"})

    stored = _memory_row_by_source(_settings(tmp_path), MemorySource.AGENT_INFERRED.value)
    assert stored["client"] == "claude-desktop"
    assert stored["session_id"] != ""


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------


async def test_an_import_stores_client_cli(settings_for_tmp: Settings, tmp_path: Path) -> None:
    """Import runs only from the CLI -- there is no MCP import tool -- so its writer names
    ``"cli"`` outright, the same value the CLI's other writers default to."""
    export = _chatgpt_export(tmp_path)

    await cmd_import(argparse.Namespace(path=str(export)), settings_for_tmp, "ignored")

    stored = _memory_row_by_source(settings_for_tmp, MemorySource.USER_STATED.value)
    assert stored["origin_kind"] == OriginKind.IMPORT.value
    assert stored["client"] == "cli"


# ---------------------------------------------------------------------------
# consolidate (facts)
# ---------------------------------------------------------------------------


async def test_consolidated_facts_carry_the_owners_authorship() -> None:
    """Consolidation's ``upsert_fact`` path must set
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
