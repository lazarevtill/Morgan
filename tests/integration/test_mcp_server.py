"""The MCP server must expose Morgan's memory to an external client."""

from __future__ import annotations

import json

from morgan_brain.mcp_server import TOOL_NAMES, build_server


def test_exposes_exactly_the_five_tools():
    assert sorted(TOOL_NAMES) == ["ask_morgan", "facts", "forget", "recall", "remember"]


async def test_remember_then_recall_through_the_server(tmp_path, monkeypatch):
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    server = build_server()

    await server.call_tool(
        "remember", {"text": "the Harbor mirror blocked the deploy", "project": "acme"}
    )
    out = await server.call_tool("recall", {"query": "harbor", "project": "acme"})
    assert "Harbor mirror" in json.dumps(out)


async def test_recall_is_project_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    server = build_server()
    await server.call_tool("remember", {"text": "company secret", "project": "acme"})
    out = await server.call_tool("recall", {"query": "secret", "project": "personal"})
    assert "company secret" not in json.dumps(out)


async def test_writes_survive_a_new_server_instance(tmp_path, monkeypatch):
    """A daemon restart must not lose what a client stored."""
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    await build_server().call_tool("remember", {"text": "survives", "project": "p"})
    out = await build_server().call_tool("recall", {"query": "survives", "project": "p"})
    assert "survives" in json.dumps(out)


async def test_forget_reports_are_honest(tmp_path, monkeypatch):
    """An MCP client deserves the same truth the CLI's ``forget`` prints -- including which
    tables were skipped (Task 17's ``ForgetReport``)."""
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    server = build_server()
    await server.call_tool("remember", {"text": "ephemeral", "project": "p"})
    out = await server.call_tool("forget", {"project": "p"})
    assert out["memories"] == 1
    assert "tables_skipped" in out
    assert "warnings" in out


async def test_facts_and_ask_morgan_are_project_scoped_from_the_argument(tmp_path, monkeypatch):
    """The one place the CLI's git-root cwd detection must NOT be copied: project comes from
    the tool argument, never the server's own cwd."""
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    server = build_server()
    out = await server.call_tool("facts", {"project": "acme"})
    assert out["project"] == "acme"
