"""What each MCP tool declares about itself.

A client decides from these hints whether a call may run without asking, and read-only is
the one that lets it: a client that trusts the claim runs the tool unprompted. So it is made
for the tools that only read, and every tool states every hint. A tool left without them is
judged by each client's own defaults, which is not a decision this server should hand away.
"""

from __future__ import annotations

from morgan_brain.surfaces.mcp_server import build_server

HINTS = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")


async def _declared(tmp_path, monkeypatch):
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    return {tool.name: tool.annotations for tool in await build_server().mcp.list_tools()}


async def test_only_recall_and_facts_declare_themselves_read_only(tmp_path, monkeypatch):
    """``ask_morgan`` reads like a question, but the turn stores both halves of the exchange."""
    declared = await _declared(tmp_path, monkeypatch)

    read_only = {name for name, hints in declared.items() if hints and hints.readOnlyHint}

    assert read_only == {"recall", "facts"}


async def test_forget_is_the_one_destructive_tool(tmp_path, monkeypatch):
    declared = await _declared(tmp_path, monkeypatch)

    destructive = {name for name, hints in declared.items() if hints and hints.destructiveHint}

    assert destructive == {"forget"}


async def test_every_tool_states_every_hint(tmp_path, monkeypatch):
    declared = await _declared(tmp_path, monkeypatch)

    unstated = {
        name: [hint for hint in HINTS if hints is None or getattr(hints, hint) is None]
        for name, hints in declared.items()
    }

    assert {name: missing for name, missing in unstated.items() if missing} == {}
