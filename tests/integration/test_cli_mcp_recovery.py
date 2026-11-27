"""Actual subprocess CLI and MCP SDK interoperability on disposable synthetic memory.

Hash embeddings verify persistence and scope, not semantic retrieval quality. Closing
the stdio server before restore is part of the public maintenance contract.
"""

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def cli(root, env, *args, code=0):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "morgan_brain.surfaces.cli",
        *args,
        "--json",
        cwd=root,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == code, stderr.decode()
    assert "Traceback" not in stderr.decode()
    return json.loads(stdout)


async def call(client, name, arguments, *, error=False):
    result = await client.call_tool(name, arguments)
    assert result.isError is error, result
    text = "\n".join(part.text for part in result.content if part.type == "text")
    return text if error else json.loads(text)


async def test_cli_to_stdio_to_reopened_cli_and_snapshot_recovery(tmp_path):
    env = {
        **os.environ,
        "MORGAN_DATA_DIR": str(tmp_path / "data"),
        "MORGAN_SNAPSHOT_DIR": str(tmp_path / "snapshots"),
        "MORGAN_OWNER_USER_ID": "synthetic-owner",
        "MORGAN_EMBEDDING_BACKEND": "hash",
        "MORGAN_EMBEDDING_DIM": "16",
    }
    alpha = "Synthetic Harbor note: проверить зеркало."
    beta = "Synthetic separate project note."
    await cli(tmp_path, env, "remember", alpha, "--project", "alpha", "--event-id", "alpha-note")
    await cli(tmp_path, env, "remember", beta, "--project", "beta", "--event-id", "beta-note")
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "morgan_brain.surfaces.mcp_server", "--transport", "stdio"],
        env=env,
        cwd=str(tmp_path),
    )
    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(parameters))
        client = await stack.enter_async_context(
            ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30))
        )
        await client.initialize()
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        readers = {"recall", "facts", "evidence", "inspect_context"}
        assert set(tools) == readers | {"remember", "forget", "ask_morgan"}
        assert {name for name, t in tools.items() if t.annotations.readOnlyHint} == readers
        assert tools["forget"].annotations.destructiveHint is True
        assert "project" in tools["remember"].inputSchema["properties"]

        evidence = await call(
            client, "evidence", {"ids": ["alpha-note", "beta-note"], "project": "alpha"}
        )
        assert evidence["version"] == "morgan.evidence.v1"
        assert evidence["missing_ids"] == ["beta-note"]
        assert evidence["results"][0]["content"] == alpha
        recall = await call(client, "recall", {"query": "Harbor", "project": "alpha"})
        assert {row["id"] for row in recall["results"]} == {"alpha-note"}
        cross = await call(client, "recall", {"query": "note", "all_projects": True})
        assert {row["id"] for row in cross["results"]} == {"alpha-note", "beta-note"}

        await call(
            client,
            "remember",
            {"text": "Synthetic MCP note", "project": "alpha", "event_id": "mcp-note"},
        )
        rejected = await call(
            client,
            "remember",
            {"text": "Conflicting replacement", "project": "alpha", "event_id": "mcp-note"},
            error=True,
        )
        assert "Traceback" not in rejected
        inspected = await call(client, "inspect_context", {"ids": ["mcp-note"], "project": "alpha"})
        assert inspected["action_authority"] == "none"
        assert (await call(client, "facts", {"project": "alpha"}))["facts"] == []

    # New CLI processes see the MCP write; another configured owner sees none of it.
    reopened = await cli(tmp_path, env, "recall", "note", "--project", "alpha")
    assert {row["id"] for row in reopened["results"]} == {"alpha-note", "mcp-note"}
    other = await cli(
        tmp_path,
        {**env, "MORGAN_OWNER_USER_ID": "other-owner"},
        "evidence",
        "alpha-note",
        "--project",
        "alpha",
    )
    assert other["missing_ids"] == ["alpha-note"]

    forgotten = await cli(tmp_path, env, "forget", "--project", "alpha")
    assert forgotten["memories"] == 2
    assert (await cli(tmp_path, env, "recall", "note", "--project", "alpha"))["abstained"]
    retained = await cli(tmp_path, env, "recall", "note", "--project", "beta")
    assert retained["results"][0]["content"] == beta
    await cli(tmp_path, env, "restore", forgotten["snapshot"], code=2)
    assert (await cli(tmp_path, env, "recall", "note", "--project", "alpha"))["abstained"]
    restored = await cli(tmp_path, env, "restore", forgotten["snapshot"], "--yes")
    assert restored["after"]["memories"] == 3
    final = await cli(tmp_path, env, "evidence", "alpha-note", "mcp-note", "--project", "alpha")
    assert final["missing_ids"] == []
    assert {row["content"] for row in final["results"]} == {alpha, "Synthetic MCP note"}
