"""``morgan-mcp --transport stdio``: every byte on stdout is JSON-RPC.

The server is driven by hand over pipes rather than through an MCP client, because the
client library skips lines it cannot parse -- which is precisely the failure this guards
against. A model server that is down makes the embedder probe log a warning; that warning
must land on stderr, never in the protocol stream.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


def _frame(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


def test_every_stdout_line_is_a_jsonrpc_message(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MORGAN_")}
    env.update(
        MORGAN_DATA_DIR=str(tmp_path),
        MORGAN_LLM_ENDPOINT="http://127.0.0.1:1/v1",
        PYTHONUNBUFFERED="1",
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "morgan_brain.mcp_server", "--transport", "stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=tmp_path,
    )
    assert proc.stdin is not None
    frames = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "remember", "arguments": {"text": "x", "project": "p"}},
        },
    ]
    assert proc.stdout is not None
    for f in frames:
        proc.stdin.write(_frame(f))
    proc.stdin.flush()
    # Keep stdin open until the last reply has arrived: EOF ends the server, and a tool
    # call still in flight when it ends would simply never be answered.
    lines: list[str] = []
    while True:
        line = proc.stdout.readline().decode()
        if not line:
            break
        lines.append(line.rstrip("\n"))
        if line.startswith("{") and json.loads(line).get("id") == 3:
            break
    _, stderr = proc.communicate(timeout=60)

    assert lines, stderr.decode()
    messages = [json.loads(ln) for ln in lines if ln.strip()]  # raises on any stray log line
    assert all(m.get("jsonrpc") == "2.0" for m in messages)
    by_id = {m["id"]: m for m in messages if "id" in m}
    assert {t["name"] for t in by_id[2]["result"]["tools"]} >= {"remember", "recall", "forget"}
    assert by_id[3]["result"]["isError"] is True
    assert "127.0.0.1:1" in by_id[3]["result"]["content"][0]["text"]
    assert "embedding-dim-probe" in stderr.decode()
