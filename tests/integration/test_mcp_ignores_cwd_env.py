"""``morgan-mcp`` reads no ``./.env``: its working directory is its client's, not the owner's.

A client starts the server in whatever folder it has open. On 2026-09-19 a stale ``.env`` in
such a folder pointed a server at another database, and the brain forked. The server is run
here exactly as a client runs it -- a subprocess speaking JSON-RPC over stdio -- from a folder
whose ``.env`` names a decoy database; the memory must land in the one the user's own config
file names.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys


def _frame(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


def test_a_cwd_env_does_not_move_the_mcp_servers_database(tmp_path):
    user_db, decoy_db = tmp_path / "user-brain", tmp_path / "decoy-brain"
    config_home = tmp_path / "config"
    (config_home / "morgan").mkdir(parents=True)
    (config_home / "morgan" / ".env").write_text(
        f"MORGAN_DATA_DIR={user_db}\nMORGAN_EMBEDDING_BACKEND=hash\n", encoding="utf-8"
    )
    client_folder = tmp_path / "client-folder"
    client_folder.mkdir()
    (client_folder / ".env").write_text(f"MORGAN_DATA_DIR={decoy_db}\n", encoding="utf-8")

    # No MORGAN_* from the environment: a variable there would beat both files and prove
    # nothing. XDG_DATA_HOME too, so a server that read neither file still stays in tmp_path.
    env = {k: v for k, v in os.environ.items() if not k.startswith("MORGAN_")}
    env.update(
        XDG_CONFIG_HOME=str(config_home),
        XDG_DATA_HOME=str(tmp_path / "share"),
        PYTHONUNBUFFERED="1",
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "morgan_brain.surfaces.mcp_server", "--transport", "stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=client_folder,
    )
    assert proc.stdin is not None and proc.stdout is not None
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
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "remember", "arguments": {"text": "kept here", "project": "p"}},
        },
    ]
    for f in frames:
        proc.stdin.write(_frame(f))
    proc.stdin.flush()
    # stdin stays open until the tool call is answered: EOF ends the server.
    reply = None
    while True:
        line = proc.stdout.readline().decode()
        if not line:
            break
        if line.startswith("{") and json.loads(line).get("id") == 2:
            reply = json.loads(line)
            break
    _, stderr = proc.communicate(timeout=60)

    assert reply is not None, stderr.decode()
    assert reply["result"]["isError"] is False, reply
    assert not decoy_db.exists()
    database = user_db / "morgan.db"
    assert database.is_file()
    conn = sqlite3.connect(database)
    try:
        stored = [row[0] for row in conn.execute("SELECT content FROM memories")]
    finally:
        conn.close()
    assert stored == ["kept here"]
