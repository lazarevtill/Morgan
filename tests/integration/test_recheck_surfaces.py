"""Exercise the actual HTTP app, including SDK transport security and Morgan auth."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from starlette.testclient import TestClient

from morgan_brain.config import Settings
from morgan_brain.surfaces.mcp_server import build_server
from tests.fakes import raw_model_server

_INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "recheck", "version": "1"},
    },
}


@pytest.mark.parametrize("bind", ["0.0.0.0", "10.20.30.40"])
def test_authenticated_remote_http_can_initialize(tmp_path, monkeypatch, bind):
    server = build_server(
        Settings(data_dir=str(tmp_path), embedding_backend="hash", api_key="synthetic-key")
    )
    captured = []

    async def capture_app(uvicorn_server):
        captured.append(uvicorn_server.config.app)

    monkeypatch.setattr("uvicorn.Server.serve", capture_app)
    asyncio.run(server.run_http_async(bind, 8090))
    with TestClient(captured[0], client=("10.20.30.41", 12345)) as client:
        headers = {"Host": "morgan.example:8090", "Accept": "application/json, text/event-stream"}
        assert client.post("/mcp", headers=headers, json=_INIT).status_code == 401
        headers["Authorization"] = "Bearer wrong"
        assert client.post("/mcp", headers=headers, json=_INIT).status_code == 401
        headers["Authorization"] = "Bearer synthetic-key"
        response = client.post("/mcp", headers=headers, json=_INIT)
        assert response.status_code == 200, response.text
        assert '"name":"morgan"' in response.text


def _refuse_chat(request):
    body = b'{"error":"synthetic refusal Bearer test-credential-only"}'
    return (
        b"HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )


def test_cli_chat_refusal_is_json_and_leaves_no_turn(tmp_path):
    with raw_model_server(_refuse_chat) as url:
        env = {
            **os.environ,
            "MORGAN_DATA_DIR": str(tmp_path),
            "MORGAN_EMBEDDING_BACKEND": "hash",
            "MORGAN_LLM_ENDPOINT": url,
            "MORGAN_LLM_API_KEY": "test-credential-only",
        }

        def run(*arguments):
            return subprocess.run(
                [sys.executable, "-m", "morgan_brain.surfaces.cli", *arguments],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

        failed = run("ask", "synthetic question", "--project", "audit", "--json")
        assert failed.returncode == 1
        message = json.loads(failed.stdout)["error"]
        assert "HTTP 401" in message and "MORGAN_LLM_API_KEY" in message
        assert "test-credential-only" not in failed.stdout + failed.stderr
        assert "Traceback" not in failed.stdout + failed.stderr
        recalled = run("recall", "synthetic question", "--project", "audit", "--json")
        assert recalled.returncode == 0, recalled.stderr
        assert json.loads(recalled.stdout)["results"] == []


async def test_mcp_chat_refusal_is_safe_and_leaves_no_turn(tmp_path, caplog):
    with raw_model_server(_refuse_chat) as url:
        server = build_server(
            Settings(
                data_dir=str(tmp_path),
                embedding_backend="hash",
                llm_endpoint=url,
                llm_api_key="test-credential-only",
            )
        )
        async with create_connected_server_and_client_session(server.mcp) as session:
            failed = await session.call_tool(
                "ask_morgan", {"text": "synthetic question", "project": "audit"}
            )
            assert failed.isError
            message = failed.content[0].text
            assert "HTTP 401" in message and "MORGAN_LLM_API_KEY" in message
            assert "test-credential-only" not in message + caplog.text
            recalled = await session.call_tool(
                "recall", {"query": "synthetic question", "project": "audit"}
            )
            assert not recalled.isError
            assert json.loads(recalled.content[0].text)["results"] == []


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("broken", ["setting", "dotenv", "data_dir", "snapshot_dir", "code_roots"])
def test_cli_configuration_errors_follow_output_contract(tmp_path, as_json, broken):
    env = {
        **os.environ,
        "MORGAN_DATA_DIR": str(tmp_path / "data"),
        "MORGAN_EMBEDDING_BACKEND": "hash",
    }
    if broken == "setting":
        # Invalid settings must name the field without echoing arbitrary input values.
        env["MORGAN_EMBEDDING_DIM"] = "private-invalid-value"
    elif broken == "dotenv":
        (tmp_path / ".env").write_bytes(b"\xff\xfe")
    else:
        env[f"MORGAN_{broken.upper()}"] = "~morgan-no-such-owner-9d198f/private-invalid-value"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "morgan_brain.surfaces.cli",
            "facts",
            *(["--json"] if as_json else []),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1
    if as_json:
        message = json.loads(result.stdout)["error"]
        assert result.stderr == ""
    else:
        assert result.stdout == ""
        message = result.stderr
    assert "configuration" in message
    if broken == "setting":
        assert "MORGAN_EMBEDDING_DIM" in message
    elif broken != "dotenv":
        assert f"MORGAN_{broken.upper()}" in message
    assert "private-invalid-value" not in result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("bind", ["127.0.0.1", "127.0.0.2", "::1", "localhost.localdomain"])
def test_loopback_http_keeps_host_protection(tmp_path, monkeypatch, bind):
    server = build_server(Settings(data_dir=str(tmp_path), embedding_backend="hash"))
    captured = []

    async def capture_app(uvicorn_server):
        captured.append(uvicorn_server.config.app)

    monkeypatch.setattr("uvicorn.Server.serve", capture_app)
    asyncio.run(server.run_http_async(bind, 8090))
    with TestClient(captured[0], client=("127.0.0.1", 12345)) as client:
        response = client.post(
            "/mcp",
            headers={"Host": "attacker.invalid", "Accept": "application/json, text/event-stream"},
            json=_INIT,
        )
        assert response.status_code == 421
        host = f"[{bind}]" if ":" in bind else bind
        response = client.post(
            "/mcp",
            headers={"Host": f"{host}:8090", "Accept": "application/json, text/event-stream"},
            json=_INIT,
        )
        assert response.status_code == 200, response.text
