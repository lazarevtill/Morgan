"""On a database waiting for `morgan migrate`, a write command refuses before it does any work.

Refusing only at the gate is too late for three of them. ``ask`` would call the model -- up to
a cold host's whole load time -- and write both halves of the exchange into the session history,
outside the gate, before the gate refused the memories; the owner would never see the reply,
and the history would feed it to the next turn. ``consolidate`` would pay for a model call, and
``import`` would read the whole export. Each says why and does nothing else.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from morgan_brain.composition import build_memory_context, sqlite_path
from morgan_brain.config import Settings
from morgan_brain.memory import migrations
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory
from morgan_brain.surfaces.cli.commands import cmd_ask, cmd_consolidate, cmd_import

_REASON = "writes are blocked until `morgan migrate` runs: 1 step pending (3 a heavy step)"

#: One literal count per table ``store/tables.py`` registers -- every table a write lands in,
#: the session history ``ask`` writes outside the gate included. Literal, because a table name
#: cannot be a bound parameter.
_COUNT_SQL = {
    "memories": "SELECT count(*) FROM memories",
    "facts": "SELECT count(*) FROM facts",
    "memory_entities": "SELECT count(*) FROM memory_entities",
    "vec_meta": "SELECT count(*) FROM vec_meta",
    "vec_items": "SELECT count(*) FROM vec_items",
    "fts_memories": "SELECT count(*) FROM fts_memories",
    "session_history": "SELECT count(*) FROM session_history",
}


@pytest.mark.parametrize("command", ["ask", "consolidate", "import"])
async def test_a_write_command_refuses_before_any_request_or_row(tmp_path, monkeypatch, command):
    with _counting_model_server() as (endpoint, requests):
        settings = await _a_read_only_database(tmp_path, monkeypatch, endpoint)
        before = _rows(settings)

        with pytest.raises(migrations.DatabaseNeedsMigration) as exc:
            await _run(command, settings, _export(tmp_path))

    assert exc.value.reason == _REASON
    assert requests == []
    assert _rows(settings) == before


async def test_an_import_is_refused_before_its_export_is_read(tmp_path, monkeypatch):
    with _counting_model_server() as (endpoint, _requests):
        settings = await _a_read_only_database(tmp_path, monkeypatch, endpoint)

        with pytest.raises(migrations.DatabaseNeedsMigration):
            await _run("import", settings, tmp_path / "an export that was never written.json")


async def _run(command: str, settings: Settings, export: Path) -> dict[str, Any]:
    if command == "ask":
        return await cmd_ask(argparse.Namespace(text="what blocked the deploy?"), settings, "p")
    if command == "consolidate":
        return await cmd_consolidate(argparse.Namespace(all_projects=False), settings, "p")
    return await cmd_import(argparse.Namespace(path=str(export)), settings, "p")


async def _a_read_only_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> Settings:
    """One memory in project ``p`` at ``user_version`` 2, under code whose step 3 is heavy.

    The chat model is *endpoint*; embeddings are the local hash, so any request the server
    sees is one a command sent to the model.
    """
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("MORGAN_LLM_ENDPOINT", endpoint)

    version_two = migrations._STEPS[:2]
    monkeypatch.setattr(migrations, "_STEPS", version_two)
    ctx = build_memory_context(Settings())
    try:
        await ctx.gate.store(
            Memory(
                user_id=ctx.settings.owner_user_id,
                project="p",
                content="The Harbor mirror blocked the deploy.",
            )
        )
    finally:
        ctx.conn.close()

    heavy = migrations.Step(3, "a heavy step", True, lambda c, s: None)
    monkeypatch.setattr(migrations, "_STEPS", (*version_two, heavy))
    return Settings()


def _export(tmp_path: Path) -> Path:
    """A one-turn ChatGPT export, in the export's own shape."""
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


def _rows(settings: Settings) -> dict[str, int]:
    conn = open_db(sqlite_path(settings.temporal_db_url))
    try:
        return {t: int(conn.execute(sql).fetchone()[0]) for t, sql in _COUNT_SQL.items()}
    finally:
        conn.close()


@contextmanager
def _counting_model_server() -> Iterator[tuple[str, list[str]]]:
    """A chat server that answers every completion and records the path of every request.

    It answers rather than failing, so a command that reached it would go on to do whatever
    it does after the model replies -- which is what the tests must see it never do.
    """
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            requests.append(self.path)
            self._reply(
                200,
                {
                    "id": "c",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Noted."},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        def _reply(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            """The suite's output is not a request log."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
