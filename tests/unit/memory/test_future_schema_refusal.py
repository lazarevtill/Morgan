"""Future schema admission fails before DDL, provider construction or header changes."""

import json
import sqlite3
import subprocess
import sys

import pytest

from morgan_brain.composition import (
    build_evidence_context,
    build_memory_context,
    build_memory_module,
)
from morgan_brain.config import Settings
from morgan_brain.memory import migrations
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.errors import DatabaseSchemaTooNew
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory


def test_future_empty_morgan_schema_cannot_be_restamped_or_initialized():
    conn = open_db(":memory:")
    conn.execute("CREATE TABLE future_lineage (id TEXT PRIMARY KEY, opaque TEXT)")
    conn.execute("PRAGMA user_version=999")
    conn.commit()
    before = conn.serialize()
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        with pytest.raises(DatabaseSchemaTooNew, match="newer"):
            build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4)
        assert conn.serialize() == before
        assert not any(
            statement.lstrip().upper().startswith(("CREATE", "INSERT", "UPDATE", "ALTER"))
            for statement in statements
        )
    finally:
        conn.close()


def test_future_file_refused_before_provider_or_journal_header_change(tmp_path, monkeypatch):
    path = tmp_path / "future.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE future_lineage (id TEXT PRIMARY KEY, opaque TEXT)")
    conn.execute("PRAGMA user_version=999")
    conn.commit()
    conn.close()
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("Provider construction must not occur for unsupported future schema")

    monkeypatch.setattr("morgan_brain.composition.build_embedder", forbidden)
    settings = Settings(
        temporal_db_url="sqlite:///" + str(path), embedding_backend="hash", embedding_dim=4
    )
    with pytest.raises(DatabaseSchemaTooNew, match="newer"):
        build_memory_context(settings)
    assert path.read_bytes() == before
    assert not path.with_name(path.name + "-wal").exists()


@pytest.mark.parametrize("operation", ["pending", "stamp_if_new", "upgrade", "migrate"])
def test_migration_entrypoints_refuse_future_without_writes(operation):
    conn = open_db(":memory:")
    conn.execute("PRAGMA user_version=999")
    before = conn.serialize()
    try:
        with pytest.raises(DatabaseSchemaTooNew, match="newer"):
            if operation in ("upgrade", "migrate"):
                getattr(migrations, operation)(conn, None)
            else:
                getattr(migrations, operation)(conn)
        assert conn.serialize() == before
    finally:
        conn.close()


def future_file(tmp_path):
    path = tmp_path / "unsupported.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE future_lineage (id TEXT PRIMARY KEY, opaque TEXT)")
    conn.execute("PRAGMA user_version=999")
    conn.commit()
    conn.close()
    return path, Settings(
        temporal_db_url="sqlite:///" + str(path), embedding_backend="hash", embedding_dim=4
    )


def test_source_only_future_refusal_keeps_unknown_source_bytes(tmp_path):
    path, settings = future_file(tmp_path)
    before = path.read_bytes()
    with pytest.raises(DatabaseSchemaTooNew) as refused:
        build_evidence_context(settings)
    assert refused.value.database_version == 999
    assert refused.value.supported_version == migrations.code_version()
    assert path.read_bytes() == before
    assert not path.with_name(path.name + "-wal").exists()


def test_cli_future_refusal_is_clean_json_without_traceback(tmp_path, monkeypatch, capsys):
    from morgan_brain.surfaces.cli import __main__ as cli

    path, settings = future_file(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(cli, "settings_for", lambda surface: settings)
    assert (
        cli.main(["remember", "Synthetic rejected input", "--project", "personal", "--json"]) == 1
    )
    captured = capsys.readouterr()
    error = json.loads(captured.out)["error"]
    assert "user_version 999" in error and "supported schema" in error
    assert "Traceback" not in captured.err
    assert path.read_bytes() == before


async def test_mcp_future_refusal_reaches_protocol_client_as_error(tmp_path):
    from mcp.shared.memory import create_connected_server_and_client_session

    from morgan_brain.surfaces.mcp_server import build_server

    path, settings = future_file(tmp_path)
    before = path.read_bytes()
    async with create_connected_server_and_client_session(build_server(settings).mcp) as session:
        result = await session.call_tool(
            "remember", {"text": "Synthetic rejected input", "project": "personal"}
        )
    assert result.isError
    message = result.content[0].text
    assert "user_version 999" in message and "supported schema" in message
    assert "Traceback" not in message
    assert path.read_bytes() == before


async def test_future_version_on_synthetic_current_copy_keeps_lineage_and_schema(tmp_path):
    path = tmp_path / "current-source.db"
    conn = open_db(str(path))
    module = build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4)
    await module.store(Memory(id="source", user_id="owner", content="Synthetic preserved source"))
    conn.close()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA user_version=999")
    conn.commit()
    expected_rows = conn.execute("SELECT * FROM memories").fetchall()
    expected_schema = conn.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall()
    conn.close()
    future_copy = tmp_path / "future-copy.db"
    future_copy.write_bytes(path.read_bytes())
    before = future_copy.read_bytes()
    settings = Settings(
        temporal_db_url="sqlite:///" + str(future_copy), embedding_backend="hash", embedding_dim=4
    )
    for factory in (build_memory_context, build_evidence_context):
        with pytest.raises(DatabaseSchemaTooNew):
            factory(settings)
        assert future_copy.read_bytes() == before
    conn = sqlite3.connect(future_copy)
    try:
        assert conn.execute("SELECT * FROM memories").fetchall() == expected_rows
        assert (
            conn.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall()
            == expected_schema
        )
    finally:
        conn.close()


@pytest.mark.parametrize(
    "first",
    [
        "morgan_brain.memory.store.db",
        "morgan_brain.memory.migrations",
        "morgan_brain.composition",
        "morgan_brain.memory.store.projects",
        "morgan_brain.memory.store.spaces",
        "morgan_brain.memory.store.vectors",
        "morgan_brain.memory.store.entities",
        "morgan_brain.memory.store.episodic",
    ],
)
def test_schema_guard_import_order_and_transaction_reexports_in_fresh_process(first):
    program = (
        f"import importlib; importlib.import_module({first!r})"
        + """
from morgan_brain.memory.store import db, transactions
from morgan_brain.memory import migrations
from morgan_brain import composition
assert db.read_transaction is transactions.read_transaction
assert db.write_transaction is transactions.write_transaction
conn=db.open_db(":memory:")
conn.execute("PRAGMA user_version=999")
try:
    migrations.require_supported_version(conn)
except ValueError as error:
    assert "newer" in str(error)
else:
    raise AssertionError("Future schema must be refused")
finally:
    conn.close()
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
