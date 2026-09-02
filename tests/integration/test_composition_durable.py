"""The assembled context uses one durable database, and every store is in it."""

import sqlite3

from morgan_brain.composition import build_app_context, build_memory_context
from morgan_brain.config import Settings


def test_every_store_shares_one_database_file(tmp_path, monkeypatch):
    """History must be reachable from the memory connection, or forget() cannot reach it."""
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    ctx = build_memory_context(Settings())
    ctx.conn.close()
    dbs = sorted(p.name for p in tmp_path.glob("*.db"))
    assert dbs == ["morgan.db"], f"expected one database, found {dbs}"

    conn = sqlite3.connect(tmp_path / "morgan.db")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "memories",
        "facts",
        "fts_memories",
        "memory_entities",
        "vec_meta",
        "mem_entity_nodes",
        "session_history",
    } <= tables, tables


def test_app_context_needs_no_model_to_build(tmp_path, monkeypatch):
    """Building the context must not call the model: `ask` fails on the call, never on
    construction, so `doctor` and the memory commands work with the server down."""
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("MORGAN_LLM_ENDPOINT", "http://127.0.0.1:1/v1")
    ctx = build_app_context(Settings())
    assert ctx.chat is not None and ctx.consolidator is not None
    ctx.conn.close()
