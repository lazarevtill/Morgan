"""The assembled app must use durable stores, not in-memory ones."""

import sqlite3

from morgan_brain.composition import build_app_context
from morgan_brain.config import Settings
from morgan_brain.modules.memory.stores.sqlite_vector import SqliteVectorIndex


def test_app_context_uses_the_sqlite_vector_index(tmp_path, monkeypatch):
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    ctx = build_app_context(Settings())
    assert isinstance(ctx.vectors, SqliteVectorIndex), type(ctx.vectors)


def test_every_store_shares_one_database_file(tmp_path, monkeypatch):
    """signals and history must be reachable from the memory connection, or forget() cannot work.

    This includes ``prompt_versions`` (the ``LocalPromptRegistry``, holding champion-preprompt
    versions): it shares the same connection as every other store -- the one-database invariant
    (Task 13A) -- so backup, at-rest encryption, and "does forget() reach everything" stay a
    single-file story. It is NOT touched by forget()'s DELETE statements, though: a promoted
    champion may embed text mined from a now-forgotten conversation and can only be reviewed
    and rolled back by hand, never silently deleted (see ``ForgetReport.champions_flagged``).
    Sharing the connection and being exempt from erasure are independent decisions.
    """
    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    build_app_context(Settings())
    dbs = sorted(p.name for p in tmp_path.glob("*.db"))
    assert dbs == ["morgan.db"], f"expected one database, found {dbs}"

    conn = sqlite3.connect(tmp_path / "morgan.db")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "memories",
        "facts",
        "interaction_signals",
        "session_history",
        "prompt_versions",
    } <= tables, tables


def test_brain_api_starts_and_stops_the_bus(tmp_path, monkeypatch):
    """Nothing called bus.start() before this task, so queued cold-path work never ran."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MORGAN_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MORGAN_EMBEDDING_BACKEND", "hash")
    from morgan_brain.apps.brain_api.app import create_app

    app = create_app()
    with TestClient(app) as client:  # __enter__ runs the lifespan
        assert client.get("/health").status_code == 200
        assert app.state.ctx.bus.is_running is True
    assert app.state.ctx.bus.is_running is False
