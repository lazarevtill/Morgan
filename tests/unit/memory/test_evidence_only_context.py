"""Source-only binding uses the same gate without assembling the full memory module."""

import sqlite3

import pytest

from morgan_brain.composition import build_evidence_context, build_memory_module
from morgan_brain.config import Settings
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory, MemoryQuery


async def test_evidence_context_fails_clearly_for_unsupported_full_operations(tmp_path):
    path = tmp_path / "morgan.db"
    conn = open_db(str(path))
    build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4)
    conn.close()
    context = build_evidence_context(Settings(temporal_db_url=f"sqlite:///{path.as_posix()}"))
    try:
        result = await context.gate.evidence(
            user_id="owner", project="personal", evidence_ids=["absent"]
        )
        assert result.missing_ids == ["absent"]
        with pytest.raises(RuntimeError, match="evidence reads only"):
            await context.gate.recall(MemoryQuery(user_id="owner", text="tea"))
        with pytest.raises(RuntimeError, match="evidence reads only"):
            await context.gate.store(Memory(user_id="owner", content="tea"))
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            context.conn.execute("DELETE FROM memories")
    finally:
        context.conn.close()


def test_missing_source_schema_errors_without_creating_tables(tmp_path):
    path = tmp_path / "other.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE unrelated (id TEXT)")
    conn.commit()
    conn.close()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Morgan evidence source schema"):
        build_evidence_context(Settings(temporal_db_url=f"sqlite:///{path.as_posix()}"))
    assert path.read_bytes() == before


@pytest.mark.parametrize("version", [9, 10])
async def test_evidence_read_never_creates_or_changes_erasure_metadata(tmp_path, version):
    path = tmp_path / "morgan.db"
    conn = open_db(str(path))
    module = build_memory_module(conn, embedder=FakeEmbedder(dim=4), dim=4)
    await module.store(Memory(id="source", user_id="owner", content="Synthetic source"))
    if version == 9:
        conn.execute("DROP TABLE erasure_state")
    else:
        conn.execute("UPDATE erasure_state SET generation=7")
    conn.execute(f"PRAGMA user_version={version}")
    conn.commit()
    conn.close()
    before = path.read_bytes()
    context = build_evidence_context(Settings(temporal_db_url=f"sqlite:///{path.as_posix()}"))
    try:
        result = await context.gate.evidence(
            user_id="owner", project="personal", evidence_ids=["source"]
        )
        assert result.records[0].content == "Synthetic source"
        assert context.conn.execute("PRAGMA user_version").fetchone()[0] == version
        exists = context.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='erasure_state'"
        ).fetchone()
        assert (exists is not None) == (version == 10)
        if version == 10:
            assert context.conn.execute("SELECT generation FROM erasure_state").fetchone()[0] == 7
    finally:
        context.conn.close()
    assert path.read_bytes() == before
