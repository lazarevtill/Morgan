"""Re-embedding a sample is the only way to find a vector the server got wrong.

The stored blob is self-consistent: nothing in the database disagrees with it. Only asking the
model again, and comparing, finds the row that was embedded by a model having a bad moment.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from morgan_brain.config import Settings
from morgan_brain.memory.store import spaces
from morgan_brain.memory.store import vectors as vectors_store
from morgan_brain.memory.store.db import open_db, open_readonly
from morgan_brain.models import PERSONAL_PROJECT
from morgan_brain.surfaces.cli.doctor import audit_vectors, build_doctor_report
from tests.fakes import _unit_vector, vector_audit_server

_DIM = 16
_USER = "owner"
_PROJECT = PERSONAL_PROJECT
#: Refuses at once (a reserved port), so a probe of the chat endpoint -- not under test here --
#: never waits out a real timeout.
_CLOSED = "http://127.0.0.1:1/v1"


def _clock() -> datetime:
    return datetime(2026, 9, 21, tzinfo=UTC)


async def _seed(tmp_path: Path, rows: int) -> str:
    """*rows* memories, ids and content both ``"id-0"``..``"id-{rows-1}"`` -- so a scripted
    fake server can key its answers off exactly the text ``doctor`` sends it -- each paired
    with the vector ``tests.fakes``'s own servers answer for that same text: a clean sample
    compares a stored vector against itself under ``serve="same"``. Named ``morgan.db``: the
    one full-report test builds ``Settings(data_dir=tmp_path)`` and needs its default
    ``temporal_db_url`` to find this same file."""
    path = str(tmp_path / "morgan.db")
    conn = open_db(path)
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT NOT NULL)")
    index = vectors_store.SqliteVectorIndex(conn, dim=_DIM)
    for i in range(rows):
        memory_id = f"id-{i}"
        conn.execute("INSERT INTO memories (id, content) VALUES (?, ?)", (memory_id, memory_id))
        await index.upsert(
            vectors_store.VectorRecord(
                id=memory_id,
                user_id=_USER,
                project=_PROJECT,
                vector=_unit_vector(memory_id, _DIM),
            )
        )
    conn.commit()
    conn.close()
    return path


async def _audit(tmp_path: Path, *, rows: int, serve: Any, clients: int = 1) -> dict[str, Any]:
    path = await _seed(tmp_path, rows)
    wrong = None if serve == "same" else serve
    with vector_audit_server(wrong=wrong, embedding_dim=_DIM) as url:
        settings = Settings(data_dir=str(tmp_path), embedding_endpoint=url, embedding_dim=_DIM)
        conn = open_readonly(path)
        try:
            return await audit_vectors(
                conn,
                settings,
                table_name="vec_items",
                project=_PROJECT,
                all_projects=False,
                clients=clients,
            )
        finally:
            conn.close()


async def test_a_clean_sample_reports_its_minimum(tmp_path):
    audit = await _audit(tmp_path, rows=10, serve="same")
    assert audit["below_tolerance"] == [] and audit["min"] >= 0.995


async def test_one_wrong_row_is_named(tmp_path):
    audit = await _audit(tmp_path, rows=10, serve={"id-7": "wrong"})
    assert audit["below_tolerance"] == ["id-7"]


async def test_two_clients_name_the_one_that_disagreed(tmp_path):
    audit = await _audit(tmp_path, rows=10, clients=2, serve={"client-2": {"id-3": "wrong"}})
    assert audit["disagreements"] == ["id-3"]
    assert audit["per_client"]["client-2"]["below_tolerance"] == ["id-3"]


async def test_a_zero_or_non_finite_vector_is_reported_failing_not_raised(tmp_path):
    audit = await _audit(tmp_path, rows=10, serve={"id-1": "zero", "id-2": "nan"})
    assert set(audit["below_tolerance"]) == {"id-1", "id-2"}
    assert audit["min"] is not None and audit["min"] >= 0.995  # the other 8 rows compare clean


async def test_vectors_writes_nothing(tmp_path):
    path = await _seed(tmp_path, 5)
    setup = open_db(path)
    spaces.create_schema(setup)
    spaces.register(setup, model="m", dims=_DIM, table_name="vec_items", clock=_clock)
    setup.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    setup.commit()
    setup.close()
    before = _read_bytes(path)

    conn = open_readonly(path)
    try:
        tables_before = _tables(conn)
        with vector_audit_server(wrong={"id-2": "wrong"}, embedding_dim=_DIM) as url:
            settings = Settings(
                data_dir=str(Path(path).parent), embedding_endpoint=url, embedding_dim=_DIM
            )
            audit = await audit_vectors(
                conn, settings, table_name="vec_items", project=_PROJECT, all_projects=False
            )
        assert audit["below_tolerance"] == ["id-2"]
        assert spaces.active(conn).fingerprint is None
        assert _tables(conn) == tables_before
    finally:
        conn.close()

    assert _read_bytes(path) == before


def _read_bytes(path: str) -> bytes:
    """A plain function, not an inline ``Path.read_bytes()``: ruff's ASYNC240 flags a blocking
    pathlib call written directly in an ``async def`` test body."""
    return Path(path).read_bytes()


def _tables(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    rows = conn.execute("SELECT type, name FROM sqlite_master ORDER BY type, name").fetchall()
    return [(str(r["type"]), str(r["name"])) for r in rows]


async def test_the_stderr_notice_appears_and_stdout_stays_pure_json(tmp_path, capsys):
    settings_project = _PROJECT
    path = await _seed(tmp_path, 5)
    with vector_audit_server(embedding_dim=_DIM) as url:
        settings = Settings(
            data_dir=str(tmp_path),
            llm_endpoint=_CLOSED,
            embedding_endpoint=url,
            embedding_dim=_DIM,
        )
        conn = open_db(path)
        spaces.create_schema(conn)
        spaces.register(conn, model="m", dims=_DIM, table_name="vec_items", clock=_clock)
        conn.close()
        report = await build_doctor_report(
            settings, project=settings_project, all_projects=False, vectors=True, clients=1
        )

    captured = capsys.readouterr()
    assert "sending" in captured.err and "embedding host" in captured.err
    # Not a blanket "stdout is empty": structlog, uninitialised in this unit test (production
    # always calls configure_logging() first), still writes its own lines to stdout here. What
    # matters is that doctor's own notice -- the one that could corrupt --json -- never does.
    assert "sending" not in captured.out and "embedding host" not in captured.out
    assert report["vector_audit"] is not None
