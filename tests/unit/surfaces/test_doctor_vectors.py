"""Re-embedding a sample is the only way to find a vector the server got wrong.

The stored blob is self-consistent: nothing in the database disagrees with it. Only asking the
model again, and comparing, finds the row that was embedded by a model having a bad moment.

Every test goes through ``build_doctor_report`` -- the path ``doctor`` actually runs, gating
included -- never a test-only shortcut around it: a regression in the production glue (the
gate drawing the sample only for ``all_projects``, say, or the wrong connection reaching
``CheckedEmbedder``) must be able to fail one of these.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from morgan_brain.config import Settings
from morgan_brain.memory.store import spaces
from morgan_brain.memory.store import vectors as vectors_store
from morgan_brain.memory.store.db import open_db, open_readonly
from morgan_brain.models import PERSONAL_PROJECT
from morgan_brain.surfaces.cli.doctor import build_doctor_report
from tests.fakes import _unit_vector, model_server, vector_audit_server

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
    compares a stored vector against itself under ``serve="same"``. Named ``morgan.db``: every
    test here builds ``Settings(data_dir=tmp_path)`` and needs its default ``temporal_db_url``
    to find this same file. Registers an active embedding space too, so every test's call to
    ``build_doctor_report`` finds one without repeating that setup itself.
    """
    path = str(tmp_path / "morgan.db")
    conn = open_db(path)
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT NOT NULL)")
    index = vectors_store.SqliteVectorIndex(conn, dim=_DIM)
    spaces.create_schema(conn)
    spaces.register(conn, model="m", dims=_DIM, table_name="vec_items", clock=_clock)
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


def _settings(tmp_path: Path, url: str) -> Settings:
    return Settings(
        data_dir=str(tmp_path), llm_endpoint=_CLOSED, embedding_endpoint=url, embedding_dim=_DIM
    )


async def _audit(tmp_path: Path, *, rows: int, serve: Any, clients: int = 1) -> dict[str, Any]:
    """The path ``doctor --vectors [--clients N]`` actually runs: seed, probe, gate, re-embed,
    compare -- through ``build_doctor_report``, reading ``report["vector_audit"]`` back."""
    await _seed(tmp_path, rows)
    wrong = None if serve == "same" else serve
    with vector_audit_server(wrong=wrong, embedding_dim=_DIM) as url:
        report = await build_doctor_report(
            _settings(tmp_path, url),
            project=_PROJECT,
            all_projects=False,
            vectors=True,
            clients=clients,
        )
    return report["vector_audit"]


async def test_a_clean_sample_reports_its_minimum(tmp_path):
    audit = await _audit(tmp_path, rows=10, serve="same")
    assert audit["below_tolerance"] == [] and audit["min"] >= 0.995


async def test_one_wrong_row_is_named(tmp_path):
    audit = await _audit(tmp_path, rows=10, serve={"id-7": "wrong"})
    assert audit["below_tolerance"] == ["id-7"]
    # A client's own min must be the worst comparable cosine it saw, not the best -- the
    # exact negation the fake server answers for "wrong" puts it at -1.0 exactly.
    assert audit["min"] == pytest.approx(-1.0)
    assert audit["per_client"]["client-1"]["min"] == pytest.approx(-1.0)


async def test_two_clients_name_the_one_that_disagreed(tmp_path):
    audit = await _audit(tmp_path, rows=10, clients=2, serve={"client-2": {"id-3": "wrong"}})
    assert audit["disagreements"] == ["id-3"]
    assert audit["per_client"]["client-2"]["below_tolerance"] == ["id-3"]
    # client-2's own min shows the failing row; client-1, which answered everything
    # normally, has nothing below tolerance and so nothing to lower its own min.
    assert audit["per_client"]["client-2"]["min"] == pytest.approx(-1.0)
    assert audit["per_client"]["client-1"]["min"] == pytest.approx(1.0)
    assert audit["min"] == pytest.approx(-1.0)


async def test_a_zero_or_non_finite_vector_is_reported_failing_not_raised(tmp_path):
    audit = await _audit(tmp_path, rows=10, serve={"id-1": "zero", "id-2": "nan"})
    assert set(audit["below_tolerance"]) == {"id-1", "id-2"}
    assert audit["min"] is not None and audit["min"] >= 0.995  # the other 8 rows compare clean


async def test_a_nan_comparison_between_clients_counts_as_a_disagreement(tmp_path):
    """Huge (but individually finite) components overflow inside ``fingerprint.cosine``'s
    own arithmetic -- both norms and the dot product become infinite, and ``inf / inf`` is
    ``nan``. A NaN cosine compares ``False`` against both ``<`` and ``>=``, so the old
    ``cosine < tolerance`` check silently called this a pass; ``not (cosine >= tolerance)``
    calls it a failure instead."""
    audit = await _audit(tmp_path, rows=10, clients=2, serve={"id-4": "overflow"})
    assert audit["disagreements"] == ["id-4"]


async def test_vectors_writes_nothing(tmp_path):
    path = await _seed(tmp_path, 5)
    setup = open_db(path)
    setup.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    setup.commit()
    setup.close()
    before = _read_bytes(path)
    tables_before = _tables_readonly(path)

    with vector_audit_server(wrong={"id-2": "wrong"}, embedding_dim=_DIM) as url:
        report = await build_doctor_report(
            _settings(tmp_path, url), project=_PROJECT, all_projects=False, vectors=True, clients=1
        )

    assert report["vector_audit"]["below_tolerance"] == ["id-2"]
    assert _tables_readonly(path) == tables_before
    assert _read_bytes(path) == before
    conn = open_readonly(path)
    try:
        assert spaces.active(conn).fingerprint is None
    finally:
        conn.close()


async def test_a_failed_sample_read_is_reported_as_what_it_was(tmp_path, monkeypatch):
    """``audit_sample`` raising is a read failure, not an empty archive -- the two must
    not collapse into the same reason."""
    await _seed(tmp_path, 5)

    def _boom(*_args: object, **_kwargs: object) -> list[Any]:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr("morgan_brain.surfaces.cli.doctor.vectors_store.audit_sample", _boom)
    with vector_audit_server(embedding_dim=_DIM) as url:
        report = await build_doctor_report(
            _settings(tmp_path, url), project=_PROJECT, all_projects=False, vectors=True, clients=1
        )

    assert report["vector_audit"] is None
    assert "disk I/O error" in report["vector_audit_reason"]
    assert "no stored vectors" not in report["vector_audit_reason"]


async def test_the_audit_is_skipped_when_the_embedding_host_refused(tmp_path):
    """The plain embedding probe already answered this run's question -- a client sent to
    re-embed 180 rows against a host that just refused would each wait out the full import
    budget (600 s by default) to report the same thing."""
    await _seed(tmp_path, 5)
    with model_server(embeddings=False) as url:  # a chat-only server refuses every embed: 501
        report = await build_doctor_report(
            _settings(tmp_path, url), project=_PROJECT, all_projects=False, vectors=True, clients=1
        )

    assert report["embedding_provider"] == "refused"
    assert report["vector_audit"] is None
    assert "refused" in report["vector_audit_reason"]


async def test_the_audit_is_skipped_when_the_embedding_host_is_unreachable(tmp_path):
    await _seed(tmp_path, 5)
    settings = Settings(
        data_dir=str(tmp_path), llm_endpoint=_CLOSED, embedding_endpoint=_CLOSED, embedding_dim=_DIM
    )

    report = await build_doctor_report(
        settings, project=_PROJECT, all_projects=False, vectors=True, clients=1
    )

    assert report["embedding_provider"] == "unreachable"
    assert report["vector_audit"] is None
    assert "unreachable" in report["vector_audit_reason"]


async def test_vector_audit_keys_are_always_present_even_without_vectors(tmp_path):
    """A ``--json`` consumer reading ``vector_audit_reason`` beside ``vector_audit`` must
    never get a ``KeyError`` depending on whether ``--vectors`` happened to be passed."""
    await _seed(tmp_path, 5)
    with vector_audit_server(embedding_dim=_DIM) as url:
        report = await build_doctor_report(
            _settings(tmp_path, url), project=_PROJECT, all_projects=False, vectors=False
        )

    assert report["vector_audit"] is None
    assert report["vector_audit_reason"] == "not requested; pass --vectors"


def _read_bytes(path: str) -> bytes:
    """A plain function, not an inline ``Path.read_bytes()``: ruff's ASYNC240 flags a blocking
    pathlib call written directly in an ``async def`` test body."""
    return Path(path).read_bytes()


def _tables(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    rows = conn.execute("SELECT type, name FROM sqlite_master ORDER BY type, name").fetchall()
    return [(str(r["type"]), str(r["name"])) for r in rows]


def _tables_readonly(path: str) -> list[tuple[str, str]]:
    conn = open_readonly(path)
    try:
        return _tables(conn)
    finally:
        conn.close()


async def test_the_stderr_notice_appears_and_stdout_stays_pure_json(tmp_path, capsys):
    await _seed(tmp_path, 5)
    with vector_audit_server(embedding_dim=_DIM) as url:
        report = await build_doctor_report(
            _settings(tmp_path, url), project=_PROJECT, all_projects=False, vectors=True, clients=1
        )

    captured = capsys.readouterr()
    assert "sending" in captured.err and "embedding host" in captured.err
    # Not a blanket "stdout is empty": structlog, uninitialised in this unit test (production
    # always calls configure_logging() first), still writes its own lines to stdout here. What
    # matters is that doctor's own notice -- the one that could corrupt --json -- never does.
    assert "sending" not in captured.out and "embedding host" not in captured.out
    assert report["vector_audit"] is not None
