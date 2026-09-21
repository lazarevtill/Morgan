"""doctor embeds the five fingerprint strings as its probe, and says whether the model answering
is the one that wrote the stored vectors.

A model swapped for another of the same width is invisible to every width check; the recorded
fingerprint is what catches it. doctor compares and prints -- it never records one: recording
is a write, and a fingerprint recorded against whichever model happens to answer doctor would
bless the wrong one.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from morgan_brain.composition import build_memory_context, sqlite_path
from morgan_brain.config import Settings
from morgan_brain.memory import fingerprint
from morgan_brain.memory.store import spaces
from morgan_brain.memory.store.db import open_db
from morgan_brain.surfaces.cli.doctor import build_doctor_report
from morgan_brain.surfaces.cli.render import _render_doctor
from tests.fakes import _unit_vector, model_server

_DIM = 8


@pytest.fixture
def servers() -> Iterator[tuple[str, str]]:
    """A chat server and an embedding server, both answering at once."""
    with model_server() as chat, model_server(embedding_dim=_DIM) as embeddings:
        yield chat, embeddings


def _settings(tmp_path: Path, servers: tuple[str, str]) -> Settings:
    chat, embeddings = servers
    return Settings(
        data_dir=str(tmp_path),
        embedding_backend="provider",
        embedding_model="a-model",
        embedding_dim=_DIM,
        llm_endpoint=chat,
        embedding_endpoint=embeddings,
        doctor_probe_timeout_seconds=5.0,
    )


async def _report(settings: Settings) -> dict[str, Any]:
    return await build_doctor_report(settings, project="p", all_projects=False)


def _space(settings: Settings) -> spaces.EmbeddingSpace:
    conn = open_db(sqlite_path(settings.temporal_db_url))
    try:
        space = spaces.active(conn)
    finally:
        conn.close()
    assert space is not None
    return space


def _record(settings: Settings, vectors: list[list[float]]) -> None:
    conn = open_db(sqlite_path(settings.temporal_db_url))
    try:
        space = spaces.active(conn)
        assert space is not None
        spaces.record_fingerprint(
            conn, space.id, vectors, clock=lambda: datetime(2026, 9, 21, tzinfo=UTC)
        )
    finally:
        conn.close()


async def test_a_fingerprint_the_model_still_answers_matches(tmp_path, servers):
    settings = _settings(tmp_path, servers)
    build_memory_context(settings).conn.close()
    _record(settings, [_unit_vector(s, _DIM) for s in fingerprint.STRINGS])

    report = await _report(settings)

    space = report["embedding_space"]
    assert {k: space[k] for k in ("id", "model", "dims")} == {
        "id": 1,
        "model": "a-model",
        "dims": 8,
    }
    assert space["fingerprint"] == "matches (min cosine 1.0000)"
    assert space["strings_digest"] == fingerprint.DIGEST
    assert report["embedding_provider"] == "reachable"
    assert "embedding_space: 1 (a-model, 8 dims): fingerprint matches (min cosine 1.0000)" in (
        _render_doctor(report)
    )


async def test_a_space_with_no_fingerprint_is_unrecorded_and_doctor_records_none(tmp_path, servers):
    settings = _settings(tmp_path, servers)
    build_memory_context(settings).conn.close()

    report = await _report(settings)

    assert report["embedding_space"]["fingerprint"] == "unrecorded"
    assert _space(settings).fingerprint is None


async def test_a_same_width_model_that_answers_differently_is_a_mismatch(tmp_path, servers):
    settings = _settings(tmp_path, servers)
    build_memory_context(settings).conn.close()
    another_model = [
        _unit_vector(f"{s} as another model reads it", _DIM) for s in fingerprint.STRINGS
    ]
    _record(settings, another_model)
    recorded = _space(settings).fingerprint

    report = await _report(settings)

    verdict = report["embedding_space"]["fingerprint"]
    assert verdict.startswith("MISMATCH (min cosine 0.")
    assert _space(settings).fingerprint == recorded


async def test_a_model_of_another_width_is_a_mismatch_by_width(tmp_path, servers):
    settings = _settings(tmp_path, servers)
    build_memory_context(settings).conn.close()
    _record(settings, [_unit_vector(s, _DIM) for s in fingerprint.STRINGS])

    with model_server(embedding_dim=4) as narrower:
        report = await _report(settings.model_copy(update={"embedding_endpoint": narrower}))

    verdict = report["embedding_space"]["fingerprint"]
    assert verdict.startswith("MISMATCH") and "4" in verdict and "8" in verdict


async def test_no_answer_leaves_the_fingerprint_unchecked_with_the_reason(tmp_path, servers):
    settings = _settings(tmp_path, servers)
    build_memory_context(settings).conn.close()
    _record(settings, [_unit_vector(s, _DIM) for s in fingerprint.STRINGS])

    report = await _report(
        settings.model_copy(update={"embedding_endpoint": "http://127.0.0.1:1/v1"})
    )

    assert report["embedding_provider"] == "unreachable"
    assert report["embedding_space"]["fingerprint"].startswith("not checked")
