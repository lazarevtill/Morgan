"""An import survives a connection the embedding host drops, and loses nothing to it.

Two dropped connections ended a real import on 2026-09-19: one lost request was the end of the
run. An import now asks for the import budget, so a drop is retried, and the memory whose
request was dropped is stored like every other -- in every index, because a memory visible to
one index and not another is found by one search and missed by the next.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from morgan_brain.composition import sqlite_path
from morgan_brain.config import Settings
from morgan_brain.memory.store.db import open_db
from morgan_brain.providers.factory import build_embedder
from morgan_brain.providers.wire import ProviderUnreachable
from morgan_brain.surfaces.cli.commands import cmd_import
from tests.fakes import Calls, flaky_model_server

_TURNS = 4


async def test_an_import_survives_a_dropped_connection(tmp_path):
    export = _export(tmp_path / "export.json")
    calls = Calls()
    # The third embedding request is dropped: two memories are stored before it and one after,
    # so the drop is in the middle of the import.
    with flaky_model_server(
        fail_times=1, status=None, after=2, embedding_dim=4, calls=calls
    ) as url:
        settings = _settings(tmp_path, url)
        result = await cmd_import(argparse.Namespace(path=str(export)), settings, "ignored")

    assert result["memories"] == _TURNS
    assert calls.total == _TURNS + 1, "the dropped request was not retried exactly once"
    indexed = _ids_in_every_index(settings)
    memories = indexed.pop("memories")
    assert len(memories) == _TURNS
    for index, ids in indexed.items():
        assert ids == memories, f"{index} is missing {sorted(memories - ids)}"


async def test_the_same_drop_ends_a_command_which_has_the_shorter_budget(tmp_path):
    """What the import survives, a command does not, under these settings: the import's
    survival is its own budget's doing, not a retry any call would have made."""
    with (
        flaky_model_server(fail_times=1, status=None, embedding_dim=4) as url,
        pytest.raises(ProviderUnreachable, match="dropped"),
    ):
        await build_embedder(_settings(tmp_path, url)).embed("x")


def _settings(tmp_path: Path, url: str) -> Settings:
    """A command's budget too short to wait out one backoff; an import's long enough."""
    return Settings(
        data_dir=str(tmp_path / "data"),
        embedding_backend="provider",
        embedding_endpoint=url,
        embedding_model="an-embedding-model",
        embedding_dim=4,
        embedding_retry_budget_seconds=0.5,
        embedding_import_retry_budget_seconds=10.0,
        embedding_retry_backoff_seconds=0.6,
        embedding_timeout_seconds=5.0,
    )


def _export(path: Path) -> Path:
    """One conversation, not held out, of *_TURNS* user turns, each naming an entity away from
    its sentence start so the entity index has a row for every memory."""
    conversation = {
        "conversation_id": "c2",
        "title": "t",
        "mapping": {
            f"m{i}": {
                "id": f"m{i}",
                "message": {
                    "id": f"m{i}",
                    "author": {"role": "user"},
                    "create_time": 1700000000.0 + i,
                    "content": {
                        "content_type": "text",
                        "parts": [f"we moved the registry to Harbor on day {i}"],
                    },
                },
            }
            for i in range(_TURNS)
        },
    }
    path.write_text(json.dumps([conversation]), encoding="utf-8")
    return path


def _ids_in_every_index(settings: Settings) -> dict[str, set[str]]:
    conn = open_db(sqlite_path(settings.temporal_db_url))
    try:
        return {
            "memories": _ids(conn, "SELECT id FROM memories"),
            "vectors": _ids(
                conn,
                "SELECT m.id FROM vec_meta AS m JOIN vec_items AS v ON v.rowid = m.rowid",
            ),
            "fts": _ids(conn, "SELECT memory_id FROM fts_memories"),
            "entities": _ids(conn, "SELECT DISTINCT memory_id FROM memory_entities"),
        }
    finally:
        conn.close()


def _ids(conn, sql: str) -> set[str]:
    return {row[0] for row in conn.execute(sql)}
