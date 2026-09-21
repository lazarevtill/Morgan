"""``morgan doctor`` -- answer "is this install working" without a model server.

Every probe is independent and failure-tolerant, so one broken thing reports itself
rather than hiding the rest. A table that was never created is named as skipped, never
counted as an honest zero.

The chat server and the embedding server are probed separately, because they are often two
servers and fail on their own: ``provider`` is the chat endpoint, which ``ask`` and
``consolidate`` need, and ``embedding_provider`` is where every ``remember`` and ``recall``
embeds -- ``"not used"`` under the hash backend, which calls no server.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

from morgan_brain.composition import (
    build_memory_module,
    sqlite_path,
)
from morgan_brain.config import Settings
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.store.db import open_db
from morgan_brain.providers.factory import (
    check_embeddings_reachable,
    check_llm_reachable,
    embedding_endpoint_of,
)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


def _collect_local_probes(
    settings: Settings, *, project: str, all_projects: bool
) -> dict[str, Any]:
    """Every *local* probe: filesystem, SQLite, sqlite-vec, FTS5, row counts.

    Synchronous on purpose -- each probe blocks -- so ``build_doctor_report`` hands the
    whole body to a worker thread.
    """
    db_path = sqlite_path(settings.temporal_db_url)
    hash_backend = settings.embedding_backend == "hash"
    report: dict[str, Any] = {
        "database": db_path if db_path == ":memory:" else str(Path(db_path).resolve()),
        # The first question after "why is my brain empty?" is "which config did it read?"
        # These are the files this surface read, in order, and whether each was there.
        "env_files": list(settings.env_files_read),
        "project": project,
        "all_projects": all_projects,
        "embedding_backend": settings.embedding_backend,
        "embedding_dim": settings.embedding_dim,
        "embedding_endpoint": None if hash_backend else embedding_endpoint_of(settings).url,
        "llm_endpoint": settings.llm_endpoint,
        "llm_model": settings.llm_model,
        "sqlite_vec": None,
        "fts5": False,
        "provider": "unreachable",
        "embedding_provider": "not used" if hash_backend else "unreachable",
        "vector_rows": None,
        "memory_rows": None,
        "fts_rows": None,
    }

    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        conn = open_db(db_path)
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the diagnostic tool
        report["error"] = f"failed to open database: {exc}"
        return report

    try:
        row = conn.execute("SELECT vec_version()").fetchone()
        report["sqlite_vec"] = row[0] if row else None
    except Exception as exc:  # noqa: BLE001
        report["sqlite_vec_error"] = str(exc)

    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __morgan_fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS __morgan_fts5_probe")
        report["fts5"] = True
    except sqlite3.OperationalError:
        report["fts5"] = False

    # Build the real schema (idempotent CREATE ... IF NOT EXISTS) so row counts are always
    # meaningful -- 0 on a fresh install rather than "table doesn't exist yet".
    try:
        build_memory_module(
            conn, embedder=FakeEmbedder(dim=settings.embedding_dim), dim=settings.embedding_dim
        )
    except Exception as exc:  # noqa: BLE001
        report["schema_error"] = str(exc)

    # A table name cannot be a bound parameter, so each count is a literal statement chosen
    # by key. The project filter is a bound flag for the same reason.
    count_sql = {
        "memories": "SELECT COUNT(*) FROM memories WHERE user_id = ? AND (? OR project = ?)",
        "fts_memories": (
            "SELECT COUNT(*) FROM fts_memories WHERE user_id = ? AND (? OR project = ?)"
        ),
        "vec_meta": "SELECT COUNT(*) FROM vec_meta WHERE user_id = ? AND (? OR project = ?)",
    }
    params = (settings.owner_user_id, all_projects, project)

    def _count(table: str) -> int | None:
        """The row count, or None when the table is absent or unreadable -- independently
        caught, like every other probe: `doctor` is the command you run *because*
        something is broken."""
        if not _table_exists(conn, table):
            return None
        try:
            row = conn.execute(count_sql[table], params).fetchone()
        except sqlite3.Error as exc:
            report.setdefault("count_errors", {})[table] = str(exc)
            return None
        return int(row[0])

    report["memory_rows"] = _count("memories")
    report["fts_rows"] = _count("fts_memories")
    report["vector_rows"] = _count("vec_meta")
    conn.close()
    return report


async def build_doctor_report(
    settings: Settings, *, project: str, all_projects: bool
) -> dict[str, Any]:
    report = await asyncio.to_thread(
        _collect_local_probes, settings, project=project, all_projects=all_projects
    )
    chat, embeddings = await asyncio.gather(
        check_llm_reachable(settings), _embeddings_answer(settings)
    )
    report["provider"] = "reachable" if chat else "unreachable"
    if embeddings is not None:
        report["embedding_provider"] = "reachable" if embeddings else "unreachable"
    return report


async def _embeddings_answer(settings: Settings) -> bool | None:
    """Whether the embedding endpoint answers; None under the hash backend, which has none."""
    if settings.embedding_backend == "hash":
        return None
    return await check_embeddings_reachable(settings)
