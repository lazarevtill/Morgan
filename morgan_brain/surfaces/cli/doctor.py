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


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    """A table's own column names, read with the table name bound rather than interpolated
    (``pragma_table_info`` takes it as a table-valued function argument, unlike ``PRAGMA
    table_info(...)``, which cannot take a bound parameter at all)."""
    return {str(r[0]) for r in conn.execute("SELECT name FROM pragma_table_info(?)", (table,))}


#: The table and column migration step 4 (``memories``/``facts`` author) and step 6
#: (``vec_items`` status) add. Checked by ``_missing_provenance`` below: a column absent on a
#: database that has not been through that step yet is ordinary before ``morgan migrate``, not
#: a bug, and so is a table entirely absent because schema-building partially failed -- either
#: way the count reports ``None`` with a reason instead of a wrong zero.
_PROVENANCE_CHECKS: tuple[tuple[str, str], ...] = (
    ("memories", "author_id"),
    ("facts", "author_id"),
    ("vec_items", "status"),
)


def _missing_provenance(conn: sqlite3.Connection, *, user_id: str) -> tuple[int | None, str | None]:
    """Rows carrying the signature of a pre-phase-0 process still writing into a migrated
    database: an empty ``author_id`` on ``memories`` or ``facts`` (step 4 backfilled every
    existing row) or a NULL ``vec_items.status`` (step 6 backfilled it) -- every phase-0
    writer sets both, so a row missing either after ``migrate`` did not come from this code.
    Subsumes the spec's "unknown origin written after the migration" case: an old writer also
    leaves ``author_id`` empty, and nothing records when the migration ran.

    Returns ``(None, reason)`` when a checked table has not been through the step that added
    its column yet, *or* is entirely absent -- ``build_memory_module`` builds ``memories``
    before ``vec_items`` (``composition.py``), so a partial failure (``schema_error``,
    ``sqlite_vec_error``) can leave the later ones missing outright while the earlier ones
    exist. Either way a count would be meaningless, not an honest zero: an absent table is
    named in the reason exactly like a present one missing its column, never silently
    skipped into the total the way ``0`` rows would be.
    """
    missing = [
        f"{table}.{column}" if _table_exists(conn, table) else table
        for table, column in _PROVENANCE_CHECKS
        if not _table_exists(conn, table) or column not in _column_names(conn, table)
    ]
    if missing:
        return None, (
            f"not migrated yet or missing entirely: {', '.join(missing)}; run `morgan migrate`, "
            "or see schema_error/sqlite_vec_error above if it should already exist"
        )
    # Every checked table exists and has its column -- `missing` above already ruled out the
    # only two ways this count could be meaningless.
    total = 0
    for table, column in _PROVENANCE_CHECKS:
        empty = "IS NULL" if table == "vec_items" else "= ''"
        # `table` and `column` come only from `_PROVENANCE_CHECKS` above, never from a caller.
        row = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE user_id = ? AND {column} {empty}",  # noqa: S608 # nosec B608
            (user_id,),
        ).fetchone()
        total += int(row[0])
    return total, None


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
        "rows": None,
        "rows_all_projects": None,
        "rows_by_project": None,
        "rows_missing_provenance": None,
        "rows_missing_provenance_reason": None,
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
    # by key. The project filter is a bound flag for the same reason: "every project" and
    # "this one" are one statement, not two, so the two scopes can never drift apart.
    count_sql = {
        "memories": "SELECT COUNT(*) FROM memories WHERE user_id = ? AND (? OR project = ?)",
        "fts_memories": (
            "SELECT COUNT(*) FROM fts_memories WHERE user_id = ? AND (? OR project = ?)"
        ),
        "vec_meta": "SELECT COUNT(*) FROM vec_meta WHERE user_id = ? AND (? OR project = ?)",
        "facts": "SELECT COUNT(*) FROM facts WHERE user_id = ? AND (? OR project = ?)",
        "session_history": (
            "SELECT COUNT(*) FROM session_history WHERE user_id = ? AND (? OR project = ?)"
        ),
        "memory_entities": (
            "SELECT COUNT(*) FROM memory_entities WHERE user_id = ? AND (? OR project = ?)"
        ),
    }

    def _count(table: str, *, scope_all: bool) -> int | None:
        """The row count, or None when the table is absent or unreadable -- independently
        caught, like every other probe: `doctor` is the command you run *because*
        something is broken."""
        if not _table_exists(conn, table):
            return None
        try:
            row = conn.execute(
                count_sql[table], (settings.owner_user_id, scope_all, project)
            ).fetchone()
        except sqlite3.Error as exc:
            report.setdefault("count_errors", {})[table] = str(exc)
            return None
        return int(row[0])

    def _by_project() -> dict[str, int] | None:
        """Every project this user has a memory in, and how many -- what told the owner on
        2026-09-21 that a scoped zero was not the whole database: `doctor` from the wrong
        directory prints one project's count next to every project's, in one report."""
        if not _table_exists(conn, "memories"):
            return None
        try:
            rows = conn.execute(
                "SELECT project, COUNT(*) FROM memories WHERE user_id = ? GROUP BY project",
                (settings.owner_user_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            report.setdefault("count_errors", {})["memories_by_project"] = str(exc)
            return None
        return {str(r[0]): int(r[1]) for r in rows}

    report["rows"] = {
        "scope": "all projects" if all_projects else f"project {project!r}",
        "memories": _count("memories", scope_all=all_projects),
        "fts": _count("fts_memories", scope_all=all_projects),
        "vectors": _count("vec_meta", scope_all=all_projects),
    }
    report["rows_all_projects"] = {
        "memories": _count("memories", scope_all=True),
        "fts": _count("fts_memories", scope_all=True),
        "vectors": _count("vec_meta", scope_all=True),
        "facts": _count("facts", scope_all=True),
        "history": _count("session_history", scope_all=True),
        "entities": _count("memory_entities", scope_all=True),
    }
    report["rows_by_project"] = _by_project()
    try:
        missing, reason = _missing_provenance(conn, user_id=settings.owner_user_id)
    except sqlite3.Error as exc:
        report.setdefault("count_errors", {})["rows_missing_provenance"] = str(exc)
        missing, reason = None, None
    report["rows_missing_provenance"] = missing
    report["rows_missing_provenance_reason"] = reason
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
