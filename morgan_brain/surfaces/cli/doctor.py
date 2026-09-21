"""``morgan doctor`` -- what this install reads, where it sends text, and whether each model
server answers, changing nothing.

doctor only reads. It opens the database with sqlite-vec loaded and runs SELECTs: it builds no
store, creates no table and runs no migration step, so a database another install still writes
to, or one waiting for ``morgan migrate``, is left as it was found. A table that was never
created is named absent, never counted as an honest zero, and a database file that does not
exist is reported missing, not created.

Every probe is independent and failure-tolerant, so one broken thing reports itself rather
than hiding the rest.

The chat server and the embedding server are probed separately, because they are often two
servers and fail on their own: ``provider`` is the chat endpoint, which ``ask`` and
``consolidate`` need, and ``embedding_provider`` is where every ``remember`` and ``recall``
embeds -- ``"not used"`` under the hash backend, which calls no server. Each is ``reachable``;
``slow``, when it answered after ``MORGAN_DOCTOR_SLOW_AFTER_SECONDS``; or ``unreachable``, when
no answer came within ``MORGAN_DOCTOR_PROBE_TIMEOUT_SECONDS`` or the answer was an error. A
host that answers slowly is never called unreachable: the embedding host loads its model on the
first request after idle, so slow is its normal first answer. The embedding probe embeds the
five fingerprint strings, and ``embedding_space`` compares their vectors with the fingerprint
the database recorded -- compares only: recording one is a write.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from morgan_brain.composition import sqlite_path
from morgan_brain.config import Settings
from morgan_brain.memory import fingerprint, migrations, snapshot
from morgan_brain.memory.store import projects, spaces
from morgan_brain.memory.store.db import open_db
from morgan_brain.providers.factory import (
    Probe,
    chat_endpoint_of,
    check_embeddings_reachable,
    check_llm_reachable,
    embedding_endpoint_of,
)
from morgan_brain.surfaces.cli.payloads import embedding_space_to_dict, migration_status_to_dict

#: What the chat endpoint receives, by the command that sends it (``app/chat.py`` builds the
#: prompt; ``memory/knowledge/consolidation.py`` reads up to 50 memories per project).
_CHAT_CARRIES = {
    "ask": "the question, the memories recalled for it and the recent history",
    "consolidate": "up to 50 memories per project, with the project's current facts",
}


def _embedding_carries(settings: Settings) -> dict[str, str]:
    """What the embedding endpoint receives, by the command that sends it."""
    return {
        "remember": "the memory's text",
        "recall": "the query",
        "import": "every imported message",
        "doctor --vectors": "a sample of stored memories",
        "a process's first embedding call": (
            f"up to {settings.embedding_fingerprint_sample_rows} stored memories, while the "
            "embedding space's fingerprint is unrecorded"
        ),
    }


def _host(url: str) -> str:
    """The host alone: no scheme, port or path, and so none of the userinfo a URL can carry."""
    try:
        host = urlsplit(url).hostname
    except ValueError:
        host = None
    return host or "(no host)"


def _data_flow(settings: Settings) -> list[dict[str, Any]]:
    """One entry per endpoint: its host, the settings that address it, and what each command
    sends there. Embeddings sent to the chat endpoint make one entry carrying both."""
    routes = [(chat_endpoint_of(settings), _CHAT_CARRIES)]
    if settings.embedding_backend != "hash":
        routes.append((embedding_endpoint_of(settings), _embedding_carries(settings)))
    flows: dict[str, dict[str, Any]] = {}
    for endpoint, carries in routes:
        flow = flows.setdefault(
            endpoint.url.rstrip("/"),
            {"host": _host(endpoint.url), "settings": [], "carries": {}},
        )
        if endpoint.setting not in flow["settings"]:
            flow["settings"].append(endpoint.setting)
        flow["carries"].update(carries)
    return list(flows.values())


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
#: a bug, and so is a table absent from a database no command has opened since it was made --
#: either way the count reports ``None`` with a reason instead of a wrong zero.
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
    its column yet, *or* is absent -- doctor creates no table, so one the database does not
    have stays missing. Either way a count would be meaningless, not an honest zero: an absent
    table is named in the reason exactly like a present one missing its column, never silently
    skipped into the total the way ``0`` rows would be.
    """
    missing = [
        f"{table}.{column}" if _table_exists(conn, table) else table
        for table, column in _PROVENANCE_CHECKS
        if not _table_exists(conn, table) or column not in _column_names(conn, table)
    ]
    if missing:
        return None, (
            f"not migrated yet or missing entirely: {', '.join(missing)}; run `morgan migrate`"
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


@dataclass
class _Local:
    """What the local probes found, and the active embedding space, read here so the model's
    answer can be compared with it once the probe returns -- after the connection is closed."""

    report: dict[str, Any]
    space: spaces.EmbeddingSpace | None = None


def _collect_library_probes(report: dict[str, Any]) -> None:
    """sqlite-vec and FTS5, which belong to this Python's SQLite rather than to any database:
    probed on a connection of their own, in memory, so neither probe touches the owner's file
    and both answer when there is no database yet."""
    try:
        conn = open_db(":memory:")
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the diagnostic tool
        report["sqlite_vec_error"] = str(exc)
        return
    try:
        row = conn.execute("SELECT vec_version()").fetchone()
        report["sqlite_vec"] = row[0] if row else None
    except Exception as exc:  # noqa: BLE001
        report["sqlite_vec_error"] = str(exc)
    try:
        conn.execute("CREATE VIRTUAL TABLE __morgan_fts5_probe USING fts5(x)")
        report["fts5"] = True
    except sqlite3.OperationalError:
        report["fts5"] = False
    finally:
        conn.close()


def _snapshots(settings: Settings) -> dict[str, Any]:
    """How many snapshots there are, the newest, and the bytes they take -- Morgan never
    deletes one, so this only grows until the owner does."""
    found = snapshot.list_snapshots(Path(settings.snapshot_dir))
    return {
        "dir": settings.snapshot_dir,
        "count": len(found),
        "newest": found[-1].path.name if found else None,
        "bytes": sum(s.bytes for s in found),
    }


def _project_line(project: projects.Project) -> dict[str, Any]:
    """A ``projects`` row's classification and switches. Not its remote: a remote URL can
    carry a token, and this report is the kind of thing that gets pasted into an issue."""
    return {
        "name": project.name,
        "classification": project.classification,
        "capture_enabled": project.capture_enabled,
        "paused_until": project.paused_until,
        "retention_days": project.retention_days,
        "consolidate_enabled": project.consolidate_enabled,
    }


#: The report's lines that read the database, each with a ``*_reason`` beside it for when it
#: could not: the database is missing or will not open.
_DATABASE_REASONS: tuple[str, ...] = (
    "embedding_space_reason",
    "migration_reason",
    "rows_missing_provenance_reason",
    "projects_reason",
)


def _database_unread(report: dict[str, Any], why: str) -> None:
    """The database was not read: *why* is its error, and the reason on every line that
    would have read it -- so none of them is empty without saying so."""
    report["database_error"] = why
    for reason in _DATABASE_REASONS:
        report[reason] = why


def _collect_local_probes(settings: Settings, *, project: str, all_projects: bool) -> _Local:
    """Every *local* probe: filesystem, SQLite, sqlite-vec, FTS5, row counts, the migration
    state, snapshots, the ``projects`` rows and the active embedding space.

    Synchronous on purpose -- each probe blocks -- so ``build_doctor_report`` hands the
    whole body to a worker thread.
    """
    db_path = sqlite_path(settings.temporal_db_url)
    hash_backend = settings.embedding_backend == "hash"
    report: dict[str, Any] = {
        "database": db_path if db_path == ":memory:" else str(Path(db_path).resolve()),
        # "no database yet at <path>" or why it would not open; None when it opened.
        "database_error": None,
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
        "data_flow": _data_flow(settings),
        "sqlite_vec": None,
        "fts5": False,
        "provider": "unreachable",
        "provider_probe": None,
        "embedding_provider": "not used" if hash_backend else "unreachable",
        "embedding_probe": None,
        "embedding_space": None,
        "embedding_space_reason": None,
        "migration": None,
        "migration_reason": None,
        "snapshots": None,
        "rows": None,
        "rows_all_projects": None,
        "rows_by_project": None,
        "rows_missing_provenance": None,
        "rows_missing_provenance_reason": None,
        "projects": None,
        "projects_reason": None,
        # Named, not walked: the walk over them is phase 1a.
        "code_roots": [
            {"path": root, "is_directory": Path(root).is_dir()} for root in settings.code_roots
        ],
    }
    _collect_library_probes(report)
    try:
        report["snapshots"] = _snapshots(settings)
    except Exception as exc:  # noqa: BLE001
        report.setdefault("probe_errors", {})["snapshots"] = str(exc)

    # sqlite3.connect creates a file that is not there; doctor must not be what creates it.
    # Only nothing at the path is missing: whatever else is there -- a directory, a file that
    # is not a database -- is opened, fails to open, and is reported as that.
    if db_path != ":memory:" and not Path(db_path).exists():
        _database_unread(report, f"no database yet at {db_path}")
        return _Local(report)

    try:
        conn = open_db(db_path, busy_timeout_ms=settings.db_busy_timeout_ms)
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the diagnostic tool
        _database_unread(report, f"failed to open database: {exc}")
        return _Local(report)
    try:
        return _read_database(conn, report, settings, project=project, all_projects=all_projects)
    finally:
        conn.close()


def _read_database(
    conn: sqlite3.Connection,
    report: dict[str, Any],
    settings: Settings,
    *,
    project: str,
    all_projects: bool,
) -> _Local:
    """Every probe of the database itself: SELECTs and PRAGMA reads, nothing else."""
    local = _Local(report)
    try:
        if migrations._holds_morgan_tables(conn):
            report["migration"] = migration_status_to_dict(
                user_version=int(conn.execute("PRAGMA user_version").fetchone()[0]),
                code_version=len(migrations._STEPS),
                pending=migrations.pending(conn),
            )
        else:
            report["migration_reason"] = (
                "the database holds no Morgan tables yet; the first command that opens it "
                "writes the latest schema"
            )
    except sqlite3.Error as exc:
        report.setdefault("probe_errors", {})["migration"] = str(exc)

    try:
        if not _table_exists(conn, "embedding_spaces"):
            report["embedding_space_reason"] = (
                "no embedding_spaces table in this database yet; `morgan migrate` creates it "
                "and registers the space"
            )
        else:
            local.space = spaces.active(conn)
            if local.space is None:
                report["embedding_space_reason"] = "no active embedding space registered yet"
    except sqlite3.Error as exc:
        report.setdefault("probe_errors", {})["embedding_space"] = str(exc)

    try:
        if _table_exists(conn, "projects"):
            report["projects"] = [_project_line(p) for p in projects.all(conn)]
        else:
            report["projects_reason"] = "no projects table in this database yet"
    except sqlite3.Error as exc:
        report.setdefault("probe_errors", {})["projects"] = str(exc)

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
    return local


def _verdict(probe: Probe, settings: Settings) -> str:
    """``unreachable`` only when no answer came in time or the answer was an error: a host
    that answered successfully is ``reachable``, or ``slow`` when that took longer than
    ``MORGAN_DOCTOR_SLOW_AFTER_SECONDS`` -- however much longer."""
    if probe.status is None or not 200 <= probe.status < 300:
        return "unreachable"
    return "slow" if probe.seconds > settings.doctor_slow_after_seconds else "reachable"


def _probe_to_dict(probe: Probe, settings: Settings) -> dict[str, Any]:
    return {
        "seconds": round(probe.seconds, 3),
        "timeout_seconds": settings.doctor_probe_timeout_seconds,
        "slow_after_seconds": settings.doctor_slow_after_seconds,
        "error": probe.error,
    }


def _fingerprint_verdict(
    space: spaces.EmbeddingSpace, probe: Probe | None, settings: Settings
) -> str:
    """``matches (min cosine 0.9993)``, ``unrecorded`` or ``MISMATCH (min cosine 0.41)`` --
    or ``not checked`` with the reason, when there were no fresh vectors to compare. Compared
    string by string against ``MORGAN_EMBEDDING_FINGERPRINT_TOLERANCE``, as the first-call
    check compares; nothing is recorded."""
    if space.fingerprint is None:
        return "unrecorded"
    if probe is None:
        return "not checked: the hash backend calls no model"
    if probe.vectors is None:
        return f"not checked: {probe.error}"
    widths = sorted({len(v) for v in probe.vectors})
    if widths != [space.dims]:
        answered = "/".join(str(w) for w in widths)
        return f"MISMATCH (the model answers {answered}-wide vectors; the space is {space.dims})"
    try:
        stored = spaces.unpack(space.fingerprint, dims=space.dims)
    except Exception as exc:  # noqa: BLE001 -- a report, not a crash
        return f"not checked: the recorded fingerprint does not read ({type(exc).__name__})"
    try:
        comparison = fingerprint.compare(probe.vectors, stored)
    except ValueError as exc:
        return f"MISMATCH ({exc})"
    tolerance = settings.embedding_fingerprint_tolerance
    # Counted as passes, never as `c < tolerance` failures, so a cosine that is not a number
    # counts as a failure.
    passing = sum(1 for c in comparison.per_string if c >= tolerance)
    word = "matches" if passing == len(comparison.per_string) else "MISMATCH"
    return f"{word} (min cosine {comparison.min_cosine:.4f})"


async def build_doctor_report(
    settings: Settings, *, project: str, all_projects: bool
) -> dict[str, Any]:
    # The database is read, and its connection closed, in a worker thread while the two
    # servers are asked; the space read there is compared with the answer afterwards.
    local, chat, embeddings = await asyncio.gather(
        asyncio.to_thread(
            _collect_local_probes, settings, project=project, all_projects=all_projects
        ),
        check_llm_reachable(settings),
        _embeddings_answer(settings),
    )
    report = local.report
    report["provider"] = _verdict(chat, settings)
    report["provider_probe"] = _probe_to_dict(chat, settings)
    if embeddings is not None:
        report["embedding_provider"] = _verdict(embeddings, settings)
        report["embedding_probe"] = _probe_to_dict(embeddings, settings)
    if local.space is not None:
        report["embedding_space"] = embedding_space_to_dict(
            local.space,
            fingerprint=_fingerprint_verdict(local.space, embeddings, settings),
            strings_digest=fingerprint.DIGEST,
        )
    return report


async def _embeddings_answer(settings: Settings) -> Probe | None:
    """The embedding endpoint's answer to the five fingerprint strings; None under the hash
    backend, which has no endpoint."""
    if settings.embedding_backend == "hash":
        return None
    return await check_embeddings_reachable(settings)
