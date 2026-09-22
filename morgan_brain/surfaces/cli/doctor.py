"""``morgan doctor`` -- what this install reads, where it sends text, and whether each model
server answers, changing nothing.

doctor only reads. It opens the database read-only (``store/db.py::open_readonly``: SQLite's
read-only mode, sqlite-vec loaded, no pragma that writes) and runs SELECTs: it builds no store,
creates no table, runs no migration step and never switches the journal mode, so a database
another install still writes to, one waiting for ``morgan migrate`` or one ``morgan restore``
just put in place is left as it was found, byte for byte. A table that was never created is
named absent, never counted as an honest zero; a database file that does not exist is reported
missing, not created; and one that cannot be read read-only is reported as that, never opened
another way.

Every probe is independent and failure-tolerant, so one broken thing reports itself rather
than hiding the rest.

The chat server and the embedding server are probed separately, because they are often two
servers and fail on their own: ``provider`` is the chat endpoint, which ``ask`` and
``consolidate`` need, and ``embedding_provider`` is where every ``remember`` and ``recall``
embeds -- ``"not used"`` under the hash backend, which calls no server. Each is ``reachable``;
``slow``, when it answered after ``MORGAN_DOCTOR_SLOW_AFTER_SECONDS`` or answered a 429 or
another 5xx; ``refused``, when it answered and refused the request (a 401 or 403 for the key,
another 4xx, a 501, or a 200 that is not an embeddings response for the endpoint); or
``unreachable``, when no answer came within ``MORGAN_DOCTOR_PROBE_TIMEOUT_SECONDS`` or no
connection was made. A host that answered is never called unreachable: the embedding host
loads its model on the first request after idle, so slow is its normal first answer. The
embedding probe embeds the five fingerprint strings, and ``embedding_space`` compares their
vectors with the fingerprint the database recorded -- compares only: recording one is a write.
"""

from __future__ import annotations

import asyncio
import itertools
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from morgan_brain.composition import space_width_mismatch, sqlite_path
from morgan_brain.config import Settings
from morgan_brain.memory import fingerprint, migrations, snapshot
from morgan_brain.memory.store import projects, spaces
from morgan_brain.memory.store import vectors as vectors_store
from morgan_brain.memory.store.db import open_db, open_readonly
from morgan_brain.providers.factory import (
    Probe,
    build_embedder,
    chat_endpoint_of,
    check_embeddings_reachable,
    check_llm_reachable,
    embedding_endpoint_of,
)
from morgan_brain.providers.wire import is_refusal
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
    """Rows carrying the signature of an older Morgan's process still writing into a migrated
    database: an empty ``author_id`` on ``memories`` or ``facts`` (step 4 backfilled every
    existing row) or a NULL ``vec_items.status`` (step 6 backfilled it) -- every writer in this
    code sets both, so a row missing either after ``migrate`` did not come from this code.
    That covers a row of unknown origin written after the migration too: an old writer also
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
    #: ``doctor --vectors``'s sample, drawn while the connection is still open -- id, text and
    #: stored vector, per ``vectors_store.audit_sample`` -- and re-embedded only after it
    #: closes, the same way the embedding-space fingerprint check is compared afterward.
    vector_sample: list[tuple[str, str, list[float]]] = field(default_factory=list)


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


def _collect_local_probes(
    settings: Settings, *, project: str, all_projects: bool, vectors: bool
) -> _Local:
    """Every *local* probe: filesystem, SQLite, sqlite-vec, FTS5, row counts, the migration
    state, snapshots, the ``projects`` rows, the active embedding space and, with *vectors*,
    ``doctor --vectors``'s sample.

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
        # What every memory command refuses on when the active space is another width; None
        # when the two agree or there is no space to compare with.
        "embedding_dim_error": None,
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
        # Named, not walked: nothing walks them yet.
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

    # Read-only, and never retried with a connection that writes: a database doctor cannot
    # read that way is reported as that.
    try:
        conn = open_readonly(db_path, busy_timeout_ms=settings.db_busy_timeout_ms)
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the diagnostic tool
        _database_unread(report, f"failed to open database read-only: {exc}")
        return _Local(report)
    try:
        return _read_database(
            conn, report, settings, project=project, all_projects=all_projects, vectors=vectors
        )
    finally:
        conn.close()


def _read_database(
    conn: sqlite3.Connection,
    report: dict[str, Any],
    settings: Settings,
    *,
    project: str,
    all_projects: bool,
    vectors: bool,
) -> _Local:
    """Every probe of the database itself: SELECTs and PRAGMA reads, nothing else.

    One probe per helper below, in the order the report names them. Each catches its own
    failure, so a table that is missing or unreadable costs its own line and not the rest of
    the report: `doctor` is the command you run *because* something is broken.
    """
    local = _Local(report)
    _probe_migration(conn, report)
    _probe_embedding_space(conn, report, local)
    if vectors and local.space is not None and settings.embedding_backend != "hash":
        local.vector_sample = _vector_sample(
            conn, report, settings, space=local.space, project=project, all_projects=all_projects
        )
    _probe_projects(conn, report)
    _probe_row_counts(conn, report, settings, project=project, all_projects=all_projects)
    return local


def _probe_migration(conn: sqlite3.Connection, report: dict[str, Any]) -> None:
    """The database's ``user_version`` against the steps this build knows -- or, on a file that
    holds no Morgan table yet, why there is no migration state to read."""
    try:
        if migrations.holds_morgan_tables(conn):
            report["migration"] = migration_status_to_dict(
                user_version=int(conn.execute("PRAGMA user_version").fetchone()[0]),
                code_version=migrations.code_version(),
                pending=migrations.pending(conn),
            )
        else:
            report["migration_reason"] = (
                "the database holds no Morgan tables yet; the first command that opens it "
                "writes the latest schema"
            )
    except sqlite3.Error as exc:
        report.setdefault("probe_errors", {})["migration"] = str(exc)


def _probe_embedding_space(conn: sqlite3.Connection, report: dict[str, Any], local: _Local) -> None:
    """The active embedding space, kept on *local* for the comparison made after the connection
    is closed -- or why there is no space to compare the model against."""
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


def _vector_sample(
    conn: sqlite3.Connection,
    report: dict[str, Any],
    settings: Settings,
    *,
    space: spaces.EmbeddingSpace,
    project: str,
    all_projects: bool,
) -> list[tuple[str, str, list[float]]]:
    """``doctor --vectors``'s sample of *space*'s table, drawn while the connection is open;
    empty when that table cannot be read."""
    try:
        return vectors_store.audit_sample(
            conn,
            table_name=space.table_name,
            n=settings.vector_audit_sample_rows,
            user_id=settings.owner_user_id,
            project=project,
            all_projects=all_projects,
        )
    except sqlite3.Error as exc:
        report.setdefault("probe_errors", {})["vector_sample"] = str(exc)
        return []


def _probe_projects(conn: sqlite3.Connection, report: dict[str, Any]) -> None:
    """One line per row of ``projects``, or why the table has none to read."""
    try:
        if _table_exists(conn, "projects"):
            report["projects"] = [_project_line(p) for p in projects.list_all(conn)]
        else:
            report["projects_reason"] = "no projects table in this database yet"
    except sqlite3.Error as exc:
        report.setdefault("probe_errors", {})["projects"] = str(exc)


# A table name cannot be a bound parameter, so each count is a literal statement chosen
# by key. The project filter is a bound flag for the same reason: "every project" and
# "this one" are one statement, not two, so the two scopes can never drift apart.
_COUNT_SQL = {
    "memories": "SELECT COUNT(*) FROM memories WHERE user_id = ? AND (? OR project = ?)",
    "fts_memories": "SELECT COUNT(*) FROM fts_memories WHERE user_id = ? AND (? OR project = ?)",
    "vec_meta": "SELECT COUNT(*) FROM vec_meta WHERE user_id = ? AND (? OR project = ?)",
    "facts": "SELECT COUNT(*) FROM facts WHERE user_id = ? AND (? OR project = ?)",
    "session_history": (
        "SELECT COUNT(*) FROM session_history WHERE user_id = ? AND (? OR project = ?)"
    ),
    "memory_entities": (
        "SELECT COUNT(*) FROM memory_entities WHERE user_id = ? AND (? OR project = ?)"
    ),
}


def _count(
    conn: sqlite3.Connection,
    report: dict[str, Any],
    table: str,
    *,
    user_id: str,
    project: str,
    scope_all: bool,
) -> int | None:
    """The row count, or None when the table is absent or unreadable -- independently
    caught, like every other probe: `doctor` is the command you run *because*
    something is broken."""
    if not _table_exists(conn, table):
        return None
    try:
        row = conn.execute(_COUNT_SQL[table], (user_id, scope_all, project)).fetchone()
    except sqlite3.Error as exc:
        report.setdefault("count_errors", {})[table] = str(exc)
        return None
    return int(row[0])


def _memories_by_project(
    conn: sqlite3.Connection, report: dict[str, Any], *, user_id: str
) -> dict[str, int] | None:
    """Every project this user has a memory in, and how many -- what told the owner on
    2026-09-21 that a scoped zero was not the whole database: `doctor` from the wrong
    directory prints one project's count next to every project's, in one report."""
    if not _table_exists(conn, "memories"):
        return None
    try:
        rows = conn.execute(
            "SELECT project, COUNT(*) FROM memories WHERE user_id = ? GROUP BY project",
            (user_id,),
        ).fetchall()
    except sqlite3.Error as exc:
        report.setdefault("count_errors", {})["memories_by_project"] = str(exc)
        return None
    return {str(r[0]): int(r[1]) for r in rows}


def _probe_row_counts(
    conn: sqlite3.Connection,
    report: dict[str, Any],
    settings: Settings,
    *,
    project: str,
    all_projects: bool,
) -> None:
    """What is stored: this scope's rows, every project's, the per-project breakdown, and how
    many rows predate provenance."""
    user_id = settings.owner_user_id

    def count(table: str, *, scope_all: bool) -> int | None:
        return _count(conn, report, table, user_id=user_id, project=project, scope_all=scope_all)

    report["rows"] = {
        "scope": "all projects" if all_projects else f"project {project!r}",
        "memories": count("memories", scope_all=all_projects),
        "fts": count("fts_memories", scope_all=all_projects),
        "vectors": count("vec_meta", scope_all=all_projects),
    }
    report["rows_all_projects"] = {
        "memories": count("memories", scope_all=True),
        "fts": count("fts_memories", scope_all=True),
        "vectors": count("vec_meta", scope_all=True),
        "facts": count("facts", scope_all=True),
        "history": count("session_history", scope_all=True),
        "entities": count("memory_entities", scope_all=True),
    }
    report["rows_by_project"] = _memories_by_project(conn, report, user_id=user_id)
    try:
        missing, reason = _missing_provenance(conn, user_id=user_id)
    except sqlite3.Error as exc:
        report.setdefault("count_errors", {})["rows_missing_provenance"] = str(exc)
        missing, reason = None, None
    report["rows_missing_provenance"] = missing
    report["rows_missing_provenance_reason"] = reason


def _verdict(probe: Probe, settings: Settings) -> str:
    """``unreachable`` only when no answer came: none within the timeout, or no connection.
    A host that answered is never unreachable. It is ``refused`` when the answer refuses the
    request (``is_refusal``: a 401 or 403 for the key, another 4xx, a redirect or a 501 for
    the endpoint), as an embedding call's ``ProviderRefused`` classifies it, or is a 2xx that
    is not what was asked for (an embeddings request answered with a web page); ``slow`` when it
    answered after ``MORGAN_DOCTOR_SLOW_AFTER_SECONDS`` -- however much after -- or answered
    a 429 or another 5xx, which a retry may mend, as a host loading its model answers; and
    ``reachable`` otherwise."""
    if probe.status is None:
        return "unreachable"
    if 200 <= probe.status < 300:
        if probe.error is not None:
            return "refused"
        return "slow" if probe.seconds > settings.doctor_slow_after_seconds else "reachable"
    return "refused" if is_refusal(probe.status) else "slow"


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


#: The header ``doctor --vectors --clients N`` tags each of its N concurrent embedders'
#: requests with, so a fake server in a test -- or a real one's access log -- can tell one
#: client's requests from another's. Read by ``tests.fakes.vector_audit_server``.
_CLIENT_HEADER = "X-Morgan-Audit-Client"

#: Printed once, to stderr, before ``doctor --vectors`` sends the sample's text anywhere --
#: never to stdout, which carries ``--json``.
_SENDS_TEXT_NOTICE = (
    "doctor --vectors: sending the sampled memories' text to the embedding host to re-embed "
    "and compare against what is stored."
)

#: Carried in the report so ``--json`` says it too, not only the rendered text. Two causes, not
#: one: `wall_seconds` times the whole pass, so it includes Morgan's own per-request client
#: setup (a fresh one per row -- see `_client_pass`), which grows with --clients on this
#: process's own event loop even against a perfectly parallel host; and Ollama specifically may
#: also serialise concurrent clients rather than answer them in parallel. Either can make wall
#: time rise with --clients, and this number cannot tell them apart -- it is not evidence that
#: the host did or did not serialise anything.
_OLLAMA_NOTE = (
    "wall_seconds includes this process's own per-request client setup, which grows with "
    "--clients on its own; Ollama specifically may also serialise concurrent clients rather "
    "than answer them in parallel. A slower wall time with --clients than without it can be "
    "either, or both -- not evidence on its own that the host serialised anything."
)


def _safe_cosine(a: list[float], b: list[float]) -> float | None:
    """``fingerprint.cosine``, or ``None`` on a zero or non-finite vector -- reported as a
    failing row, never raised."""
    try:
        return fingerprint.cosine(a, b)
    except ValueError:
        return None


def _fails(cosine: float | None, tolerance: float) -> bool:
    """Whether *cosine* counts as a failing comparison: missing (``_safe_cosine`` already
    turned a zero or non-finite vector into ``None``), or not clearly at or above *tolerance*.

    Written as ``not (cosine >= tolerance)`` rather than ``cosine < tolerance``: two finite,
    non-zero vectors can still overflow inside ``fingerprint.cosine`` (huge components make
    the dot product and both norms infinite, and ``inf / inf`` is ``nan``), and a NaN result
    compares ``False`` against *both* ``<`` and ``>=`` -- so ``cosine < tolerance`` silently
    calls it a pass, while ``not (cosine >= tolerance)`` correctly calls it a failure.
    """
    return cosine is None or not (cosine >= tolerance)


def _any_pair_disagrees(fresh: list[list[float]], tolerance: float) -> bool:
    """Whether any two of *fresh* -- one row's vector, fresh from each client -- fall below
    *tolerance* against each other, or cannot be compared at all."""
    for one, other in itertools.combinations(fresh, 2):
        if _fails(_safe_cosine(one, other), tolerance):
            return True
    return False


async def _client_pass(
    settings: Settings, texts: list[str], label: str
) -> tuple[list[list[float]] | None, float, str | None]:
    """One audit client's whole pass over the sample: one embedder, one ``embed`` call per
    row, timed as the whole pass.

    Not one ``embed_batch`` call carrying every row: ``MORGAN_EMBEDDING_TIMEOUT_SECONDS``
    bounds a single attempt, sized for one call's answer, the way every other caller under the
    import budget sends one. A sample of 180 in one request routinely runs past it -- measured
    against a real host, every attempt died at the attempt timeout before the server finished,
    and the retry sent the same doomed request again, spending the whole budget without ever
    succeeding. One row per request is what the budget was measured against.

    Never raises -- a client that could not be reached at all reports its own error instead of
    failing the other clients' comparisons.
    """
    embedder = build_embedder(settings, conn=None, budget="import", headers={_CLIENT_HEADER: label})
    started = time.monotonic()
    fresh: list[list[float]] = []
    try:
        for text in texts:
            fresh.append(await embedder.embed(text))
    except Exception as exc:  # noqa: BLE001 -- report the failing client, never crash the audit
        return None, time.monotonic() - started, f"{type(exc).__name__}: {exc}"
    return fresh, time.monotonic() - started, None


async def _compare_sample(
    sample: list[tuple[str, str, list[float]]], settings: Settings, *, clients: int
) -> dict[str, Any]:
    """Re-embed *sample* under *clients* concurrent, independent embedders and compare every
    answer against what is stored -- and, with more than one client, against each other.

    Top-level ``min``/``median``/``below_tolerance`` pool every client's row comparisons
    together: the honest reading of "is it safe to import" is the worst any client saw, and it
    collapses to that one client's own numbers when ``clients`` is 1.
    """
    clients = max(1, clients)
    if not sample:
        return {
            "sampled": 0,
            "min": None,
            "median": None,
            "below_tolerance": [],
            "per_client": {},
            "disagreements": [],
            "note": _OLLAMA_NOTE,
        }
    print(_SENDS_TEXT_NOTICE, file=sys.stderr)
    ids = [row_id for row_id, _, _ in sample]
    texts = [text for _, text, _ in sample]
    stored = {row_id: vector for row_id, _, vector in sample}
    tolerance = settings.embedding_fingerprint_tolerance
    labels = [f"client-{n}" for n in range(1, clients + 1)]

    results = await asyncio.gather(*(_client_pass(settings, texts, label) for label in labels))

    per_client: dict[str, Any] = {}
    fresh_by_client: dict[str, list[list[float]]] = {}
    pooled: list[float] = []
    pooled_below: list[str] = []
    for label, (fresh, wall_seconds, error) in zip(labels, results, strict=True):
        if error is not None or fresh is None:
            per_client[label] = {
                "min": None,
                "median": None,
                "below_tolerance": [],
                "wall_seconds": round(wall_seconds, 3),
                "error": error,
            }
            continue
        fresh_by_client[label] = fresh
        # min/median are computed over every row this client's cosine could be computed for --
        # a failing (below-tolerance) row's cosine is a real, comparable number and belongs in
        # both populations, the same way the pooled ones already include it. Only a `None`
        # (zero or non-finite vector) has nothing to contribute.
        cosines: list[float] = []
        below: list[str] = []
        for row_id, vector in zip(ids, fresh, strict=True):
            cosine = _safe_cosine(vector, stored[row_id])
            if cosine is not None:
                cosines.append(cosine)
                pooled.append(cosine)
            if _fails(cosine, tolerance):
                below.append(row_id)
                if row_id not in pooled_below:
                    pooled_below.append(row_id)
        per_client[label] = {
            "min": min(cosines) if cosines else None,
            "median": statistics.median(cosines) if cosines else None,
            "below_tolerance": below,
            "wall_seconds": round(wall_seconds, 3),
        }

    disagreements: list[str] = []
    ok_labels = list(fresh_by_client)
    if len(ok_labels) >= 2:
        for idx, row_id in enumerate(ids):
            at_row = [fresh_by_client[label][idx] for label in ok_labels]
            if _any_pair_disagrees(at_row, tolerance):
                disagreements.append(row_id)

    return {
        "sampled": len(sample),
        "min": min(pooled) if pooled else None,
        "median": statistics.median(pooled) if pooled else None,
        "below_tolerance": pooled_below,
        "per_client": per_client,
        "disagreements": disagreements,
        "note": _OLLAMA_NOTE,
    }


async def _run_vector_audit(
    report: dict[str, Any], settings: Settings, local: _Local, *, clients: int
) -> dict[str, Any]:
    """``vector_audit`` and ``vector_audit_reason``, paired like ``embedding_space`` and its
    own reason: exactly one of them is not ``None``, so ``--json`` and the render both know
    why there are no numbers without guessing from an absent key. Called only when
    ``--vectors`` was passed; ``build_doctor_report`` sets both keys to ``None`` plus "not
    requested" on its own otherwise, so the pair is present in every report, asked for or not.

    Reads ``report["embedding_provider"]``, already set by the caller from the same run's
    embedding probe: an unreachable or refused host would otherwise make every client wait out
    the full import budget (``MORGAN_EMBEDDING_IMPORT_RETRY_BUDGET_SECONDS``, 600 s by default)
    only to report the same thing the probe already knows in one request.
    """
    if report.get("database_error"):
        return {
            "vector_audit": None,
            "vector_audit_reason": f"database not read: {report['database_error']}",
        }
    if settings.embedding_backend == "hash":
        return {
            "vector_audit": None,
            "vector_audit_reason": "the hash backend embeds locally; there is no host to audit",
        }
    embedding_verdict = report.get("embedding_provider")
    if embedding_verdict in ("unreachable", "refused"):
        return {
            "vector_audit": None,
            "vector_audit_reason": f"embedding host {embedding_verdict}; see embedding_provider",
        }
    if local.space is None:
        return {
            "vector_audit": None,
            "vector_audit_reason": report.get("embedding_space_reason")
            or "no active embedding space registered yet",
        }
    sample_error = report.get("probe_errors", {}).get("vector_sample")
    if sample_error:
        return {
            "vector_audit": None,
            "vector_audit_reason": f"failed to read the sample: {sample_error}",
        }
    if not local.vector_sample:
        return {"vector_audit": None, "vector_audit_reason": "no stored vectors in scope to sample"}
    audit = await _compare_sample(local.vector_sample, settings, clients=clients)
    return {"vector_audit": audit, "vector_audit_reason": None}


async def build_doctor_report(
    settings: Settings, *, project: str, all_projects: bool, vectors: bool = False, clients: int = 1
) -> dict[str, Any]:
    # The database is read, and its connection closed, in a worker thread while the two
    # servers are asked; the space read there is compared with the answer afterwards.
    local, chat, embeddings = await asyncio.gather(
        asyncio.to_thread(
            _collect_local_probes,
            settings,
            project=project,
            all_projects=all_projects,
            vectors=vectors,
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
        report["embedding_dim_error"] = space_width_mismatch(local.space, settings)
    # vector_audit/vector_audit_reason are always present, like embedding_space and its own
    # reason -- a --json consumer that reads one without checking for the other first must
    # never get a KeyError depending on whether --vectors happened to be passed.
    if vectors:
        report.update(await _run_vector_audit(report, settings, local, clients=clients))
    else:
        report["vector_audit"] = None
        report["vector_audit_reason"] = "not requested; pass --vectors"
    return report


async def _embeddings_answer(settings: Settings) -> Probe | None:
    """The embedding endpoint's answer to the five fingerprint strings; None under the hash
    backend, which has no endpoint."""
    if settings.embedding_backend == "hash":
        return None
    return await check_embeddings_reachable(settings)
