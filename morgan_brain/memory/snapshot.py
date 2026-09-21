"""Verified copies of the one database -- restorable, or it is not a snapshot at all.

``VACUUM INTO`` writes a consistent copy while other processes hold the file open under WAL,
which a filesystem copy of a live database cannot promise: a copy made mid-checkpoint can land
mid-write. A copy nobody checked is worse than none, so every snapshot is opened and
``PRAGMA quick_check``'d before it is handed back to the caller; one that fails is deleted on
the spot. Morgan never deletes a snapshot for any other reason -- every one that passes its
check stays on disk until the owner removes it.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from morgan_brain.memory.store.db import open_db

#: ``VACUUM INTO``'s destination has no bound-parameter form -- see ``take()`` -- so the reason
#: is restricted to this shape before it ever reaches SQL text. ``\Z`` rather than ``$``: ``$``
#: also matches immediately before a trailing newline, which would let ``"ok\n"`` through.
_REASON_RE = re.compile(r"^[a-z0-9-]{1,32}\Z")

#: The content tables a human reading ``morgan snapshot`` cares about. The index tables
#: (``vec_items``, ``vec_meta``, ``fts_memories``) are derived, so their row counts are not the
#: point -- and counting a ``vec0`` virtual table needs sqlite-vec loaded, which the read-only
#: connection ``_describe`` opens on the finished copy deliberately does not do (see there).
#: Written as literal SQL, one per name, rather than built from the tuple: no interpolation
#: means no S608/B608 finding to reason about here at all.
_COUNT_SQL: dict[str, str] = {
    "memories": "SELECT count(*) FROM memories",
    "facts": "SELECT count(*) FROM facts",
    "session_history": "SELECT count(*) FROM session_history",
}


class NotEnoughSpace(Exception):
    """Refused before writing anything: *path*'s volume has less free space than the database
    being copied."""

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)
        self.path = path


class SnapshotCorrupt(Exception):
    """A finished copy failed its own ``PRAGMA quick_check``. *path* has already been deleted."""

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)
        self.path = path


@dataclass
class SnapshotResult:
    path: Path
    bytes: int
    user_version: int
    counts: dict[str, int]


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


def _readonly_connect(path: Path) -> sqlite3.Connection:
    """A read-only connection to *path*. Describing or checking a copy must never mutate it --
    a read-write open flips ``journal_mode`` to WAL and leaves ``-wal``/``-shm`` sidecars next
    to what is meant to be a static archive file."""
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def _quick_check(path: Path) -> str:
    conn = _readonly_connect(path)
    try:
        return str(conn.execute("PRAGMA quick_check").fetchone()[0])
    finally:
        conn.close()


def _describe(path: Path) -> tuple[int, dict[str, int]]:
    """``user_version`` and a row count per content table (``_COUNT_SQL``), skipping any table
    the copy does not have."""
    conn = _readonly_connect(path)
    try:
        user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        counts = {
            table: int(conn.execute(sql).fetchone()[0])
            for table, sql in _COUNT_SQL.items()
            if _table_exists(conn, table)
        }
        return user_version, counts
    finally:
        conn.close()


def take(
    db_path: str,
    *,
    into: Path,
    reason: str,
    clock: Callable[[], datetime],
    busy_timeout_ms: int = 5000,
) -> SnapshotResult:
    """Write a verified ``VACUUM INTO`` copy of *db_path* under *into*, named by time and
    reason.

    In order: make *into*; refuse with ``NotEnoughSpace`` when *into*'s volume has less free
    space than *db_path*'s size; ``VACUUM INTO`` the timestamped destination; ``quick_check``
    the result, deleting it and raising ``SnapshotCorrupt`` on anything but ``"ok"``; read
    ``user_version`` and the content-table counts off the finished copy. This is the one path
    on which Morgan deletes a snapshot -- see the module docstring.
    """
    if not _REASON_RE.match(reason):
        raise ValueError(
            f"reason {reason!r} must match {_REASON_RE.pattern!r} -- VACUUM INTO's destination "
            "has no bound-parameter form, so the reason is validated before it reaches the SQL "
            "text below"
        )

    into.mkdir(parents=True, exist_ok=True)

    needed = Path(db_path).stat().st_size
    free = _free_bytes(into)
    if free < needed:
        raise NotEnoughSpace(
            into,
            f"only {free} bytes free under {into}, need {needed} for a snapshot of {db_path} "
            "-- point MORGAN_SNAPSHOT_DIR somewhere with more room",
        )

    stamp = clock().strftime("%Y%m%dT%H%M%SZ")
    dest = into / f"morgan-{stamp}-{reason}.db"

    # `reason` is validated above (`_REASON_RE`: `[a-z0-9-]{1,32}`) and `into` is a path this
    # process constructed, so `dest` cannot contain a single quote -- escaped anyway because
    # VACUUM INTO's destination is interpolated text, never a bound parameter, and a caller of
    # this function is not the one who gets to decide that stays true. (Neither ruff's S608
    # nor bandit's B608 fire on this line -- both pattern-match SELECT/INSERT/UPDATE/DELETE,
    # not VACUUM -- confirmed by running both; no suppression is added because none is needed.)
    escaped_dest = str(dest).replace("'", "''")
    conn = open_db(db_path, busy_timeout_ms=busy_timeout_ms)
    try:
        conn.execute(f"VACUUM INTO '{escaped_dest}'")
    finally:
        conn.close()

    check = _quick_check(dest)
    if check != "ok":
        dest.unlink(missing_ok=True)
        raise SnapshotCorrupt(dest, f"{dest} failed PRAGMA quick_check: {check}")

    user_version, counts = _describe(dest)
    return SnapshotResult(
        path=dest, bytes=dest.stat().st_size, user_version=user_version, counts=counts
    )


def list_snapshots(into: Path) -> list[SnapshotResult]:
    """Every snapshot under *into*, oldest first -- the timestamped filename sorts that way.

    Morgan never deletes a snapshot outside ``take()``'s own failed-check path, so this list
    only ever grows.
    """
    if not into.is_dir():
        return []
    results = []
    for path in sorted(into.glob("morgan-*.db")):
        user_version, counts = _describe(path)
        results.append(
            SnapshotResult(
                path=path, bytes=path.stat().st_size, user_version=user_version, counts=counts
            )
        )
    return results
