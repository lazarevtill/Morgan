"""Verified copies of the one database -- restorable, or it is not a snapshot at all.

``VACUUM INTO`` writes a consistent copy while other processes hold the file open under WAL,
which a filesystem copy of a live database cannot promise: a copy made mid-checkpoint can land
mid-write. A copy nobody checked is worse than none, so every snapshot is opened and
``PRAGMA quick_check``'d before it is handed back to the caller; one that fails is deleted on
the spot. Morgan never deletes a snapshot for any other reason -- every one that passes its
check stays on disk until the owner removes it.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from morgan_brain.memory import migrations
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


class SnapshotTooNew(Exception):
    """*path* was written by a Morgan with more migration steps than this one knows.

    Restoring it would run this database backwards: rows and columns a newer ``_STEPS`` added
    would sit in a database this build never upgrades to expect them. Refused before anything
    is touched -- upgrade morgan before restoring a snapshot written by a newer one.
    """

    def __init__(self, path: Path, snapshot_version: int, code_version: int) -> None:
        message = (
            f"{path} is at user_version {snapshot_version}, but this morgan only knows "
            f"{code_version} migration step(s) -- upgrade morgan before restoring a snapshot "
            "written by a newer one"
        )
        super().__init__(message)
        self.path = path
        self.snapshot_version = snapshot_version
        self.code_version = code_version


@dataclass
class SnapshotResult:
    path: Path
    bytes: int
    user_version: int
    counts: dict[str, int]


@dataclass
class RestoreResult:
    before: dict[str, int]
    after: dict[str, int]
    #: The safety copy of the database taken before it was replaced -- see ``restore()``.
    safety: SnapshotResult


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


def _require_ok(path: Path) -> None:
    """Raise ``SnapshotCorrupt`` unless *path* passes ``PRAGMA quick_check``.

    A plain function, not inlined at each call site: ``restore()`` calls this from inside a
    ``try``/``except`` that cleans up its own scratch file, and a ``raise`` written directly
    inside that block reads as though it might be caught there too.
    """
    check = _quick_check(path)
    if check != "ok":
        raise SnapshotCorrupt(path, f"{path} failed PRAGMA quick_check: {check}")


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


def _free_destination(into: Path, stamp: str, reason: str) -> Path:
    """The first name under *into* for this *stamp* and *reason* that nothing already
    occupies.

    Two snapshots with the same reason in the same UTC second are not hypothetical: a
    restore always takes its safety copy under the fixed reason ``before-restore``, so a
    second restore inside one second is the first caller to collide. ``VACUUM INTO`` refuses
    to write over an existing file with a bare ``sqlite3.OperationalError``, and Morgan never
    overwrites a snapshot that already passed its check -- so the smallest free numeric
    suffix is added instead.
    """
    candidate = into / f"morgan-{stamp}-{reason}.db"
    suffix = 2
    while candidate.exists():
        candidate = into / f"morgan-{stamp}-{reason}-{suffix}.db"
        suffix += 1
    return candidate


def take(
    db_path: str,
    *,
    into: Path,
    reason: str,
    clock: Callable[[], datetime],
    busy_timeout_ms: int = 5000,
) -> SnapshotResult:
    """Write a verified ``VACUUM INTO`` copy of *db_path* under *into*, named by time and
    reason (a numeric suffix is added when that name is already taken -- see
    ``_free_destination``).

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
    dest = _free_destination(into, stamp, reason)

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


def restore(
    db_path: str,
    *,
    source: Path,
    into: Path,
    clock: Callable[[], datetime],
    busy_timeout_ms: int = 5000,
) -> RestoreResult:
    """Replace *db_path* with a copy of *source*, behind a safety snapshot of *db_path*
    taken first. *source* is only ever read -- a restore does not delete the snapshot it
    restores from (SPEC-phase0 SS3.2: Morgan never deletes a snapshot).

    In order: ``quick_check`` *source* itself, raising ``SnapshotCorrupt`` on anything but
    ``"ok"`` -- a restore never reads from a copy that failed its own check; refuse with
    ``SnapshotTooNew`` when *source*'s ``user_version`` is ahead of what this build's
    ``migrations._STEPS`` knows how to read, naming both numbers; count *db_path*'s rows
    before anything changes; ``take()`` a safety copy of *db_path* under the fixed reason
    ``"before-restore"`` -- taken unconditionally, because a restore is the one command whose
    own mistake cannot be undone by running it again; copy *source* to a scratch file next to
    *db_path* (``f"{db_path}.restoring"``) and ``quick_check`` the copy; drop *db_path*'s own
    ``-wal``/``-shm`` -- their content is already captured durably in the safety snapshot
    above, and removing them now, before the swap, closes the window a crash could otherwise
    land in: a restored file paired with the *old* database's WAL, which a later
    ``PRAGMA journal_mode=WAL`` open could try to replay onto it; ``os.replace`` the scratch
    copy onto *db_path* -- same directory as *db_path*, so the swap is a same-volume rename
    even when *source* itself lives on a different drive; ``quick_check`` the result; count
    rows after. The scratch copy is removed on any failure between its creation and the swap.

    On Windows, ``os.replace`` raises ``PermissionError`` while another process still holds
    *db_path* open -- caught and re-raised naming the fix (close the sessions running
    ``morgan-mcp``) rather than forced.
    """
    _require_ok(source)

    snapshot_version, _ = _describe(source)
    code_version = len(migrations._STEPS)
    if snapshot_version > code_version:
        raise SnapshotTooNew(source, snapshot_version, code_version)

    _, before = _describe(Path(db_path))

    safety = take(
        db_path, into=into, reason="before-restore", clock=clock, busy_timeout_ms=busy_timeout_ms
    )

    # Same directory as db_path, never db_path itself: `source` is copied here rather than
    # moved into place directly, so it survives the restore untouched at its own path.
    scratch = Path(f"{db_path}.restoring")
    try:
        shutil.copyfile(source, scratch)
        _require_ok(scratch)

        Path(f"{db_path}-wal").unlink(missing_ok=True)
        Path(f"{db_path}-shm").unlink(missing_ok=True)

        try:
            os.replace(scratch, db_path)
        except PermissionError as exc:
            raise PermissionError(
                f"could not replace {db_path}: another process still has it open -- close any "
                "running morgan-mcp sessions and try again"
            ) from exc
    except BaseException:
        scratch.unlink(missing_ok=True)
        raise

    _require_ok(Path(db_path))

    _, after = _describe(Path(db_path))
    return RestoreResult(before=before, after=after, safety=safety)
