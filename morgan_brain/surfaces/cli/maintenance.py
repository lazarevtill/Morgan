"""``morgan snapshot`` and ``morgan restore`` -- a verified VACUUM INTO copy of the whole
database, and putting one back.

Neither is project-scoped: each acts on the whole database file, not one project's rows, so
these handlers -- unlike every other verb -- ignore ``--project``/``--all-projects``, and
their subparsers in ``__main__.py`` offer neither flag.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from morgan_brain.composition import sqlite_path
from morgan_brain.config import Settings
from morgan_brain.memory import snapshot
from morgan_brain.surfaces.cli.payloads import restore_to_dict, snapshot_to_dict


def _utcnow() -> datetime:
    return datetime.now(UTC)


async def cmd_snapshot(
    args: argparse.Namespace, settings: Settings, project: str
) -> dict[str, Any]:
    """Take a snapshot, or list the ones already on disk under ``--list``.

    *project* is accepted only because ``main()`` computes it for every verb before
    dispatching -- a snapshot covers the whole database, so it is never read here.
    """
    into = Path(settings.snapshot_dir)
    if args.list:
        return {"snapshots": [snapshot_to_dict(r) for r in snapshot.list_snapshots(into)]}

    db_path = sqlite_path(settings.temporal_db_url)
    result = snapshot.take(
        db_path,
        into=into,
        reason=args.reason,
        clock=_utcnow,
        busy_timeout_ms=settings.db_busy_timeout_ms,
    )
    return snapshot_to_dict(result)


def restore_preview(args: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    """What ``--yes`` would replace, without touching anything.

    ``__main__.py`` prints this and stops (exit code 2) when ``--yes`` is missing --
    restoring is the one command whose own mistake a rerun cannot undo, so it is never
    performed on the strength of a bare invocation.
    """
    return {"database": sqlite_path(settings.temporal_db_url), "snapshot": str(args.file)}


async def cmd_restore(args: argparse.Namespace, settings: Settings, project: str) -> dict[str, Any]:
    """Replace the whole database with the snapshot at ``args.file``, behind a safety
    snapshot of the database as it was, taken first.

    *project* is accepted only because ``main()`` computes it for every verb before
    dispatching -- like ``morgan snapshot``, a restore replaces the whole database, so it is
    never read here. Only reached with ``--yes``: ``__main__.py`` intercepts the bare form
    before this handler is ever called (see ``restore_preview``).
    """
    into = Path(settings.snapshot_dir)
    db_path = sqlite_path(settings.temporal_db_url)
    result = snapshot.restore(
        db_path,
        source=Path(args.file),
        into=into,
        clock=_utcnow,
        busy_timeout_ms=settings.db_busy_timeout_ms,
    )
    return restore_to_dict(result)
