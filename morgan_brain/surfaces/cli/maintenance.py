"""``morgan snapshot`` -- a verified VACUUM INTO copy of the whole database.

Not project-scoped: a snapshot is of the whole database file, not one project's rows, so this
handler -- unlike every other verb -- ignores ``--project``/``--all-projects``, and its
subparser in ``__main__.py`` offers neither flag.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from morgan_brain.composition import sqlite_path
from morgan_brain.config import Settings
from morgan_brain.memory import snapshot
from morgan_brain.surfaces.cli.payloads import snapshot_to_dict


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
