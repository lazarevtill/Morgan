"""``morgan snapshot``, ``morgan restore`` and ``morgan migrate`` -- a verified VACUUM INTO
copy of the whole database, putting one back, and running the migration steps an open may not.
A migration that ran ends by registering the settings' embedding space where none is active
and checking it: the fingerprint is recorded then when the embedding server answers.

None is project-scoped: each acts on the whole database file, not one project's rows, so
these handlers -- unlike every other verb -- ignore ``--project``/``--all-projects``, and
their subparsers in ``__main__.py`` offer neither flag.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from morgan_brain.composition import (
    migration_stores,
    register_the_settings_space,
    sqlite_path,
)
from morgan_brain.config import Settings
from morgan_brain.memory import migrations, snapshot
from morgan_brain.memory.checked_embedder import CheckedEmbedder
from morgan_brain.memory.store import spaces
from morgan_brain.memory.store.db import open_db
from morgan_brain.providers.factory import build_embedder
from morgan_brain.providers.wire import ProviderRefused, ProviderUnreachable
from morgan_brain.surfaces.cli.payloads import (
    embedding_space_to_dict,
    migration_plan_to_dict,
    migration_to_dict,
    restore_to_dict,
    snapshot_to_dict,
)


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


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


async def cmd_migrate(args: argparse.Namespace, settings: Settings, project: str) -> dict[str, Any]:
    """List the pending migration steps, or run every one of them behind a snapshot.

    *project* is accepted only because ``main()`` computes it for every verb before
    dispatching -- a migration covers the whole database, so it is never read here.

    Once a wave has run, the embedding space is registered and checked (``_check_the_space``),
    after the wave's commit: the migration stands whatever the embedding server does.
    """
    result = _migrate(settings, dry_run=args.dry_run)
    # Only the result of a wave that ran carries "steps"; a plan (dry run, nothing pending)
    # changed nothing, and has nothing to check.
    if "steps" in result:
        try:
            result["embedding_space"] = await _check_the_space(settings)
        except Exception as exc:
            raise RuntimeError(
                f"migrated to user_version {result['user_version']} (the snapshot taken first "
                f"is {result['snapshot']}), but {exc}"
            ) from exc
    return result


async def _check_the_space(settings: Settings) -> dict[str, Any] | None:
    """Register the settings' embedding space where none is active, then check it.

    Registered by the function every open runs (``register_the_settings_space``), which also
    refuses a vector table of another width than ``MORGAN_EMBEDDING_DIM``. Checked by the same
    first-call check every process makes (``CheckedEmbedder.verify``): the fingerprint is
    recorded when the stored sample matches, or compared when it was recorded before. An
    embedding server that gives no answer -- none at all, or none within the retry budget --
    or refuses the request leaves the space unverified for the first call to verify, and
    that is said with the reason. ``None`` for the hash backend, which no model answers and
    nothing registers.
    """
    if settings.embedding_backend != "provider":
        return None
    conn = open_db(
        sqlite_path(settings.temporal_db_url), busy_timeout_ms=settings.db_busy_timeout_ms
    )
    try:
        register_the_settings_space(conn, settings)
        before = spaces.active(conn)
        embedder = build_embedder(settings, conn=conn)
        if before is None or not isinstance(embedder, CheckedEmbedder):
            return None
        try:
            await embedder.verify()
        except (ProviderUnreachable, ProviderRefused) as exc:
            return embedding_space_to_dict(before, fingerprint="unverified", reason=str(exc))
        after = spaces.active(conn) or before
        return embedding_space_to_dict(
            after, fingerprint="recorded" if before.fingerprint is None else "matches"
        )
    finally:
        conn.close()


def _migrate(settings: Settings, *, dry_run: bool) -> dict[str, Any]:
    """``cmd_migrate``'s work, which is blocking file and database I/O from start to end.

    In order: open the database and read what is pending, building no store, so ``--dry-run``
    writes nothing -- not even a table -- and stops there; nothing pending stops there too.
    Otherwise a ``migrate`` snapshot, taken before anything changes (``VACUUM INTO`` cannot run
    inside the transaction that follows); then the stores, and every pending step in one write
    transaction (``migrations.migrate``); then ``PRAGMA quick_check`` once it has committed. A
    step that raises rolls the whole wave back, and the error names the snapshot, which stays.
    """
    db_path = sqlite_path(settings.temporal_db_url)
    if not Path(db_path).is_file():
        raise FileNotFoundError(f"no database at {db_path}: nothing to migrate")
    code_version = len(migrations._STEPS)

    conn = open_db(db_path, busy_timeout_ms=settings.db_busy_timeout_ms)
    try:
        from_version = _user_version(conn)
        steps = migrations.pending(conn)
        if dry_run or not steps:
            return migration_plan_to_dict(
                database=db_path,
                dry_run=dry_run,
                user_version=from_version,
                code_version=code_version,
                pending=steps,
            )

        taken = snapshot.take(
            db_path,
            into=Path(settings.snapshot_dir),
            reason="migrate",
            clock=_utcnow,
            busy_timeout_ms=settings.db_busy_timeout_ms,
        )
        stores = migration_stores(conn)
        try:
            applied = migrations.migrate(conn, stores)
        except Exception as exc:
            raise RuntimeError(
                f"migration failed and was rolled back, so the database is as it was at "
                f"user_version {from_version}: {exc} (the snapshot taken first is {taken.path})"
            ) from exc
        check = str(conn.execute("PRAGMA quick_check").fetchone()[0])
        user_version = _user_version(conn)
    finally:
        conn.close()

    if check != "ok":
        raise RuntimeError(
            f"{db_path} failed PRAGMA quick_check after migrating: {check} -- "
            f"`morgan restore {taken.path} --yes` puts back the database as it was"
        )
    _, after = snapshot._describe(Path(db_path))
    return migration_to_dict(
        database=db_path,
        snapshot=taken,
        from_version=from_version,
        user_version=user_version,
        code_version=code_version,
        applied=applied,
        after=after,
        quick_check=check,
    )
