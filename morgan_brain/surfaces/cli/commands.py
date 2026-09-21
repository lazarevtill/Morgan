"""The command handlers, one per verb.

Each returns its payload and raises nothing for the caller to interpret: the CLI
renders them and the MCP server returns them, which is why both surfaces import from
here rather than reimplementing a single memory operation between them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from morgan_brain.app.chatgpt_import import (
    ARCHIVE_PROJECT,
    HOLDOUT_PROJECT,
    import_chatgpt,
)
from morgan_brain.composition import (
    build_app_context,
    build_memory_context,
    sqlite_path,
    utcnow,
)
from morgan_brain.config import Settings
from morgan_brain.memory import snapshot
from morgan_brain.memory.store import projects as projects_store
from morgan_brain.models import PERSONAL_PROJECT, Memory, MemoryQuery, MemorySource, OriginKind
from morgan_brain.surfaces.cli.doctor import build_doctor_report
from morgan_brain.surfaces.cli.payloads import (
    fact_to_dict,
    forget_result,
    merge_forget_reports,
    recall_result,
)


async def cmd_remember(
    args: argparse.Namespace,
    settings: Settings,
    project: str | None,
    *,
    client: str = "cli",
    session_id: str = "",
) -> dict[str, Any]:
    """*client* and *session_id* default to the CLI's own values; the MCP server passes its
    caller's ``clientInfo.name`` and its own per-process session id instead.

    *project* is ``None`` exactly when the caller named none: an MCP call with no ``project``
    argument, or the CLI run outside a git repository (``surfaces.cli.__main__`` passes the
    detected repository name through as a string, never ``None``, when one was found). This
    is the one place that resolves it to ``PERSONAL_PROJECT`` and the one place the result
    says so, in ``project_defaulted`` -- there is no other way a caller learns a write landed
    in the personal project by default rather than by name.
    """
    project_defaulted = project is None
    resolved_project = project if project is not None else PERSONAL_PROJECT
    ctx = build_memory_context(settings)
    try:
        memory = Memory(
            user_id=settings.owner_user_id,
            project=resolved_project,
            content=args.text,
            source=MemorySource.USER_STATED,
            origin_kind=OriginKind.REMEMBER,
            client=client,
            session_id=session_id,
            cwd=str(Path.cwd()),
            author_id=settings.owner_user_id,
        )
        memory_id = await ctx.gate.store(memory)
    finally:
        ctx.conn.close()
    return {
        "stored": True,
        "id": memory_id,
        "project": resolved_project,
        "project_defaulted": project_defaulted,
        "content": args.text,
    }


async def cmd_recall(args: argparse.Namespace, settings: Settings, project: str) -> dict[str, Any]:
    ctx = build_memory_context(settings)
    try:
        outcome = await ctx.gate.recall(
            MemoryQuery(
                user_id=settings.owner_user_id,
                project=project,
                all_projects=args.all_projects,
                text=args.query,
                top_k=args.top_k,
            )
        )
    finally:
        ctx.conn.close()
    return recall_result(outcome, project=project, all_projects=args.all_projects)


async def cmd_facts(args: argparse.Namespace, settings: Settings, project: str) -> dict[str, Any]:
    ctx = build_memory_context(settings)
    try:
        facts = await ctx.gate.current_facts(
            user_id=settings.owner_user_id,
            subject=args.subject,
            project=project,
            all_projects=args.all_projects,
        )
    finally:
        ctx.conn.close()
    return {
        "project": project,
        "all_projects": args.all_projects,
        "facts": [fact_to_dict(f) for f in facts],
    }


async def cmd_forget(args: argparse.Namespace, settings: Settings, project: str) -> dict[str, Any]:
    """Erase everything stored under a project (or every project), behind a snapshot taken
    first -- forget has no undo, so the snapshot is the undo.

    ``require_writable()`` is called explicitly, before the snapshot: on a database waiting
    for ``morgan migrate``, ``gate.forget()`` would refuse anyway, but only after the
    snapshot had already been written. Checking first means a forget that cannot run leaves
    no snapshot behind.
    """
    ctx = build_memory_context(settings)
    try:
        ctx.gate.require_writable()
        taken = snapshot.take(
            sqlite_path(settings.temporal_db_url),
            into=Path(settings.snapshot_dir),
            reason="forget",
            clock=utcnow,
            busy_timeout_ms=settings.db_busy_timeout_ms,
        )
        if args.all_projects:
            projects = await ctx.gate.distinct_projects(settings.owner_user_id)
            if not projects:
                projects = [project]
            reports = [
                await ctx.gate.forget(user_id=settings.owner_user_id, project=p) for p in projects
            ]
            report = merge_forget_reports(reports)
        else:
            projects = [project]
            report = await ctx.gate.forget(user_id=settings.owner_user_id, project=project)
    finally:
        ctx.conn.close()
    result = forget_result(report, project=project, all_projects=args.all_projects)
    result["projects"] = projects
    result["snapshot"] = str(taken.path)
    return result


async def cmd_ask(
    args: argparse.Namespace,
    settings: Settings,
    project: str,
    *,
    client: str = "cli",
    session_id: str = "",
) -> dict[str, Any]:
    """*client* and *session_id* default to the CLI's own values; the MCP server passes its
    caller's ``clientInfo.name`` and its own per-process session id instead -- the same
    convention ``cmd_remember`` uses."""
    ctx = build_app_context(settings)
    try:
        reply = await ctx.chat.ask(
            user_id=settings.owner_user_id,
            project=project,
            text=args.text,
            caller_client=client,
            caller_session_id=session_id,
        )
    finally:
        ctx.conn.close()
    return {"project": project, "response": reply, "model_used": settings.llm_model}


async def cmd_consolidate(
    args: argparse.Namespace, settings: Settings, project: str
) -> dict[str, Any]:
    """Turn recent episodic memories into durable valid-time facts, per project.

    On demand rather than on a schedule: the owner runs it when a session is over, or from
    cron if they want it nightly. Nothing in the core runs a model unasked.
    """
    ctx = build_app_context(settings)
    try:
        # Before the model call a proposal costs: on a database waiting for `morgan migrate`
        # every fact it proposed would be refused.
        ctx.gate.require_writable()
        if args.all_projects:
            projects = await ctx.gate.distinct_projects(settings.owner_user_id) or [project]
            # A project with no `projects` row -- the seed never reached it, or a database
            # from before this switch existed -- is consolidated as it always was; only a row
            # that says `consolidate_enabled = 0` skips.
            projects = [
                p
                for p in projects
                if (row := projects_store.get(ctx.conn, p)) is None or row.consolidate_enabled
            ]
        else:
            projects = [project]
        applied: dict[str, list[dict[str, Any]]] = {}
        for p in projects:
            ops = await ctx.consolidator.consolidate(settings.owner_user_id, project=p)
            applied[p] = [
                {
                    "op": op.op.value,
                    "subject": op.subject,
                    "predicate": op.predicate,
                    "object": op.object,
                }
                for op in ops
            ]
    finally:
        ctx.conn.close()
    return {"project": project, "all_projects": args.all_projects, "applied": applied}


async def cmd_doctor(args: argparse.Namespace, settings: Settings, project: str) -> dict[str, Any]:
    return await build_doctor_report(settings, project=project, all_projects=args.all_projects)


async def cmd_import(args: argparse.Namespace, settings: Settings, project: str) -> dict[str, Any]:
    """Seed memory from a ChatGPT export.

    *project* is ignored on purpose: the import decides where each conversation lands from
    the holdout rule, so honouring a caller's project would put held-out conversations
    somewhere consolidation can reach.
    """

    def progress(done: int, total: int) -> None:
        # stderr, because stdout is the --json contract. A seed of a few thousand turns is
        # minutes of embedding calls, and a silent run is indistinguishable from a hung one.
        print(f"\rimporting conversation {done}/{total}", end="", file=sys.stderr, flush=True)

    # The import budget: thousands of embedding calls against a host that may be cold or
    # flapping, and one dropped connection must not end the run.
    ctx = build_memory_context(settings, budget="import")
    try:
        # Before the export is read: on a database waiting for `morgan migrate` every memory
        # in it would be refused.
        ctx.gate.require_writable()
        report = await import_chatgpt(
            Path(args.path), gate=ctx.gate, user_id=settings.owner_user_id, progress=progress
        )
    finally:
        print(file=sys.stderr)
        ctx.conn.close()
    return {
        "archive_project": ARCHIVE_PROJECT,
        "holdout_project": HOLDOUT_PROJECT,
        "conversations": report.conversations,
        "held_out": report.held_out,
        "memories": report.memories,
        "skipped_turns": report.skipped_turns,
    }
