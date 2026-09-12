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
)
from morgan_brain.config import Settings
from morgan_brain.models import Memory, MemoryQuery, MemorySource
from morgan_brain.surfaces.cli.doctor import build_doctor_report
from morgan_brain.surfaces.cli.payloads import (
    fact_to_dict,
    forget_result,
    memory_to_dict,
    merge_forget_reports,
)


async def cmd_remember(
    args: argparse.Namespace, settings: Settings, project: str
) -> dict[str, Any]:
    ctx = build_memory_context(settings)
    try:
        memory = Memory(
            user_id=settings.owner_user_id,
            project=project,
            content=args.text,
            source=MemorySource.USER_STATED,
        )
        memory_id = await ctx.gate.store(memory)
    finally:
        ctx.conn.close()
    return {"stored": True, "id": memory_id, "project": project, "content": args.text}


async def cmd_recall(args: argparse.Namespace, settings: Settings, project: str) -> dict[str, Any]:
    ctx = build_memory_context(settings)
    try:
        results = await ctx.gate.recall(
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
    return {
        "project": project,
        "all_projects": args.all_projects,
        "results": [memory_to_dict(m) for m in results],
    }


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
    ctx = build_memory_context(settings)
    try:
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
    return result


async def cmd_ask(args: argparse.Namespace, settings: Settings, project: str) -> dict[str, Any]:
    ctx = build_app_context(settings)
    try:
        reply = await ctx.chat.ask(user_id=settings.owner_user_id, project=project, text=args.text)
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
        if args.all_projects:
            projects = await ctx.gate.distinct_projects(settings.owner_user_id) or [project]
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

    ctx = build_memory_context(settings)
    try:
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
