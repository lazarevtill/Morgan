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

import structlog

from morgan_brain.app.chat import TurnRequest
from morgan_brain.app.chatgpt_import import (
    ARCHIVE_PROJECT,
    HOLDOUT_PROJECT,
    import_chatgpt,
)
from morgan_brain.app.strict_context import StrictContextError
from morgan_brain.composition import (
    MemoryContext,
    build_app_context,
    build_evidence_context,
    build_memory_context,
    sqlite_path,
    utcnow,
)
from morgan_brain.config import Settings
from morgan_brain.memory import snapshot
from morgan_brain.models import PERSONAL_PROJECT, Memory, MemoryQuery, MemorySource, OriginKind
from morgan_brain.surfaces.cli.doctor import build_doctor_report
from morgan_brain.surfaces.cli.payloads import (
    fact_to_dict,
    forget_result,
    memory_to_dict,
    merge_forget_reports,
    recall_result,
)
from morgan_brain.surfaces.cli.project import Repository, classify
from morgan_brain.surfaces.cli.validation import (
    effective_time,
    event_identity,
    evidence_ids,
    revision_parents,
)

#: Warnings the owner should see beside a command that otherwise succeeded. stderr, through
#: the one logging configuration, because stdout carries ``--json``.
log = structlog.get_logger("cli")


async def _record_the_repository(
    ctx: MemoryContext, settings: Settings, project: str, repository: Repository | None
) -> None:
    """Record what repository *project* is, when this invocation was run from inside one.

    *repository* comes from ``surfaces.cli.__main__`` and only for a write command whose
    project was named by the enclosing git repository: never for ``--project``, never for the
    personal default outside a repository, and never from ``morgan-mcp``, which cannot see its
    client's checkout (the server may run on another machine). The label is recomputed on
    every such write, because both the repository's remote and ``MORGAN_WORK_REMOTE_GLOBS``
    change without Morgan hearing about it, and nothing sets it by hand.

    It runs after the command's own write, so a command that failed -- an unreachable model,
    a refused embedding -- records nothing, and on a database waiting for ``morgan migrate``
    the write is refused before this is reached.

    Two things it will not do. A repository whose config could not be read is left alone
    rather than relabelled: `classify(None, ...)` is `unclassified`, and writing that would
    erase a correct label on every write until the file happens to parse again. And a failure
    here never fails the command: the command's contract is its write, which has already
    committed -- a lost race for the write lock must not report a stored memory, or a model
    answer already paid for, as an error. It says so on stderr instead, naming the project and
    never the remote or the root, which are the owner's data.
    """
    if repository is None or not repository.remote_readable:
        return
    try:
        await ctx.gate.record_project(
            user_id=settings.owner_user_id,
            project=project,
            classification=classify(repository.remote, settings.work_remote_globs),
            remote=repository.remote,
            root=str(repository.root),
        )
    except Exception as exc:  # noqa: BLE001 -- bookkeeping never fails the write it follows
        log.warning("project.not-recorded", project=project, error=str(exc))


async def cmd_remember(
    args: argparse.Namespace,
    settings: Settings,
    project: str | None,
    *,
    client: str = "cli",
    session_id: str = "",
    repository: Repository | None = None,
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
    source = MemorySource(getattr(args, "source", "unknown"))
    author_id = getattr(args, "author_id", "")
    identity = event_identity(getattr(args, "event_id", None))
    effective_at = effective_time(getattr(args, "effective_at", None))
    parents = revision_parents(getattr(args, "revises_event_ids", None))
    if parents and (
        source is MemorySource.UNKNOWN
        or not isinstance(author_id, str)
        or not author_id.strip()
        or effective_at is None
    ):
        raise ValueError("Corrections require known source, reported author and effective_at")
    event_fields: dict[str, Any] = {"revises_event_ids": parents}
    if identity is not None:
        event_fields["id"] = identity
    if effective_at is not None:
        event_fields["created_at"] = effective_at
    ctx = build_memory_context(settings)
    try:
        memory = Memory(
            user_id=settings.owner_user_id,
            project=resolved_project,
            content=args.text,
            source=source,
            origin_kind=OriginKind.REMEMBER,
            client=client,
            session_id=session_id,
            cwd=str(Path.cwd()),
            author_id=author_id,
            **event_fields,
        )
        if identity is not None:
            existing = await ctx.gate.evidence(
                user_id=settings.owner_user_id,
                project=resolved_project,
                evidence_ids=[identity],
            )
            if existing.records:
                captured = existing.records[0]
                if captured.origin_kind is OriginKind.REMEMBER and captured.client == client:
                    # Retry capture context is historical metadata, not caller identity.
                    # The core still checks every assertion field before accepting replay.
                    memory.session_id = captured.session_id
                    memory.cwd = captured.cwd
        memory_id = await ctx.gate.store(memory)
        stored_memory = await ctx.gate.get(memory_id, user_id=settings.owner_user_id)
        await _record_the_repository(ctx, settings, resolved_project, repository)
    finally:
        ctx.conn.close()
    return {
        "stored": True,
        "id": memory_id,
        "project": resolved_project,
        "project_defaulted": project_defaulted,
        "content": args.text,
        "source": memory.source.value,
        "author_id": memory.author_id,
        "effective_at": (
            stored_memory.created_at.isoformat()
            if stored_memory is not None and stored_memory.created_at is not None
            else None
        ),
        "recorded_at": (
            stored_memory.recorded_at.isoformat()
            if stored_memory is not None and stored_memory.recorded_at is not None
            else None
        ),
        "revision_root_id": stored_memory.revision_root_id if stored_memory else None,
        "revises_event_ids": parents,
    }


async def cmd_recall(args: argparse.Namespace, settings: Settings, project: str) -> dict[str, Any]:
    effective_at = effective_time(getattr(args, "effective_at", None))
    ctx = build_memory_context(settings)
    try:
        outcome = await ctx.gate.recall(
            MemoryQuery(
                user_id=settings.owner_user_id,
                project=project,
                all_projects=args.all_projects,
                text=args.query,
                top_k=args.top_k,
                effective_at=effective_at,
            )
        )
    finally:
        ctx.conn.close()
    result = recall_result(outcome, project=project, all_projects=args.all_projects)
    result["effective_at"] = effective_at.isoformat() if effective_at else None
    return result


async def cmd_evidence(
    args: argparse.Namespace, settings: Settings, project: str
) -> dict[str, Any]:
    """Fetch bounded durable evidence in one named scope, without any model call."""
    requested = evidence_ids(args.ids)
    effective_at = effective_time(getattr(args, "effective_at", None))
    if getattr(args, "all_projects", False):
        raise ValueError("evidence requires one project; --all-projects is not supported")
    ctx = build_evidence_context(settings)
    try:
        resolved = await ctx.gate.evidence(
            user_id=settings.owner_user_id,
            project=project,
            evidence_ids=requested,
            effective_at=effective_at,
        )
        return {
            "version": resolved.schema_version,
            "project": project,
            "effective_at": effective_at.isoformat() if effective_at else None,
            "requested_ids": resolved.requested_ids,
            "missing_ids": resolved.missing_ids,
            "results": [memory_to_dict(memory) for memory in resolved.records],
        }
    finally:
        ctx.conn.close()


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
    repository: Repository | None = None,
) -> dict[str, Any]:
    """*client* and *session_id* default to the CLI's own values; the MCP server passes its
    caller's ``clientInfo.name`` and its own per-process session id instead -- the same
    convention ``cmd_remember`` uses, and so is *repository* (``_record_the_repository``)."""
    source = MemorySource(getattr(args, "source", "unknown"))
    author_id = getattr(args, "author_id", "")
    strict_context = getattr(args, "strict_context", False)
    strict_context_is_bool = isinstance(strict_context, bool)
    if not strict_context_is_bool:
        raise ValueError("strict_context must be a boolean")
    if strict_context and settings.strict_context_backend == "disabled":
        raise StrictContextError("token_counter_unavailable")
    ctx = build_app_context(settings)
    try:
        detailed = None
        if strict_context:
            detailed = await ctx.chat.ask_evidence(
                TurnRequest(
                    user_id=settings.owner_user_id,
                    project=project,
                    text=args.text,
                    caller_client=client,
                    caller_session_id=session_id,
                    source=source,
                    author_id=author_id,
                )
            )
            reply = detailed.answer
        else:
            default_result = await ctx.chat.ask_with_provenance(
                TurnRequest(
                    user_id=settings.owner_user_id,
                    project=project,
                    text=args.text,
                    caller_client=client,
                    caller_session_id=session_id,
                    source=source,
                    author_id=author_id,
                )
            )
            reply = default_result.answer
        await _record_the_repository(ctx, settings, project, repository)
    finally:
        try:
            close = getattr(ctx.chat, "aclose", None)
            if callable(close):
                await close()
        finally:
            ctx.conn.close()
    payload = {
        "project": project,
        "response": reply,
        "model_used": settings.llm_model if strict_context else default_result.model_used,
        "source": source.value,
        "author_id": author_id,
        "response_source": MemorySource.AGENT_INFERRED.value,
        "response_author_id": (
            f"model:{settings.llm_model}" if strict_context else default_result.response_author_id
        ),
    }

    if detailed is not None:
        payload.update(detailed.model_dump(mode="json"))
    return payload


async def cmd_consolidate(
    args: argparse.Namespace,
    settings: Settings,
    project: str,
    *,
    repository: Repository | None = None,
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
            # The owner's per-project switch: only a project whose row turns consolidation
            # off is skipped.
            projects = [
                p
                for p in projects
                if await ctx.gate.consolidate_enabled(user_id=settings.owner_user_id, project=p)
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
        await _record_the_repository(ctx, settings, project, repository)
    finally:
        try:
            await ctx.chat.aclose()
        finally:
            ctx.conn.close()
    return {"project": project, "all_projects": args.all_projects, "applied": applied}


async def cmd_doctor(args: argparse.Namespace, settings: Settings, project: str) -> dict[str, Any]:
    # args.clients is None unless the caller passed --clients (argparse default); _dispatch
    # already rejected that combined with a missing --vectors, so here it just means "1".
    return await build_doctor_report(
        settings,
        project=project,
        all_projects=args.all_projects,
        vectors=args.vectors,
        clients=args.clients if args.clients is not None else 1,
    )


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
            Path(args.path),
            gate=ctx.gate,
            user_id=settings.owner_user_id,
            progress=progress,
            canary_every=settings.import_canary_every,
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
