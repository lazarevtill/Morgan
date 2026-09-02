"""``morgan`` -- the terminal client.

remember/recall/facts/forget/doctor are direct memory operations: they go through
``composition.build_memory_context`` (a MemoryGate over the real database) and need no chat
model (``MORGAN_EMBEDDING_BACKEND=hash`` removes the embedding call too). ``ask`` and
``consolidate`` go through ``build_app_context`` and need a reachable model server.

Every command accepts ``--project`` (default: the current git repository's directory name),
``--all-projects`` where it makes sense, and ``--json`` for scripting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from morgan_brain.cli.project import detect_project
from morgan_brain.composition import (
    build_app_context,
    build_memory_context,
    build_memory_module,
    sqlite_path,
)
from morgan_brain.config import Settings, get_settings, user_config_file
from morgan_brain.logging_setup import configure_logging
from morgan_brain.memory.db import open_db
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import ForgetReport
from morgan_brain.models import Memory, MemoryQuery, MemorySource, TemporalFact
from morgan_brain.providers.factory import check_llm_reachable

# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _memory_to_dict(m: Memory) -> dict[str, Any]:
    return {
        "id": m.id,
        "project": m.project,
        "kind": m.kind.value,
        "content": m.content,
        "source": m.source.value,
        "importance": m.importance,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


def _fact_to_dict(f: TemporalFact) -> dict[str, Any]:
    return {
        "id": f.id,
        "project": f.project,
        "subject": f.subject,
        "predicate": f.predicate,
        "object": f.object,
        "confidence": f.confidence,
        "source": f.source.value,
        "valid_from": f.valid_from.isoformat() if f.valid_from else None,
        "last_confirmed": f.last_confirmed.isoformat() if f.last_confirmed else None,
    }


def _merge_forget_reports(reports: list[ForgetReport]) -> ForgetReport:
    """Sum ``--all-projects`` reports into one. ``tables_skipped`` is deduplicated: the same
    optional table is either present or absent for the whole database, not per project."""
    merged = ForgetReport()
    skipped: set[str] = set()
    for r in reports:
        merged.memories += r.memories
        merged.facts += r.facts
        merged.history += r.history
        merged.index_entries += r.index_entries
        skipped.update(r.tables_skipped)
    merged.tables_skipped = sorted(skipped)
    return merged


def _forget_result(report: ForgetReport, *, project: str, all_projects: bool) -> dict[str, Any]:
    """The one output that must not lie: a skipped table prints as "not present", never as
    a silent 0."""
    warnings: list[str] = []
    if report.tables_skipped:
        warnings.append(
            "not present in this database, so nothing was erased from (not an error): "
            + ", ".join(report.tables_skipped)
        )
    return {
        "project": project,
        "all_projects": all_projects,
        "memories": report.memories,
        "facts": report.facts,
        "history": report.history,
        "index_entries": report.index_entries,
        "tables_skipped": list(report.tables_skipped),
        "warnings": warnings,
    }


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _collect_local_probes(
    settings: Settings, *, project: str, all_projects: bool
) -> dict[str, Any]:
    """Every *local* probe: filesystem, SQLite, sqlite-vec, FTS5, row counts.

    Synchronous on purpose -- each probe blocks -- so ``build_doctor_report`` hands the
    whole body to a worker thread.
    """
    db_path = sqlite_path(settings.temporal_db_url)
    config_file = user_config_file()
    report: dict[str, Any] = {
        "database": db_path if db_path == ":memory:" else str(Path(db_path).resolve()),
        # The first question after "why is my brain empty?" is "which config did it read?"
        "config_file": str(config_file),
        "config_file_present": config_file.is_file(),
        "project": project,
        "all_projects": all_projects,
        "embedding_backend": settings.embedding_backend,
        "embedding_dim": settings.embedding_dim,
        "llm_endpoint": settings.llm_endpoint,
        "llm_model": settings.llm_model,
        "sqlite_vec": None,
        "fts5": False,
        "provider": "unreachable",
        "vector_rows": None,
        "memory_rows": None,
        "fts_rows": None,
    }

    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        conn = open_db(db_path)
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the diagnostic tool
        report["error"] = f"failed to open database: {exc}"
        return report

    try:
        row = conn.execute("SELECT vec_version()").fetchone()
        report["sqlite_vec"] = row[0] if row else None
    except Exception as exc:  # noqa: BLE001
        report["sqlite_vec_error"] = str(exc)

    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __morgan_fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS __morgan_fts5_probe")
        report["fts5"] = True
    except sqlite3.OperationalError:
        report["fts5"] = False

    # Build the real schema (idempotent CREATE ... IF NOT EXISTS) so row counts are always
    # meaningful -- 0 on a fresh install rather than "table doesn't exist yet".
    try:
        build_memory_module(
            conn, embedder=FakeEmbedder(dim=settings.embedding_dim), dim=settings.embedding_dim
        )
    except Exception as exc:  # noqa: BLE001
        report["schema_error"] = str(exc)

    # A table name cannot be a bound parameter, so each count is a literal statement chosen
    # by key. The project filter is a bound flag for the same reason.
    count_sql = {
        "memories": "SELECT COUNT(*) FROM memories WHERE user_id = ? AND (? OR project = ?)",
        "fts_memories": (
            "SELECT COUNT(*) FROM fts_memories WHERE user_id = ? AND (? OR project = ?)"
        ),
        "vec_meta": "SELECT COUNT(*) FROM vec_meta WHERE user_id = ? AND (? OR project = ?)",
    }
    params = (settings.owner_user_id, all_projects, project)

    def _count(table: str) -> int | None:
        """The row count, or None when the table is absent or unreadable -- independently
        caught, like every other probe: `doctor` is the command you run *because*
        something is broken."""
        if not _table_exists(conn, table):
            return None
        try:
            row = conn.execute(count_sql[table], params).fetchone()
        except sqlite3.Error as exc:
            report.setdefault("count_errors", {})[table] = str(exc)
            return None
        return int(row[0])

    report["memory_rows"] = _count("memories")
    report["fts_rows"] = _count("fts_memories")
    report["vector_rows"] = _count("vec_meta")
    conn.close()
    return report


async def build_doctor_report(
    settings: Settings, *, project: str, all_projects: bool
) -> dict[str, Any]:
    report = await asyncio.to_thread(
        _collect_local_probes, settings, project=project, all_projects=all_projects
    )
    report["provider"] = "reachable" if await check_llm_reachable(settings) else "unreachable"
    return report


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


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
        "results": [_memory_to_dict(m) for m in results],
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
        "facts": [_fact_to_dict(f) for f in facts],
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
            report = _merge_forget_reports(reports)
        else:
            projects = [project]
            report = await ctx.gate.forget(user_id=settings.owner_user_id, project=project)
    finally:
        ctx.conn.close()
    result = _forget_result(report, project=project, all_projects=args.all_projects)
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


# ---------------------------------------------------------------------------
# Human-readable rendering
# ---------------------------------------------------------------------------


def _render_remember(data: dict[str, Any]) -> str:
    return f"Stored memory {data['id']} in project {data['project']!r}."


def _render_recall(data: dict[str, Any]) -> str:
    if not data["results"]:
        return (
            f"No memories found (project={data['project']!r}, all_projects={data['all_projects']})."
        )
    return "\n".join(
        f"{i + 1}. [{r['kind']}/{r['project']}] {r['content']}"
        for i, r in enumerate(data["results"])
    )


def _render_facts(data: dict[str, Any]) -> str:
    if not data["facts"]:
        return (
            f"No currently-valid facts "
            f"(project={data['project']!r}, all_projects={data['all_projects']})."
        )
    return "\n".join(
        f"{f['subject']} {f['predicate']} {f['object']} (confidence={f['confidence']:.2f}, "
        f"project={f['project']})"
        for f in data["facts"]
    )


def _render_forget(data: dict[str, Any]) -> str:
    scope = "all projects" if data["all_projects"] else f"project {data['project']!r}"
    lines = [
        f"Forgot {scope}: memories={data['memories']} facts={data['facts']} "
        f"history={data['history']} index={data['index_entries']}"
    ]
    lines.extend(f"WARNING: {w}" for w in data["warnings"])
    return "\n".join(lines)


def _render_consolidate(data: dict[str, Any]) -> str:
    lines = []
    for project, ops in data["applied"].items():
        lines.append(f"{project}: {len(ops)} fact operation(s)")
        lines.extend(
            f"  {op['op']:<6} {op['subject']} {op['predicate']} {op['object']}" for op in ops
        )
    return "\n".join(lines) or "Nothing to consolidate."


def _render_ask(data: dict[str, Any]) -> str:
    return str(data["response"])


def _render_doctor(data: dict[str, Any]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in data.items())


_RENDERERS = {
    "remember": (cmd_remember, _render_remember),
    "recall": (cmd_recall, _render_recall),
    "facts": (cmd_facts, _render_facts),
    "forget": (cmd_forget, _render_forget),
    "ask": (cmd_ask, _render_ask),
    "consolidate": (cmd_consolidate, _render_consolidate),
    "doctor": (cmd_doctor, _render_doctor),
}

# Commands where --all-projects is meaningless: a write or a single chat turn always
# targets exactly one project.
_SINGLE_PROJECT_ONLY = {"remember", "ask"}


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--project",
        default=None,
        help="Project to scope this command to (default: the current git repository's "
        "directory name; DEFAULT_PROJECT outside a repo).",
    )
    sp.add_argument(
        "--all-projects",
        action="store_true",
        help="Cross every project the owner has stored data under, instead of just --project.",
    )
    sp.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of human text."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="morgan", description="Talk to your local Morgan brain.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_remember = sub.add_parser("remember", help="Store a memory.")
    p_remember.add_argument("text", help="What to remember.")
    _add_common(p_remember)

    p_recall = sub.add_parser("recall", help="Search memory (vector + keyword + entity).")
    p_recall.add_argument("query", help="Search text.")
    p_recall.add_argument("--top-k", type=int, default=8, help="Maximum results to return.")
    _add_common(p_recall)

    p_facts = sub.add_parser("facts", help="List currently-valid facts.")
    p_facts.add_argument("--subject", default=None, help="Filter to facts about this subject.")
    _add_common(p_facts)

    p_forget = sub.add_parser("forget", help="Erase everything stored under a project.")
    _add_common(p_forget)

    p_ask = sub.add_parser("ask", help="Ask the assistant (a chat turn; needs a reachable model).")
    p_ask.add_argument("text", help="Your message.")
    _add_common(p_ask)

    p_cons = sub.add_parser(
        "consolidate",
        help="Turn recent episodic memories into durable facts (needs a reachable model).",
    )
    _add_common(p_cons)

    p_doctor = sub.add_parser("doctor", help="Diagnose the local Morgan installation.")
    _add_common(p_doctor)

    return parser


async def _dispatch(args: argparse.Namespace, settings: Settings, project: str) -> int:
    if args.command in _SINGLE_PROJECT_ONLY and args.all_projects:
        message = (
            f"morgan {args.command}: --all-projects is not valid here "
            "(a write / chat turn always targets exactly one project)."
        )
        # Anything that can fail must still emit JSON under --json, or a script parsing
        # stdout can't tell a rejected flag from a crash.
        if args.json:
            print(json.dumps({"error": message}, ensure_ascii=False))
        else:
            print(message, file=sys.stderr)
        return 2

    handler, renderer = _RENDERERS[args.command]
    try:
        data = await handler(args, settings, project)
    except Exception as exc:  # noqa: BLE001 -- a CLI user gets a clean message, not a traceback
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        # ensure_ascii=False: a substantially non-Latin corpus stays readable in JSON output.
        print(json.dumps(data, indent=2, sort_keys=False, default=str, ensure_ascii=False))
    else:
        print(renderer(data))
    return 0


def main(argv: list[str] | None = None) -> int:
    # stdout is the --json contract; every log line belongs on stderr.
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    project = args.project or detect_project(Path.cwd())
    return asyncio.run(_dispatch(args, settings, project))


if __name__ == "__main__":
    sys.exit(main())
