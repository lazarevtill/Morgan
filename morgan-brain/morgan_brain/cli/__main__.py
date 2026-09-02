"""``morgan`` -- the terminal client over the local memory + reasoning stack.

remember/recall/facts/forget/doctor are direct memory operations: they go through
``composition.build_memory_context`` (a MemoryGate over the real database, no LLM router
required) so they work with no model server running (``MORGAN_EMBEDDING_BACKEND=hash``).
``ask`` is a full chat turn: it goes through ``composition.build_app_context``, the same
production wiring brain-api uses, so it does need a reachable LLM endpoint.

Every command accepts ``--project`` (default: the current git repository's directory name,
via ``cli.project.detect_project``), ``--all-projects`` (the explicit cross-project escape
hatch, where it makes sense), and ``--json`` (machine-readable output for scripting).
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
    CHAMPION_PROMPT_NAME,
    build_app_context,
    build_memory_context,
    build_vector_index,
    sqlite_path,
)
from morgan_brain.config import Settings, get_settings, user_config_file
from morgan_brain.interfaces.memory import ForgetReport
from morgan_brain.learning.history import session_key
from morgan_brain.learning.receipts import ReceiptStore
from morgan_brain.logging_setup import configure_logging
from morgan_brain.models.memory import Memory, MemoryQuery, MemorySource, TemporalFact
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.stores.db import open_db
from morgan_brain.modules.memory.stores.episodic import EpisodicStore
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
    """Sum ``--all-projects`` forget() reports into one honest total.

    ``tables_skipped`` is deduplicated (the same optional table is either present or absent
    for the whole database, not per project) rather than repeated once per project.

    ``vectors_erased`` is an AND, not a majority: one project whose vector delete failed makes
    the whole sweep incomplete, and the merged report has to say so.
    """
    merged = ForgetReport()
    skipped: set[str] = set()
    vector_errors: list[str] = []
    for r in reports:
        merged.memories += r.memories
        merged.facts += r.facts
        merged.signals += r.signals
        merged.history += r.history
        merged.index_entries += r.index_entries
        merged.persona_nodes += r.persona_nodes
        merged.vectors_erased = merged.vectors_erased and r.vectors_erased
        if r.vector_error:
            vector_errors.append(r.vector_error)
        merged.champions_flagged.extend(r.champions_flagged)
        skipped.update(r.tables_skipped)
    merged.tables_skipped = sorted(skipped)
    merged.vector_error = "; ".join(vector_errors) or None
    return merged


def _forget_result(
    report: ForgetReport, settings: Settings, *, project: str, all_projects: bool
) -> dict[str, Any]:
    """Turn a ``ForgetReport`` into the CLI's output shape -- the one place that must not
    lie: a skipped table prints as "not present", never as a silent 0, and the vector result
    is whatever ``forget()`` actually achieved.

    ``vectors_erased`` is read off the report rather than inferred from ``vector_backend``.
    Inferring it was wrong in both directions once ``forget()`` began deleting from external
    stores: it claimed failure where the delete had in fact succeeded, and it could never have
    reported a delete that was attempted and refused.
    """
    vectors_erased = report.vectors_erased
    warnings: list[str] = []
    if report.tables_skipped:
        warnings.append(
            "not present in this database, so nothing was erased from (not an error): "
            + ", ".join(report.tables_skipped)
        )
    if not vectors_erased:
        warnings.append(
            f"vector_backend={settings.vector_backend!r}: vectors were NOT erased "
            f"({report.vector_error or 'delete did not run'}) -- this project's text may "
            "still be retrievable from the vector store and must be removed there."
        )
    return {
        "project": project,
        "all_projects": all_projects,
        "memories": report.memories,
        "facts": report.facts,
        "signals": report.signals,
        "history": report.history,
        "index_entries": report.index_entries,
        "persona_nodes": report.persona_nodes,
        "tables_skipped": list(report.tables_skipped),
        "champions_flagged": list(report.champions_flagged),
        "vector_backend": settings.vector_backend,
        "vectors_erased": vectors_erased,
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
    """Run every *local* doctor probe: filesystem, SQLite, sqlite-vec, FTS5, row counts.

    Synchronous on purpose. Each probe blocks -- ``Path.resolve()`` stats, every
    ``conn.execute()`` is a blocking sqlite call -- so this whole body is the thing that
    must not run on the event loop. ``build_doctor_report`` hands it to a worker thread.
    Splitting here keeps that boundary in one place instead of scattering per-call
    offloads through a function that is blocking from top to bottom.
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
        "vector_backend": settings.vector_backend,
        "llm_endpoint": settings.llm_endpoint,
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
        FtsIndex(conn)
        EntityIndex(conn)
        EpisodicStore(conn)
        if settings.vector_backend == "sqlite":
            build_vector_index(settings, conn)
    except Exception as exc:  # noqa: BLE001
        report["schema_error"] = str(exc)

    # A table name cannot be a bound parameter, so each count is a literal statement chosen by
    # key rather than a name interpolated into SQL. The project filter is a bound flag for the
    # same reason: no part of these queries is assembled from data.
    count_sql = {
        "memories": "SELECT COUNT(*) FROM memories WHERE user_id = ? AND (? OR project = ?)",
        "fts_memories": (
            "SELECT COUNT(*) FROM fts_memories WHERE user_id = ? AND (? OR project = ?)"
        ),
        "vec_meta": "SELECT COUNT(*) FROM vec_meta WHERE user_id = ? AND (? OR project = ?)",
    }
    params = (settings.owner_user_id, all_projects, project)

    def _count(table: str) -> int | None:
        """Return the row count, or None when the table is absent or unreadable.

        Independently caught, like every other probe above: a legacy database whose schema
        migration failed leaves a table present but unqueryable, and `doctor` is the command
        you run *because* something is broken. Aborting here would throw away the diagnostics
        already gathered -- the provider check, the sqlite-vec and FTS5 probes, the resolved
        paths -- which is precisely the report you need to see.
        """
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
    # vec_meta only exists for the sqlite vector backend -- under qdrant/memory, vectors
    # aren't tracked in this database at all, so counting here would be misleading.
    report["vector_rows"] = _count("vec_meta") if settings.vector_backend == "sqlite" else None

    conn.close()
    return report


async def build_doctor_report(
    settings: Settings, *, project: str, all_projects: bool
) -> dict[str, Any]:
    """Build the ``doctor`` report.

    Every probe is independently caught so one broken subsystem (e.g. FTS5 genuinely
    unavailable) reports its own failure instead of aborting every other check --
    "genuinely diagnostic, not decorative" is the whole point.

    The local probes are blocking, so they run in a worker thread; the one genuinely
    async probe -- can we reach the configured LLM endpoint -- runs here. ``provider``
    is pre-seeded by the local pass, so assigning it back keeps its place in the report.
    """
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


async def cmd_receipts(
    args: argparse.Namespace, settings: Settings, project: str
) -> dict[str, Any]:
    """Why the champion preprompt is what it is.

    Not project-scoped: the champion is one document per user, and so is its history.
    Rejections are listed alongside promotions -- a history of only the promotions cannot
    explain the promotions that did not happen, and a candidate refused for gate
    integrity is a very different event from one that simply scored worse.
    """
    ctx = build_memory_context(settings)
    try:
        store = ReceiptStore(ctx.conn)
        rows = store.recent(prompt_name=args.prompt or None, limit=args.limit)
    finally:
        ctx.conn.close()
    return {
        "prompt": args.prompt,
        "receipts": [
            {
                "at": r.created_at.isoformat(),
                "prompt": r.prompt_name,
                "verdict": r.verdict,
                "reason": r.reason,
                "champion_version": r.champion_version,
                "champion_score": r.champion_score,
                "candidate": r.candidate_hash,
                "candidate_score": r.candidate_score,
                "gate": r.gate_fingerprint,
                "judge": r.judge_model,
            }
            for r in rows
        ],
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
    result = _forget_result(report, settings, project=project, all_projects=args.all_projects)
    result["projects"] = projects
    return result


async def cmd_ask(args: argparse.Namespace, settings: Settings, project: str) -> dict[str, Any]:
    ctx = build_app_context(settings)
    await ctx.bus.start()
    try:
        user_id = settings.owner_user_id
        hkey = session_key(user_id, None)
        history = ctx.history_store.recent(hkey, project=project) if ctx.history_store else []
        champion = await ctx.prompt_registry.champion(CHAMPION_PROMPT_NAME)
        system_override = champion.body if champion is not None else ""
        result, turn_id = await ctx.orchestrator.handle_turn_with_id(
            user_id=user_id,
            project=project,
            text=args.text,
            session_id=None,
            history=history,
            system_override=system_override,
        )
    finally:
        await ctx.bus.stop()
    return {
        "project": project,
        "response": result.text,
        "model_used": result.model_used,
        "turn_id": turn_id,
    }


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
    lines = [
        f"{i + 1}. [{r['kind']}/{r['project']}] {r['content']}"
        for i, r in enumerate(data["results"])
    ]
    return "\n".join(lines)


def _render_facts(data: dict[str, Any]) -> str:
    if not data["facts"]:
        return (
            f"No currently-valid facts "
            f"(project={data['project']!r}, all_projects={data['all_projects']})."
        )
    lines = [
        f"{f['subject']} {f['predicate']} {f['object']} (confidence={f['confidence']:.2f}, "
        f"project={f['project']})"
        for f in data["facts"]
    ]
    return "\n".join(lines)


def _render_forget(data: dict[str, Any]) -> str:
    scope = "all projects" if data["all_projects"] else f"project {data['project']!r}"
    summary = (
        f"Forgot {scope}: memories={data['memories']} facts={data['facts']} "
        f"signals={data['signals']} history={data['history']} "
        f"index={data['index_entries']} persona={data['persona_nodes']}"
    )
    lines = [summary]
    for w in data["warnings"]:
        lines.append(f"WARNING: {w}")
    return "\n".join(lines)


def _render_receipts(data: dict[str, Any]) -> str:
    rows = data["receipts"]
    if not rows:
        return "No promotion decisions recorded yet."
    lines = []
    for r in rows:
        champ = "none" if r["champion_version"] is None else f"v{r['champion_version']}"
        scores = f"{_fmt_score(r['champion_score'])} -> {_fmt_score(r['candidate_score'])}"
        lines.append(
            f"{r['at']}  {r['verdict']:<8} {r['prompt']}  champion={champ}  {scores}\n"
            f"    {r['reason']}\n"
            f"    candidate={r['candidate']}  gate={r['gate'] or 'n/a'}  "
            f"judge={r['judge'] or 'n/a'}"
        )
    return "\n".join(lines)


def _fmt_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


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
    "doctor": (cmd_doctor, _render_doctor),
    "receipts": (cmd_receipts, _render_receipts),
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

    p_ask = sub.add_parser(
        "ask", help="Ask the assistant (a full chat turn; needs a reachable LLM)."
    )
    p_ask.add_argument("text", help="Your message.")
    _add_common(p_ask)

    p_doctor = sub.add_parser("doctor", help="Diagnose the local Morgan installation.")
    _add_common(p_doctor)

    p_receipts = sub.add_parser(
        "receipts", help="Why the champion preprompt is what it is (promotions AND rejections)."
    )
    p_receipts.add_argument("--prompt", default=None, help="Filter to one prompt name.")
    p_receipts.add_argument("--limit", type=int, default=20, help="How many decisions to show.")
    _add_common(p_receipts)

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
        # ensure_ascii=False: an owner whose corpus is substantially Russian gets readable
        # Cyrillic in JSON output instead of \uXXXX escapes.
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
