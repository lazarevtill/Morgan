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
import os
import sys
from pathlib import Path

from morgan_brain.config import Settings, get_settings
from morgan_brain.logging_setup import configure_logging
from morgan_brain.surfaces.cli.commands import (
    cmd_ask,
    cmd_consolidate,
    cmd_doctor,
    cmd_facts,
    cmd_forget,
    cmd_import,
    cmd_recall,
    cmd_remember,
)
from morgan_brain.surfaces.cli.maintenance import (
    cmd_migrate,
    cmd_restore,
    cmd_snapshot,
    restore_preview,
)
from morgan_brain.surfaces.cli.project import detect_project
from morgan_brain.surfaces.cli.render import RENDERERS

#: Every verb the CLI accepts, and the handler that answers it. The renderer for each
#: lives in ``render.RENDERERS`` under the same key.
HANDLERS = {
    "remember": cmd_remember,
    "recall": cmd_recall,
    "facts": cmd_facts,
    "forget": cmd_forget,
    "ask": cmd_ask,
    "consolidate": cmd_consolidate,
    "doctor": cmd_doctor,
    "import": cmd_import,
    "snapshot": cmd_snapshot,
    "restore": cmd_restore,
    "migrate": cmd_migrate,
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

    p_recall = sub.add_parser("recall", help="Search memory by meaning and by keyword.")
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

    # No --project/--all-projects: a snapshot is of the whole database, not one project.
    p_snapshot = sub.add_parser(
        "snapshot", help="Write a verified VACUUM INTO copy of the whole database."
    )
    p_snapshot.add_argument(
        "--reason",
        default="manual",
        help="Short reason recorded in the snapshot's filename (default: manual).",
    )
    p_snapshot.add_argument(
        "--list", action="store_true", help="List existing snapshots instead of taking one."
    )
    p_snapshot.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of human text."
    )

    # No --project/--all-projects: like snapshot, a restore replaces the whole database file.
    p_restore = sub.add_parser(
        "restore",
        help="Replace the whole database with a snapshot, behind a safety snapshot taken first.",
    )
    p_restore.add_argument("file", help="Path to the snapshot file to restore.")
    p_restore.add_argument(
        "--yes", action="store_true", help="Actually replace the database (default: preview only)."
    )
    p_restore.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of human text."
    )

    # No --project/--all-projects: a migration covers the whole database, like snapshot.
    p_migrate = sub.add_parser(
        "migrate",
        help="Run the pending migration steps, heavy ones included, behind a snapshot.",
    )
    p_migrate.add_argument(
        "--dry-run", action="store_true", help="List the pending steps and change nothing."
    )
    p_migrate.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of human text."
    )

    # No --project: the destination follows the holdout rule, not the caller's working
    # directory, and offering a flag that cannot be honoured would be worse than omitting it.
    p_import = sub.add_parser(
        "import", help="Seed memory from a ChatGPT conversations.json export."
    )
    p_import.add_argument("path", help="Path to the export's conversations.json.")
    p_import.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of human text."
    )

    # No --project: the skill is the same for every project, and it goes into the agents'
    # own folders rather than into the memory.
    p_skill = sub.add_parser(
        "install-skill",
        help="Teach the coding agents installed here when to recall and what to remember.",
    )
    p_skill.add_argument("--yes", action="store_true", help="Write without asking first.")
    p_skill.add_argument(
        "--mcp-server",
        default="morgan",
        help="The name morgan-mcp is registered under in Claude Code (default: morgan).",
    )
    p_skill.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of human text."
    )

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

    if args.command == "restore" and not args.yes:
        # A restore is the one command whose own mistake a rerun cannot undo, so the bare
        # form never touches anything -- it only prints what --yes would replace.
        preview = restore_preview(args, settings)
        if args.json:
            print(json.dumps(preview, ensure_ascii=False))
        else:
            print(
                f"Would replace {preview['database']} with {preview['snapshot']}. "
                "Pass --yes to actually restore."
            )
        return 2

    handler, renderer = HANDLERS[args.command], RENDERERS[args.command]
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
    if args.command == "install-skill":
        # Touches no memory and needs no settings: it writes into the agents' own folders.
        from morgan_brain.surfaces.cli.install_skill import run

        return run(
            yes=args.yes,
            as_json=args.json,
            mcp_server=args.mcp_server,
            home=Path.home(),
            env=os.environ,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    settings = get_settings()
    # getattr: not every verb takes --project. `import` decides the destination from the
    # holdout rule, so it deliberately has no such flag to read.
    project = getattr(args, "project", None) or detect_project(Path.cwd())
    return asyncio.run(_dispatch(args, settings, project))


if __name__ == "__main__":
    sys.exit(main())
