"""Prepare an unapproved local focus-plan draft from an explicit existing memory database."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path

from morgan_brain.composition import open_evidence_context
from morgan_brain.daily_brief import BriefRequest, prepare_brief
from morgan_brain.local_notes import NoteWorkspace
from morgan_brain.tasks import TaskKey


def main() -> None:
    """Read explicit evidence and write a new private draft; never enqueue or grant authority."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-db", type=Path, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--evidence", action="append", required=True)
    parser.add_argument(
        "--follow-revisions",
        action="store_true",
        help="Follow selected tasks to one current unambiguous version.",
    )
    parser.add_argument(
        "--unavailability",
        action="append",
        default=[],
        help="Confirmed constraint evidence ID; repeat up to eight times.",
    )
    parser.add_argument("--timezone", help="Explicit IANA zone for the planning day.")
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = BriefRequest(
        TaskKey(args.user, args.project, args.task),
        tuple(args.evidence),
        args.day,
        args.budget,
        follow_revisions=args.follow_revisions,
        unavailability_ids=tuple(args.unavailability),
        timezone=args.timezone,
    )
    context = open_evidence_context(args.memory_db)
    try:
        draft = asyncio.run(prepare_brief(context.gate, request))
    finally:
        context.conn.close()
    output = NoteWorkspace(args.output.parent)
    try:
        output.publish(args.output.name, draft.task.payload)
    finally:
        output.close()
    print(
        json.dumps(
            {
                "draft_written": True,
                "planner": draft.planner,
                "planned_minutes": draft.planned_minutes,
                "selected": len(draft.selected_ids),
                "deferred": len(draft.deferred_ids),
                "excluded": len(draft.excluded_ids),
                "approved": False,
                "blocked": len(draft.blocked_ids),
                "resolved_unavailability": dict(draft.unavailability_resolved_ids),
                "resolved_sources": dict(draft.resolved_ids),
            }
        )
    )


if __name__ == "__main__":
    main()
