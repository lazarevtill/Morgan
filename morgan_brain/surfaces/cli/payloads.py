"""The shape of every command's result.

One payload per command, built once and used twice: serialised directly under
``--json``, which is a contract other programs parse, and handed to ``render`` for
the human form. Keeping the two readings of the same dictionary together is what
stops them drifting apart.
"""

from __future__ import annotations

from typing import Any

from morgan_brain.memory.gate import ForgetReport
from morgan_brain.models import Memory, TemporalFact


def memory_to_dict(m: Memory) -> dict[str, Any]:
    return {
        "id": m.id,
        "project": m.project,
        "kind": m.kind.value,
        "content": m.content,
        "source": m.source.value,
        "importance": m.importance,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


def fact_to_dict(f: TemporalFact) -> dict[str, Any]:
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


def merge_forget_reports(reports: list[ForgetReport]) -> ForgetReport:
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


def forget_result(report: ForgetReport, *, project: str, all_projects: bool) -> dict[str, Any]:
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
