"""The human-readable form of each payload.

Output only. Every function here takes the dictionary ``payloads`` built and returns a
string; none of them reach for a store, a setting or a model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


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


#: The human-readable form of each verb's payload. Rendering only -- which handler
#: produced the payload is the dispatcher's business, not this module's.
def _render_import(data: dict[str, Any]) -> str:
    return (
        f"Imported {data['memories']} memories from {data['conversations']} conversations "
        f"into {data['archive_project']!r}; {data['held_out']} conversations held out in "
        f"{data['holdout_project']!r}, {data['skipped_turns']} turns skipped."
    )


RENDERERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "remember": _render_remember,
    "recall": _render_recall,
    "facts": _render_facts,
    "forget": _render_forget,
    "ask": _render_ask,
    "consolidate": _render_consolidate,
    "doctor": _render_doctor,
    "import": _render_import,
}
