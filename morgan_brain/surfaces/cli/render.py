"""The human-readable form of each payload.

Output only. Every function here takes the dictionary ``payloads`` built and returns a
string; none of them reach for a store, a setting or a model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _render_remember(data: dict[str, Any]) -> str:
    return f"Stored memory {data['id']} in project {data['project']!r}."


#: Why an empty recall is empty, in words: the owner acts differently on each.
_ABSTAINED = {
    "empty": "empty",
    "declined": "declined: nothing stood out above the background",
}


def _render_recall(data: dict[str, Any]) -> str:
    if not data["results"]:
        scope = "any project" if data["all_projects"] else f"project {data['project']!r}"
        why = _ABSTAINED.get(data["reason"], str(data["reason"]))
        return f"No memories found in {scope} ({why})."
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
        f"history={data['history']}",
        f"Snapshot (undo with `morgan restore`): {data['snapshot']}",
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


def _render_snapshot(data: dict[str, Any]) -> str:
    # "snapshots" is present only under --list -- the one discriminator between the two
    # shapes cmd_snapshot returns.
    if "snapshots" in data:
        if not data["snapshots"]:
            return "No snapshots yet."
        return "\n".join(
            f"{s['path']} ({s['bytes']} bytes, user_version={s['user_version']}, "
            f"counts={s['counts']})"
            for s in data["snapshots"]
        )
    return (
        f"Wrote {data['path']} ({data['bytes']} bytes, user_version={data['user_version']}, "
        f"counts={data['counts']})"
    )


def _render_restore(data: dict[str, Any]) -> str:
    return (
        f"Restored: before={data['before']} after={data['after']} "
        f"(the database as it was is saved at {data['safety_snapshot']})"
    )


def _step_line(step: dict[str, Any]) -> str:
    return f"{step['number']} {step['name']} ({'heavy' if step['heavy'] else 'light'})"


def _render_migrate(data: dict[str, Any]) -> str:
    # "steps" is present only once steps have run -- the one discriminator between the two
    # shapes cmd_migrate returns.
    if "steps" not in data:
        if not data["pending"]:
            return (
                f"Nothing to migrate: {data['database']} is at user_version "
                f"{data['user_version']}, and this morgan knows {data['code_version']} steps."
            )
        lines = [
            f"{data['database']} is at user_version {data['user_version']}; this morgan knows "
            f"{data['code_version']} steps. Pending:"
        ]
        lines.extend(f"  {_step_line(s)}" for s in data["pending"])
        if data["dry_run"]:
            lines.append("Dry run: nothing was changed.")
        return "\n".join(lines)
    lines = [f"Snapshot first: {data['snapshot']}"]
    for s in data["steps"]:
        counts = ", ".join(f"{table}={n}" for table, n in s["counts"].items())
        lines.append(f"  {_step_line(s)}: {counts or 'no rows counted'}")
    lines.append(
        f"user_version {data['from_version']} -> {data['user_version']}, quick_check "
        f"{data['quick_check']}; rows before={data['before']} after={data['after']}"
    )
    space = data.get("embedding_space")
    if space is not None:
        lines.append(_space_line(space))
    return "\n".join(lines)


def _space_line(space: dict[str, Any]) -> str:
    if space["fingerprint"] == "unverified":
        return f"space {space['id']} unverified; the first call will verify it ({space['reason']})"
    return (
        f"embedding space {space['id']} ({space['model']}, {space['dims']} dims): "
        f"fingerprint {space['fingerprint']}"
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
    "snapshot": _render_snapshot,
    "restore": _render_restore,
    "migrate": _render_migrate,
}
