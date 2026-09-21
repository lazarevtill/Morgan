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
        why = _ABSTAINED.get(data["reason"])
        return f"No memories found in {scope} ({why})." if why else f"No memories found in {scope}."
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


def _render_row_counts(rows: dict[str, Any], totals: dict[str, Any] | None) -> list[str]:
    """The scoped row counts, each beside its total -- what told the owner on 2026-09-21 that
    a project-scoped zero was not the whole database. ``memories: 3 in project 'personal' (5
    across all projects)``; under ``--all-projects`` the scope and the total are the same
    number, so only one is shown.
    """
    scope = rows["scope"]
    lines = []
    for key in ("memories", "fts", "vectors"):
        value = rows[key]
        if scope == "all projects":
            lines.append(f"{key}: {value} across all projects")
        else:
            total = totals.get(key) if totals else None
            lines.append(f"{key}: {value} in {scope} ({total} across all projects)")
    return lines


def _render_probe(name: str, verdict: str, probe: dict[str, Any] | None) -> str:
    """``embedding_provider: slow (4.2 s; MORGAN_DOCTOR_SLOW_AFTER_SECONDS=2.0)`` -- the
    verdict, how long the answer took, and, when it is not plainly reachable, why."""
    if probe is None:
        return f"{name}: {verdict}"
    if verdict == "unreachable":
        return f"{name}: unreachable ({probe['error']})"
    parts = [f"{probe['seconds']:.1f} s"]
    # Named only when the clock is why it is slow: a 503 answered at once is slow for what it
    # said, and its status follows.
    if verdict == "slow" and probe["seconds"] > probe["slow_after_seconds"]:
        parts.append(f"MORGAN_DOCTOR_SLOW_AFTER_SECONDS={probe['slow_after_seconds']}")
    if probe["error"]:
        parts.append(str(probe["error"]))
    return f"{name}: {verdict} ({'; '.join(parts)})"


def _render_data_flow(flow: dict[str, Any]) -> str:
    what = "; ".join(f"{command}: {text}" for command, text in flow["carries"].items())
    return f"data_flow: {flow['host']} ({', '.join(flow['settings'])}) receives {what}"


def _render_migration(migration: dict[str, Any] | None, reason: str | None) -> str:
    if migration is None:
        return f"migration: {reason}"
    head = f"migration: user_version {migration['user_version']} of {migration['code_version']}"
    if not migration["pending"]:
        return f"{head}, nothing pending"
    steps = ", ".join(_step_line(s) for s in migration["pending"])
    return f"{head}; pending: {steps} -- `morgan migrate` runs them, doctor runs none"


def _render_snapshots(snapshots: dict[str, Any]) -> str:
    if not snapshots["count"]:
        return f"snapshots: none in {snapshots['dir']}"
    return (
        f"snapshots: {snapshots['count']} in {snapshots['dir']}, newest {snapshots['newest']}, "
        f"{snapshots['bytes']} bytes in all"
    )


def _render_project(project: dict[str, Any]) -> str:
    capture = "on" if project["capture_enabled"] else "off"
    if project["paused_until"]:
        capture += f", paused until {project['paused_until']}"
    if project["retention_days"] is not None:
        capture += f", kept {project['retention_days']} days"
    consolidate = "on" if project["consolidate_enabled"] else "off"
    return (
        f"project {project['name']!r}: {project['classification']}, capture {capture}, "
        f"consolidate {consolidate}"
    )


def _fmt_cosine(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _render_vector_audit(value: dict[str, Any]) -> str:
    """``doctor --vectors``'s report: the pooled numbers, then each client's own, then any id
    the clients disagreed on, then the note about Ollama serialising concurrent clients."""
    lines = [
        f"vector_audit: {value['sampled']} sampled, min {_fmt_cosine(value['min'])}, "
        f"median {_fmt_cosine(value['median'])}, {len(value['below_tolerance'])} below "
        "tolerance"
        + (f" ({', '.join(value['below_tolerance'])})" if value["below_tolerance"] else "")
    ]
    for label, stats in value["per_client"].items():
        if stats.get("error"):
            lines.append(f"  {label}: error ({stats['error']}), {stats['wall_seconds']:.3f} s")
            continue
        lines.append(
            f"  {label}: min {_fmt_cosine(stats['min'])}, median {_fmt_cosine(stats['median'])}, "
            f"{len(stats['below_tolerance'])} below tolerance, {stats['wall_seconds']:.3f} s"
        )
    if value["disagreements"]:
        lines.append(f"  disagreements between clients: {', '.join(value['disagreements'])}")
    lines.append(f"  {value['note']}")
    return "\n".join(lines)


def _render_doctor_line(key: str, value: Any, data: dict[str, Any]) -> list[str]:
    """The lines one key of doctor's report prints as. A key whose value another key's line
    already carries prints none."""
    if key in (
        "database_error",
        "rows_all_projects",
        "provider_probe",
        "embedding_probe",
        "embedding_space_reason",
        "migration_reason",
        "projects_reason",
        "vector_audit_reason",
    ):
        return []  # folded into the line of the key it describes
    if key == "database":
        error = data.get("database_error")
        if error is None:
            return [f"database: {value}"]
        missing = error.startswith("no database yet")
        return [f"database: {value} ({'no database yet' if missing else error})"]
    if key == "env_files":
        return [f"env_file: {f['path']} ({'present' if f['present'] else 'absent'})" for f in value]
    if key == "data_flow":
        return [_render_data_flow(flow) for flow in value]
    if key in ("provider", "embedding_provider"):
        probe_key = "provider_probe" if key == "provider" else "embedding_probe"
        return [_render_probe(key, value, data.get(probe_key))]
    if key == "embedding_space":
        if value is None:
            return [f"embedding_space: none ({data.get('embedding_space_reason')})"]
        return [
            f"embedding_space: {value['id']} ({value['model']}, {value['dims']} dims): "
            f"fingerprint {value['fingerprint']}; strings sha256 {value['strings_digest']}"
        ]
    if key == "migration":
        return [_render_migration(value, data.get("migration_reason"))]
    if key == "vector_audit":
        if value is None:
            return [f"vector_audit: none ({data.get('vector_audit_reason')})"]
        return [_render_vector_audit(value)]
    if key == "snapshots" and value is not None:
        return [_render_snapshots(value)]
    if key == "rows" and value is not None:
        return _render_row_counts(value, data.get("rows_all_projects"))
    if key == "projects":
        if value is None:
            return [f"projects: none ({data.get('projects_reason')})"]
        return [_render_project(p) for p in value] or ["projects: none recorded yet"]
    if key == "code_roots":
        return [
            f"code_root: {r['path']}" + ("" if r["is_directory"] else " (not a directory)")
            for r in value
        ]
    return [f"{key}: {value}"]


def _render_doctor(data: dict[str, Any]) -> str:
    return "\n".join(line for k, v in data.items() for line in _render_doctor_line(k, v, data))


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
