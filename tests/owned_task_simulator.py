"""Owned synthetic fixture state only; never Morgan memory or external actions."""

import hashlib
import json
import sqlite3
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def initialize(
    path: Path,
    owner: str,
    project: str,
    task_id: str,
    steps: list[str],
    completed: list[str],
):
    if path.exists():
        raise ValueError("existing synthetic state; no reset")
    if not owner or not project or not task_id or not steps or len(set(steps)) != len(steps):
        raise ValueError("explicit bounded fixture scope/sequence required")
    if len(steps) > 8 or any(not s or len(s) > 128 for s in steps):
        raise ValueError("step bounds")
    if completed != steps[: len(completed)]:
        raise ValueError("initial completion must be a prefix")
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE fixture (owner TEXT, project TEXT, task_id TEXT, steps TEXT,
                              completed INTEGER, version INTEGER,
                              PRIMARY KEY(owner,project,task_id));
        CREATE TABLE effects (request_id TEXT PRIMARY KEY, owner TEXT, project TEXT, task_id TEXT,
                              step_id TEXT, before_version INTEGER, after_version INTEGER);
        """)
        db.execute(
            "INSERT INTO fixture VALUES(?,?,?,?,?,0)",
            (owner, project, task_id, json.dumps(steps), len(completed)),
        )


def observe(path: Path, owner: str, project: str, task_id: str):
    """Read-only current fixture observation; its fingerprint is not authentication."""
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        row = db.execute(
            "SELECT steps,completed,version FROM fixture WHERE owner=? AND project=? AND task_id=?",
            (owner, project, task_id),
        ).fetchone()
    if row is None:
        raise ValueError("synthetic scope unavailable")
    steps, count, version = json.loads(row[0]), row[1], row[2]
    result = {
        "observation_id": "synthetic-state-v" + str(version),
        "origin": "owned_stateful_simulator_read",
        "owner": owner,
        "project": project,
        "task_id": task_id,
        "steps": steps,
        "completed_steps": steps[:count],
        "version": version,
        "scope": "synthetic fixture only; no real-world completion claim",
    }
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    if before != after:
        raise ValueError("fixture changed during observation")
    return {
        **result,
        "observation_sha256": hashlib.sha256(canonical(result)).hexdigest(),
        "read_only_db_sha256": after,
    }


def execute(path: Path, decision: dict, current: dict, request_id: str, expected_version: int):
    """Commit one valid next-step and its receipt together in one SQLite transaction."""
    if not request_id or len(request_id) > 128:
        raise ValueError("request identity bound")
    effect = {"effect": None, "refusal": None}
    if decision["project"] != current["project"]:
        return {**effect, "refusal": "cross_project"}
    if decision["kind"] != "resume_step":
        return effect
    if current.get("simulation_authorized") is not True:
        return {**effect, "refusal": "current_request_does_not_authorize_simulation"}
    if decision["task_id"] != current["task_id"]:
        return {**effect, "refusal": "unknown_task"}
    if decision["step_id"] not in current.get("available_steps", {}).get(current["task_id"], []):
        return {**effect, "refusal": "unknown_task_or_step"}
    with sqlite3.connect(path) as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM effects WHERE request_id=?", (request_id,)).fetchone():
            raise ValueError("request already executed; no replay")
        row = db.execute(
            "SELECT steps,completed,version FROM fixture WHERE owner=? AND project=? AND task_id=?",
            (current["owner"], current["project"], current["task_id"]),
        ).fetchone()
        if row is None:
            return {**effect, "refusal": "synthetic_scope_unavailable"}
        steps, count, version = json.loads(row[0]), row[1], row[2]
        if version != expected_version:
            return {**effect, "refusal": "stale_observation_version"}
        if count >= len(steps):
            return {**effect, "refusal": "all_steps_completed"}
        if decision["step_id"] != steps[count]:
            return {**effect, "refusal": "not_next_unfinished_step"}
        next_version = version + 1
        db.execute(
            "UPDATE fixture SET completed=?,version=? WHERE owner=? AND project=? AND task_id=?",
            (
                count + 1,
                next_version,
                current["owner"],
                current["project"],
                current["task_id"],
            ),
        )
        db.execute(
            "INSERT INTO effects VALUES(?,?,?,?,?,?,?)",
            (
                request_id,
                current["owner"],
                current["project"],
                current["task_id"],
                decision["step_id"],
                version,
                next_version,
            ),
        )
        effect["effect"] = {
            "owner": current["owner"],
            "project": current["project"],
            "task_id": current["task_id"],
            "step_id": decision["step_id"],
            "before_version": version,
            "after_version": next_version,
        }
    return effect


def receipts(path: Path):
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT request_id,owner,project,task_id,step_id,before_version,after_version "
            "FROM effects ORDER BY after_version"
        ).fetchall()
    return [
        dict(
            zip(
                [
                    "request_id",
                    "owner",
                    "project",
                    "task_id",
                    "step_id",
                    "before_version",
                    "after_version",
                ],
                row,
                strict=True,
            )
        )
        for row in rows
    ]
