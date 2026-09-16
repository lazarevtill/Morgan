"""``morgan install-skill`` -- teach the owner's coding agents when to use Morgan.

The MCP tools say what Morgan can do. Nothing told an agent *when* to recall, or what is worth
remembering, and a memory nobody consults is not one. This writes a single skill, packaged
beside this module, to every coding agent installed here:

* Claude Code reads ``skills/<name>/SKILL.md`` from its config home;
* Codex reads SKILL.md skills from ``~/.agents/skills``, the folder agents share;
* OpenCode reads them from its XDG config folder;
* Cursor reads them from ``~/.cursor/skills``;
* OpenResearch runs its sessions with their own agent configuration, so they never load the
  skills above. Only its upload store reaches them, and an upload is exactly one
  ``user-skills/global/<name>/SKILL.md`` in its data folder.

For Claude Code it also allows the MCP tools that declare themselves read-only, so recalling
at the start of a task does not stop for a prompt. The rules are derived from the tools' own
declarations; a tool that writes or destroys stays behind the prompt.

Every path is listed first and nothing is written without a yes. A skill of the same name
that this command did not write is left alone, and a settings file it cannot parse is not
rewritten.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, TextIO

SKILL_NAME = "morgan"

#: The line that marks a skill this command wrote, so a rerun may replace it.
MARKER = "<!-- Written by `morgan install-skill`; run it again to update. -->"

Status = Literal["create", "update", "unchanged", "conflict", "add", "invalid"]

#: The statuses that write something when applied.
_WRITES: frozenset[str] = frozenset({"create", "update", "add"})


@dataclass(frozen=True)
class Action:
    """One thing the installer would do, and whether it can."""

    kind: Literal["skill", "permissions"]
    agent: str
    path: Path
    status: Status
    #: For permissions, the allow rules this action adds.
    rules: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    actions: list[Action] = field(default_factory=list)
    #: Agents looked for and not installed here.
    missing: list[str] = field(default_factory=list)


def skill_text() -> str:
    """The packaged skill, as written to every agent."""
    return files("morgan_brain.surfaces.cli").joinpath("skill/SKILL.md").read_text("utf-8")


def _skill_status(path: Path, text: str) -> Status:
    if not path.exists():
        return "create"
    existing = path.read_text(encoding="utf-8", errors="replace")
    if MARKER not in existing:
        return "conflict"
    return "unchanged" if existing == text else "update"


def _agent_skill_folders(home: Path, env: Mapping[str, str]) -> list[tuple[str, Path, Path]]:
    """``(agent, folder whose presence means it is installed, skill file)`` for each agent."""
    claude = Path(env["CLAUDE_CONFIG_DIR"]) if env.get("CLAUDE_CONFIG_DIR") else home / ".claude"
    codex = Path(env["CODEX_HOME"]) if env.get("CODEX_HOME") else home / ".codex"
    xdg_config = Path(env["XDG_CONFIG_HOME"]) if env.get("XDG_CONFIG_HOME") else home / ".config"
    xdg_data = Path(env["XDG_DATA_HOME"]) if env.get("XDG_DATA_HOME") else home / ".local" / "share"
    openresearch = (
        Path(env["ORX_DATA_DIR"]) if env.get("ORX_DATA_DIR") else xdg_data / "openresearch"
    )
    skill = Path(SKILL_NAME) / "SKILL.md"
    return [
        ("Claude Code", claude, claude / "skills" / skill),
        ("Codex", codex, home / ".agents" / "skills" / skill),
        ("OpenCode", xdg_config / "opencode", xdg_config / "opencode" / "skills" / skill),
        ("Cursor", home / ".cursor", home / ".cursor" / "skills" / skill),
        ("OpenResearch", openresearch, openresearch / "user-skills" / "global" / skill),
    ]


def _read_only_rules(mcp_server: str) -> tuple[str, ...]:
    # Imported here: the MCP server module loads the MCP SDK, which no other CLI verb needs.
    from morgan_brain.surfaces.mcp_server import READ_ONLY_TOOLS

    return tuple(f"mcp__{mcp_server}__{tool}" for tool in READ_ONLY_TOOLS)


def _permissions_action(settings: Path, rules: tuple[str, ...]) -> Action:
    agent = "Claude Code"
    if not settings.exists():
        return Action("permissions", agent, settings, "add", rules)
    try:
        data = json.loads(settings.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return Action("permissions", agent, settings, "invalid")
    permissions = data.get("permissions", {}) if isinstance(data, dict) else None
    allow = permissions.get("allow", []) if isinstance(permissions, dict) else None
    if not isinstance(allow, list):
        return Action("permissions", agent, settings, "invalid")
    missing = tuple(rule for rule in rules if rule not in allow)
    return Action("permissions", agent, settings, "add" if missing else "unchanged", missing)


def plan(*, home: Path, env: Mapping[str, str], mcp_server: str) -> Plan:
    """What installing would do here. Reads only."""
    text = skill_text()
    result = Plan()
    for agent, installed, skill in _agent_skill_folders(home, env):
        if not installed.is_dir():
            result.missing.append(agent)
            continue
        result.actions.append(Action("skill", agent, skill, _skill_status(skill, text)))
        if agent == "Claude Code":
            result.actions.append(
                _permissions_action(installed / "settings.json", _read_only_rules(mcp_server))
            )
    return result


def _add_rules(settings: Path, rules: tuple[str, ...]) -> None:
    data: dict[str, Any] = (
        json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
    )
    allow = data.setdefault("permissions", {}).setdefault("allow", [])
    allow.extend(rule for rule in rules if rule not in allow)
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def apply(result: Plan) -> None:
    """Carry out every action that writes. Conflicts and unparseable settings are skipped."""
    text = skill_text()
    for action in result.actions:
        if action.status not in _WRITES:
            continue
        if action.kind == "skill":
            action.path.parent.mkdir(parents=True, exist_ok=True)
            action.path.write_bytes(text.encode("utf-8"))
        else:
            _add_rules(action.path, action.rules)


_EXPLAIN: dict[str, str] = {
    "create": "",
    "update": "replaces the skill an earlier run wrote",
    "unchanged": "already current",
    "conflict": "a skill of this name that morgan did not write; left alone",
    "add": "",
    "invalid": "not a JSON settings file morgan can edit; left alone",
}


def _describe(action: Action) -> str:
    note = _EXPLAIN[action.status]
    if action.kind == "permissions" and action.rules:
        note = "allow " + ", ".join(action.rules)
    line = f"  {action.status:<9} {action.agent:<13} {action.path}"
    return f"{line}  ({note})" if note else line


def _report(result: Plan, *, applied: bool) -> dict[str, Any]:
    return {
        "applied": applied,
        "actions": [
            {
                "agent": a.agent,
                "kind": a.kind,
                "path": str(a.path),
                "status": a.status,
                "rules": list(a.rules),
            }
            for a in result.actions
        ],
        "missing": result.missing,
    }


def run(
    *,
    yes: bool,
    as_json: bool,
    mcp_server: str,
    home: Path,
    env: Mapping[str, str],
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """List what installing would do, ask, then do it. Returns the process exit code."""
    result = plan(home=home, env=env, mcp_server=mcp_server)
    pending = [a for a in result.actions if a.status in _WRITES]

    if not as_json:
        stdout.write("morgan install-skill:\n")
        stdout.writelines(_describe(action) + "\n" for action in result.actions)
        if result.missing:
            stdout.write("  not installed here: " + ", ".join(result.missing) + "\n")

    if not pending:
        if as_json:
            stdout.write(json.dumps(_report(result, applied=False), indent=2) + "\n")
        else:
            stdout.write("Nothing to write.\n")
        return 0

    if not yes:
        if as_json:
            stdout.write(json.dumps(_report(result, applied=False), indent=2) + "\n")
            stderr.write("morgan install-skill: nothing written; pass --yes to write.\n")
            return 1
        # The list goes to buffered stdout and the question to stderr; without the flush a
        # piped terminal shows the question before the list it asks about.
        stdout.flush()
        stderr.write("Write these? [y/N] ")
        stderr.flush()
        if stdin.readline().strip().lower() not in {"y", "yes"}:
            stdout.write("Nothing written.\n")
            return 1

    apply(result)
    if as_json:
        stdout.write(json.dumps(_report(result, applied=True), indent=2) + "\n")
    else:
        stdout.write(f"Wrote {len(pending)} of {len(result.actions)}.\n")
    return 0
