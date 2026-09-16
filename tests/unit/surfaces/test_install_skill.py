"""`morgan install-skill` -- teaching the owner's coding agents when to use Morgan.

The MCP tools say what Morgan can do; nothing told an agent *when* to recall or what is
worth remembering, so the memory went unused. The installer writes one skill into every
coding agent it finds, and into OpenResearch's upload store, whose sessions do not load the
agents' own skills. It lists every path first and writes nothing without a yes, and it never
overwrites a skill of the same name that it did not write.

Every test runs against a temporary home directory.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from morgan_brain.surfaces.cli.__main__ import main
from morgan_brain.surfaces.cli.install_skill import apply, plan, run, skill_text


def _agents(home: Path, *names: str) -> None:
    folders = {
        "claude": ".claude",
        "codex": ".codex",
        "opencode": ".config/opencode",
        "cursor": ".cursor",
        "openresearch": ".local/share/openresearch",
    }
    for name in names:
        (home / folders[name]).mkdir(parents=True)


def _skill_paths(result) -> dict[str, Path]:
    return {a.agent: a.path for a in result.actions if a.kind == "skill"}


def test_only_agents_that_are_installed_get_a_skill(tmp_path):
    _agents(tmp_path, "claude", "cursor")

    result = plan(home=tmp_path, env={}, mcp_server="morgan")

    assert _skill_paths(result) == {
        "Claude Code": tmp_path / ".claude/skills/morgan/SKILL.md",
        "Cursor": tmp_path / ".cursor/skills/morgan/SKILL.md",
    }
    assert set(result.missing) == {"Codex", "OpenCode", "OpenResearch"}


def test_each_agent_gets_the_skill_where_it_looks_for_skills(tmp_path):
    _agents(tmp_path, "codex", "opencode", "openresearch")

    paths = _skill_paths(plan(home=tmp_path, env={}, mcp_server="morgan"))

    assert paths == {
        # Codex reads SKILL.md skills from the folder the agents share, not its own home.
        "Codex": tmp_path / ".agents/skills/morgan/SKILL.md",
        "OpenCode": tmp_path / ".config/opencode/skills/morgan/SKILL.md",
        # Its sessions run with their own agent configuration; only uploads reach them.
        "OpenResearch": tmp_path / ".local/share/openresearch/user-skills/global/morgan/SKILL.md",
    }


def test_relocated_agent_homes_are_followed(tmp_path):
    env = {
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude-elsewhere"),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
        "ORX_DATA_DIR": str(tmp_path / "orx"),
    }
    for folder in ("claude-elsewhere", "xdg/opencode", "orx"):
        (tmp_path / folder).mkdir(parents=True)

    paths = _skill_paths(plan(home=tmp_path, env=env, mcp_server="morgan"))

    assert paths == {
        "Claude Code": tmp_path / "claude-elsewhere/skills/morgan/SKILL.md",
        "OpenCode": tmp_path / "xdg/opencode/skills/morgan/SKILL.md",
        "OpenResearch": tmp_path / "orx/user-skills/global/morgan/SKILL.md",
    }


def test_applying_writes_the_packaged_skill(tmp_path):
    _agents(tmp_path, "claude", "openresearch")

    apply(plan(home=tmp_path, env={}, mcp_server="morgan"))

    for path in _skill_paths(plan(home=tmp_path, env={}, mcp_server="morgan")).values():
        assert path.read_text(encoding="utf-8") == skill_text()
    assert skill_text().startswith("---\nname: morgan\n")


def test_a_second_run_finds_nothing_to_do(tmp_path):
    _agents(tmp_path, "claude")
    apply(plan(home=tmp_path, env={}, mcp_server="morgan"))

    again = plan(home=tmp_path, env={}, mcp_server="morgan")

    assert {a.status for a in again.actions} == {"unchanged"}


def test_its_own_older_skill_is_updated(tmp_path):
    _agents(tmp_path, "claude")
    apply(plan(home=tmp_path, env={}, mcp_server="morgan"))
    written = tmp_path / ".claude/skills/morgan/SKILL.md"
    written.write_text(skill_text().replace("Recall before you work", "Old heading"), "utf-8")

    result = plan(home=tmp_path, env={}, mcp_server="morgan")
    apply(result)

    assert [a.status for a in result.actions if a.kind == "skill"] == ["update"]
    assert written.read_text(encoding="utf-8") == skill_text()


def test_a_morgan_skill_it_did_not_write_is_left_alone(tmp_path):
    _agents(tmp_path, "claude")
    foreign = tmp_path / ".claude/skills/morgan/SKILL.md"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("---\nname: morgan\ndescription: someone else's\n---\n", "utf-8")

    result = plan(home=tmp_path, env={}, mcp_server="morgan")
    apply(result)

    assert [a.status for a in result.actions if a.kind == "skill"] == ["conflict"]
    assert "someone else's" in foreign.read_text(encoding="utf-8")


def test_claude_code_may_run_exactly_the_read_only_tools_unprompted(tmp_path):
    _agents(tmp_path, "claude")
    settings = tmp_path / ".claude/settings.json"
    settings.write_text(
        json.dumps({"model": "opus", "permissions": {"allow": ["Bash(ls)"], "deny": ["X"]}}),
        "utf-8",
    )

    apply(plan(home=tmp_path, env={}, mcp_server="morgan"))

    written = json.loads(settings.read_text(encoding="utf-8"))
    assert written["model"] == "opus"
    assert written["permissions"]["deny"] == ["X"]
    assert written["permissions"]["allow"] == [
        "Bash(ls)",
        "mcp__morgan__recall",
        "mcp__morgan__facts",
    ]


def test_the_rules_follow_the_name_the_server_is_registered_under(tmp_path):
    _agents(tmp_path, "claude")

    apply(plan(home=tmp_path, env={}, mcp_server="brain"))

    allow = json.loads((tmp_path / ".claude/settings.json").read_text("utf-8"))["permissions"]
    assert allow["allow"] == ["mcp__brain__recall", "mcp__brain__facts"]


def test_settings_it_cannot_parse_are_not_rewritten(tmp_path):
    _agents(tmp_path, "claude")
    settings = tmp_path / ".claude/settings.json"
    settings.write_text("{ not json", "utf-8")

    result = plan(home=tmp_path, env={}, mcp_server="morgan")
    apply(result)

    assert [a.status for a in result.actions if a.kind == "permissions"] == ["invalid"]
    assert settings.read_text(encoding="utf-8") == "{ not json"


def test_nothing_is_written_without_a_yes(tmp_path):
    _agents(tmp_path, "claude")
    out, err = io.StringIO(), io.StringIO()

    code = run(
        yes=False, as_json=False, mcp_server="morgan",
        home=tmp_path, env={}, stdin=io.StringIO("n\n"), stdout=out, stderr=err,
    )  # fmt: skip

    assert code == 1
    assert ".claude" in out.getvalue()  # the plan was shown
    assert not (tmp_path / ".claude/skills").exists()
    assert not (tmp_path / ".claude/settings.json").exists()


def test_a_yes_at_the_prompt_writes(tmp_path):
    _agents(tmp_path, "claude")

    code = run(
        yes=False, as_json=False, mcp_server="morgan",
        home=tmp_path, env={}, stdin=io.StringIO("y\n"), stdout=io.StringIO(),
        stderr=io.StringIO(),
    )  # fmt: skip

    assert code == 0
    assert (tmp_path / ".claude/skills/morgan/SKILL.md").exists()


def test_the_cli_installs_with_yes_and_reports_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for variable in ("CLAUDE_CONFIG_DIR", "CODEX_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("ORX_DATA_DIR", raising=False)
    _agents(tmp_path, "cursor")

    code = main(["install-skill", "--yes", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["applied"] is True
    assert [a["agent"] for a in report["actions"]] == ["Cursor"]
    assert (tmp_path / ".cursor/skills/morgan/SKILL.md").exists()
