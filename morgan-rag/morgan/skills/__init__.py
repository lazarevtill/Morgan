"""
Morgan Skill/Plugin System.

Provides a skill framework ported from Claude Code's skills and OpenClaw's
plugin patterns. Skills are markdown files with YAML frontmatter that define
reusable prompt templates with variable substitution.

Usage:
    from morgan.skills import Skill, SkillRegistry, SkillExecutor
    from morgan.skills.loader import load_skills_from_dir

    # Load skills from a directory
    skills = load_skills_from_dir("/path/to/skills/")

    # Register and look up skills
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    skill = registry.get("web_research")

    # Execute a skill through the agent system
    executor = SkillExecutor(spawner=AgentSpawner(run_fn=my_run_fn))
    result = await executor.execute(skill, {"query": "latest AI news"})
"""

from morgan.skills.loader import Skill, load_skill_from_file, load_skills_from_dir
from morgan.skills.registry import SkillRegistry
from morgan.skills.executor import SkillExecutor

__all__ = [
    "Skill",
    "SkillRegistry",
    "SkillExecutor",
    "load_skill_from_file",
    "load_skills_from_dir",
]
