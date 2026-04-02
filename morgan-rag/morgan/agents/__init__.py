"""
Morgan Agent/Subagent System.

Provides an agent spawning framework ported from Claude Code's AgentTool pattern.
Agents are defined as dataclasses with configurable system prompts, tools, and
execution parameters. They can be defined in code (built-in) or loaded from
markdown files with YAML frontmatter.

Usage:
    from morgan.agents import AgentDefinition, AgentResult, AgentSpawner
    from morgan.agents.builtin import BUILTIN_AGENTS
    from morgan.agents.loader import load_agents_from_dir

    # Spawn a built-in agent
    spawner = AgentSpawner()
    result = await spawner.spawn(BUILTIN_AGENTS[0], prompt="Research topic X")

    # Load user-defined agents from disk
    agents = load_agents_from_dir("/path/to/agents/")
"""

from morgan.agents.base import AgentDefinition, AgentResult
from morgan.agents.spawner import AgentSpawner
from morgan.agents.loader import load_agents_from_dir, parse_frontmatter

__all__ = [
    "AgentDefinition",
    "AgentResult",
    "AgentSpawner",
    "load_agents_from_dir",
    "parse_frontmatter",
]
