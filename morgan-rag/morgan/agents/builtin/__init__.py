"""
Built-in agent definitions for Morgan.

Provides pre-configured agent definitions for common tasks:
- ResearcherAgent: Web and memory research
- CoderAgent: Code writing and execution
- PlannerAgent: Task planning and decomposition
"""

from morgan.agents.builtin.researcher import ResearcherAgent
from morgan.agents.builtin.coder import CoderAgent
from morgan.agents.builtin.planner import PlannerAgent

BUILTIN_AGENTS = [
    ResearcherAgent,
    CoderAgent,
    PlannerAgent,
]

__all__ = [
    "BUILTIN_AGENTS",
    "ResearcherAgent",
    "CoderAgent",
    "PlannerAgent",
]
