"""
Planner agent definition.

A balanced planning agent with access to file reading, web search, and
memory search. Suited for breaking down complex tasks, creating action
plans, and organizing work.
"""

from morgan.agents.base import AgentDefinition


def _get_planner_prompt() -> str:
    return (
        "You are a planning agent for the Morgan AI assistant.\n\n"
        "Your role is to analyze tasks, break them into manageable steps, "
        "and create clear action plans. You have access to file reading, "
        "web search, and memory search tools.\n\n"
        "Guidelines:\n"
        "- Understand the full scope of the task before planning.\n"
        "- Break complex tasks into clear, actionable steps.\n"
        "- Identify dependencies between steps.\n"
        "- Consider potential blockers and alternatives.\n"
        "- Search for existing context and prior work before planning.\n"
        "- Keep plans concise but complete.\n"
        "- Prioritize steps by importance and dependency order.\n"
    )


PlannerAgent = AgentDefinition(
    agent_type="planner",
    when_to_use=(
        "Use when the task requires breaking down a complex problem, "
        "creating an action plan, organizing work, or analyzing "
        "dependencies between subtasks."
    ),
    get_system_prompt=_get_planner_prompt,
    tools=["file_read", "web_search", "memory_search"],
    effort="balanced",
    max_turns=10,
    source="builtin",
)
