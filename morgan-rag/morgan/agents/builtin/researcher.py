"""
Researcher agent definition.

A thorough research agent with access to web search, memory search, and
file reading capabilities. Suited for information gathering, fact-checking,
and deep-dive research tasks.
"""

from morgan.agents.base import AgentDefinition


def _get_researcher_prompt() -> str:
    return (
        "You are a thorough research agent for the Morgan AI assistant.\n\n"
        "Your role is to gather, synthesize, and present information from "
        "multiple sources. You have access to web search, memory search, "
        "and file reading tools.\n\n"
        "Guidelines:\n"
        "- Search broadly first, then narrow down to specific details.\n"
        "- Cross-reference information from multiple sources when possible.\n"
        "- Cite your sources and note any conflicting information.\n"
        "- Prioritize accuracy over speed.\n"
        "- Summarize findings clearly and concisely.\n"
        "- Flag any information you are uncertain about.\n"
    )


ResearcherAgent = AgentDefinition(
    agent_type="researcher",
    when_to_use=(
        "Use when the task requires gathering information from the web, "
        "searching through stored memories, or reading files to answer "
        "questions or compile research."
    ),
    get_system_prompt=_get_researcher_prompt,
    tools=["web_search", "memory_search", "file_read"],
    effort="thorough",
    max_turns=15,
    source="builtin",
)
