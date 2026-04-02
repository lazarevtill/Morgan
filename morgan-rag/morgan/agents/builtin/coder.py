"""
Coder agent definition.

A thorough coding agent with access to bash execution, file I/O, and
calculation tools. Suited for writing code, debugging, running scripts,
and performing technical tasks.
"""

from morgan.agents.base import AgentDefinition


def _get_coder_prompt() -> str:
    return (
        "You are an expert coding agent for the Morgan AI assistant.\n\n"
        "Your role is to write, debug, and execute code to accomplish "
        "technical tasks. You have access to bash execution, file reading, "
        "file writing, and calculator tools.\n\n"
        "Guidelines:\n"
        "- Write clean, well-documented code.\n"
        "- Test your code by running it when possible.\n"
        "- Handle errors gracefully and explain failures clearly.\n"
        "- Prefer simple, readable solutions over clever ones.\n"
        "- Read existing code before modifying it.\n"
        "- Use the calculator for precise numerical computations.\n"
        "- Explain your approach before writing complex code.\n"
    )


CoderAgent = AgentDefinition(
    agent_type="coder",
    when_to_use=(
        "Use when the task requires writing code, running shell commands, "
        "reading or modifying files, debugging, or performing calculations."
    ),
    get_system_prompt=_get_coder_prompt,
    tools=["bash", "file_read", "file_write", "calculator"],
    effort="thorough",
    max_turns=20,
    source="builtin",
)
