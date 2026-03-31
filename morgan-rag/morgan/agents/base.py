"""
Agent base definitions.

Provides the core dataclasses for the agent/subagent system:
- AgentDefinition: Describes an agent's capabilities and configuration
- AgentResult: Captures the outcome of an agent execution
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AgentDefinition:
    """
    Defines an agent's identity, capabilities, and behavior.

    Ported from Claude Code's AgentTool pattern. The get_system_prompt field
    is a callable (closure) that returns the system prompt string, allowing
    dynamic prompt generation based on captured state.

    Attributes:
        agent_type: Unique identifier for this agent type (e.g., "researcher").
        when_to_use: Description of when this agent should be invoked.
        get_system_prompt: Callable returning the system prompt string.
        tools: List of tool names this agent can use.
        disallowed_tools: List of tool names this agent must not use.
        model: Optional model override (None uses the default).
        effort: Effort level: "minimal", "balanced", or "thorough".
        max_turns: Maximum number of turns for the agent's execution loop.
        source: Origin of this definition ("builtin", file path, etc.).
    """

    agent_type: str
    when_to_use: str
    get_system_prompt: Callable[[], str]
    tools: List[str]
    disallowed_tools: List[str] = field(default_factory=list)
    model: Optional[str] = None
    effort: str = "balanced"
    max_turns: int = 10
    source: str = "builtin"


@dataclass
class AgentResult:
    """
    Captures the outcome of an agent execution.

    Attributes:
        output: The agent's text output.
        agent_type: The type of agent that produced this result.
        success: Whether the execution completed successfully.
        error: Error message if success is False.
        metadata: Additional execution metadata (turns used, tokens, etc.).
    """

    output: str
    agent_type: str
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
