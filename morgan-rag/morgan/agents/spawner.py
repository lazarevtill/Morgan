"""
Agent spawner.

Manages the execution of agent definitions by calling either a user-provided
run function or falling back to Morgan's LLM service.
"""

import logging
from typing import Any, Callable, Coroutine, List, Optional

from morgan.agents.base import AgentDefinition, AgentResult

logger = logging.getLogger(__name__)


class AgentSpawner:
    """
    Spawns and executes agents based on their definitions.

    Accepts an optional run_fn for custom execution logic. When no run_fn is
    provided, defaults to calling get_llm_service().agenerate().

    The run_fn signature is:
        async def run_fn(
            prompt: str,
            system_prompt: str,
            tools: List[str],
            model: Optional[str],
        ) -> str

    Usage:
        # With custom run function
        spawner = AgentSpawner(run_fn=my_custom_runner)
        result = await spawner.spawn(agent_def, prompt="Do something")

        # With default LLM service
        spawner = AgentSpawner()
        result = await spawner.spawn(agent_def, prompt="Do something")
    """

    def __init__(
        self,
        run_fn: Optional[
            Callable[
                [str, str, List[str], Optional[str]],
                Coroutine[Any, Any, str],
            ]
        ] = None,
    ):
        self._run_fn = run_fn

    async def spawn(
        self,
        definition: AgentDefinition,
        prompt: str,
        context: Optional[str] = None,
    ) -> AgentResult:
        """
        Spawn an agent and execute it.

        Args:
            definition: The agent definition to execute.
            prompt: The user prompt to send to the agent.
            context: Optional context to append to the prompt.

        Returns:
            AgentResult with the agent's output or error information.
        """
        # Build the full prompt with optional context
        full_prompt = prompt
        if context:
            full_prompt = f"{prompt}\n\nContext:\n{context}"

        system_prompt = definition.get_system_prompt()

        try:
            if self._run_fn is not None:
                output = await self._run_fn(
                    full_prompt,
                    system_prompt,
                    definition.tools,
                    definition.model,
                )
            else:
                output = await self._default_run(
                    full_prompt, system_prompt, definition
                )

            return AgentResult(
                output=output,
                agent_type=definition.agent_type,
                success=True,
            )
        except Exception as e:
            logger.error(
                "Agent '%s' failed: %s", definition.agent_type, str(e)
            )
            return AgentResult(
                output="",
                agent_type=definition.agent_type,
                success=False,
                error=str(e),
            )

    async def _default_run(
        self,
        prompt: str,
        system_prompt: str,
        definition: AgentDefinition,
    ) -> str:
        """
        Default execution using Morgan's LLM service.

        Imports get_llm_service lazily to avoid circular imports and allow
        the spawner to work without a running LLM service when a custom
        run_fn is provided.
        """
        from morgan.services import get_llm_service

        llm = get_llm_service()
        response = await llm.agenerate(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        return response.content
