"""
Skill executor.

Creates an AgentDefinition from a Skill and delegates execution to
the AgentSpawner. This bridges the skill/plugin system with the
existing agent infrastructure.
"""

import logging
from typing import Dict, Optional

from morgan.agents.base import AgentDefinition, AgentResult
from morgan.agents.spawner import AgentSpawner
from morgan.skills.loader import Skill

logger = logging.getLogger(__name__)


class SkillExecutor:
    """
    Executes skills by converting them to AgentDefinitions and running
    them through an AgentSpawner.

    Usage:
        spawner = AgentSpawner(run_fn=my_run_fn)
        executor = SkillExecutor(spawner=spawner)
        result = await executor.execute(skill, {"query": "AI news"})
    """

    def __init__(self, spawner: Optional[AgentSpawner] = None) -> None:
        """
        Initialize the SkillExecutor.

        Args:
            spawner: An AgentSpawner to use for execution. If None, a
                     default spawner (using get_llm_service) is created.
        """
        self._spawner = spawner or AgentSpawner()

    async def execute(
        self,
        skill: Skill,
        args: Optional[Dict[str, str]] = None,
        context: Optional[str] = None,
    ) -> AgentResult:
        """
        Execute a skill with the given arguments.

        Renders the skill's prompt template with the provided args, wraps
        it in an AgentDefinition, and runs it through the spawner.

        Args:
            skill: The Skill to execute.
            args: Dictionary of argument values for template substitution.
            context: Optional context string to pass to the agent.

        Returns:
            AgentResult from the agent execution.
        """
        prompt = skill.get_prompt(args)
        agent_type = skill.agent or f"skill:{skill.name}"

        # Capture prompt in closure for get_system_prompt
        def get_system_prompt(p: str = prompt) -> str:
            return p

        definition = AgentDefinition(
            agent_type=agent_type,
            when_to_use=skill.when_to_use or skill.description,
            get_system_prompt=get_system_prompt,
            tools=skill.allowed_tools,
            model=skill.model,
            effort=skill.effort,
            source=skill.source,
        )

        logger.info(
            "Executing skill '%s' (agent_type=%s)",
            skill.name,
            agent_type,
        )
        return await self._spawner.spawn(
            definition, prompt=prompt, context=context
        )
