"""
Skill registry.

Provides a central registry for discovering and retrieving skills by name.
"""

import logging
from typing import Dict, List, Optional

from morgan.skills.loader import Skill

logger = logging.getLogger(__name__)


class SkillRegistry:
    """
    Central registry for skill definitions.

    Manages a collection of skills indexed by name. Supports registration,
    lookup, and listing (all or user-invocable only).

    Usage:
        registry = SkillRegistry()
        registry.register(my_skill)
        skill = registry.get("web_research")
        all_skills = registry.list_all()
        user_skills = registry.list_user_invocable()
    """

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """
        Register a skill in the registry.

        If a skill with the same name already exists, it is overwritten and
        a warning is logged.

        Args:
            skill: The Skill instance to register.
        """
        if skill.name in self._skills:
            logger.warning(
                "Overwriting existing skill '%s' (source: %s) with new "
                "definition (source: %s)",
                skill.name,
                self._skills[skill.name].source,
                skill.source,
            )
        self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        """
        Retrieve a skill by name.

        Args:
            name: The unique name of the skill.

        Returns:
            The Skill instance, or None if not found.
        """
        return self._skills.get(name)

    def list_all(self) -> List[Skill]:
        """
        Return all registered skills.

        Returns:
            List of all Skill instances, sorted by name.
        """
        return sorted(self._skills.values(), key=lambda s: s.name)

    def list_user_invocable(self) -> List[Skill]:
        """
        Return only skills that are marked as user-invocable.

        Returns:
            List of user-invocable Skill instances, sorted by name.
        """
        return sorted(
            (s for s in self._skills.values() if s.user_invocable),
            key=lambda s: s.name,
        )
