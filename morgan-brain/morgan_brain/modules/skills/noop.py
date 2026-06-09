"""Phase 1 SkillEngine: selects nothing. Real skill selection + SkillOpt arrive in Phase 3."""

from __future__ import annotations

from morgan_brain.interfaces.skills import Skill
from morgan_brain.models.perception import FusedPerception


class NoopSkillEngine:
    async def select(self, perception: FusedPerception) -> list[Skill]:
        return []

    async def get(self, name: str) -> Skill | None:
        return None

    async def deploy(self, skill: Skill) -> None:
        return None
