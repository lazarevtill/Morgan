"""Skills contract — procedural memory as trainable markdown (SkillOpt).

A skill is a markdown doc with YAML frontmatter. The engine selects skills by trigger match and
injects the active ``best_skill.md`` into the reasoning context. Training happens offline in the
learning-worker behind a validation gate; the request path only ever *reads* skills.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from morgan_brain.models.perception import FusedPerception


class Skill(BaseModel):
    name: str
    version: int = 1
    triggers: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    model: str | None = None
    body: str = ""  # the markdown injected into context


@runtime_checkable
class SkillEngine(Protocol):
    async def select(self, perception: FusedPerception) -> list[Skill]:
        """Return skills whose triggers match the current intent."""
        ...

    async def get(self, name: str) -> Skill | None: ...

    async def deploy(self, skill: Skill) -> None:
        """Install/replace a (validated) skill version. Called by the learning-worker only."""
        ...
