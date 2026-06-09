"""Personalization contract — request-path adaptation. Reads UserModel + FusedPerception,
selects the traits relevant to *this* turn (budget-aware), and produces context for Reasoning.
Stateless: it writes nothing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from morgan_brain.models.perception import FusedPerception
from morgan_brain.models.user import UserModel


class PersonalizedContext(BaseModel):
    """The system-prompt fragment + signals injected for one turn (<= budget of context window)."""

    system_fragment: str = ""
    selected_traits: list[str] = Field(default_factory=list)
    tone: str = "neutral"
    proactive_threshold: float = 1.0  # higher = less likely to volunteer suggestions


@runtime_checkable
class Personalizer(Protocol):
    async def build(
        self, *, user_model: UserModel, perception: FusedPerception
    ) -> PersonalizedContext: ...
