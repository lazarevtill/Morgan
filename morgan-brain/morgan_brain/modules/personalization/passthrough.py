"""Phase 1 Personalizer: renders the user's communication preferences into a short system
fragment. No trait selection yet (Phase 2) — it simply surfaces what little the UserModel holds."""

from __future__ import annotations

from morgan_brain.interfaces.personalization import PersonalizedContext
from morgan_brain.models.perception import FusedPerception
from morgan_brain.models.user import UserModel


class PassthroughPersonalizer:
    async def build(
        self, *, user_model: UserModel, perception: FusedPerception
    ) -> PersonalizedContext:
        prefs = user_model.comm_prefs
        bits = []
        if prefs.length != "balanced":
            bits.append(f"prefers {prefs.length} replies")
        if prefs.tone != "neutral":
            bits.append(f"tone: {prefs.tone}")
        if prefs.formality != "neutral":
            bits.append(f"formality: {prefs.formality}")
        if prefs.code_vs_prose != "balanced":
            bits.append(prefs.code_vs_prose.replace("_", " "))
        fragment = "; ".join(bits)
        return PersonalizedContext(system_fragment=fragment, tone=prefs.tone)
