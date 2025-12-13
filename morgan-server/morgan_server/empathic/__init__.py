"""
Empathic Engine for Morgan.

This module provides emotional intelligence, personality, and relationship management
to make Morgan feel human and emotionally aware.
"""

from morgan_server.empathic.emotional import (
    EmotionalTone,
    EmotionalDetection,
    EmotionalPattern,
    EmotionalAdjustment,
    EmotionalIntelligence,
)

from morgan_server.empathic.personality import (
    PersonalityTrait,
    ConversationalStyle,
    PersonalityConfig,
    PersonalityApplication,
    PersonalitySystem,
)

from morgan_server.empathic.roleplay import (
    RoleplayTone,
    ResponseStyle,
    RoleplayConfig,
    RoleplayContext,
    RoleplayResponse,
    RoleplaySystem,
)

__all__ = [
    # Emotional Intelligence
    "EmotionalTone",
    "EmotionalDetection",
    "EmotionalPattern",
    "EmotionalAdjustment",
    "EmotionalIntelligence",
    # Personality System
    "PersonalityTrait",
    "ConversationalStyle",
    "PersonalityConfig",
    "PersonalityApplication",
    "PersonalitySystem",
    # Roleplay System
    "RoleplayTone",
    "ResponseStyle",
    "RoleplayConfig",
    "RoleplayContext",
    "RoleplayResponse",
    "RoleplaySystem",
]
