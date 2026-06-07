"""Shared domain models. Everything that persists is keyed by ``user_id``."""

from morgan_brain.models.base import Entity, Identified, UserScoped
from morgan_brain.models.emotion import EmotionState, SentimentScore
from morgan_brain.models.memory import Memory, MemoryQuery, MemorySource, TemporalFact
from morgan_brain.models.message import Conversation, Message, Role
from morgan_brain.models.perception import FusedPerception, Intent, Modality
from morgan_brain.models.user import CommunicationPrefs, RelationshipStage, Trait, UserModel

__all__ = [
    "Identified", "UserScoped", "Entity",
    "EmotionState", "SentimentScore",
    "Memory", "MemoryQuery", "MemorySource", "TemporalFact",
    "Message", "Conversation", "Role",
    "FusedPerception", "Intent", "Modality",
    "UserModel", "Trait", "CommunicationPrefs", "RelationshipStage",
]
