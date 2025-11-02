"""
Domain Entities - Core Business Objects
========================================

Entities are objects that have identity and lifecycle.
They are mutable and can change state over time.

Characteristics:
    - Have unique identity (ID)
    - Mutable state
    - Contain business logic
    - Validate their own invariants
"""

from morgan_v2.domain.entities.conversation import Conversation, ConversationTurn, Message
from morgan_v2.domain.entities.user import User, UserProfile, UserPreferences
from morgan_v2.domain.entities.knowledge import Document, DocumentChunk, Source
from morgan_v2.domain.entities.emotion import EmotionalState, EmotionalContext, MoodHistory
from morgan_v2.domain.entities.relationship import CompanionProfile, RelationshipMilestone
from morgan_v2.domain.entities.memory import Memory, MemoryContext, MemoryCluster

__all__ = [
    # Conversation
    "Conversation",
    "ConversationTurn",
    "Message",
    # User
    "User",
    "UserProfile",
    "UserPreferences",
    # Knowledge
    "Document",
    "DocumentChunk",
    "Source",
    # Emotion
    "EmotionalState",
    "EmotionalContext",
    "MoodHistory",
    # Relationship
    "CompanionProfile",
    "RelationshipMilestone",
    # Memory
    "Memory",
    "MemoryContext",
    "MemoryCluster",
]
