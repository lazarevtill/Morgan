"""
Domain Layer - Pure Business Logic
===================================

This layer contains the core business logic and rules.
It has NO dependencies on external frameworks or libraries.

Rules:
    - No imports from other layers (application, infrastructure, interfaces)
    - Only Python standard library + dataclasses/Pydantic
    - Framework-agnostic
    - All business rules live here

Components:
    entities/        - Core business objects (mutable)
    value_objects/   - Immutable domain values
    repositories/    - Abstract interfaces for data access
    services/        - Domain services (business logic)
    events/          - Domain events
"""

from morgan_v2.domain.entities import (
    Conversation,
    ConversationTurn,
    User,
    UserProfile,
    Document,
    DocumentChunk,
    EmotionalState,
    CompanionProfile,
    Memory,
)

from morgan_v2.domain.value_objects import (
    EmotionType,
    IntensityLevel,
    CommunicationStyle,
    SearchQuery,
    SearchResult,
    Embedding,
)

__all__ = [
    # Entities
    "Conversation",
    "ConversationTurn",
    "User",
    "UserProfile",
    "Document",
    "DocumentChunk",
    "EmotionalState",
    "CompanionProfile",
    "Memory",
    # Value Objects
    "EmotionType",
    "IntensityLevel",
    "CommunicationStyle",
    "SearchQuery",
    "SearchResult",
    "Embedding",
]
