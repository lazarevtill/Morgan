"""
Core assistant types and data structures.

Defines fundamental types used throughout the assistant system,
following immutable dataclass patterns for thread safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set

from morgan.emotions.types import EmotionResult
from morgan.learning.types import LearningPattern, UserPreference


class MessageRole(str, Enum):
    """Message role in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MemoryType(str, Enum):
    """Types of memory storage."""

    SHORT_TERM = "short_term"  # Current conversation
    WORKING = "working"  # Currently processing
    LONG_TERM = "long_term"  # Historical conversations
    CONSOLIDATED = "consolidated"  # Important patterns


class ContextPruningStrategy(str, Enum):
    """Strategies for pruning conversation context."""

    SLIDING_WINDOW = "sliding_window"  # Keep last N messages
    IMPORTANCE_BASED = "importance_based"  # Keep most important
    RECENCY_WEIGHTED = "recency_weighted"  # Weight by recency
    HYBRID = "hybrid"  # Combination


@dataclass(frozen=True)
class Message:
    """A single message in a conversation."""

    role: MessageRole
    content: str
    timestamp: datetime
    message_id: str

    # Optional enrichments
    emotion: Optional[EmotionResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Context tracking
    tokens: Optional[int] = None
    importance_score: float = 1.0

    def __post_init__(self) -> None:
        """Validate message data."""
        if not self.content.strip():
            raise ValueError("Message content cannot be empty")
        if not 0.0 <= self.importance_score <= 1.0:
            raise ValueError(f"Importance score must be 0-1, got {self.importance_score}")


@dataclass(frozen=True)
class EmotionalState:
    """Current emotional state of a user."""

    user_id: str
    timestamp: datetime

    # Current emotional context
    current_emotions: Optional[EmotionResult] = None
    emotion_trajectory: List[EmotionResult] = field(default_factory=list)

    # Emotional patterns
    detected_patterns: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UserProfile:
    """User profile with preferences and patterns."""

    user_id: str
    created_at: datetime
    last_active: datetime

    # Preferences
    preferences: List[UserPreference] = field(default_factory=list)

    # Learning patterns
    patterns: List[LearningPattern] = field(default_factory=list)

    # Context
    conversation_count: int = 0
    total_messages: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationContext:
    """Complete conversation context."""

    messages: List[Message]
    user_id: str
    session_id: str

    # User context
    user_profile: Optional[UserProfile] = None
    emotional_state: Optional[EmotionalState] = None

    # Conversation metadata
    total_tokens: int = 0
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        """Get number of messages in context."""
        return len(self.messages)

    @property
    def user_messages(self) -> List[Message]:
        """Get only user messages."""
        return [m for m in self.messages if m.role == MessageRole.USER]

    @property
    def assistant_messages(self) -> List[Message]:
        """Get only assistant messages."""
        return [m for m in self.messages if m.role == MessageRole.ASSISTANT]


@dataclass(frozen=True)
class SearchSource:
    """A source document from RAG search."""

    content: str
    source: str
    score: float
    chunk_id: str
    document_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssistantResponse:
    """Response from the assistant."""

    content: str
    response_id: str
    timestamp: datetime

    # Enrichments
    sources: List[SearchSource] = field(default_factory=list)
    emotion: Optional[EmotionResult] = None

    # Metrics
    confidence: float = 1.0
    generation_time_ms: float = 0.0
    total_tokens: Optional[int] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate response data."""
        if not self.content.strip():
            raise ValueError("Response content cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")


@dataclass(frozen=True)
class MemoryEntry:
    """An entry in the memory system."""

    entry_id: str
    user_id: str
    session_id: str
    memory_type: MemoryType

    # Content
    message: Message

    # Indexing
    created_at: datetime
    expires_at: Optional[datetime] = None

    # Metadata
    importance_score: float = 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantMetrics:
    """Performance metrics for assistant operations."""

    correlation_id: str
    operation: str
    started_at: datetime

    # Timing breakdown (in milliseconds)
    emotion_detection_ms: float = 0.0
    memory_retrieval_ms: float = 0.0
    rag_search_ms: float = 0.0
    context_building_ms: float = 0.0
    response_generation_ms: float = 0.0
    learning_update_ms: float = 0.0
    total_duration_ms: float = 0.0

    # Counts
    messages_retrieved: int = 0
    rag_sources_found: int = 0

    # Flags
    used_cache: bool = False
    degraded_mode: bool = False

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "correlation_id": self.correlation_id,
            "operation": self.operation,
            "timing": {
                "emotion_detection_ms": round(self.emotion_detection_ms, 2),
                "memory_retrieval_ms": round(self.memory_retrieval_ms, 2),
                "rag_search_ms": round(self.rag_search_ms, 2),
                "context_building_ms": round(self.context_building_ms, 2),
                "response_generation_ms": round(self.response_generation_ms, 2),
                "learning_update_ms": round(self.learning_update_ms, 2),
                "total_duration_ms": round(self.total_duration_ms, 2),
            },
            "counts": {
                "messages_retrieved": self.messages_retrieved,
                "rag_sources_found": self.rag_sources_found,
            },
            "flags": {
                "used_cache": self.used_cache,
                "degraded_mode": self.degraded_mode,
            },
            "metadata": self.metadata,
        }


@dataclass
class ProcessingContext:
    """Context for processing a single request."""

    correlation_id: str
    user_id: str
    session_id: str
    message: str
    timestamp: datetime

    # Gathered data during processing
    detected_emotion: Optional[EmotionResult] = None
    retrieved_memories: List[Message] = field(default_factory=list)
    rag_sources: List[SearchSource] = field(default_factory=list)
    conversation_context: Optional[ConversationContext] = None
    user_profile: Optional[UserProfile] = None
    emotional_state: Optional[EmotionalState] = None

    # Metrics
    metrics: Optional[AssistantMetrics] = None
