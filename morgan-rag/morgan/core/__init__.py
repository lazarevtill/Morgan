"""Core RAG and assistant functionality."""

# Main assistant
from morgan.core.assistant import MorganAssistant, AssistantError

# Memory system
from morgan.core.memory import MemorySystem, MemoryError, MemoryRetrievalError, MemoryStorageError

# Context management
from morgan.core.context import ContextManager, ContextError, ContextOverflowError

# Response generation
from morgan.core.response_generator import ResponseGenerator, GenerationError, ValidationError

# Search/RAG
from morgan.core.search import (
    MultiStageSearch,
    SearchConfig,
    SearchResult,
    SearchMetrics,
    SearchGranularity,
    ReciprocalRankFusion,
)

# Types
from morgan.core.types import (
    # Message types
    Message,
    MessageRole,

    # Context types
    ConversationContext,
    EmotionalState,
    UserProfile,

    # Response types
    AssistantResponse,
    SearchSource,

    # Memory types
    MemoryEntry,
    MemoryType,

    # Strategy types
    ContextPruningStrategy,

    # Metrics
    AssistantMetrics,
    ProcessingContext,
)

__all__ = [
    # Main assistant
    "MorganAssistant",
    "AssistantError",

    # Memory
    "MemorySystem",
    "MemoryError",
    "MemoryRetrievalError",
    "MemoryStorageError",

    # Context
    "ContextManager",
    "ContextError",
    "ContextOverflowError",

    # Response generation
    "ResponseGenerator",
    "GenerationError",
    "ValidationError",

    # Search
    "MultiStageSearch",
    "SearchConfig",
    "SearchResult",
    "SearchMetrics",
    "SearchGranularity",
    "ReciprocalRankFusion",

    # Types - Messages
    "Message",
    "MessageRole",

    # Types - Context
    "ConversationContext",
    "EmotionalState",
    "UserProfile",

    # Types - Response
    "AssistantResponse",
    "SearchSource",

    # Types - Memory
    "MemoryEntry",
    "MemoryType",

    # Types - Strategy
    "ContextPruningStrategy",

    # Types - Metrics
    "AssistantMetrics",
    "ProcessingContext",
]
