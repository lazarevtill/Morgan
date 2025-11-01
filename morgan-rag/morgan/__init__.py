"""
Morgan RAG - Human-First AI Assistant

A modular AI assistant with advanced emotional intelligence using KISS
principles. Clean architecture with focused single-responsibility modules.

Modular Architecture:
- models/: Unified model management (local + remote)
- storage/: Unified data persistence
- config/: Centralized configuration management
- utils/: Shared utilities following DRY principles
- core/: Core orchestration and assistant logic
- emotional/: Emotional intelligence engine
- companion/: Relationship management
- memory/: Conversation memory processing
- search/: Multi-stage search engine
- vectorization/: Hierarchical embeddings
- jina/: Jina AI integration
- background/: Background processing

Requirements addressed: 23.1, 23.2, 23.3, 23.4, 23.5
"""

__version__ = "1.0.0"
__author__ = "Morgan RAG Team"

# Core components - simple imports
from morgan.core.assistant import MorganAssistant
from morgan.core.knowledge import KnowledgeBase
from morgan.core.memory import ConversationMemory
from morgan.core.search import SmartSearch


def create_assistant(config_path: str = None) -> MorganAssistant:
    """
    Create a Morgan assistant instance.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Ready-to-use Morgan assistant

    Example:
        >>> morgan = create_assistant()
        >>> response = morgan.ask("How do I deploy Docker?")
        >>> print(response.answer)
    """
    return MorganAssistant(config_path=config_path)


# Simple exports for easy usage
__all__ = [
    "MorganAssistant",
    "KnowledgeBase",
    "ConversationMemory",
    "SmartSearch",
    "create_assistant"
]
