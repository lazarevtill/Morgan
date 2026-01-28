"""
Domain entities for Morgan Core.
Following DDD principles to separate domain logic from infrastructure and application layers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class KnowledgeChunk:
    """
    A piece of knowledge with metadata.
    """

    content: str
    source: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    ingested_at: datetime = field(default_factory=datetime.utcnow)
    embedding_type: str = "legacy"  # "legacy" or "hierarchical"


@dataclass
class ConversationTurn:
    """
    A single turn in a conversation.
    """

    turn_id: str
    conversation_id: str
    timestamp: datetime
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    feedback_rating: Optional[int] = None
    feedback_comment: Optional[str] = None
    emotional_tone: Optional[str] = None
    empathy_level: float = 0.0


@dataclass
class Conversation:
    """
    A complete conversation between human and Morgan.
    """

    conversation_id: str
    started_at: datetime
    topic: Optional[str] = None
    turns: List[ConversationTurn] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    user_id: Optional[str] = None

    @property
    def total_turns(self) -> int:
        return len(self.turns)

    @property
    def average_rating(self) -> float:
        ratings = [
            turn.feedback_rating
            for turn in self.turns
            if turn.feedback_rating is not None
        ]
        return sum(ratings) / len(ratings) if ratings else 0.0
