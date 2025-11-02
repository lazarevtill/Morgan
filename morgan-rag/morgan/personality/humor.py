"""
Humor Detection and Generation for Morgan RAG.

Detects user humor preferences and generates appropriate humorous responses
based on personality traits, context, and relationship dynamics.
"""

import uuid
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logger import get_logger
from ..emotional.models import (
    ConversationContext, EmotionalState, EmotionType
)
from .traits import PersonalityProfile, PersonalityTrait, TraitLevel

logger = get_logger(__name__)


class HumorStyle(Enum):
    """Types of humor styles."""
    WITTY = "witty"  # Clever wordplay and observations
    PLAYFUL = "playful"  # Light-hearted and fun
    DRY = "dry"  # Deadpan and understated
    SELF_DEPRECATING = "self_deprecating"  # Self-referential humor
    OBSERVATIONAL = "observational"  # Commentary on everyday situations
    PUNNY = "punny"  # Puns and wordplay
    GENTLE = "gentle"  # Kind and non-offensive
    NONE = "none"  # No humor preferred


class HumorTiming(Enum):
    """When to use humor."""
    OPENING = "opening"  # At conversation start
    TRANSITION = "transition"  # Between topics
    EXPLANATION = "explanation"  # During explanations
    ENCOURAGEMENT = "encouragement"  # When providing support
    CLOSING = "closing"  # At conversation end
    NEVER = "never"  # User doesn't appreciate humor


@dataclass
class HumorPreference:
    """User's humor preferences."""
    preferred_styles: List[HumorStyle] = field(default_factory=list)
    avoided_styles: List[HumorStyle] = field(default_factory=list)
    preferred_timing: List[HumorTiming] = field(default_factory=list)
    humor_frequency: float = 0.3  # 0.0 to 1.0
    appropriateness_threshold: float = 0.7  # 0.0 to 1.0
    confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HumorAttempt:
    """Record of a humor attempt and its reception."""
    attempt_id: str
    user_id: str
    humor_style: HumorStyle
    content: str
    context: str
    user_reaction: Optional[str] = None
    success_score: Optional[float] = None  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Initialize attempt ID if not provided."""
        if not self.attempt_id:
            self.attempt_id = str(uuid.uuid4())


@dataclass
class HumorSuggestion:
    """Suggestion for humorous content."""
    suggestion_id: str
    humor_style: HumorStyle
    content: str
    appropriateness_score: float  # 0.0 to 1.0
    timing: HumorTiming
    reasoning: str
    context_factors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize suggestion ID if not provided."""
        if not self.suggestion_id:
            self.suggestion_id = str(uuid.uuid4())


class HumorDetector:
    """
    Detects user humor preferences and reactions.
    
    Analyzes user messages and responses to identify humor preferences,
    timing preferences, and reaction patterns.
    """
    
    # Humor indicators in user messages
    HUMOR_INDICATORS = {
        HumorStyle.WITTY: [
            r'\b(clever|smart|brilliant|genius)\b',
            r'\b(ironic|irony|sarcastic|sarcasm)\b',
            r'[😏😉🤓]'
        ],
        HumorStyle.PLAYFUL: [
            r'\b(fun|funny|hilarious|amusing|entertaining)\b',
            r'\b(silly|goofy|playful|cheerful)\b',
            r'[😄😆😂🤣😊]'
        ],
        HumorStyle.DRY: [
            r'\b(dry|deadpan|understated|subtle)\b',
            r'\b(matter.of.fact|straightforward)\b',
            r'[😐😑]'
        ],
        HumorStyle.PUNNY: [
            r'\b(pun|wordplay|play.on.words)\b',
            r'[🙄😅]'
        ],
        HumorStyle.GENTLE: [
            r'\b(sweet|kind|gentle|wholesome)\b',
            r'\b(nice|pleasant|friendly)\b',
            r'[😌☺️🙂]'
        ]
    }
    
    # Positive reaction indicators
    POSITIVE_REACTIONS = [
        r'\b(haha|lol|lmao|rofl)\b',
        r'\b(funny|hilarious|amusing|clever)\b',
        r'\b(love|like|enjoy|appreciate)\b.*\b(humor|joke|funny)\b',
        r'[😂🤣😄😆😊😁]'
    ]
    
    # Negative reaction indicators
    NEGATIVE_REACTIONS = [
        r'\b(not funny|unfunny|inappropriate)\b',
        r'\b(serious|professional|formal)\b.*\b(please|prefer)\b',
        r'\b(stop|enough|no more)\b.*\b(joke|humor|funny)\b',
        r'[😐😑🙄😒]'
    ]
    
    def __init__(self):
        """Initialize humor detector."""
        logger.info("Humor detector initialized")
    
    def analyze_humor_preferences(
        self,
        user_id: str,
        conversation_history: List[str],
        user_reactions: List[str],
        personality_profile: Optional[PersonalityProfile] = None
    ) -> HumorPreference:
        """
        Analyze user humor preferences from conversation history.
        
        Args:
            user_id: User identifier
            conversation_history: List of user messages
            user_reactions: List of user reactions to humo