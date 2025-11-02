"""
Emotional feedback processing module.

Processes user feedback to improve emotional intelligence and communication
effectiveness through sentiment analysis, satisfaction tracking, and
adaptive response refinement.
"""

import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import json

from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.services.llm_service import get_llm_service
from morgan.utils.cache import FileCache
from morgan.emotional.models import (
    EmotionalState, ConversationContext, EmpatheticResponse
)

logger = get_logger(__name__)


class FeedbackType(Enum):
    """Types of feedback that can be processed."""
    EXPLICIT_RATING = "explicit_rating"
    IMPLICIT_BEHAVIORAL = "implicit_behavioral"
    TEXTUAL_FEEDBACK = "textual_feedback"
    EMOTIONAL_RESPONSE = "emotional_response"
    ENGAGEMENT_METRICS = "engagement_metrics"


class FeedbackSentiment(Enum):
    """Sentiment of feedback."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class FeedbackAnalysis:
    """Analysis result of user feedback."""
    feedback_type: FeedbackType
    sentiment: FeedbackSentiment
    satisfaction_score: float  # 0.0 to 1.0
    specific_aspects: Dict[str, float]  # Aspect -> score mapping
    improvement_suggestions: List[str]
    confidence_score: float
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FeedbackPattern:
    """Pattern detected in user feedback over time."""
    pattern_type: str
    description: str
    frequency: int
    trend: str  # improving, declining, stable
    confidence: float
    examples: List[str]


class EmotionalFeedbackProcessor:
    """
    Emotional feedback processing system.
    
    Features:
    - Multi-modal feedback analysis (explicit, implicit, textual)
    - Sentiment analysis and satisfaction scoring
    - Feedback pattern recognition and trend analysis
    - Adaptive response improvement based on feedback
    - Emotional intelligence refinement through feedback loops
    """
    
    def __init__(self):
        """Initialize emotional feedback processor."""
        self.settings = get_settings()
        self.llm_service = get_llm_service()
        
        # Setup cache for feedback data
        cache_dir = se