"""
Non-verbal cue detection module.

Detects and interprets non-verbal communication cues from text patterns,
timing, and behavioral indicators to enhance emotional understanding
and communication effectiveness.
"""

import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re

from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.emotional.models import (
    EmotionalState, ConversationContext
)

logger = get_logger(__name__)


class NonVerbalCueType(Enum):
    """Types of non-verbal cues that can be detected."""
    PUNCTUATION_EMPHASIS = "punctuation_emphasis"
    CAPITALIZATION = "capitalization"
    REPETITION = "repetition"
    SPACING_PATTERNS = "spacing_patterns"
    TIMING_PATTERNS = "timing_patterns"
    MESSAGE_LENGTH = "message_length"
    EMOJI_USAGE = "emoji_usage"
    TYPING_PATTERNS = "typing_patterns"


class EmotionalIntensity(Enum):
    """Emotional intensity levels detected from non-verbal cues."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class NonVerbalCue:
    """A detected non-verbal communication cue."""
    cue_type: NonVerbalCueType
    intensity: EmotionalIntensity
    confidence: float
    description: str
    indicators: List[str]
    emotional_implication: str
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class NonVerbalAnalysis:
    """Complete analysis of non-verbal cues in communication."""
    detected_cues: List[NonVerbalCue]
    overall_intensity: EmotionalIntensity
    emotional_state_indicators: Dict[str, float]
    communication_urgency: float
    engagement_level: float
    confidence_score: float
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)


class NonVerbalCueDetector:
    """
    Non-verbal cue detection system.
    
    Features:
    - Text pattern analysis for emotional indicators
    - Punctuation and capitalization pattern detection
    - Timing pattern analysis for urgency and engagement
    - Repetition and emphasis detection
    - Emoji and emoticon interpretation
    - Message structure and length analysis
    - Behavioral pattern recognition
    """
    
    def __init__(self):
        """Initialize non-verbal cue detector."""
        self.settings = get_settings()
        
        # Pattern storage for learning
        self.user_patterns: Dict[str, List[NonVerbalAnalysis]] = {}
        
        # Emoji emotion mapping
        self.emoji_emotions = {
            "😊": ("joy", 0.7), "😄": ("joy", 0.9), "😃": ("joy", 0.8),
            "😢": ("sadness", 0.8), "😭": ("sadness", 0.9), "😔": ("sadness", 0.6),
            "😠": ("anger", 0.8), "😡": (