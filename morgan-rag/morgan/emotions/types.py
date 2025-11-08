"""
Core emotion types and data structures.

Defines the fundamental types used throughout the emotion detection system,
following the Plutchik emotion model with 8 basic emotions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set


class EmotionType(str, Enum):
    """
    Eight basic emotions from Plutchik's wheel of emotions.

    These fundamental emotions can combine to form complex emotional states.
    """

    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"

    @classmethod
    def all_emotions(cls) -> Set[EmotionType]:
        """Get all emotion types."""
        return {emotion for emotion in cls}

    def opposite(self) -> EmotionType:
        """Get the opposite emotion on Plutchik's wheel."""
        opposites = {
            self.JOY: self.SADNESS,
            self.SADNESS: self.JOY,
            self.ANGER: self.FEAR,
            self.FEAR: self.ANGER,
            self.TRUST: self.DISGUST,
            self.DISGUST: self.TRUST,
            self.SURPRISE: self.ANTICIPATION,
            self.ANTICIPATION: self.SURPRISE,
        }
        return opposites[self]


class EmotionIntensity(float):
    """
    Emotion intensity on a 0-1 scale.

    0.0 = Not present
    0.1-0.3 = Subtle
    0.4-0.6 = Moderate
    0.7-0.9 = Strong
    1.0 = Overwhelming
    """

    def __new__(cls, value: float) -> EmotionIntensity:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Intensity must be between 0 and 1, got {value}")
        return super().__new__(cls, value)

    @property
    def level(self) -> str:
        """Get descriptive intensity level."""
        if self == 0.0:
            return "absent"
        elif self < 0.3:
            return "subtle"
        elif self < 0.6:
            return "moderate"
        elif self < 0.9:
            return "strong"
        else:
            return "overwhelming"

    @property
    def is_significant(self) -> bool:
        """Check if intensity is significant enough to matter."""
        return self >= 0.3


@dataclass(frozen=True)
class Emotion:
    """A single detected emotion with its intensity."""

    emotion_type: EmotionType
    intensity: EmotionIntensity
    confidence: float  # 0-1, how confident we are in this detection

    def __post_init__(self) -> None:
        """Validate emotion data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

    @property
    def is_significant(self) -> bool:
        """Check if this emotion is significant enough to matter."""
        return self.intensity.is_significant and self.confidence >= 0.5

    def __str__(self) -> str:
        return (
            f"{self.emotion_type.value}:"
            f"{self.intensity:.2f}({self.intensity.level})"
        )


@dataclass
class EmotionTrigger:
    """A detected trigger for an emotional response."""

    trigger_text: str
    trigger_type: str  # keyword, phrase, pattern, context
    related_emotions: List[EmotionType]
    confidence: float
    position: int  # Character position in text

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass
class EmotionPattern:
    """A detected pattern in emotional expression."""

    pattern_type: str  # recurring, escalating, alternating, suppressed
    emotions_involved: List[EmotionType]
    frequency: int
    first_seen: datetime
    last_seen: datetime
    confidence: float
    description: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.last_seen < self.first_seen:
            raise ValueError("last_seen cannot be before first_seen")


@dataclass
class EmotionContext:
    """Contextual information for emotion analysis."""

    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message_index: int = 0
    previous_emotions: List[Emotion] = field(default_factory=list)
    time_since_last_message: Optional[float] = None  # seconds
    conversation_topic: Optional[str] = None

    @property
    def has_history(self) -> bool:
        """Check if we have emotional history for this user."""
        return len(self.previous_emotions) > 0


@dataclass
class EmotionResult:
    """
    Complete result of emotion detection analysis.

    This is the main output from the emotion detection system.
    """

    # Primary emotions detected (sorted by intensity)
    primary_emotions: List[Emotion]

    # Dominant emotion (highest intensity * confidence)
    dominant_emotion: Optional[Emotion]

    # Overall emotional valence: -1 (negative) to +1 (positive)
    valence: float

    # Overall emotional arousal: 0 (calm) to 1 (excited)
    arousal: float

    # Detected triggers
    triggers: List[EmotionTrigger] = field(default_factory=list)

    # Detected patterns
    patterns: List[EmotionPattern] = field(default_factory=list)

    # Context used for analysis
    context: Optional[EmotionContext] = None

    # Analysis timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Processing time in milliseconds
    processing_time_ms: float = 0.0

    # Any warnings or notes
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate result data."""
        if not -1.0 <= self.valence <= 1.0:
            raise ValueError(f"Valence must be between -1 and 1, got {self.valence}")
        if not 0.0 <= self.arousal <= 1.0:
            raise ValueError(f"Arousal must be between 0 and 1, got {self.arousal}")

    @property
    def is_crisis(self) -> bool:
        """
        Detect if this emotional state indicates a potential crisis.

        Crisis indicators:
        - Very high fear, sadness, or anger
        - Multiple intense negative emotions
        - Specific trigger patterns
        """
        if not self.primary_emotions:
            return False

        crisis_emotions = {EmotionType.FEAR, EmotionType.SADNESS, EmotionType.ANGER}
        intense_negative = [
            e for e in self.primary_emotions
            if e.emotion_type in crisis_emotions and e.intensity >= 0.8
        ]

        return len(intense_negative) >= 2 or any(
            e.intensity >= 0.9 for e in intense_negative
        )

    @property
    def emotional_summary(self) -> str:
        """Get a human-readable summary of the emotional state."""
        if not self.primary_emotions:
            return "neutral"

        if self.dominant_emotion:
            summary = f"{self.dominant_emotion.emotion_type.value}"
            if len(self.primary_emotions) > 1:
                others = [
                    e.emotion_type.value
                    for e in self.primary_emotions[1:3]
                    if e.is_significant
                ]
                if others:
                    summary += f" with {', '.join(others)}"
            return summary

        return "mixed emotions"

    def get_emotions_dict(self) -> Dict[str, float]:
        """Get emotions as a dictionary mapping type to intensity."""
        return {
            emotion.emotion_type.value: float(emotion.intensity)
            for emotion in self.primary_emotions
        }
