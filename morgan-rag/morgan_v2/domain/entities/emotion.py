"""
Emotion Domain Entities
=======================

Core business objects for emotional intelligence.
Pure domain logic with no external dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from uuid import uuid4


class EmotionType(str, Enum):
    """Core emotion types based on psychological research"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


class IntensityLevel(str, Enum):
    """Emotional intensity levels"""
    VERY_LOW = "very_low"      # 0.0-0.2
    LOW = "low"                # 0.2-0.4
    MODERATE = "moderate"      # 0.4-0.6
    HIGH = "high"              # 0.6-0.8
    VERY_HIGH = "very_high"    # 0.8-1.0


@dataclass
class EmotionalState:
    """
    Represents a user's emotional state at a point in time.

    Business Rules:
        - Intensity must be between 0.0 and 1.0
        - Confidence must be between 0.0 and 1.0
        - Primary emotion is required
        - Secondary emotions are optional

    Example:
        >>> state = EmotionalState(
        ...     primary_emotion=EmotionType.JOY,
        ...     intensity=0.85,
        ...     confidence=0.92
        ... )
        >>> state.is_strong_emotion()
        True
    """
    primary_emotion: EmotionType
    intensity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    secondary_emotions: List[EmotionType] = field(default_factory=list)
    emotional_indicators: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    context: Optional[str] = None

    def __post_init__(self):
        """Validate business invariants"""
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"Intensity must be between 0 and 1, got {self.intensity}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

    @property
    def intensity_level(self) -> IntensityLevel:
        """Convert numeric intensity to categorical level"""
        if self.intensity < 0.2:
            return IntensityLevel.VERY_LOW
        elif self.intensity < 0.4:
            return IntensityLevel.LOW
        elif self.intensity < 0.6:
            return IntensityLevel.MODERATE
        elif self.intensity < 0.8:
            return IntensityLevel.HIGH
        else:
            return IntensityLevel.VERY_HIGH

    def is_strong_emotion(self, threshold: float = 0.6) -> bool:
        """Check if emotion is strong enough to act upon"""
        return self.intensity >= threshold and self.confidence >= 0.5

    def is_reliable(self, confidence_threshold: float = 0.7) -> bool:
        """Check if emotional detection is reliable"""
        return self.confidence >= confidence_threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "primary_emotion": self.primary_emotion.value,
            "intensity": self.intensity,
            "intensity_level": self.intensity_level.value,
            "confidence": self.confidence,
            "secondary_emotions": [e.value for e in self.secondary_emotions],
            "emotional_indicators": self.emotional_indicators,
            "detected_at": self.detected_at.isoformat(),
            "context": self.context,
        }


@dataclass
class EmotionalContext:
    """
    Rich context around an emotional state.

    Includes triggers, patterns, and historical context.
    """
    current_state: EmotionalState
    triggers: List[str] = field(default_factory=list)
    historical_pattern: Optional[str] = None
    mood_trend: Optional[str] = None  # "improving", "declining", "stable"
    duration: Optional[float] = None  # How long in this state (seconds)

    def is_mood_shift(self, previous_state: EmotionalState) -> bool:
        """Detect if there's been a significant mood shift"""
        if previous_state.primary_emotion != self.current_state.primary_emotion:
            return True

        intensity_change = abs(
            self.current_state.intensity - previous_state.intensity
        )
        return intensity_change > 0.3

    def get_empathy_level(self) -> float:
        """
        Calculate appropriate empathy level for response.

        Business Rule: Stronger emotions need higher empathy.
        """
        base_empathy = self.current_state.intensity

        # Increase empathy for negative emotions
        if self.current_state.primary_emotion in [
            EmotionType.SADNESS,
            EmotionType.ANGER,
            EmotionType.FEAR,
        ]:
            base_empathy *= 1.2

        # Cap at 1.0
        return min(base_empathy, 1.0)


@dataclass
class MoodHistory:
    """
    Track emotional states over time for pattern recognition.

    Business Logic:
        - Detect mood patterns
        - Identify triggers
        - Track emotional recovery
    """
    user_id: str
    states: List[EmotionalState] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    id: str = field(default_factory=lambda: str(uuid4()))

    def add_state(self, state: EmotionalState) -> None:
        """Add a new emotional state to history"""
        self.states.append(state)

        # Keep only last 100 states for memory efficiency
        if len(self.states) > 100:
            self.states = self.states[-100:]

    def get_dominant_emotion(self, hours: int = 24) -> Optional[EmotionType]:
        """Get the most common emotion in recent history"""
        cutoff = datetime.utcnow().timestamp() - (hours * 3600)
        recent_states = [
            s for s in self.states
            if s.detected_at.timestamp() > cutoff
        ]

        if not recent_states:
            return None

        # Count emotion occurrences
        emotion_counts: Dict[EmotionType, int] = {}
        for state in recent_states:
            emotion_counts[state.primary_emotion] = (
                emotion_counts.get(state.primary_emotion, 0) + 1
            )

        return max(emotion_counts, key=emotion_counts.get)

    def get_mood_trend(self, hours: int = 24) -> str:
        """
        Analyze mood trend over time.

        Returns:
            "improving", "declining", or "stable"
        """
        cutoff = datetime.utcnow().timestamp() - (hours * 3600)
        recent_states = [
            s for s in self.states
            if s.detected_at.timestamp() > cutoff
        ]

        if len(recent_states) < 3:
            return "stable"

        # Calculate average intensity for first and second half
        mid = len(recent_states) // 2
        first_half_avg = sum(s.intensity for s in recent_states[:mid]) / mid
        second_half_avg = sum(s.intensity for s in recent_states[mid:]) / (len(recent_states) - mid)

        # Check for positive vs negative emotions
        first_half_positive = sum(
            1 for s in recent_states[:mid]
            if s.primary_emotion in [EmotionType.JOY, EmotionType.SURPRISE]
        )
        second_half_positive = sum(
            1 for s in recent_states[mid:]
            if s.primary_emotion in [EmotionType.JOY, EmotionType.SURPRISE]
        )

        # Trend based on positive emotion ratio
        first_ratio = first_half_positive / mid if mid > 0 else 0
        second_ratio = second_half_positive / (len(recent_states) - mid) if len(recent_states) > mid else 0

        if second_ratio > first_ratio + 0.2:
            return "improving"
        elif second_ratio < first_ratio - 0.2:
            return "declining"
        else:
            return "stable"

    def get_average_intensity(self, hours: int = 24) -> float:
        """Calculate average emotional intensity over time"""
        cutoff = datetime.utcnow().timestamp() - (hours * 3600)
        recent_states = [
            s for s in self.states
            if s.detected_at.timestamp() > cutoff
        ]

        if not recent_states:
            return 0.0

        return sum(s.intensity for s in recent_states) / len(recent_states)
