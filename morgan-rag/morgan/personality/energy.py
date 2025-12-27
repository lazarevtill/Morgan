"""
Energy Level Matching for Morgan RAG.

Matches communication energy levels with user's current energy state
based on emotional indicators, time patterns, and personality traits.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from ..emotional.models import ConversationContext, EmotionalState
from ..utils.logger import get_logger
from .traits import PersonalityProfile, PersonalityTrait, TraitLevel

logger = get_logger(__name__)


class EnergyLevel(Enum):
    """Levels of communication energy."""

    VERY_LOW = 1  # Calm, subdued, minimal enthusiasm
    LOW = 2  # Relaxed, gentle, understated
    MODERATE = 3  # Balanced, steady, neutral energy
    HIGH = 4  # Enthusiastic, animated, energetic
    VERY_HIGH = 5  # Highly excited, very animated, maximum enthusiasm


class TimeOfDay(Enum):
    """Time periods that affect energy levels."""

    EARLY_MORNING = "early_morning"  # 5:00-8:00
    MORNING = "morning"  # 8:00-12:00
    AFTERNOON = "afternoon"  # 12:00-17:00
    EVENING = "evening"  # 17:00-21:00
    LATE_NIGHT = "late_night"  # 21:00-5:00


@dataclass
class EnergyIndicator:
    """Indicator of energy level in text or behavior."""

    indicator_type: str  # "vocabulary", "punctuation", "structure", "emotional"
    pattern: str
    energy_score: float  # 0.0 (very low) to 1.0 (very high)
    confidence: float  # 0.0 to 1.0


@dataclass
class EnergyAnalysis:
    """Analysis of energy level from text and context."""

    detected_level: EnergyLevel
    confidence: float
    indicators: List[EnergyIndicator] = field(default_factory=list)
    time_factors: List[str] = field(default_factory=list)
    emotional_factors: List[str] = field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EnergyAdjustment:
    """Adjustment to match energy levels."""

    adjustment_id: str
    from_level: EnergyLevel
    to_level: EnergyLevel
    original_text: str
    adjusted_text: str
    adjustment_techniques: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Initialize adjustment ID if not provided."""
        if not self.adjustment_id:
            self.adjustment_id = str(uuid.uuid4())


@dataclass
class EnergyPreference:
    """User's energy level preferences and patterns."""

    user_id: str
    default_energy: EnergyLevel = EnergyLevel.MODERATE
    time_preferences: Dict[TimeOfDay, EnergyLevel] = field(default_factory=dict)
    emotional_energy_patterns: Dict[str, EnergyLevel] = field(default_factory=dict)
    energy_variability: float = 0.5  # How much energy varies (0.0-1.0)
    adaptation_sensitivity: float = 0.7  # How quickly to match user energy
    confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def get_time_preference(self, time_of_day: TimeOfDay) -> EnergyLevel:
        """Get preferred energy level for time of day."""
        return self.time_preferences.get(time_of_day, self.default_energy)

    def get_emotional_preference(self, emotion: str) -> EnergyLevel:
        """Get preferred energy level for emotional state."""
        return self.emotional_energy_patterns.get(emotion, self.default_energy)


class EnergyLevelMatcher:
    """
    Matches communication energy levels with user's current state.

    Analyzes user energy from text patterns, emotional state, time of day,
    and personality traits to determine appropriate response energy level.
    """

    def __init__(self):
        """Initialize energy level matcher."""
        self._energy_patterns = self._initialize_energy_patterns()
        logger.info("Energy level matcher initialized")

    def _initialize_energy_patterns(self) -> Dict[str, List[EnergyIndicator]]:
        """Initialize patterns for detecting energy levels."""
        return {
            "high_energy_words": [
                EnergyIndicator(
                    "vocabulary", "amazing|awesome|fantastic|incredible", 0.8, 0.9
                ),
                EnergyIndicator(
                    "vocabulary", "excited|thrilled|pumped|energized", 0.9, 0.95
                ),
                EnergyIndicator("vocabulary", "love|adore|obsessed", 0.7, 0.8),
            ],
            "low_energy_words": [
                EnergyIndicator("vocabulary", "tired|exhausted|drained", 0.2, 0.9),
                EnergyIndicator("vocabulary", "calm|peaceful|quiet", 0.3, 0.7),
                EnergyIndicator("vocabulary", "okay|fine|whatever", 0.4, 0.6),
            ],
            "punctuation_patterns": [
                EnergyIndicator("punctuation", r"!{2,}", 0.8, 0.8),
                EnergyIndicator("punctuation", r"\?{2,}", 0.7, 0.7),
                EnergyIndicator("punctuation", r"\.{3,}", 0.3, 0.6),
            ],
            "structure_patterns": [
                EnergyIndicator("structure", "ALL_CAPS", 0.9, 0.8),
                EnergyIndicator("structure", "short_sentences", 0.6, 0.5),
                EnergyIndicator("structure", "long_rambling", 0.4, 0.6),
            ],
        }

    def analyze_user_energy(
        self,
        text: str,
        context: Optional[ConversationContext] = None,
        emotional_state: Optional[EmotionalState] = None,
        current_time: Optional[datetime] = None,
    ) -> EnergyAnalysis:
        """
        Analyze user's energy level from text and context.

        Args:
            text: User's message text
            context: Optional conversation context
            emotional_state: Optional emotional state
            current_time: Optional current time (defaults to now)

        Returns:
            EnergyAnalysis with detected energy level and reasoning
        """
        logger.debug("Analyzing user energy from text: %s", text[:50])

        if current_time is None:
            current_time = datetime.now()

        indicators = []
        energy_scores = []

        # Analyze vocabulary patterns
        for pattern_group in self._energy_patterns.values():
            for indicator in pattern_group:
                if indicator.indicator_type == "vocabulary":
                    if re.search(indicator.pattern, text.lower()):
                        indicators.append(indicator)
                        energy_scores.append(indicator.energy_score)
                elif indicator.indicator_type == "punctuation":
                    if re.search(indicator.pattern, text):
                        indicators.append(indicator)
                        energy_scores.append(indicator.energy_score)

        # Analyze structural patterns
        if text.isupper():
            cap_indicator = EnergyIndicator("structure", "ALL_CAPS", 0.9, 0.8)
            indicators.append(cap_indicator)
            energy_scores.append(0.9)

        # Calculate average energy score
        if energy_scores:
            avg_energy = sum(energy_scores) / len(energy_scores)
        else:
            avg_energy = 0.5  # Default moderate energy

        # Map energy score to energy level
        detected_level = self._score_to_energy_level(avg_energy)

        # Calculate confidence based on number of indicators
        confidence = min(0.9, 0.3 + (len(indicators) * 0.1))

        # Add time factors
        time_factors = []
        time_of_day = self._get_time_of_day(current_time)
        if time_of_day == TimeOfDay.EARLY_MORNING:
            time_factors.append("Early morning - typically lower energy")
        elif time_of_day == TimeOfDay.LATE_NIGHT:
            time_factors.append("Late night - potentially tired")

        # Add emotional factors
        emotional_factors = []
        if emotional_state:
            if emotional_state.primary_emotion in ["joy", "excitement"]:
                emotional_factors.append("Positive emotions suggest higher energy")
            elif emotional_state.primary_emotion in ["sadness", "fatigue"]:
                emotional_factors.append("Negative emotions suggest lower energy")

        reasoning = f"Detected {len(indicators)} energy indicators"
        if time_factors:
            reasoning += f", time factors: {', '.join(time_factors)}"
        if emotional_factors:
            reasoning += f", emotional factors: {', '.join(emotional_factors)}"

        analysis = EnergyAnalysis(
            detected_level=detected_level,
            confidence=confidence,
            indicators=indicators,
            time_factors=time_factors,
            emotional_factors=emotional_factors,
            reasoning=reasoning,
        )

        logger.debug(
            "Energy analysis complete: %s (confidence: %.2f)",
            detected_level.name,
            confidence,
        )

        return analysis

    def determine_target_energy(
        self,
        user_id: str,
        user_energy_analysis: EnergyAnalysis,
        personality_profile: Optional[PersonalityProfile] = None,
        energy_preference: Optional[EnergyPreference] = None,
        context: Optional[ConversationContext] = None,
    ) -> EnergyLevel:
        """
        Determine target energy level for response.

        Args:
            user_id: User identifier
            user_energy_analysis: Analysis of user's current energy
            personality_profile: Optional personality profile
            energy_preference: Optional user energy preferences
            context: Optional conversation context

        Returns:
            Target energy level for response
        """
        logger.debug("Determining target energy for user %s", user_id)

        user_energy = user_energy_analysis.detected_level
        target_energy = user_energy

        # Adjust based on personality profile
        if personality_profile:
            extraversion = personality_profile.get_trait_level(
                PersonalityTrait.EXTRAVERSION
            )
            if extraversion == TraitLevel.HIGH:
                # Extraverts prefer slightly higher energy
                target_energy = self._adjust_energy_level(target_energy, +1)
            elif extraversion == TraitLevel.LOW:
                # Introverts prefer slightly lower energy
                target_energy = self._adjust_energy_level(target_energy, -1)

        # Adjust based on user preferences
        if energy_preference:
            # Apply adaptation sensitivity
            sensitivity = energy_preference.adaptation_sensitivity
            if sensitivity < 0.5:
                # Low sensitivity - maintain more consistent energy
                target_energy = energy_preference.default_energy
            elif sensitivity > 0.8:
                # High sensitivity - match user energy closely
                target_energy = user_energy

        # Ensure target energy is within reasonable bounds
        target_energy = self._clamp_energy_level(target_energy)

        logger.debug(
            "Target energy determined: %s -> %s",
            user_energy.name,
            target_energy.name,
        )

        return target_energy

    def adjust_response_energy(
        self,
        text: str,
        from_level: EnergyLevel,
        to_level: EnergyLevel,
        user_id: str = "",
    ) -> EnergyAdjustment:
        """
        Adjust text to match target energy level.

        Args:
            text: Original text to adjust
            from_level: Current energy level
            to_level: Target energy level
            user_id: Optional user identifier

        Returns:
            EnergyAdjustment with modified text
        """
        if from_level == to_level:
            return EnergyAdjustment(
                adjustment_id=str(uuid.uuid4()),
                from_level=from_level,
                to_level=to_level,
                original_text=text,
                adjusted_text=text,
                reasoning="No adjustment needed - energy levels match",
            )

        adjusted_text = text
        techniques = []

        # Simple energy adjustment techniques
        if to_level.value > from_level.value:
            # Increase energy
            if "." in adjusted_text:
                adjusted_text = adjusted_text.replace(".", "!")
                techniques.append("Added exclamation points")

            # Add energetic words
            if "good" in adjusted_text.lower():
                adjusted_text = adjusted_text.replace("good", "great")
                techniques.append("Enhanced positive vocabulary")

        elif to_level.value < from_level.value:
            # Decrease energy
            if "!" in adjusted_text:
                adjusted_text = adjusted_text.replace("!", ".")
                techniques.append("Reduced exclamation points")

            # Use calmer language
            if "amazing" in adjusted_text.lower():
                adjusted_text = adjusted_text.replace("amazing", "nice")
                techniques.append("Used calmer vocabulary")

        confidence = 0.7 if techniques else 0.3
        reasoning = f"Applied {len(techniques)} adjustment techniques"

        return EnergyAdjustment(
            adjustment_id=str(uuid.uuid4()),
            from_level=from_level,
            to_level=to_level,
            original_text=text,
            adjusted_text=adjusted_text,
            adjustment_techniques=techniques,
            confidence=confidence,
            reasoning=reasoning,
        )

    def get_energy_description(self, level: EnergyLevel) -> str:
        """Get human-readable description of energy level."""
        descriptions = {
            EnergyLevel.VERY_LOW: "Very calm and subdued communication",
            EnergyLevel.LOW: "Relaxed and gentle communication",
            EnergyLevel.MODERATE: "Balanced and steady energy",
            EnergyLevel.HIGH: "Enthusiastic and animated communication",
            EnergyLevel.VERY_HIGH: "Highly energetic and excited communication",
        }

        return descriptions.get(level, f"Energy level: {level.name}")

    def _score_to_energy_level(self, score: float) -> EnergyLevel:
        """Convert energy score to energy level enum."""
        if score <= 0.2:
            return EnergyLevel.VERY_LOW
        elif score <= 0.4:
            return EnergyLevel.LOW
        elif score <= 0.6:
            return EnergyLevel.MODERATE
        elif score <= 0.8:
            return EnergyLevel.HIGH
        else:
            return EnergyLevel.VERY_HIGH

    def _get_time_of_day(self, current_time: datetime) -> TimeOfDay:
        """Determine time of day category."""
        hour = current_time.hour

        if 5 <= hour < 8:
            return TimeOfDay.EARLY_MORNING
        elif 8 <= hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= hour < 17:
            return TimeOfDay.AFTERNOON
        elif 17 <= hour < 21:
            return TimeOfDay.EVENING
        else:
            return TimeOfDay.LATE_NIGHT

    def _adjust_energy_level(self, level: EnergyLevel, adjustment: int) -> EnergyLevel:
        """Adjust energy level by specified amount."""
        new_value = level.value + adjustment
        new_value = max(1, min(5, new_value))  # Clamp to valid range
        return EnergyLevel(new_value)

    def _clamp_energy_level(self, level: EnergyLevel) -> EnergyLevel:
        """Ensure energy level is within valid bounds."""
        return level  # Already validated by enum
