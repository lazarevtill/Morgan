"""
Communication Style Adaptation for Morgan RAG.

Adapts communication style based on user personality traits, preferences,
and contextual factors to provide more personalized interactions.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from ..intelligence.core.models import ConversationContext, EmotionalState
from ..utils.logger import get_logger
from .traits import PersonalityProfile, PersonalityTrait, TraitLevel

logger = get_logger(__name__)


class StyleDimension(Enum):
    """Dimensions of communication style."""

    FORMALITY = "formality"
    DIRECTNESS = "directness"
    WARMTH = "warmth"
    COMPLEXITY = "complexity"
    ENTHUSIASM = "enthusiasm"
    SUPPORTIVENESS = "supportiveness"


class StyleIntensity(Enum):
    """Intensity levels for style dimensions."""

    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class StyleProfile:
    """Communication style profile with multiple dimensions."""

    formality: StyleIntensity = StyleIntensity.MODERATE
    directness: StyleIntensity = StyleIntensity.MODERATE
    warmth: StyleIntensity = StyleIntensity.MODERATE
    complexity: StyleIntensity = StyleIntensity.MODERATE
    enthusiasm: StyleIntensity = StyleIntensity.MODERATE
    supportiveness: StyleIntensity = StyleIntensity.MODERATE

    def get_dimension(self, dimension: StyleDimension) -> StyleIntensity:
        """Get intensity for a specific style dimension."""
        return getattr(self, dimension.value, StyleIntensity.MODERATE)

    def set_dimension(self, dimension: StyleDimension, intensity: StyleIntensity):
        """Set intensity for a specific style dimension."""
        setattr(self, dimension.value, intensity)


@dataclass
class StyleAdaptation:
    """Represents a communication style adaptation."""

    adaptation_id: str
    user_id: str
    dimension: StyleDimension
    from_intensity: StyleIntensity
    to_intensity: StyleIntensity
    confidence: float  # 0.0 to 1.0
    reasoning: str
    context_factors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Initialize adaptation ID if not provided."""
        if not self.adaptation_id:
            self.adaptation_id = str(uuid.uuid4())


@dataclass
class StyleRecommendation:
    """Recommendation for communication style adjustments."""

    user_id: str
    recommended_style: StyleProfile
    adaptations: List[StyleAdaptation]
    overall_confidence: float
    reasoning: str
    context_summary: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CommunicationStyleAdapter:
    """
    Adapts communication style based on user personality and context.

    Uses personality traits, emotional state, and conversation context
    to determine optimal communication style adjustments.
    """

    # Style adaptation rules based on personality traits
    TRAIT_STYLE_MAPPINGS = {
        PersonalityTrait.EXTRAVERSION: {
            TraitLevel.VERY_LOW: {
                StyleDimension.ENTHUSIASM: StyleIntensity.LOW,
                StyleDimension.DIRECTNESS: StyleIntensity.LOW,
                StyleDimension.WARMTH: StyleIntensity.MODERATE,
            },
            TraitLevel.LOW: {
                StyleDimension.ENTHUSIASM: StyleIntensity.LOW,
                StyleDimension.DIRECTNESS: StyleIntensity.MODERATE,
                StyleDimension.WARMTH: StyleIntensity.MODERATE,
            },
            TraitLevel.MODERATE: {
                StyleDimension.ENTHUSIASM: StyleIntensity.MODERATE,
                StyleDimension.DIRECTNESS: StyleIntensity.MODERATE,
                StyleDimension.WARMTH: StyleIntensity.MODERATE,
            },
            TraitLevel.HIGH: {
                StyleDimension.ENTHUSIASM: StyleIntensity.HIGH,
                StyleDimension.DIRECTNESS: StyleIntensity.HIGH,
                StyleDimension.WARMTH: StyleIntensity.HIGH,
            },
            TraitLevel.VERY_HIGH: {
                StyleDimension.ENTHUSIASM: StyleIntensity.VERY_HIGH,
                StyleDimension.DIRECTNESS: StyleIntensity.HIGH,
                StyleDimension.WARMTH: StyleIntensity.VERY_HIGH,
            },
        },
        PersonalityTrait.AGREEABLENESS: {
            TraitLevel.VERY_LOW: {
                StyleDimension.DIRECTNESS: StyleIntensity.VERY_HIGH,
                StyleDimension.WARMTH: StyleIntensity.LOW,
                StyleDimension.SUPPORTIVENESS: StyleIntensity.LOW,
            },
            TraitLevel.LOW: {
                StyleDimension.DIRECTNESS: StyleIntensity.HIGH,
                StyleDimension.WARMTH: StyleIntensity.LOW,
                StyleDimension.SUPPORTIVENESS: StyleIntensity.MODERATE,
            },
            TraitLevel.MODERATE: {
                StyleDimension.DIRECTNESS: StyleIntensity.MODERATE,
                StyleDimension.WARMTH: StyleIntensity.MODERATE,
                StyleDimension.SUPPORTIVENESS: StyleIntensity.MODERATE,
            },
            TraitLevel.HIGH: {
                StyleDimension.DIRECTNESS: StyleIntensity.LOW,
                StyleDimension.WARMTH: StyleIntensity.HIGH,
                StyleDimension.SUPPORTIVENESS: StyleIntensity.HIGH,
            },
            TraitLevel.VERY_HIGH: {
                StyleDimension.DIRECTNESS: StyleIntensity.VERY_LOW,
                StyleDimension.WARMTH: StyleIntensity.VERY_HIGH,
                StyleDimension.SUPPORTIVENESS: StyleIntensity.VERY_HIGH,
            },
        },
        PersonalityTrait.CONSCIENTIOUSNESS: {
            TraitLevel.VERY_LOW: {
                StyleDimension.FORMALITY: StyleIntensity.LOW,
                StyleDimension.COMPLEXITY: StyleIntensity.LOW,
            },
            TraitLevel.LOW: {
                StyleDimension.FORMALITY: StyleIntensity.LOW,
                StyleDimension.COMPLEXITY: StyleIntensity.MODERATE,
            },
            TraitLevel.MODERATE: {
                StyleDimension.FORMALITY: StyleIntensity.MODERATE,
                StyleDimension.COMPLEXITY: StyleIntensity.MODERATE,
            },
            TraitLevel.HIGH: {
                StyleDimension.FORMALITY: StyleIntensity.HIGH,
                StyleDimension.COMPLEXITY: StyleIntensity.HIGH,
            },
            TraitLevel.VERY_HIGH: {
                StyleDimension.FORMALITY: StyleIntensity.VERY_HIGH,
                StyleDimension.COMPLEXITY: StyleIntensity.HIGH,
            },
        },
        PersonalityTrait.OPENNESS: {
            TraitLevel.VERY_LOW: {
                StyleDimension.COMPLEXITY: StyleIntensity.LOW,
                StyleDimension.FORMALITY: StyleIntensity.HIGH,
            },
            TraitLevel.LOW: {
                StyleDimension.COMPLEXITY: StyleIntensity.LOW,
                StyleDimension.FORMALITY: StyleIntensity.MODERATE,
            },
            TraitLevel.MODERATE: {
                StyleDimension.COMPLEXITY: StyleIntensity.MODERATE,
                StyleDimension.FORMALITY: StyleIntensity.MODERATE,
            },
            TraitLevel.HIGH: {
                StyleDimension.COMPLEXITY: StyleIntensity.HIGH,
                StyleDimension.FORMALITY: StyleIntensity.LOW,
            },
            TraitLevel.VERY_HIGH: {
                StyleDimension.COMPLEXITY: StyleIntensity.VERY_HIGH,
                StyleDimension.FORMALITY: StyleIntensity.LOW,
            },
        },
        PersonalityTrait.NEUROTICISM: {
            TraitLevel.VERY_LOW: {
                StyleDimension.SUPPORTIVENESS: StyleIntensity.MODERATE,
                StyleDimension.DIRECTNESS: StyleIntensity.HIGH,
            },
            TraitLevel.LOW: {
                StyleDimension.SUPPORTIVENESS: StyleIntensity.MODERATE,
                StyleDimension.DIRECTNESS: StyleIntensity.MODERATE,
            },
            TraitLevel.MODERATE: {
                StyleDimension.SUPPORTIVENESS: StyleIntensity.MODERATE,
                StyleDimension.DIRECTNESS: StyleIntensity.MODERATE,
            },
            TraitLevel.HIGH: {
                StyleDimension.SUPPORTIVENESS: StyleIntensity.HIGH,
                StyleDimension.DIRECTNESS: StyleIntensity.LOW,
            },
            TraitLevel.VERY_HIGH: {
                StyleDimension.SUPPORTIVENESS: StyleIntensity.VERY_HIGH,
                StyleDimension.DIRECTNESS: StyleIntensity.VERY_LOW,
            },
        },
    }

    def __init__(self):
        """Initialize communication style adapter."""
        logger.info("Communication style adapter initialized")

    def adapt_style(
        self,
        user_id: str,
        personality_profile: PersonalityProfile,
        context: ConversationContext,
        emotional_state: Optional[EmotionalState] = None,
        current_style: Optional[StyleProfile] = None,
    ) -> StyleRecommendation:
        """
        Adapt communication style based on user profile and context.

        Args:
            user_id: User identifier
            personality_profile: User's personality profile
            context: Current conversation context
            emotional_state: Current emotional state
            current_style: Current style profile

        Returns:
            StyleRecommendation: Recommended style adaptations
        """
        logger.info("Adapting communication style for user %s", user_id)

        if current_style is None:
            current_style = StyleProfile()

        # Generate base style from personality traits
        base_style = self._generate_base_style(personality_profile)

        # Apply contextual adjustments
        contextual_style = self._apply_contextual_adjustments(
            base_style, context, emotional_state
        )

        # Generate adaptations
        adaptations = self._generate_adaptations(
            user_id, current_style, contextual_style
        )

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            personality_profile, adaptations
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            personality_profile, context, emotional_state, adaptations
        )

        # Generate context summary
        context_summary = self._generate_context_summary(context, emotional_state)

        recommendation = StyleRecommendation(
            user_id=user_id,
            recommended_style=contextual_style,
            adaptations=adaptations,
            overall_confidence=overall_confidence,
            reasoning=reasoning,
            context_summary=context_summary,
        )

        logger.info(
            "Generated %d style adaptations for user %s with confidence %.2f",
            len(adaptations),
            user_id,
            overall_confidence,
        )

        return recommendation

    def _generate_base_style(
        self, personality_profile: PersonalityProfile
    ) -> StyleProfile:
        """Generate base style from personality traits."""
        style = StyleProfile()

        # Apply trait-based style mappings
        for trait, trait_score in personality_profile.trait_scores.items():
            if trait in self.TRAIT_STYLE_MAPPINGS:
                trait_level = trait_score.level
                style_mappings = self.TRAIT_STYLE_MAPPINGS[trait].get(trait_level, {})

                for dimension, intensity in style_mappings.items():
                    # Weight by trait confidence
                    weighted_intensity = self._weight_intensity(
                        intensity, trait_score.confidence
                    )

                    # Combine with existing dimension value
                    current_intensity = style.get_dimension(dimension)
                    combined_intensity = self._combine_intensities(
                        current_intensity, weighted_intensity
                    )

                    style.set_dimension(dimension, combined_intensity)

        return style

    def _apply_contextual_adjustments(
        self,
        base_style: StyleProfile,
        context: ConversationContext,
        emotional_state: Optional[EmotionalState],
    ) -> StyleProfile:
        """Apply contextual adjustments to base style."""
        adjusted_style = StyleProfile(
            formality=base_style.formality,
            directness=base_style.directness,
            warmth=base_style.warmth,
            complexity=base_style.complexity,
            enthusiasm=base_style.enthusiasm,
            supportiveness=base_style.supportiveness,
        )

        # Adjust based on emotional state
        if emotional_state:
            adjusted_style = self._adjust_for_emotion(adjusted_style, emotional_state)

        # Adjust based on conversation context
        adjusted_style = self._adjust_for_context(adjusted_style, context)

        return adjusted_style

    def _adjust_for_emotion(
        self, style: StyleProfile, emotional_state: EmotionalState
    ) -> StyleProfile:
        """Adjust style based on emotional state."""
        emotion = emotional_state.primary_emotion.value
        intensity = emotional_state.intensity

        if emotion in ["sadness", "fear", "anger"]:
            # Increase supportiveness and warmth for negative emotions
            style.supportiveness = self._increase_intensity(
                style.supportiveness, intensity
            )
            style.warmth = self._increase_intensity(style.warmth, intensity)

            # Decrease directness for sensitive emotions
            if emotion in ["sadness", "fear"]:
                style.directness = self._decrease_intensity(style.directness, intensity)

        elif emotion == "joy":
            # Match enthusiasm for positive emotions
            style.enthusiasm = self._increase_intensity(style.enthusiasm, intensity)
            style.warmth = self._increase_intensity(style.warmth, intensity)

        elif emotion == "surprise":
            # Moderate adjustments for surprise
            style.supportiveness = self._increase_intensity(
                style.supportiveness, intensity * 0.5
            )

        return style

    def _adjust_for_context(
        self, style: StyleProfile, context: ConversationContext
    ) -> StyleProfile:
        """Adjust style based on conversation context."""
        # Check for work/professional context
        if hasattr(context, "context_type"):
            if context.context_type == "professional":
                style.formality = self._increase_intensity(style.formality, 0.3)
                style.directness = self._increase_intensity(style.directness, 0.2)
            elif context.context_type == "personal":
                style.warmth = self._increase_intensity(style.warmth, 0.3)
                style.formality = self._decrease_intensity(style.formality, 0.2)

        # Check for learning context
        if hasattr(context, "is_learning_context"):
            if context.is_learning_context:
                style.supportiveness = self._increase_intensity(
                    style.supportiveness, 0.3
                )
                style.complexity = self._adjust_complexity_for_learning(
                    style.complexity, context
                )

        return style

    def _adjust_complexity_for_learning(
        self, current_complexity: StyleIntensity, context: ConversationContext
    ) -> StyleIntensity:
        """Adjust complexity for learning contexts."""
        # Check user's indicated skill level
        if hasattr(context, "user_skill_level"):
            skill_level = context.user_skill_level
            if skill_level == "beginner":
                return StyleIntensity.LOW
            elif skill_level == "intermediate":
                return StyleIntensity.MODERATE
            elif skill_level == "advanced":
                return StyleIntensity.HIGH

        return current_complexity

    def _generate_adaptations(
        self, user_id: str, current_style: StyleProfile, target_style: StyleProfile
    ) -> List[StyleAdaptation]:
        """Generate list of style adaptations needed."""
        adaptations = []

        for dimension in StyleDimension:
            current_intensity = current_style.get_dimension(dimension)
            target_intensity = target_style.get_dimension(dimension)

            if current_intensity != target_intensity:
                # Calculate confidence based on intensity difference
                intensity_diff = abs(current_intensity.value - target_intensity.value)
                confidence = min(intensity_diff / 4.0, 1.0)

                # Generate reasoning for this adaptation
                reasoning = self._generate_adaptation_reasoning(
                    dimension, current_intensity, target_intensity
                )

                adaptation = StyleAdaptation(
                    adaptation_id=str(uuid.uuid4()),
                    user_id=user_id,
                    dimension=dimension,
                    from_intensity=current_intensity,
                    to_intensity=target_intensity,
                    confidence=confidence,
                    reasoning=reasoning,
                )

                adaptations.append(adaptation)

        return adaptations

    def _generate_adaptation_reasoning(
        self,
        dimension: StyleDimension,
        from_intensity: StyleIntensity,
        to_intensity: StyleIntensity,
    ) -> str:
        """Generate reasoning for a specific adaptation."""
        direction = (
            "increase" if to_intensity.value > from_intensity.value else "decrease"
        )

        reasoning_templates = {
            StyleDimension.FORMALITY: {
                "increase": "User personality suggests preference for more formal communication",
                "decrease": "User personality suggests preference for more casual communication",
            },
            StyleDimension.DIRECTNESS: {
                "increase": "User traits indicate comfort with direct communication",
                "decrease": "User traits suggest preference for gentler, less direct approach",
            },
            StyleDimension.WARMTH: {
                "increase": "User profile indicates appreciation for warmer interactions",
                "decrease": "User profile suggests preference for more neutral tone",
            },
            StyleDimension.COMPLEXITY: {
                "increase": "User traits suggest ability to handle more complex information",
                "decrease": "User profile indicates preference for simpler explanations",
            },
            StyleDimension.ENTHUSIASM: {
                "increase": "User personality suggests appreciation for more energetic responses",
                "decrease": "User traits indicate preference for calmer, measured responses",
            },
            StyleDimension.SUPPORTIVENESS: {
                "increase": "User profile suggests need for more supportive communication",
                "decrease": "User traits indicate comfort with less supportive approach",
            },
        }

        return reasoning_templates.get(dimension, {}).get(
            direction,
            f"{direction.capitalize()} {dimension.value} based on user profile",
        )

    def _weight_intensity(
        self, intensity: StyleIntensity, confidence: float
    ) -> StyleIntensity:
        """Weight intensity by confidence score."""
        if confidence < 0.3:
            # Low confidence - move toward moderate
            if intensity.value > 3:
                return StyleIntensity(max(intensity.value - 1, 3))
            elif intensity.value < 3:
                return StyleIntensity(min(intensity.value + 1, 3))

        return intensity

    def _combine_intensities(
        self, intensity1: StyleIntensity, intensity2: StyleIntensity
    ) -> StyleIntensity:
        """Combine two intensity values."""
        # Simple average, rounded to nearest valid intensity
        combined_value = (intensity1.value + intensity2.value) / 2
        rounded_value = round(combined_value)
        return StyleIntensity(max(1, min(5, rounded_value)))

    def _increase_intensity(
        self, current: StyleIntensity, factor: float
    ) -> StyleIntensity:
        """Increase intensity by a factor."""
        increase = max(1, round(factor * 2))  # Convert factor to 1-2 step increase
        new_value = min(5, current.value + increase)
        return StyleIntensity(new_value)

    def _decrease_intensity(
        self, current: StyleIntensity, factor: float
    ) -> StyleIntensity:
        """Decrease intensity by a factor."""
        decrease = max(1, round(factor * 2))  # Convert factor to 1-2 step decrease
        new_value = max(1, current.value - decrease)
        return StyleIntensity(new_value)

    def _calculate_overall_confidence(
        self,
        personality_profile: PersonalityProfile,
        adaptations: List[StyleAdaptation],
    ) -> float:
        """Calculate overall confidence in style recommendations."""
        if not adaptations:
            return 0.0

        # Base confidence on personality profile confidence
        base_confidence = personality_profile.overall_confidence

        # Average adaptation confidences
        adaptation_confidence = sum(a.confidence for a in adaptations) / len(
            adaptations
        )

        # Combine with weights
        overall_confidence = 0.6 * base_confidence + 0.4 * adaptation_confidence

        return overall_confidence

    def _generate_reasoning(
        self,
        personality_profile: PersonalityProfile,
        context: ConversationContext,
        emotional_state: Optional[EmotionalState],
        adaptations: List[StyleAdaptation],
    ) -> str:
        """Generate overall reasoning for style recommendations."""
        reasoning_parts = []

        # Personality-based reasoning
        if personality_profile.overall_confidence > 0.5:
            reasoning_parts.append(
                f"Based on personality analysis (confidence: "
                f"{personality_profile.overall_confidence:.2f})"
            )

        # Emotional state reasoning
        if emotional_state:
            reasoning_parts.append(
                f"Adjusted for current emotional state: "
                f"{emotional_state.primary_emotion.value}"
            )

        # Context reasoning
        if hasattr(context, "context_type"):
            reasoning_parts.append(f"Adapted for {context.context_type} context")

        # Adaptation count
        if adaptations:
            reasoning_parts.append(f"Applied {len(adaptations)} style adjustments")

        return (
            "; ".join(reasoning_parts) if reasoning_parts else "Default style applied"
        )

    def _generate_context_summary(
        self, context: ConversationContext, emotional_state: Optional[EmotionalState]
    ) -> str:
        """Generate summary of context factors."""
        summary_parts = []

        if emotional_state:
            summary_parts.append(
                f"Emotion: {emotional_state.primary_emotion.value} "
                f"(intensity: {emotional_state.intensity:.2f})"
            )

        if hasattr(context, "context_type"):
            summary_parts.append(f"Context: {context.context_type}")

        if hasattr(context, "is_learning_context") and context.is_learning_context:
            summary_parts.append("Learning context detected")

        return "; ".join(summary_parts) if summary_parts else "Standard context"

    def get_style_description(self, style: StyleProfile) -> Dict[str, str]:
        """Get human-readable description of a style profile."""
        descriptions = {}

        intensity_labels = {
            StyleIntensity.VERY_LOW: "Very Low",
            StyleIntensity.LOW: "Low",
            StyleIntensity.MODERATE: "Moderate",
            StyleIntensity.HIGH: "High",
            StyleIntensity.VERY_HIGH: "Very High",
        }

        for dimension in StyleDimension:
            intensity = style.get_dimension(dimension)
            descriptions[dimension.value] = intensity_labels[intensity]

        return descriptions
