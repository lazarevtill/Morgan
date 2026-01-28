"""
Formality Level Adjustment for Morgan RAG.

Adjusts communication formality based on user preferences, context,
and personality traits to match appropriate professional or casual tone.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from ..intelligence.core.models import ConversationContext, EmotionalState
from ..utils.logger import get_logger
from .traits import PersonalityProfile, PersonalityTrait, TraitLevel

logger = get_logger(__name__)


class FormalityLevel(Enum):
    """Levels of communication formality."""

    VERY_CASUAL = 1  # "Hey! What's up?"
    CASUAL = 2  # "Hi there! How can I help?"
    NEUTRAL = 3  # "Hello. How may I assist you?"
    FORMAL = 4  # "Good day. How may I be of service?"
    VERY_FORMAL = 5  # "Good afternoon. I would be pleased to assist you."


class ContextType(Enum):
    """Types of conversation contexts that affect formality."""

    PERSONAL = "personal"
    PROFESSIONAL = "professional"
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    SOCIAL = "social"
    CUSTOMER_SERVICE = "customer_service"
    MEDICAL = "medical"
    LEGAL = "legal"


@dataclass
class FormalityIndicator:
    """Indicator of formality level in text."""

    indicator_type: str  # "vocabulary", "structure", "punctuation"
    text_pattern: str
    formality_score: float  # -1.0 (very casual) to 1.0 (very formal)
    confidence: float  # 0.0 to 1.0


@dataclass
class FormalityAnalysis:
    """Analysis of formality level in text or context."""

    detected_level: FormalityLevel
    confidence: float
    indicators: List[FormalityIndicator] = field(default_factory=list)
    context_factors: List[str] = field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FormalityAdjustment:
    """Adjustment to make text more or less formal."""

    adjustment_id: str
    from_level: FormalityLevel
    to_level: FormalityLevel
    original_text: str
    adjusted_text: str
    adjustment_type: str  # "vocabulary", "structure", "tone"
    confidence: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Initialize adjustment ID if not provided."""
        if not self.adjustment_id:
            self.adjustment_id = str(uuid.uuid4())


@dataclass
class FormalityPreference:
    """User's formality preferences for different contexts."""

    user_id: str
    default_level: FormalityLevel = FormalityLevel.NEUTRAL
    context_preferences: Dict[ContextType, FormalityLevel] = field(default_factory=dict)
    flexibility: float = 0.5  # 0.0 (rigid) to 1.0 (very flexible)
    adaptation_speed: float = 0.3  # How quickly to adapt to user's style
    confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def get_preferred_level(self, context_type: ContextType) -> FormalityLevel:
        """Get preferred formality level for a context type."""
        return self.context_preferences.get(context_type, self.default_level)

    def set_context_preference(self, context_type: ContextType, level: FormalityLevel):
        """Set formality preference for a specific context."""
        self.context_preferences[context_type] = level
        self.last_updated = datetime.now(timezone.utc)


class FormalityLevelAdjuster:
    """
    Adjusts communication formality based on user preferences and context.

    Analyzes user communication patterns, context cues, and personality traits
    to determine and adjust appropriate formality levels.
    """

    # Formality indicators for text analysis
    FORMALITY_INDICATORS = {
        # Very casual indicators
        "very_casual": {
            "vocabulary": [
                r"\b(hey|hi|yo|sup|what's up|howdy)\b",
                r"\b(yeah|yep|nah|nope|gonna|wanna|gotta)\b",
                r"\b(awesome|cool|sweet|nice|dude|bro)\b",
                r"\b(lol|omg|btw|fyi|tbh|imo)\b",
            ],
            "structure": [
                r"^[a-z]",  # Lowercase start
                r"[.!?]{2,}",  # Multiple punctuation
                r"\b\w+\b[.!?]$",  # Single word responses
            ],
            "punctuation": [r"!!+", r"\?\?+", r"\.\.\.+"],
        },
        # Casual indicators
        "casual": {
            "vocabulary": [
                r"\b(hello|hi there|thanks|sure|okay|alright)\b",
                r"\b(pretty|quite|really|very|super)\b",
                r"\b(help|check|look|see|think|feel)\b",
            ],
            "structure": [
                r"^[A-Z][^.!?]*[.!?]$",  # Simple sentences
                r"\bI\b.*\b(think|feel|believe)\b",  # Personal opinions
            ],
        },
        # Formal indicators
        "formal": {
            "vocabulary": [
                r"\b(please|thank you|certainly|indeed|however)\b",
                r"\b(assist|provide|ensure|consider|recommend)\b",
                r"\b(appropriate|suitable|adequate|sufficient)\b",
                r"\b(furthermore|moreover|nevertheless|therefore)\b",
            ],
            "structure": [
                r"\b(I would|I shall|I will|one might|one could)\b",
                r"\b(it is|there are|there is)\b.*\b(that|which)\b",
            ],
        },
        # Very formal indicators
        "very_formal": {
            "vocabulary": [
                r"\b(distinguished|esteemed|honored|privileged)\b",
                r"\b(endeavor|facilitate|accommodate|demonstrate)\b",
                r"\b(subsequently|consequently|accordingly|henceforth)\b",
                r"\b(respectfully|cordially|sincerely|gratefully)\b",
            ],
            "structure": [
                r"\b(I am pleased to|I would be delighted to|It would be my pleasure)\b",
                r"\b(May I|Might I|Could I|Should you require)\b",
            ],
        },
    }

    # Context type indicators
    CONTEXT_INDICATORS = {
        ContextType.PROFESSIONAL: [
            r"\b(meeting|project|deadline|client|business|work|office)\b",
            r"\b(report|presentation|proposal|contract|agreement)\b",
        ],
        ContextType.ACADEMIC: [
            r"\b(research|study|paper|thesis|academic|university|professor)\b",
            r"\b(analysis|methodology|hypothesis|conclusion|bibliography)\b",
        ],
        ContextType.TECHNICAL: [
            r"\b(code|programming|software|system|database|algorithm)\b",
            r"\b(implementation|configuration|deployment|debugging)\b",
        ],
        ContextType.MEDICAL: [
            r"\b(doctor|patient|medical|health|treatment|diagnosis)\b",
            r"\b(symptoms|medication|therapy|clinic|hospital)\b",
        ],
        ContextType.LEGAL: [
            r"\b(law|legal|court|attorney|contract|agreement|clause)\b",
            r"\b(litigation|compliance|regulation|statute|jurisdiction)\b",
        ],
    }

    # Personality trait to formality mappings
    TRAIT_FORMALITY_MAPPINGS = {
        PersonalityTrait.CONSCIENTIOUSNESS: {
            TraitLevel.VERY_HIGH: FormalityLevel.FORMAL,
            TraitLevel.HIGH: FormalityLevel.FORMAL,
            TraitLevel.MODERATE: FormalityLevel.NEUTRAL,
            TraitLevel.LOW: FormalityLevel.CASUAL,
            TraitLevel.VERY_LOW: FormalityLevel.CASUAL,
        },
        PersonalityTrait.AGREEABLENESS: {
            TraitLevel.VERY_HIGH: FormalityLevel.FORMAL,
            TraitLevel.HIGH: FormalityLevel.NEUTRAL,
            TraitLevel.MODERATE: FormalityLevel.NEUTRAL,
            TraitLevel.LOW: FormalityLevel.CASUAL,
            TraitLevel.VERY_LOW: FormalityLevel.CASUAL,
        },
        PersonalityTrait.OPENNESS: {
            TraitLevel.VERY_HIGH: FormalityLevel.CASUAL,
            TraitLevel.HIGH: FormalityLevel.CASUAL,
            TraitLevel.MODERATE: FormalityLevel.NEUTRAL,
            TraitLevel.LOW: FormalityLevel.FORMAL,
            TraitLevel.VERY_LOW: FormalityLevel.FORMAL,
        },
    }

    def __init__(self):
        """Initialize formality level adjuster."""
        logger.info("Formality level adjuster initialized")

    def analyze_formality(
        self,
        text: str,
        context: Optional[ConversationContext] = None,
        context_type: Optional[ContextType] = None,
    ) -> FormalityAnalysis:
        """
        Analyze the formality level of given text.

        Args:
            text: Text to analyze
            context: Optional conversation context
            context_type: Optional explicit context type

        Returns:
            FormalityAnalysis with detected formality level and reasoning
        """
        logger.debug("Analyzing formality for text: %s", text[:50])

        # Detect formality indicators
        indicators = self._detect_formality_indicators(text)

        # Calculate formality score
        formality_score = self._calculate_formality_score(indicators)

        # Determine formality level
        detected_level = self._score_to_level(formality_score)

        # Detect context type if not provided
        if not context_type and context:
            context_type = self._detect_context_type(text, context)

        # Adjust for context
        if context_type:
            detected_level = self._adjust_for_context(detected_level, context_type)

        # Calculate confidence
        confidence = self._calculate_analysis_confidence(indicators, text)

        # Generate reasoning
        reasoning = self._generate_analysis_reasoning(
            detected_level, indicators, context_type
        )

        # Collect context factors
        context_factors = []
        if context_type:
            context_factors.append(f"Context: {context_type.value}")
        if context:
            context_factors.append("Conversation context available")

        analysis = FormalityAnalysis(
            detected_level=detected_level,
            confidence=confidence,
            indicators=indicators,
            context_factors=context_factors,
            reasoning=reasoning,
        )

        logger.debug(
            "Formality analysis complete: %s (confidence: %.2f)",
            detected_level.name,
            confidence,
        )

        return analysis

    def determine_target_formality(
        self,
        user_id: str,
        context: ConversationContext,
        personality_profile: Optional[PersonalityProfile] = None,
        user_preference: Optional[FormalityPreference] = None,
        emotional_state: Optional[EmotionalState] = None,
    ) -> FormalityLevel:
        """
        Determine target formality level for response.

        Args:
            user_id: User identifier
            context: Conversation context
            personality_profile: Optional personality profile
            user_preference: Optional user formality preferences
            emotional_state: Optional current emotional state

        Returns:
            Target formality level for response
        """
        logger.debug("Determining target formality for user %s", user_id)

        # Start with neutral default
        target_level = FormalityLevel.NEUTRAL

        # Analyze user's message formality
        if context.message_text:
            user_analysis = self.analyze_formality(context.message_text, context)
            # Match user's formality level (with some adaptation)
            target_level = user_analysis.detected_level

        # Apply user preferences if available
        if user_preference:
            context_type = self._detect_context_type(context.message_text, context)
            if context_type:
                preferred_level = user_preference.get_preferred_level(context_type)
                # Blend user's current style with their preferences
                target_level = self._blend_formality_levels(
                    target_level, preferred_level, user_preference.flexibility
                )

        # Apply personality-based adjustments
        if personality_profile:
            personality_adjustment = self._get_personality_formality_preference(
                personality_profile
            )
            target_level = self._blend_formality_levels(
                target_level, personality_adjustment, 0.3
            )

        # Apply emotional state adjustments
        if emotional_state:
            emotional_adjustment = self._adjust_for_emotion(
                target_level, emotional_state
            )
            target_level = emotional_adjustment

        logger.debug("Target formality determined: %s", target_level.name)
        return target_level

    def adjust_text_formality(
        self,
        text: str,
        target_level: FormalityLevel,
        current_level: Optional[FormalityLevel] = None,
    ) -> FormalityAdjustment:
        """
        Adjust text to match target formality level.

        Args:
            text: Original text to adjust
            target_level: Desired formality level
            current_level: Current formality level (will be detected if not provided)

        Returns:
            FormalityAdjustment with original and adjusted text
        """
        logger.debug("Adjusting text formality to %s: %s", target_level.name, text[:50])

        # Detect current level if not provided
        if current_level is None:
            analysis = self.analyze_formality(text)
            current_level = analysis.detected_level

        # If already at target level, no adjustment needed
        if current_level == target_level:
            return FormalityAdjustment(
                adjustment_id=str(uuid.uuid4()),
                from_level=current_level,
                to_level=target_level,
                original_text=text,
                adjusted_text=text,
                adjustment_type="none",
                confidence=1.0,
                reasoning="Text already at target formality level",
            )

        # Determine adjustment direction
        direction = 1 if target_level.value > current_level.value else -1

        # Apply adjustments
        adjusted_text = text
        adjustment_types = []

        if direction > 0:  # Make more formal
            adjusted_text = self._make_more_formal(adjusted_text)
            adjustment_types.append("formalization")
        else:  # Make more casual
            adjusted_text = self._make_more_casual(adjusted_text)
            adjustment_types.append("casualization")

        # Calculate confidence
        confidence = self._calculate_adjustment_confidence(
            text, adjusted_text, current_level, target_level
        )

        # Generate reasoning
        reasoning = self._generate_adjustment_reasoning(
            current_level, target_level, adjustment_types
        )

        adjustment = FormalityAdjustment(
            adjustment_id=str(uuid.uuid4()),
            from_level=current_level,
            to_level=target_level,
            original_text=text,
            adjusted_text=adjusted_text,
            adjustment_type=", ".join(adjustment_types),
            confidence=confidence,
            reasoning=reasoning,
        )

        logger.debug(
            "Formality adjustment complete: %s -> %s (confidence: %.2f)",
            current_level.name,
            target_level.name,
            confidence,
        )

        return adjustment

    def learn_user_formality_preference(
        self,
        user_id: str,
        conversation_history: List[str],
        context_history: List[ContextType],
        existing_preference: Optional[FormalityPreference] = None,
    ) -> FormalityPreference:
        """
        Learn user's formality preferences from conversation history.

        Args:
            user_id: User identifier
            conversation_history: List of user messages
            context_history: List of context types for each message
            existing_preference: Existing preference to update

        Returns:
            Updated formality preference
        """
        logger.info(
            "Learning formality preferences for user %s from %d messages",
            user_id,
            len(conversation_history),
        )

        if existing_preference:
            preference = existing_preference
        else:
            preference = FormalityPreference(user_id=user_id)

        # Analyze formality patterns by context
        context_patterns = {}
        for message, context_type in zip(conversation_history, context_history):
            if context_type not in context_patterns:
                context_patterns[context_type] = []

            analysis = self.analyze_formality(message)
            context_patterns[context_type].append(analysis.detected_level)

        # Update preferences based on patterns
        for context_type, levels in context_patterns.items():
            if levels:
                # Use most common level for this context
                level_counts = {}
                for level in levels:
                    level_counts[level] = level_counts.get(level, 0) + 1

                most_common_level = max(level_counts, key=level_counts.get)
                preference.set_context_preference(context_type, most_common_level)

        # Calculate overall default level
        all_levels = [level for levels in context_patterns.values() for level in levels]
        if all_levels:
            level_counts = {}
            for level in all_levels:
                level_counts[level] = level_counts.get(level, 0) + 1

            preference.default_level = max(level_counts, key=level_counts.get)

        # Calculate confidence and flexibility
        preference.confidence = min(len(conversation_history) / 20.0, 1.0)
        preference.flexibility = self._calculate_flexibility(context_patterns)

        logger.info(
            "Formality preference learning complete for user %s. "
            "Default level: %s, Confidence: %.2f",
            user_id,
            preference.default_level.name,
            preference.confidence,
        )

        return preference

    def _detect_formality_indicators(self, text: str) -> List[FormalityIndicator]:
        """Detect formality indicators in text."""
        import re

        indicators = []
        text_lower = text.lower()

        for formality_category, categories in self.FORMALITY_INDICATORS.items():
            for indicator_type, patterns in categories.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text_lower)
                    if matches:
                        # Map category to score
                        score_mapping = {
                            "very_casual": -1.0,
                            "casual": -0.5,
                            "formal": 0.5,
                            "very_formal": 1.0,
                        }

                        score = score_mapping.get(formality_category, 0.0)
                        confidence = min(
                            len(matches) / 3.0, 1.0
                        )  # More matches = higher confidence

                        indicator = FormalityIndicator(
                            indicator_type=indicator_type,
                            text_pattern=pattern,
                            formality_score=score,
                            confidence=confidence,
                        )
                        indicators.append(indicator)

        return indicators

    def _calculate_formality_score(self, indicators: List[FormalityIndicator]) -> float:
        """Calculate overall formality score from indicators."""
        if not indicators:
            return 0.0  # Neutral

        # Weighted average of indicator scores
        total_weighted_score = 0.0
        total_weight = 0.0

        for indicator in indicators:
            weight = indicator.confidence
            total_weighted_score += indicator.formality_score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_weighted_score / total_weight

    def _score_to_level(self, score: float) -> FormalityLevel:
        """Convert formality score to formality level."""
        if score <= -0.75:
            return FormalityLevel.VERY_CASUAL
        elif score <= -0.25:
            return FormalityLevel.CASUAL
        elif score <= 0.25:
            return FormalityLevel.NEUTRAL
        elif score <= 0.75:
            return FormalityLevel.FORMAL
        else:
            return FormalityLevel.VERY_FORMAL

    def _detect_context_type(
        self, text: str, context: ConversationContext
    ) -> Optional[ContextType]:
        """Detect conversation context type from text and context."""
        import re

        text_lower = text.lower()

        for context_type, patterns in self.CONTEXT_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return context_type

        # Default context detection based on conversation context
        # This could be enhanced with more sophisticated analysis
        return ContextType.PERSONAL

    def _adjust_for_context(
        self, detected_level: FormalityLevel, context_type: ContextType
    ) -> FormalityLevel:
        """Adjust formality level based on context type."""
        # Context-specific adjustments
        context_adjustments = {
            ContextType.PROFESSIONAL: 1,  # More formal
            ContextType.ACADEMIC: 1,  # More formal
            ContextType.LEGAL: 2,  # Much more formal
            ContextType.MEDICAL: 1,  # More formal
            ContextType.TECHNICAL: 0,  # Neutral
            ContextType.SOCIAL: -1,  # More casual
            ContextType.PERSONAL: -1,  # More casual
        }

        adjustment = context_adjustments.get(context_type, 0)
        new_level_value = max(1, min(5, detected_level.value + adjustment))

        return FormalityLevel(new_level_value)

    def _calculate_analysis_confidence(
        self, indicators: List[FormalityIndicator], text: str
    ) -> float:
        """Calculate confidence in formality analysis."""
        if not indicators:
            return 0.3  # Low confidence with no indicators

        # Base confidence on number and strength of indicators
        avg_confidence = sum(ind.confidence for ind in indicators) / len(indicators)

        # Adjust for text length (longer text = more reliable)
        length_factor = min(len(text.split()) / 20.0, 1.0)

        return min(avg_confidence * length_factor, 1.0)

    def _generate_analysis_reasoning(
        self,
        detected_level: FormalityLevel,
        indicators: List[FormalityIndicator],
        context_type: Optional[ContextType],
    ) -> str:
        """Generate reasoning for formality analysis."""
        reasoning_parts = []

        if indicators:
            reasoning_parts.append(f"Detected {len(indicators)} formality indicators")

        if context_type:
            reasoning_parts.append(f"Context type: {context_type.value}")

        reasoning_parts.append(f"Overall assessment: {detected_level.name}")

        return "; ".join(reasoning_parts)

    def _blend_formality_levels(
        self, level1: FormalityLevel, level2: FormalityLevel, blend_factor: float
    ) -> FormalityLevel:
        """Blend two formality levels based on blend factor."""
        # Simple weighted average
        blended_value = level1.value * (1 - blend_factor) + level2.value * blend_factor
        rounded_value = round(blended_value)
        return FormalityLevel(max(1, min(5, rounded_value)))

    def _get_personality_formality_preference(
        self, personality_profile: PersonalityProfile
    ) -> FormalityLevel:
        """Get formality preference based on personality traits."""
        # Start with neutral
        preference_scores = {level: 0 for level in FormalityLevel}
        preference_scores[FormalityLevel.NEUTRAL] = 1

        # Apply trait-based preferences
        for trait, trait_score in personality_profile.trait_scores.items():
            if trait in self.TRAIT_FORMALITY_MAPPINGS:
                trait_level = trait_score.level
                preferred_formality = self.TRAIT_FORMALITY_MAPPINGS[trait].get(
                    trait_level, FormalityLevel.NEUTRAL
                )

                # Weight by trait confidence
                weight = trait_score.confidence
                preference_scores[preferred_formality] += weight

        # Return level with highest score
        return max(preference_scores, key=preference_scores.get)

    def _adjust_for_emotion(
        self, current_level: FormalityLevel, emotional_state: EmotionalState
    ) -> FormalityLevel:
        """Adjust formality based on emotional state."""
        emotion = emotional_state.primary_emotion.value
        intensity = emotional_state.intensity

        # Emotional adjustments
        if emotion in ["sadness", "fear"]:
            # Be more gentle/formal when user is upset
            adjustment = max(1, round(intensity * 2))
            new_value = min(5, current_level.value + adjustment)
            return FormalityLevel(new_value)

        elif emotion == "anger":
            # Be more formal/respectful when user is angry
            adjustment = max(1, round(intensity * 1.5))
            new_value = min(5, current_level.value + adjustment)
            return FormalityLevel(new_value)

        elif emotion == "joy":
            # Can be slightly more casual when user is happy
            if intensity > 0.7:
                new_value = max(1, current_level.value - 1)
                return FormalityLevel(new_value)

        return current_level

    def _make_more_formal(self, text: str) -> str:
        """Make text more formal."""
        # Simple formalization rules
        replacements = {
            r"\bhi\b": "Hello",
            r"\bhey\b": "Hello",
            r"\bthanks\b": "Thank you",
            r"\byeah\b": "Yes",
            r"\bnope\b": "No",
            r"\bokay\b": "Very well",
            r"\bsure\b": "Certainly",
            r"\bcan't\b": "cannot",
            r"\bwon't\b": "will not",
            r"\bdon't\b": "do not",
            r"\bisn't\b": "is not",
            r"\bI'm\b": "I am",
            r"\byou're\b": "you are",
            r"\bit's\b": "it is",
        }

        import re

        adjusted_text = text
        for pattern, replacement in replacements.items():
            adjusted_text = re.sub(
                pattern, replacement, adjusted_text, flags=re.IGNORECASE
            )

        return adjusted_text

    def _make_more_casual(self, text: str) -> str:
        """Make text more casual."""
        # Simple casualization rules
        replacements = {
            r"\bGood morning\b": "Hi",
            r"\bGood afternoon\b": "Hi",
            r"\bGood evening\b": "Hi",
            r"\bHello\b": "Hi",
            r"\bThank you very much\b": "Thanks",
            r"\bThank you\b": "Thanks",
            r"\bCertainly\b": "Sure",
            r"\bVery well\b": "Okay",
            r"\bI would be pleased to\b": "I'd be happy to",
            r"\bI am\b": "I'm",
            r"\byou are\b": "you're",
            r"\bit is\b": "it's",
            r"\bcannot\b": "can't",
            r"\bwill not\b": "won't",
            r"\bdo not\b": "don't",
            r"\bis not\b": "isn't",
        }

        import re

        adjusted_text = text
        for pattern, replacement in replacements.items():
            adjusted_text = re.sub(
                pattern, replacement, adjusted_text, flags=re.IGNORECASE
            )

        return adjusted_text

    def _calculate_adjustment_confidence(
        self,
        original_text: str,
        adjusted_text: str,
        from_level: FormalityLevel,
        to_level: FormalityLevel,
    ) -> float:
        """Calculate confidence in formality adjustment."""
        # Base confidence on level difference
        level_diff = abs(to_level.value - from_level.value)
        base_confidence = max(0.5, 1.0 - (level_diff * 0.1))

        # Adjust for text change amount
        text_change_ratio = 1.0 - (
            len(set(original_text.split()) & set(adjusted_text.split()))
            / max(len(original_text.split()), 1)
        )

        # More change = lower confidence (might be over-adjusting)
        change_penalty = text_change_ratio * 0.3

        return max(0.3, base_confidence - change_penalty)

    def _generate_adjustment_reasoning(
        self,
        from_level: FormalityLevel,
        to_level: FormalityLevel,
        adjustment_types: List[str],
    ) -> str:
        """Generate reasoning for formality adjustment."""
        direction = "increased" if to_level.value > from_level.value else "decreased"
        return (
            f"Formality {direction} from {from_level.name} to {to_level.name} "
            f"using {', '.join(adjustment_types)}"
        )

    def _calculate_flexibility(
        self, context_patterns: Dict[ContextType, List[FormalityLevel]]
    ) -> float:
        """Calculate user's formality flexibility from patterns."""
        if not context_patterns:
            return 0.5  # Default moderate flexibility

        # Calculate variance in formality levels across contexts
        all_levels = [level for levels in context_patterns.values() for level in levels]
        if len(all_levels) < 2:
            return 0.3  # Low flexibility with limited data

        # Calculate standard deviation of formality levels
        mean_level = sum(level.value for level in all_levels) / len(all_levels)
        variance = sum((level.value - mean_level) ** 2 for level in all_levels) / len(
            all_levels
        )
        std_dev = variance**0.5

        # Normalize to 0-1 range (max std_dev for 5 levels is ~2)
        flexibility = min(std_dev / 2.0, 1.0)

        return flexibility

    def get_formality_description(self, level: FormalityLevel) -> str:
        """Get human-readable description of formality level."""
        descriptions = {
            FormalityLevel.VERY_CASUAL: "Very casual and relaxed communication",
            FormalityLevel.CASUAL: "Casual and friendly communication",
            FormalityLevel.NEUTRAL: "Balanced, neither formal nor casual",
            FormalityLevel.FORMAL: "Formal and professional communication",
            FormalityLevel.VERY_FORMAL: "Very formal and ceremonious communication",
        }

        return descriptions.get(level, f"Formality level: {level.name}")

    def get_context_formality_guidelines(
        self, context_type: ContextType
    ) -> Dict[str, str]:
        """Get formality guidelines for different context types."""
        guidelines = {
            ContextType.PERSONAL: {
                "recommended_level": "Casual to Neutral",
                "description": "Friendly and approachable, matching user's style",
                "avoid": "Overly formal language that creates distance",
            },
            ContextType.PROFESSIONAL: {
                "recommended_level": "Neutral to Formal",
                "description": "Professional and competent, appropriate for work contexts",
                "avoid": "Too casual language that undermines professionalism",
            },
            ContextType.ACADEMIC: {
                "recommended_level": "Formal",
                "description": "Scholarly and precise, appropriate for educational contexts",
                "avoid": "Casual language that undermines academic credibility",
            },
            ContextType.TECHNICAL: {
                "recommended_level": "Neutral to Formal",
                "description": "Clear and precise, focused on accuracy",
                "avoid": "Overly casual language that might confuse technical concepts",
            },
            ContextType.MEDICAL: {
                "recommended_level": "Formal",
                "description": "Professional and reassuring, appropriate for health contexts",
                "avoid": "Casual language that might undermine trust in medical advice",
            },
            ContextType.LEGAL: {
                "recommended_level": "Very Formal",
                "description": "Precise and professional, appropriate for legal contexts",
                "avoid": "Any casual language that might affect legal interpretation",
            },
        }

        return guidelines.get(
            context_type,
            {
                "recommended_level": "Neutral",
                "description": "Balanced approach appropriate for general contexts",
                "avoid": "Extreme formality or casualness without context",
            },
        )
