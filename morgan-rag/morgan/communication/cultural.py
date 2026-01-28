"""
Cultural emotional awareness module.

Provides cultural sensitivity and adaptation for emotional communication
across different cultural contexts, communication styles, and social norms.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.intelligence.core.models import (
    CommunicationStyle,
    ConversationContext,
    EmotionalState,
)
from morgan.services.llm import get_llm_service
from morgan.utils.cache import FileCache
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class CulturalContext(Enum):
    """Cultural context categories."""

    WESTERN_INDIVIDUALISTIC = "western_individualistic"
    EASTERN_COLLECTIVISTIC = "eastern_collectivistic"
    LATIN_EXPRESSIVE = "latin_expressive"
    NORTHERN_RESERVED = "northern_reserved"
    MIDDLE_EASTERN_HIERARCHICAL = "middle_eastern_hierarchical"
    AFRICAN_COMMUNAL = "african_communal"
    MIXED_MULTICULTURAL = "mixed_multicultural"
    UNKNOWN = "unknown"


class CommunicationDirectness(Enum):
    """Communication directness levels."""

    VERY_DIRECT = "very_direct"
    DIRECT = "direct"
    MODERATE = "moderate"
    INDIRECT = "indirect"
    VERY_INDIRECT = "very_indirect"


class EmotionalExpressiveness(Enum):
    """Emotional expressiveness levels."""

    HIGHLY_EXPRESSIVE = "highly_expressive"
    EXPRESSIVE = "expressive"
    MODERATE = "moderate"
    RESERVED = "reserved"
    HIGHLY_RESERVED = "highly_reserved"


@dataclass
class CulturalProfile:
    """Cultural communication profile."""

    cultural_context: CulturalContext
    communication_directness: CommunicationDirectness
    emotional_expressiveness: EmotionalExpressiveness
    formality_preference: str  # formal, casual, context_dependent
    hierarchy_awareness: float  # 0.0 to 1.0
    collectivism_score: float  # 0.0 (individualistic) to 1.0 (collectivistic)
    uncertainty_avoidance: float  # 0.0 to 1.0
    time_orientation: str  # monochronic, polychronic, flexible
    confidence_score: float
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CulturalAdaptation:
    """Cultural adaptation recommendations."""

    adapted_communication_style: CommunicationStyle
    tone_adjustments: List[str]
    formality_level: str
    emotional_sensitivity_notes: List[str]
    cultural_considerations: List[str]
    adaptation_confidence: float
    reasoning: str


class CulturalEmotionalAwareness:
    """
    Cultural emotional awareness system.

    Features:
    - Cultural context detection from communication patterns
    - Cultural adaptation of emotional responses
    - Cross-cultural communication sensitivity
    - Cultural norm awareness and respect
    - Adaptive formality and directness levels
    - Cultural emotional expression interpretation
    """

    def __init__(self):
        """Initialize cultural emotional awareness system."""
        self.settings = get_settings()
        self.llm_service = get_llm_service()

        # Setup cache for cultural profiles
        cache_dir = self.settings.morgan_data_dir / "cache" / "cultural"
        self.cache = FileCache(cache_dir)

        # Cultural profiles storage
        self.user_cultural_profiles: Dict[str, CulturalProfile] = {}

        # Cultural communication patterns
        self.cultural_patterns = self._initialize_cultural_patterns()

        logger.info("Cultural Emotional Awareness system initialized")

    def detect_cultural_context(
        self,
        user_id: str,
        conversation_history: List[ConversationContext],
        emotional_patterns: Dict[str, Any],
    ) -> CulturalProfile:
        """
        Detect user's cultural context from communication patterns.

        Args:
            user_id: User identifier
            conversation_history: History of conversations
            emotional_patterns: Detected emotional patterns

        Returns:
            Cultural profile with detected context
        """
        # Check for cached profile
        cached_profile = self._get_cached_cultural_profile(user_id)
        if cached_profile and self._is_profile_recent(cached_profile):
            return cached_profile

        # Analyze communication patterns
        directness_level = self._analyze_communication_directness(
            conversation_history
        )
        expressiveness_level = self._analyze_emotional_expressiveness(
            conversation_history, emotional_patterns
        )
        formality_preference = self._analyze_formality_preference(
            conversation_history
        )

        # Detect cultural indicators
        cultural_indicators = self._detect_cultural_indicators(conversation_history)

        # Determine cultural context
        cultural_context = self._determine_cultural_context(
            directness_level, expressiveness_level, cultural_indicators
        )

        # Calculate cultural dimensions
        hierarchy_awareness = self._calculate_hierarchy_awareness(
            conversation_history
        )
        collectivism_score = self._calculate_collectivism_score(
            conversation_history
        )
        uncertainty_avoidance = self._calculate_uncertainty_avoidance(
            conversation_history
        )
        time_orientation = self._determine_time_orientation(conversation_history)

        # Calculate confidence
        confidence_score = self._calculate_cultural_confidence(
            conversation_history, cultural_indicators
        )

        # Create cultural profile
        profile = CulturalProfile(
            cultural_context=cultural_context,
            communication_directness=directness_level,
            emotional_expressiveness=expressiveness_level,
            formality_preference=formality_preference,
            hierarchy_awareness=hierarchy_awareness,
            collectivism_score=collectivism_score,
            uncertainty_avoidance=uncertainty_avoidance,
            time_orientation=time_orientation,
            confidence_score=confidence_score,
        )

        # Store and cache profile
        self.user_cultural_profiles[user_id] = profile
        self._cache_cultural_profile(user_id, profile)

        logger.debug(
            "Detected cultural context for user %s: %s, confidence=%.2f",
            user_id, cultural_context.value, confidence_score
        )

        return profile

    def adapt_for_culture(
        self,
        user_id: str,
        emotional_state: EmotionalState,
        context: ConversationContext,
        cultural_profile: Optional[CulturalProfile] = None,
    ) -> CulturalAdaptation:
        """
        Adapt communication for user's cultural context.

        Args:
            user_id: User identifier
            emotional_state: Current emotional state
            context: Conversation context
            cultural_profile: User's cultural profile

        Returns:
            Cultural adaptation recommendations
        """
        if not cultural_profile:
            cultural_profile = self.user_cultural_profiles.get(user_id)
            if not cultural_profile:
                # Create default profile
                cultural_profile = self._create_default_cultural_profile()

        # Adapt communication style
        adapted_style = self._adapt_communication_style(
            cultural_profile, emotional_state
        )

        # Determine tone adjustments
        tone_adjustments = self._determine_tone_adjustments(
            cultural_profile, emotional_state
        )

        # Set formality level
        formality_level = self._determine_formality_level(
            cultural_profile, context, emotional_state
        )

        # Generate emotional sensitivity notes
        sensitivity_notes = self._generate_emotional_sensitivity_notes(
            cultural_profile, emotional_state
        )

        # Generate cultural considerations
        cultural_considerations = self._generate_cultural_considerations(
            cultural_profile, context
        )

        # Calculate adaptation confidence
        adaptation_confidence = self._calculate_adaptation_confidence(
            cultural_profile, emotional_state
        )

        # Generate reasoning
        reasoning = self._generate_adaptation_reasoning(
            cultural_profile, emotional_state, adapted_style
        )

        adaptation = CulturalAdaptation(
            adapted_communication_style=adapted_style,
            tone_adjustments=tone_adjustments,
            formality_level=formality_level,
            emotional_sensitivity_notes=sensitivity_notes,
            cultural_considerations=cultural_considerations,
            adaptation_confidence=adaptation_confidence,
            reasoning=reasoning,
        )

        logger.debug(
            "Cultural adaptation for user %s: style=%s, formality=%s",
            user_id, adapted_style.value, formality_level
        )

        return adaptation

    def get_cultural_insights(
        self, user_id: str, cultural_profile: Optional[CulturalProfile] = None
    ) -> Dict[str, Any]:
        """
        Get cultural insights for better communication.

        Args:
            user_id: User identifier
            cultural_profile: User's cultural profile

        Returns:
            Cultural insights and recommendations
        """
        if not cultural_profile:
            cultural_profile = self.user_cultural_profiles.get(user_id)
            if not cultural_profile:
                return {"error": "No cultural profile available"}

        insights = {
            "cultural_context": cultural_profile.cultural_context.value,
            "communication_preferences": {
                "directness": cultural_profile.communication_directness.value,
                "expressiveness": cultural_profile.emotional_expressiveness.value,
                "formality": cultural_profile.formality_preference,
            },
            "cultural_dimensions": {
                "hierarchy_awareness": cultural_profile.hierarchy_awareness,
                "collectivism_score": cultural_profile.collectivism_score,
                "uncertainty_avoidance": cultural_profile.uncertainty_avoidance,
                "time_orientation": cultural_profile.time_orientation,
            },
            "communication_tips": self._generate_communication_tips(cultural_profile),
            "emotional_considerations": self._generate_emotional_considerations(
                cultural_profile
            ),
            "potential_misunderstandings": self._identify_potential_misunderstandings(
                cultural_profile
            ),
            "confidence_score": cultural_profile.confidence_score,
        }

        return insights

    def _initialize_cultural_patterns(self) -> Dict[str, Any]:
        """Initialize cultural communication patterns."""
        return {
            "directness_indicators": {
                "direct": [
                    "I think",
                    "I believe",
                    "clearly",
                    "obviously",
                    "definitely",
                    "no",
                    "yes",
                ],
                "indirect": [
                    "perhaps",
                    "maybe",
                    "might",
                    "could be",
                    "it seems",
                    "I wonder",
                    "possibly",
                ],
            },
            "formality_indicators": {
                "formal": [
                    "please",
                    "thank you",
                    "would you",
                    "could you",
                    "I would appreciate",
                    "sincerely",
                ],
                "casual": [
                    "hey",
                    "yeah",
                    "ok",
                    "cool",
                    "awesome",
                    "thanks",
                    "sure",
                ],
            },
            "collectivism_indicators": {
                "collectivistic": [
                    "we",
                    "us",
                    "our",
                    "together",
                    "family",
                    "group",
                    "team",
                    "community",
                ],
                "individualistic": [
                    "I",
                    "me",
                    "my",
                    "myself",
                    "personal",
                    "individual",
                    "own",
                ],
            },
            "hierarchy_indicators": {
                "high_hierarchy": [
                    "sir",
                    "madam",
                    "respect",
                    "honor",
                    "authority",
                    "permission",
                ],
                "low_hierarchy": [
                    "equal",
                    "peer",
                    "friend",
                    "buddy",
                    "informal",
                    "casual",
                ],
            },
        }

    def _analyze_communication_directness(
        self, conversation_history: List[ConversationContext]
    ) -> CommunicationDirectness:
        """Analyze communication directness level."""
        if not conversation_history:
            return CommunicationDirectness.MODERATE

        direct_count = 0
        indirect_count = 0

        for context in conversation_history[-20:]:  # Last 20 conversations
            text_lower = context.message_text.lower()

            # Count direct indicators
            direct_count += sum(
                1
                for indicator in self.cultural_patterns["directness_indicators"][
                    "direct"
                ]
                if indicator in text_lower
            )

            # Count indirect indicators
            indirect_count += sum(
                1
                for indicator in self.cultural_patterns["directness_indicators"][
                    "indirect"
                ]
                if indicator in text_lower
            )

        # Determine directness level
        total_indicators = direct_count + indirect_count
        if total_indicators == 0:
            return CommunicationDirectness.MODERATE

        directness_ratio = direct_count / total_indicators

        if directness_ratio >= 0.8:
            return CommunicationDirectness.VERY_DIRECT
        elif directness_ratio >= 0.6:
            return CommunicationDirectness.DIRECT
        elif directness_ratio >= 0.4:
            return CommunicationDirectness.MODERATE
        elif directness_ratio >= 0.2:
            return CommunicationDirectness.INDIRECT
        else:
            return CommunicationDirectness.VERY_INDIRECT

    def _analyze_emotional_expressiveness(
        self,
        conversation_history: List[ConversationContext],
        emotional_patterns: Dict[str, Any],
    ) -> EmotionalExpressiveness:
        """Analyze emotional expressiveness level."""
        if not conversation_history:
            return EmotionalExpressiveness.MODERATE

        # Count emotional expressions
        emotional_expressions = 0
        total_messages = len(conversation_history[-20:])

        for context in conversation_history[-20:]:
            text = context.message_text

            # Count emotional indicators
            if any(
                indicator in text.lower()
                for indicator in [
                    "feel",
                    "emotion",
                    "happy",
                    "sad",
                    "angry",
                    "excited",
                    "frustrated",
                    "love",
                    "hate",
                ]
            ):
                emotional_expressions += 1

            # Count exclamation marks and emojis
            if "!" in text or any(
                emoji in text
                for emoji in ["😊", "😢", "😠", "😍", "😂", "🥰", "😔", "😡"]
            ):
                emotional_expressions += 1

        # Calculate expressiveness ratio
        expressiveness_ratio = emotional_expressions / max(1, total_messages)

        if expressiveness_ratio >= 0.8:
            return EmotionalExpressiveness.HIGHLY_EXPRESSIVE
        elif expressiveness_ratio >= 0.6:
            return EmotionalExpressiveness.EXPRESSIVE
        elif expressiveness_ratio >= 0.4:
            return EmotionalExpressiveness.MODERATE
        elif expressiveness_ratio >= 0.2:
            return EmotionalExpressiveness.RESERVED
        else:
            return EmotionalExpressiveness.HIGHLY_RESERVED

    def _analyze_formality_preference(
        self, conversation_history: List[ConversationContext]
    ) -> str:
        """Analyze formality preference."""
        if not conversation_history:
            return "moderate"

        formal_count = 0
        casual_count = 0

        for context in conversation_history[-20:]:
            text_lower = context.message_text.lower()

            # Count formal indicators
            formal_count += sum(
                1
                for indicator in self.cultural_patterns["formality_indicators"][
                    "formal"
                ]
                if indicator in text_lower
            )

            # Count casual indicators
            casual_count += sum(
                1
                for indicator in self.cultural_patterns["formality_indicators"][
                    "casual"
                ]
                if indicator in text_lower
            )

        if formal_count > casual_count * 1.5:
            return "formal"
        elif casual_count > formal_count * 1.5:
            return "casual"
        else:
            return "context_dependent"

    def _detect_cultural_indicators(
        self, conversation_history: List[ConversationContext]
    ) -> Dict[str, float]:
        """Detect cultural indicators from conversation patterns."""
        indicators = {}

        if not conversation_history:
            return indicators

        # Analyze collectivism vs individualism
        collectivistic_count = 0
        individualistic_count = 0

        for context in conversation_history[-20:]:
            text_lower = context.message_text.lower()

            collectivistic_count += sum(
                1
                for indicator in self.cultural_patterns["collectivism_indicators"][
                    "collectivistic"
                ]
                if indicator in text_lower
            )

            individualistic_count += sum(
                1
                for indicator in self.cultural_patterns["collectivism_indicators"][
                    "individualistic"
                ]
                if indicator in text_lower
            )

        total_collectivism_indicators = collectivistic_count + individualistic_count
        if total_collectivism_indicators > 0:
            indicators["collectivism_score"] = (
                collectivistic_count / total_collectivism_indicators
            )

        # Analyze hierarchy awareness
        high_hierarchy_count = 0
        low_hierarchy_count = 0

        for context in conversation_history[-20:]:
            text_lower = context.message_text.lower()

            high_hierarchy_count += sum(
                1
                for indicator in self.cultural_patterns["hierarchy_indicators"][
                    "high_hierarchy"
                ]
                if indicator in text_lower
            )

            low_hierarchy_count += sum(
                1
                for indicator in self.cultural_patterns["hierarchy_indicators"][
                    "low_hierarchy"
                ]
                if indicator in text_lower
            )

        total_hierarchy_indicators = high_hierarchy_count + low_hierarchy_count
        if total_hierarchy_indicators > 0:
            indicators["hierarchy_awareness"] = (
                high_hierarchy_count / total_hierarchy_indicators
            )

        return indicators

    def _determine_cultural_context(
        self,
        directness: CommunicationDirectness,
        expressiveness: EmotionalExpressiveness,
        indicators: Dict[str, float],
    ) -> CulturalContext:
        """Determine cultural context from communication patterns."""
        # Simple heuristic-based cultural context determination
        collectivism_score = indicators.get("collectivism_score", 0.5)
        hierarchy_awareness = indicators.get("hierarchy_awareness", 0.5)

        # High collectivism + high hierarchy
        if collectivism_score > 0.7 and hierarchy_awareness > 0.7:
            if expressiveness in [
                EmotionalExpressiveness.RESERVED,
                EmotionalExpressiveness.HIGHLY_RESERVED,
            ]:
                return CulturalContext.EASTERN_COLLECTIVISTIC
            else:
                return CulturalContext.MIDDLE_EASTERN_HIERARCHICAL

        # High collectivism + low hierarchy
        elif collectivism_score > 0.7 and hierarchy_awareness < 0.3:
            return CulturalContext.AFRICAN_COMMUNAL

        # Low collectivism + high expressiveness
        elif collectivism_score < 0.3 and expressiveness in [
            EmotionalExpressiveness.EXPRESSIVE,
            EmotionalExpressiveness.HIGHLY_EXPRESSIVE,
        ]:
            return CulturalContext.LATIN_EXPRESSIVE

        # Low collectivism + low expressiveness
        elif collectivism_score < 0.3 and expressiveness in [
            EmotionalExpressiveness.RESERVED,
            EmotionalExpressiveness.HIGHLY_RESERVED,
        ]:
            return CulturalContext.NORTHERN_RESERVED

        # Low collectivism + moderate expressiveness
        elif collectivism_score < 0.3:
            return CulturalContext.WESTERN_INDIVIDUALISTIC

        # Mixed indicators
        else:
            return CulturalContext.MIXED_MULTICULTURAL

    def _calculate_hierarchy_awareness(
        self, conversation_history: List[ConversationContext]
    ) -> float:
        """Calculate hierarchy awareness score."""
        if not conversation_history:
            return 0.5

        hierarchy_indicators = 0
        total_messages = len(conversation_history[-20:])

        for context in conversation_history[-20:]:
            text_lower = context.message_text.lower()

            # Count hierarchy-related terms
            hierarchy_indicators += sum(
                1
                for indicator in self.cultural_patterns["hierarchy_indicators"][
                    "high_hierarchy"
                ]
                if indicator in text_lower
            )

        return min(1.0, hierarchy_indicators / max(1, total_messages))

    def _calculate_collectivism_score(
        self, conversation_history: List[ConversationContext]
    ) -> float:
        """Calculate collectivism score."""
        if not conversation_history:
            return 0.5

        collectivistic_count = 0
        individualistic_count = 0

        for context in conversation_history[-20:]:
            text_lower = context.message_text.lower()

            collectivistic_count += sum(
                1
                for indicator in self.cultural_patterns["collectivism_indicators"][
                    "collectivistic"
                ]
                if indicator in text_lower
            )

            individualistic_count += sum(
                1
                for indicator in self.cultural_patterns["collectivism_indicators"][
                    "individualistic"
                ]
                if indicator in text_lower
            )

        total_indicators = collectivistic_count + individualistic_count
        if total_indicators == 0:
            return 0.5

        return collectivistic_count / total_indicators

    def _calculate_uncertainty_avoidance(
        self, conversation_history: List[ConversationContext]
    ) -> float:
        """Calculate uncertainty avoidance score."""
        if not conversation_history:
            return 0.5

        uncertainty_indicators = 0
        total_messages = len(conversation_history[-20:])

        for context in conversation_history[-20:]:
            text_lower = context.message_text.lower()

            # Count uncertainty-related terms
            if any(
                term in text_lower
                for term in [
                    "sure",
                    "certain",
                    "definitely",
                    "absolutely",
                    "guarantee",
                    "confirm",
                ]
            ):
                uncertainty_indicators += 1

        return min(1.0, uncertainty_indicators / max(1, total_messages))

    def _determine_time_orientation(
        self, conversation_history: List[ConversationContext]
    ) -> str:
        """Determine time orientation preference."""
        if not conversation_history:
            return "flexible"

        time_indicators = {"monochronic": 0, "polychronic": 0}

        for context in conversation_history[-20:]:
            text_lower = context.message_text.lower()

            # Monochronic indicators
            if any(
                term in text_lower
                for term in [
                    "schedule",
                    "time",
                    "deadline",
                    "punctual",
                    "on time",
                    "precise",
                ]
            ):
                time_indicators["monochronic"] += 1

            # Polychronic indicators
            if any(
                term in text_lower
                for term in [
                    "flexible",
                    "whenever",
                    "no rush",
                    "relationship",
                    "people first",
                ]
            ):
                time_indicators["polychronic"] += 1

        if time_indicators["monochronic"] > time_indicators["polychronic"]:
            return "monochronic"
        elif time_indicators["polychronic"] > time_indicators["monochronic"]:
            return "polychronic"
        else:
            return "flexible"

    def _calculate_cultural_confidence(
        self,
        conversation_history: List[ConversationContext],
        cultural_indicators: Dict[str, float],
    ) -> float:
        """Calculate confidence in cultural assessment."""
        confidence_factors = []

        # Conversation history factor
        history_confidence = min(1.0, len(conversation_history) / 20.0)
        confidence_factors.append(history_confidence)

        # Cultural indicators strength
        if cultural_indicators:
            avg_indicator_strength = sum(cultural_indicators.values()) / len(
                cultural_indicators
            )
            confidence_factors.append(avg_indicator_strength)
        else:
            confidence_factors.append(0.3)

        # Message richness factor
        if conversation_history:
            avg_message_length = sum(
                len(ctx.message_text) for ctx in conversation_history[-10:]
            ) / min(10, len(conversation_history))
            richness_confidence = min(1.0, avg_message_length / 100.0)
            confidence_factors.append(richness_confidence)
        else:
            confidence_factors.append(0.3)

        return sum(confidence_factors) / len(confidence_factors)

    def _create_default_cultural_profile(self) -> CulturalProfile:
        """Create default cultural profile."""
        return CulturalProfile(
            cultural_context=CulturalContext.MIXED_MULTICULTURAL,
            communication_directness=CommunicationDirectness.MODERATE,
            emotional_expressiveness=EmotionalExpressiveness.MODERATE,
            formality_preference="context_dependent",
            hierarchy_awareness=0.5,
            collectivism_score=0.5,
            uncertainty_avoidance=0.5,
            time_orientation="flexible",
            confidence_score=0.3,
        )

    def _adapt_communication_style(
        self, cultural_profile: CulturalProfile, emotional_state: EmotionalState
    ) -> CommunicationStyle:
        """Adapt communication style for cultural context."""
        # Base adaptation on cultural context
        style_mapping = {
            CulturalContext.WESTERN_INDIVIDUALISTIC: CommunicationStyle.CASUAL,
            CulturalContext.EASTERN_COLLECTIVISTIC: CommunicationStyle.FORMAL,
            CulturalContext.LATIN_EXPRESSIVE: CommunicationStyle.FRIENDLY,
            CulturalContext.NORTHERN_RESERVED: CommunicationStyle.PROFESSIONAL,
            CulturalContext.MIDDLE_EASTERN_HIERARCHICAL: CommunicationStyle.FORMAL,
            CulturalContext.AFRICAN_COMMUNAL: CommunicationStyle.FRIENDLY,
            CulturalContext.MIXED_MULTICULTURAL: CommunicationStyle.FRIENDLY,
        }

        base_style = style_mapping.get(
            cultural_profile.cultural_context, CommunicationStyle.FRIENDLY
        )

        # Adjust for emotional state
        if emotional_state.intensity > 0.7:
            if emotional_state.primary_emotion.value in ["sadness", "fear"]:
                return CommunicationStyle.FRIENDLY  # Supportive style
            elif emotional_state.primary_emotion.value == "anger":
                return CommunicationStyle.PROFESSIONAL  # Calm style

        return base_style

    def _determine_tone_adjustments(
        self, cultural_profile: CulturalProfile, emotional_state: EmotionalState
    ) -> List[str]:
        """Determine tone adjustments for cultural context."""
        adjustments = []

        # Directness adjustments
        if cultural_profile.communication_directness in [
            CommunicationDirectness.INDIRECT,
            CommunicationDirectness.VERY_INDIRECT,
        ]:
            adjustments.append("Use softer, more indirect language")
            adjustments.append("Include qualifying phrases like 'perhaps' or 'it seems'")

        # Expressiveness adjustments
        if cultural_profile.emotional_expressiveness in [
            EmotionalExpressiveness.RESERVED,
            EmotionalExpressiveness.HIGHLY_RESERVED,
        ]:
            adjustments.append("Maintain emotional restraint")
            adjustments.append("Avoid overly enthusiastic expressions")

        # Hierarchy adjustments
        if cultural_profile.hierarchy_awareness > 0.7:
            adjustments.append("Show appropriate respect and deference")
            adjustments.append("Use formal titles when appropriate")

        # Collectivism adjustments
        if cultural_profile.collectivism_score > 0.7:
            adjustments.append("Emphasize group harmony and consensus")
            adjustments.append("Consider impact on relationships and community")

        return adjustments

    def _determine_formality_level(
        self,
        cultural_profile: CulturalProfile,
        context: ConversationContext,
        emotional_state: EmotionalState,
    ) -> str:
        """Determine appropriate formality level."""
        # Start with user's preference
        base_formality = cultural_profile.formality_preference

        # Adjust for cultural context
        if cultural_profile.cultural_context in [
            CulturalContext.MIDDLE_EASTERN_HIERARCHICAL,
            CulturalContext.EASTERN_COLLECTIVISTIC,
        ]:
            if base_formality == "casual":
                return "moderate"
            else:
                return "formal"

        # Adjust for hierarchy awareness
        if cultural_profile.hierarchy_awareness > 0.8:
            return "formal"

        # Adjust for emotional state
        if emotional_state.intensity > 0.7 and emotional_state.primary_emotion.value in [
            "sadness",
            "fear",
            "anger",
        ]:
            return "supportive"  # Special formality for emotional support

        return base_formality

    def _generate_emotional_sensitivity_notes(
        self, cultural_profile: CulturalProfile, emotional_state: EmotionalState
    ) -> List[str]:
        """Generate emotional sensitivity notes."""
        notes = []

        # Cultural emotional expression norms
        if cultural_profile.emotional_expressiveness in [
            EmotionalExpressiveness.RESERVED,
            EmotionalExpressiveness.HIGHLY_RESERVED,
        ]:
            notes.append(
                "User may prefer subtle emotional acknowledgment rather than direct emotional discussion"
            )
            notes.append("Respect emotional privacy and avoid probing deeply")

        # Collectivistic emotional considerations
        if cultural_profile.collectivism_score > 0.7:
            notes.append(
                "Consider how emotions might affect family/group relationships"
            )
            notes.append("Frame emotional support in terms of community well-being")

        # High hierarchy emotional considerations
        if cultural_profile.hierarchy_awareness > 0.7:
            notes.append("Maintain respectful distance while offering support")
            notes.append("Avoid appearing to challenge user's emotional authority")

        # Current emotional state considerations
        if emotional_state.intensity > 0.7:
            if cultural_profile.cultural_context == CulturalContext.EASTERN_COLLECTIVISTIC:
                notes.append("Offer support while respecting face-saving needs")
            elif cultural_profile.cultural_context == CulturalContext.NORTHERN_RESERVED:
                notes.append("Provide practical support rather than emotional expression")

        return notes

    def _generate_cultural_considerations(
        self, cultural_profile: CulturalProfile, context: ConversationContext
    ) -> List[str]:
        """Generate cultural considerations."""
        considerations = []

        # Time orientation considerations
        if cultural_profile.time_orientation == "polychronic":
            considerations.append("Allow for flexible timing and relationship focus")
        elif cultural_profile.time_orientation == "monochronic":
            considerations.append("Respect time constraints and scheduling preferences")

        # Uncertainty avoidance considerations
        if cultural_profile.uncertainty_avoidance > 0.7:
            considerations.append("Provide clear, definitive information when possible")
            considerations.append("Acknowledge uncertainty explicitly when it exists")

        # Cultural context specific considerations
        context_considerations = {
            CulturalContext.WESTERN_INDIVIDUALISTIC: [
                "Respect individual autonomy and personal choice"
            ],
            CulturalContext.EASTERN_COLLECTIVISTIC: [
                "Consider group harmony and consensus-building"
            ],
            CulturalContext.LATIN_EXPRESSIVE: [
                "Welcome emotional expression and personal sharing"
            ],
            CulturalContext.NORTHERN_RESERVED: [
                "Maintain professional boundaries and emotional restraint"
            ],
            CulturalContext.MIDDLE_EASTERN_HIERARCHICAL: [
                "Show appropriate respect for authority and tradition"
            ],
            CulturalContext.AFRICAN_COMMUNAL: [
                "Emphasize community support and collective well-being"
            ],
        }

        considerations.extend(
            context_considerations.get(cultural_profile.cultural_context, [])
        )

        return considerations

    def _calculate_adaptation_confidence(
        self, cultural_profile: CulturalProfile, emotional_state: EmotionalState
    ) -> float:
        """Calculate confidence in cultural adaptation."""
        confidence_factors = []

        # Cultural profile confidence
        confidence_factors.append(cultural_profile.confidence_score)

        # Emotional state confidence
        confidence_factors.append(emotional_state.confidence)

        # Cultural context specificity
        if cultural_profile.cultural_context != CulturalContext.UNKNOWN:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)

        return sum(confidence_factors) / len(confidence_factors)

    def _generate_adaptation_reasoning(
        self,
        cultural_profile: CulturalProfile,
        emotional_state: EmotionalState,
        adapted_style: CommunicationStyle,
    ) -> str:
        """Generate reasoning for cultural adaptation."""
        reasoning_parts = []

        reasoning_parts.append(
            f"Adapted for {cultural_profile.cultural_context.value} cultural context"
        )

        reasoning_parts.append(
            f"Communication directness: {cultural_profile.communication_directness.value}"
        )

        reasoning_parts.append(
            f"Emotional expressiveness: {cultural_profile.emotional_expressiveness.value}"
        )

        reasoning_parts.append(f"Selected {adapted_style.value} communication style")

        if emotional_state.intensity > 0.6:
            reasoning_parts.append(
                f"Adjusted for {emotional_state.primary_emotion.value} emotion"
            )

        return "; ".join(reasoning_parts)

    def _generate_communication_tips(
        self, cultural_profile: CulturalProfile
    ) -> List[str]:
        """Generate communication tips for cultural context."""
        tips = []

        # Directness tips
        if cultural_profile.communication_directness == CommunicationDirectness.DIRECT:
            tips.append("Be clear and straightforward in communication")
        elif cultural_profile.communication_directness == CommunicationDirectness.INDIRECT:
            tips.append("Use gentle, indirect language and allow for interpretation")

        # Formality tips
        if cultural_profile.formality_preference == "formal":
            tips.append("Maintain formal language and respectful tone")
        elif cultural_profile.formality_preference == "casual":
            tips.append("Use friendly, informal language")

        # Cultural context tips
        context_tips = {
            CulturalContext.WESTERN_INDIVIDUALISTIC: [
                "Focus on personal achievements and individual goals"
            ],
            CulturalContext.EASTERN_COLLECTIVISTIC: [
                "Emphasize group harmony and collective benefits"
            ],
            CulturalContext.LATIN_EXPRESSIVE: [
                "Welcome emotional expression and personal warmth"
            ],
            CulturalContext.NORTHERN_RESERVED: [
                "Maintain professional demeanor and respect personal space"
            ],
        }

        tips.extend(context_tips.get(cultural_profile.cultural_context, []))

        return tips

    def _generate_emotional_considerations(
        self, cultural_profile: CulturalProfile
    ) -> List[str]:
        """Generate emotional considerations for cultural context."""
        considerations = []

        if cultural_profile.emotional_expressiveness == EmotionalExpressiveness.RESERVED:
            considerations.append("Respect emotional privacy and restraint")
            considerations.append("Offer support subtly without overwhelming")

        if cultural_profile.collectivism_score > 0.7:
            considerations.append("Consider impact on family and community relationships")

        if cultural_profile.hierarchy_awareness > 0.7:
            considerations.append("Maintain appropriate respect and deference")

        return considerations

    def _identify_potential_misunderstandings(
        self, cultural_profile: CulturalProfile
    ) -> List[str]:
        """Identify potential cultural misunderstandings."""
        misunderstandings = []

        # Directness misunderstandings
        if cultural_profile.communication_directness == CommunicationDirectness.INDIRECT:
            misunderstandings.append(
                "Direct communication might be perceived as rude or aggressive"
            )

        # Hierarchy misunderstandings
        if cultural_profile.hierarchy_awareness > 0.7:
            misunderstandings.append(
                "Casual or informal approach might be seen as disrespectful"
            )

        # Collectivism misunderstandings
        if cultural_profile.collectivism_score > 0.7:
            misunderstandings.append(
                "Individual-focused advice might conflict with group harmony values"
            )

        # Time orientation misunderstandings
        if cultural_profile.time_orientation == "polychronic":
            misunderstandings.append(
                "Strict time focus might seem impersonal or relationship-damaging"
            )

        return misunderstandings

    def _get_cached_cultural_profile(self, user_id: str) -> Optional[CulturalProfile]:
        """Get cached cultural profile."""
        cache_key = f"cultural_profile_{user_id}"
        cached_data = self.cache.get(cache_key)

        if cached_data:
            return CulturalProfile(
                cultural_context=CulturalContext(cached_data["cultural_context"]),
                communication_directness=CommunicationDirectness(
                    cached_data["communication_directness"]
                ),
                emotional_expressiveness=EmotionalExpressiveness(
                    cached_data["emotional_expressiveness"]
                ),
                formality_preference=cached_data["formality_preference"],
                hierarchy_awareness=cached_data["hierarchy_awareness"],
                collectivism_score=cached_data["collectivism_score"],
                uncertainty_avoidance=cached_data["uncertainty_avoidance"],
                time_orientation=cached_data["time_orientation"],
                confidence_score=cached_data["confidence_score"],
                last_updated=datetime.fromisoformat(cached_data["last_updated"]),
            )

        return None

    def _cache_cultural_profile(
        self, user_id: str, profile: CulturalProfile
    ) -> None:
        """Cache cultural profile."""
        cache_key = f"cultural_profile_{user_id}"
        profile_dict = {
            "cultural_context": profile.cultural_context.value,
            "communication_directness": profile.communication_directness.value,
            "emotional_expressiveness": profile.emotional_expressiveness.value,
            "formality_preference": profile.formality_preference,
            "hierarchy_awareness": profile.hierarchy_awareness,
            "collectivism_score": profile.collectivism_score,
            "uncertainty_avoidance": profile.uncertainty_avoidance,
            "time_orientation": profile.time_orientation,
            "confidence_score": profile.confidence_score,
            "last_updated": profile.last_updated.isoformat(),
        }
        self.cache.set(cache_key, profile_dict)

    def _is_profile_recent(self, profile: CulturalProfile) -> bool:
        """Check if cultural profile is recent enough to use."""
        # Consider profile recent if updated within last 30 days
        age = datetime.utcnow() - profile.last_updated
        return age.days < 30


# Singleton instance
_cultural_emotional_awareness_instance = None
_cultural_emotional_awareness_lock = threading.Lock()


def get_cultural_emotional_awareness() -> CulturalEmotionalAwareness:
    """
    Get singleton cultural emotional awareness instance.

    Returns:
        Shared CulturalEmotionalAwareness instance
    """
    global _cultural_emotional_awareness_instance

    if _cultural_emotional_awareness_instance is None:
        with _cultural_emotional_awareness_lock:
            if _cultural_emotional_awareness_instance is None:
                _cultural_emotional_awareness_instance = CulturalEmotionalAwareness()

    return _cultural_emotional_awareness_instance