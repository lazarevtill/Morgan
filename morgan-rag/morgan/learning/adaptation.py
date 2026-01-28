"""
Behavioral Adaptation Engine for Morgan RAG.

Adapts assistant behavior based on learned patterns and preferences,
including response style adaptation and content selection customization.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from ..intelligence.core.models import (
    CommunicationStyle,
    CompanionProfile,
    ConversationContext,
    ResponseLength,
)
from ..utils.logger import get_logger
from .preferences import PreferenceCategory, UserPreferenceProfile

logger = get_logger(__name__)


class AdaptationType(Enum):
    """Types of behavioral adaptations."""

    RESPONSE_STYLE = "response_style"
    CONTENT_SELECTION = "content_selection"
    SEARCH_WEIGHTING = "search_weighting"
    EMOTIONAL_TONE = "emotional_tone"
    INTERACTION_FLOW = "interaction_flow"
    VOCABULARY_LEVEL = "vocabulary_level"


class AdaptationStrategy(Enum):
    """Strategies for applying adaptations."""

    IMMEDIATE = "immediate"  # Apply immediately
    GRADUAL = "gradual"  # Apply gradually over time
    CONDITIONAL = "conditional"  # Apply based on conditions
    EXPERIMENTAL = "experimental"  # Test and measure effectiveness


@dataclass
class Adaptation:
    """Represents a specific behavioral adaptation."""

    adaptation_id: str
    adaptation_type: AdaptationType
    parameter: str
    old_value: Any
    new_value: Any
    confidence: float  # 0.0 to 1.0
    strategy: AdaptationStrategy
    reasoning: str
    expected_impact: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Initialize adaptation ID if not provided."""
        if not self.adaptation_id:
            self.adaptation_id = str(uuid.uuid4())


@dataclass
class AdaptationResult:
    """Result of behavioral adaptation process."""

    user_id: str
    adaptations: List[Adaptation]
    confidence_score: float  # Overall confidence in adaptations
    reasoning: str
    expected_improvements: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ResponseStyleAdapter:
    """
    Adapts response style based on user preferences and patterns.

    Handles formality level, technical depth, response length,
    and emotional tone adaptations.
    """

    def __init__(self):
        """Initialize response style adapter."""
        logger.debug("Response style adapter initialized")

    def adapt_response_style(
        self,
        user_id: str,
        context: ConversationContext,
        user_profile: CompanionProfile,
        preference_profile: UserPreferenceProfile,
    ) -> List[Adaptation]:
        """
        Adapt response style based on user preferences.

        Args:
            user_id: User identifier
            context: Current conversation context
            user_profile: User's companion profile
            preference_profile: User's preference profile

        Returns:
            List[Adaptation]: Style adaptations to apply
        """
        adaptations = []

        # Adapt formality level
        formality_adaptations = self._adapt_formality_level(
            user_id, preference_profile, user_profile
        )
        adaptations.extend(formality_adaptations)

        # Adapt technical depth
        technical_adaptations = self._adapt_technical_depth(
            user_id, preference_profile, context
        )
        adaptations.extend(technical_adaptations)

        # Adapt response length
        length_adaptations = self._adapt_response_length(
            user_id, preference_profile, user_profile
        )
        adaptations.extend(length_adaptations)

        # Adapt emotional tone
        emotional_adaptations = self._adapt_emotional_tone(
            user_id, preference_profile, user_profile, context
        )
        adaptations.extend(emotional_adaptations)

        # Adapt vocabulary level
        vocabulary_adaptations = self._adapt_vocabulary_level(
            user_id, preference_profile, context
        )
        adaptations.extend(vocabulary_adaptations)

        return adaptations

    def _adapt_formality_level(
        self,
        user_id: str,
        preference_profile: UserPreferenceProfile,
        user_profile: CompanionProfile,
    ) -> List[Adaptation]:
        """Adapt formality level based on preferences."""
        adaptations = []

        # Get formality preference
        formality_pref = preference_profile.get_preference(
            PreferenceCategory.COMMUNICATION, "formality_level"
        )

        if formality_pref:
            confidence = preference_profile.get_confidence(
                PreferenceCategory.COMMUNICATION, "formality_level"
            )

            current_style = user_profile.communication_preferences.communication_style

            # Determine target formality
            if (
                formality_pref == "formal"
                and current_style != CommunicationStyle.FORMAL
            ):
                adaptations.append(
                    Adaptation(
                        adaptation_id=str(uuid.uuid4()),
                        adaptation_type=AdaptationType.RESPONSE_STYLE,
                        parameter="formality_level",
                        old_value=current_style.value,
                        new_value="formal",
                        confidence=confidence,
                        strategy=AdaptationStrategy.GRADUAL,
                        reasoning=f"User prefers formal communication style (confidence: {confidence:.2f})",
                        expected_impact="More polite and professional responses",
                    )
                )
            elif (
                formality_pref == "casual"
                and current_style != CommunicationStyle.CASUAL
            ):
                adaptations.append(
                    Adaptation(
                        adaptation_id=str(uuid.uuid4()),
                        adaptation_type=AdaptationType.RESPONSE_STYLE,
                        parameter="formality_level",
                        old_value=current_style.value,
                        new_value="casual",
                        confidence=confidence,
                        strategy=AdaptationStrategy.GRADUAL,
                        reasoning=f"User prefers casual communication style (confidence: {confidence:.2f})",
                        expected_impact="More relaxed and friendly responses",
                    )
                )

        return adaptations

    def _adapt_technical_depth(
        self,
        user_id: str,
        preference_profile: UserPreferenceProfile,
        context: ConversationContext,
    ) -> List[Adaptation]:
        """Adapt technical depth based on preferences."""
        adaptations = []

        # Get technical depth preference
        tech_depth = preference_profile.get_preference(
            PreferenceCategory.COMMUNICATION, "technical_depth"
        )

        if tech_depth:
            confidence = preference_profile.get_confidence(
                PreferenceCategory.COMMUNICATION, "technical_depth"
            )

            if tech_depth == "high":
                adaptations.append(
                    Adaptation(
                        adaptation_id=str(uuid.uuid4()),
                        adaptation_type=AdaptationType.RESPONSE_STYLE,
                        parameter="technical_depth",
                        old_value="medium",
                        new_value="high",
                        confidence=confidence,
                        strategy=AdaptationStrategy.CONDITIONAL,
                        reasoning=f"User prefers technical explanations (confidence: {confidence:.2f})",
                        expected_impact="More detailed technical information and terminology",
                    )
                )
            elif tech_depth == "low":
                adaptations.append(
                    Adaptation(
                        adaptation_id=str(uuid.uuid4()),
                        adaptation_type=AdaptationType.RESPONSE_STYLE,
                        parameter="technical_depth",
                        old_value="medium",
                        new_value="low",
                        confidence=confidence,
                        strategy=AdaptationStrategy.CONDITIONAL,
                        reasoning=f"User prefers simple explanations (confidence: {confidence:.2f})",
                        expected_impact="Simpler language and more accessible explanations",
                    )
                )

        return adaptations

    def _adapt_response_length(
        self,
        user_id: str,
        preference_profile: UserPreferenceProfile,
        user_profile: CompanionProfile,
    ) -> List[Adaptation]:
        """Adapt response length based on preferences."""
        adaptations = []

        # Get response length preference
        length_pref = preference_profile.get_preference(
            PreferenceCategory.CONTENT, "response_length"
        )

        if length_pref:
            confidence = preference_profile.get_confidence(
                PreferenceCategory.CONTENT, "response_length"
            )

            current_length = (
                user_profile.communication_preferences.preferred_response_length
            )

            if length_pref == "brief" and current_length != ResponseLength.BRIEF:
                adaptations.append(
                    Adaptation(
                        adaptation_id=str(uuid.uuid4()),
                        adaptation_type=AdaptationType.RESPONSE_STYLE,
                        parameter="response_length",
                        old_value=current_length.value,
                        new_value="brief",
                        confidence=confidence,
                        strategy=AdaptationStrategy.IMMEDIATE,
                        reasoning=f"User prefers brief responses (confidence: {confidence:.2f})",
                        expected_impact="Shorter, more concise responses",
                    )
                )
            elif (
                length_pref == "detailed"
                and current_length != ResponseLength.COMPREHENSIVE
            ):
                adaptations.append(
                    Adaptation(
                        adaptation_id=str(uuid.uuid4()),
                        adaptation_type=AdaptationType.RESPONSE_STYLE,
                        parameter="response_length",
                        old_value=current_length.value,
                        new_value="detailed",
                        confidence=confidence,
                        strategy=AdaptationStrategy.IMMEDIATE,
                        reasoning=f"User prefers detailed responses (confidence: {confidence:.2f})",
                        expected_impact="More comprehensive and thorough responses",
                    )
                )

        return adaptations

    def _adapt_emotional_tone(
        self,
        user_id: str,
        preference_profile: UserPreferenceProfile,
        user_profile: CompanionProfile,
        context: ConversationContext,
    ) -> List[Adaptation]:
        """Adapt emotional tone based on preferences."""
        adaptations = []

        # Get emotional preferences
        intensity_pref = preference_profile.get_preference(
            PreferenceCategory.EMOTIONAL, "emotional_intensity_comfort"
        )

        if intensity_pref:
            confidence = preference_profile.get_confidence(
                PreferenceCategory.EMOTIONAL, "emotional_intensity_comfort"
            )

            if intensity_pref == "high":
                adaptations.append(
                    Adaptation(
                        adaptation_id=str(uuid.uuid4()),
                        adaptation_type=AdaptationType.EMOTIONAL_TONE,
                        parameter="emotional_intensity",
                        old_value="medium",
                        new_value="high",
                        confidence=confidence,
                        strategy=AdaptationStrategy.GRADUAL,
                        reasoning=f"User comfortable with high emotional intensity (confidence: {confidence:.2f})",
                        expected_impact="More expressive and emotionally engaged responses",
                    )
                )
            elif intensity_pref == "low":
                adaptations.append(
                    Adaptation(
                        adaptation_id=str(uuid.uuid4()),
                        adaptation_type=AdaptationType.EMOTIONAL_TONE,
                        parameter="emotional_intensity",
                        old_value="medium",
                        new_value="low",
                        confidence=confidence,
                        strategy=AdaptationStrategy.GRADUAL,
                        reasoning=f"User prefers low emotional intensity (confidence: {confidence:.2f})",
                        expected_impact="More neutral and measured emotional responses",
                    )
                )

        return adaptations

    def _adapt_vocabulary_level(
        self,
        user_id: str,
        preference_profile: UserPreferenceProfile,
        context: ConversationContext,
    ) -> List[Adaptation]:
        """Adapt vocabulary level based on user patterns."""
        adaptations = []

        # This would analyze user's vocabulary level from conversations
        # and adapt the assistant's vocabulary accordingly

        # For now, return empty list - would be implemented based on
        # vocabulary analysis from conversation patterns

        return adaptations


class ContentSelectionAdapter:
    """
    Adapts content selection and search weighting based on user preferences.

    Handles topic preferences, search result ranking, and content filtering.
    """

    def __init__(self):
        """Initialize content selection adapter."""
        logger.debug("Content selection adapter initialized")

    def adapt_content_selection(
        self,
        user_id: str,
        context: ConversationContext,
        user_profile: CompanionProfile,
        preference_profile: UserPreferenceProfile,
    ) -> List[Adaptation]:
        """
        Adapt content selection based on user preferences.

        Args:
            user_id: User identifier
            context: Current conversation context
            user_profile: User's companion profile
            preference_profile: User's preference profile

        Returns:
            List[Adaptation]: Content selection adaptations
        """
        adaptations = []

        # Adapt search weighting
        search_adaptations = self._adapt_search_weighting(
            user_id, preference_profile, context
        )
        adaptations.extend(search_adaptations)

        # Adapt content filtering
        filter_adaptations = self._adapt_content_filtering(
            user_id, preference_profile, user_profile
        )
        adaptations.extend(filter_adaptations)

        # Adapt topic prioritization
        topic_adaptations = self._adapt_topic_prioritization(
            user_id, preference_profile, context
        )
        adaptations.extend(topic_adaptations)

        return adaptations

    def _adapt_search_weighting(
        self,
        user_id: str,
        preference_profile: UserPreferenceProfile,
        context: ConversationContext,
    ) -> List[Adaptation]:
        """Adapt search result weighting based on preferences."""
        adaptations = []

        # Get topic preferences
        topic_prefs = preference_profile.preferences.get(
            PreferenceCategory.TOPICS.value, {}
        )

        if topic_prefs:
            # Create search weight adaptations for preferred topics
            for pref_key, pref_value in topic_prefs.items():
                if pref_key.startswith("interest_") and isinstance(
                    pref_value, (int, float)
                ):
                    topic = pref_key.replace("interest_", "")
                    confidence = preference_profile.get_confidence(
                        PreferenceCategory.TOPICS, pref_key
                    )

                    if pref_value > 0.3:  # Significant interest
                        adaptations.append(
                            Adaptation(
                                adaptation_id=str(uuid.uuid4()),
                                adaptation_type=AdaptationType.SEARCH_WEIGHTING,
                                parameter=f"topic_weight_{topic}",
                                old_value=1.0,
                                new_value=1.0 + pref_value,
                                confidence=confidence,
                                strategy=AdaptationStrategy.IMMEDIATE,
                                reasoning=f"User shows strong interest in {topic} (score: {pref_value:.2f})",
                                expected_impact=f"Higher ranking for {topic}-related content",
                            )
                        )

        # Adapt based on learning focus
        if preference_profile.get_preference(
            PreferenceCategory.TOPICS, "learning_focus"
        ):
            confidence = preference_profile.get_confidence(
                PreferenceCategory.TOPICS, "learning_focus"
            )

            adaptations.append(
                Adaptation(
                    adaptation_id=str(uuid.uuid4()),
                    adaptation_type=AdaptationType.SEARCH_WEIGHTING,
                    parameter="learning_content_boost",
                    old_value=1.0,
                    new_value=1.3,
                    confidence=confidence,
                    strategy=AdaptationStrategy.IMMEDIATE,
                    reasoning="User focuses on learning-related content",
                    expected_impact="Higher ranking for educational and tutorial content",
                )
            )

        return adaptations

    def _adapt_content_filtering(
        self,
        user_id: str,
        preference_profile: UserPreferenceProfile,
        user_profile: CompanionProfile,
    ) -> List[Adaptation]:
        """Adapt content filtering based on preferences."""
        adaptations = []

        # Get content preferences
        examples_pref = preference_profile.get_preference(
            PreferenceCategory.CONTENT, "examples_preferred"
        )

        if examples_pref:
            confidence = preference_profile.get_confidence(
                PreferenceCategory.CONTENT, "examples_preferred"
            )

            adaptations.append(
                Adaptation(
                    adaptation_id=str(uuid.uuid4()),
                    adaptation_type=AdaptationType.CONTENT_SELECTION,
                    parameter="prioritize_examples",
                    old_value=False,
                    new_value=True,
                    confidence=confidence,
                    strategy=AdaptationStrategy.IMMEDIATE,
                    reasoning="User prefers content with examples",
                    expected_impact="Prioritize content that includes examples and demonstrations",
                )
            )

        return adaptations

    def _adapt_topic_prioritization(
        self,
        user_id: str,
        preference_profile: UserPreferenceProfile,
        context: ConversationContext,
    ) -> List[Adaptation]:
        """Adapt topic prioritization based on user interests."""
        adaptations = []

        # Get work/personal focus preferences
        work_focus = preference_profile.get_preference(
            PreferenceCategory.TOPICS, "work_focus"
        )
        personal_focus = preference_profile.get_preference(
            PreferenceCategory.TOPICS, "personal_focus"
        )

        if work_focus and work_focus > 0.3:
            confidence = preference_profile.get_confidence(
                PreferenceCategory.TOPICS, "work_focus"
            )

            adaptations.append(
                Adaptation(
                    adaptation_id=str(uuid.uuid4()),
                    adaptation_type=AdaptationType.CONTENT_SELECTION,
                    parameter="work_content_priority",
                    old_value=1.0,
                    new_value=1.2,
                    confidence=confidence,
                    strategy=AdaptationStrategy.CONDITIONAL,
                    reasoning="User focuses on work-related topics",
                    expected_impact="Higher priority for professional and work-related content",
                )
            )

        if personal_focus and personal_focus > 0.3:
            confidence = preference_profile.get_confidence(
                PreferenceCategory.TOPICS, "personal_focus"
            )

            adaptations.append(
                Adaptation(
                    adaptation_id=str(uuid.uuid4()),
                    adaptation_type=AdaptationType.CONTENT_SELECTION,
                    parameter="personal_content_priority",
                    old_value=1.0,
                    new_value=1.2,
                    confidence=confidence,
                    strategy=AdaptationStrategy.CONDITIONAL,
                    reasoning="User focuses on personal topics",
                    expected_impact="Higher priority for personal and lifestyle content",
                )
            )

        return adaptations


class BehavioralAdaptationEngine:
    """
    Main behavioral adaptation engine.

    Coordinates response style and content selection adaptations
    based on learned user patterns and preferences.
    """

    def __init__(self):
        """Initialize behavioral adaptation engine."""
        self.response_adapter = ResponseStyleAdapter()
        self.content_adapter = ContentSelectionAdapter()

        # Track adaptation history
        self.adaptation_history: Dict[str, List[AdaptationResult]] = {}

        logger.info("Behavioral adaptation engine initialized")

    def adapt_behavior(
        self,
        user_id: str,
        context: ConversationContext,
        user_profile: CompanionProfile,
        preference_profile: UserPreferenceProfile,
    ) -> AdaptationResult:
        """
        Apply behavioral adaptations based on user preferences.

        Args:
            user_id: User identifier
            context: Current conversation context
            user_profile: User's companion profile
            preference_profile: User's preference profile

        Returns:
            AdaptationResult: Applied adaptations and results
        """
        logger.info(f"Adapting behavior for user {user_id}")

        all_adaptations = []

        # Get response style adaptations
        style_adaptations = self.response_adapter.adapt_response_style(
            user_id, context, user_profile, preference_profile
        )
        all_adaptations.extend(style_adaptations)

        # Get content selection adaptations
        content_adaptations = self.content_adapter.adapt_content_selection(
            user_id, context, user_profile, preference_profile
        )
        all_adaptations.extend(content_adaptations)

        # Calculate overall confidence
        if all_adaptations:
            overall_confidence = sum(a.confidence for a in all_adaptations) / len(
                all_adaptations
            )
        else:
            overall_confidence = 0.0

        # Generate reasoning
        reasoning = self._generate_adaptation_reasoning(all_adaptations)

        # Generate expected improvements
        expected_improvements = [a.expected_impact for a in all_adaptations]

        result = AdaptationResult(
            user_id=user_id,
            adaptations=all_adaptations,
            confidence_score=overall_confidence,
            reasoning=reasoning,
            expected_improvements=expected_improvements,
        )

        # Store in history
        if user_id not in self.adaptation_history:
            self.adaptation_history[user_id] = []
        self.adaptation_history[user_id].append(result)

        logger.info(f"Applied {len(all_adaptations)} adaptations for user {user_id}")
        return result

    def update_strategies(self, user_id: str, changes: List[Dict[str, Any]]):
        """
        Update adaptation strategies based on feedback.

        Args:
            user_id: User identifier
            changes: List of strategy changes to apply
        """
        logger.info(f"Updating adaptation strategies for user {user_id}")

        # This would update the adaptation strategies based on
        # feedback about their effectiveness

        for change in changes:
            logger.debug(f"Strategy change: {change}")

    def get_adaptation_history(self, user_id: str) -> List[AdaptationResult]:
        """
        Get adaptation history for a user.

        Args:
            user_id: User identifier

        Returns:
            List[AdaptationResult]: Adaptation history
        """
        return self.adaptation_history.get(user_id, [])

    def _generate_adaptation_reasoning(self, adaptations: List[Adaptation]) -> str:
        """Generate overall reasoning for adaptations."""
        if not adaptations:
            return "No adaptations applied - insufficient preference data"

        adaptation_types = {a.adaptation_type for a in adaptations}
        type_counts = {
            t: sum(1 for a in adaptations if a.adaptation_type == t)
            for t in adaptation_types
        }

        reasoning_parts = []
        for adapt_type, count in type_counts.items():
            reasoning_parts.append(f"{count} {adapt_type.value} adaptation(s)")

        return f"Applied {len(adaptations)} total adaptations: " + ", ".join(
            reasoning_parts
        )
