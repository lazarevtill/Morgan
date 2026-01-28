"""
Humor Detection and Generation for Morgan RAG.

Detects user humor preferences and generates appropriate humorous responses
based on personality traits, context, and relationship dynamics.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from ..utils.logger import get_logger
from .traits import PersonalityProfile

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
            r"\b(clever|smart|brilliant|genius)\b",
            r"\b(ironic|irony|sarcastic|sarcasm)\b",
            r"[😏😉🤓]",
        ],
        HumorStyle.PLAYFUL: [
            r"\b(fun|funny|hilarious|amusing|entertaining)\b",
            r"\b(silly|goofy|playful|cheerful)\b",
            r"[😄😆😂🤣😊]",
        ],
        HumorStyle.DRY: [
            r"\b(dry|deadpan|understated|subtle)\b",
            r"\b(matter.of.fact|straightforward)\b",
            r"[😐😑]",
        ],
        HumorStyle.PUNNY: [r"\b(pun|wordplay|play.on.words)\b", r"[🙄😅]"],
        HumorStyle.GENTLE: [
            r"\b(sweet|kind|gentle|wholesome)\b",
            r"\b(nice|pleasant|friendly)\b",
            r"[😌☺️🙂]",
        ],
    }

    # Positive reaction indicators
    POSITIVE_REACTIONS = [
        r"\b(haha|lol|lmao|rofl)\b",
        r"\b(funny|hilarious|amusing|clever)\b",
        r"\b(love|like|enjoy|appreciate)\b.*\b(humor|joke|funny)\b",
        r"[😂🤣😄😆😊😁]",
    ]

    # Negative reaction indicators
    NEGATIVE_REACTIONS = [
        r"\b(not funny|unfunny|inappropriate)\b",
        r"\b(serious|professional|formal)\b.*\b(please|prefer)\b",
        r"\b(stop|enough|no more)\b.*\b(joke|humor|funny)\b",
        r"[😐😑🙄😒]",
    ]

    def __init__(self):
        """Initialize humor detector."""
        logger.info("Humor detector initialized")

    def analyze_humor_preferences(
        self,
        user_id: str,
        conversation_history: List[str],
        user_reactions: List[str],
        personality_profile: Optional[PersonalityProfile] = None,
    ) -> HumorPreference:
        """
        Analyze user humor preferences from conversation history.

        Args:
            user_id: User identifier
            conversation_history: List of user messages
            user_reactions: List of user reactions to humor
            personality_profile: Optional personality profile

        Returns:
            HumorPreference object with detected preferences
        """
        logger.info(
            "Analyzing humor preferences for user %s from %d messages",
            user_id,
            len(conversation_history),
        )

        preference = HumorPreference()

        # Analyze humor style preferences from conversation history
        style_scores = self._analyze_humor_styles(conversation_history)
        preference.preferred_styles = [
            style for style, score in style_scores.items() if score > 0.3
        ]

        # Analyze reactions to determine success patterns
        if user_reactions:
            reaction_analysis = self._analyze_reactions(user_reactions)
            preference.humor_frequency = reaction_analysis.get("frequency", 0.3)
            preference.appropriateness_threshold = reaction_analysis.get(
                "threshold", 0.7
            )

        # Use personality profile to infer humor preferences
        if personality_profile:
            personality_adjustments = self._infer_from_personality(personality_profile)
            preference = self._apply_personality_adjustments(
                preference, personality_adjustments
            )

        # Analyze timing preferences
        preference.preferred_timing = self._analyze_timing_preferences(
            conversation_history, user_reactions
        )

        # Calculate overall confidence
        preference.confidence = self._calculate_preference_confidence(
            len(conversation_history), len(user_reactions), personality_profile
        )

        logger.info(
            "Humor preference analysis complete for user %s. "
            "Preferred styles: %s, Confidence: %.2f",
            user_id,
            [style.value for style in preference.preferred_styles],
            preference.confidence,
        )

        return preference

    def _analyze_humor_styles(
        self, conversation_history: List[str]
    ) -> Dict[HumorStyle, float]:
        """Analyze humor style preferences from conversation history."""
        import re

        style_scores = {style: 0.0 for style in HumorStyle}

        for message in conversation_history:
            message_lower = message.lower()

            for style, patterns in self.HUMOR_INDICATORS.items():
                for pattern in patterns:
                    if re.search(pattern, message_lower):
                        style_scores[style] += 1.0

        # Normalize scores
        total_messages = len(conversation_history)
        if total_messages > 0:
            for style in style_scores:
                style_scores[style] = min(style_scores[style] / total_messages, 1.0)

        return style_scores

    def _analyze_reactions(self, user_reactions: List[str]) -> Dict[str, float]:
        """Analyze user reactions to humor attempts."""
        import re

        positive_count = 0
        negative_count = 0
        total_reactions = len(user_reactions)

        for reaction in user_reactions:
            reaction_lower = reaction.lower()

            # Check for positive reactions
            for pattern in self.POSITIVE_REACTIONS:
                if re.search(pattern, reaction_lower):
                    positive_count += 1
                    break

            # Check for negative reactions
            for pattern in self.NEGATIVE_REACTIONS:
                if re.search(pattern, reaction_lower):
                    negative_count += 1
                    break

        # Calculate frequency and threshold
        if total_reactions > 0:
            positive_ratio = positive_count / total_reactions
            negative_ratio = negative_count / total_reactions

            # Adjust frequency based on positive reactions
            frequency = min(positive_ratio * 0.8, 0.8)  # Cap at 80%

            # Adjust threshold based on negative reactions
            threshold = max(0.5, 1.0 - negative_ratio)

            return {"frequency": frequency, "threshold": threshold}

        return {"frequency": 0.3, "threshold": 0.7}

    def _infer_from_personality(
        self, personality_profile: PersonalityProfile
    ) -> Dict[str, any]:
        """Infer humor preferences from personality traits."""
        from .traits import PersonalityTrait, TraitLevel

        adjustments = {}

        # Extraversion affects humor frequency and style
        extraversion = personality_profile.get_trait_level(
            PersonalityTrait.EXTRAVERSION
        )
        if extraversion in [TraitLevel.HIGH, TraitLevel.VERY_HIGH]:
            adjustments["frequency_boost"] = 0.2
            adjustments["preferred_styles"] = [HumorStyle.PLAYFUL, HumorStyle.WITTY]
        elif extraversion in [TraitLevel.LOW, TraitLevel.VERY_LOW]:
            adjustments["frequency_reduction"] = 0.1
            adjustments["preferred_styles"] = [HumorStyle.DRY, HumorStyle.GENTLE]

        # Openness affects humor complexity and creativity
        openness = personality_profile.get_trait_level(PersonalityTrait.OPENNESS)
        if openness in [TraitLevel.HIGH, TraitLevel.VERY_HIGH]:
            adjustments["complexity_boost"] = True
            adjustments["creative_styles"] = [HumorStyle.WITTY, HumorStyle.PUNNY]
        elif openness in [TraitLevel.LOW, TraitLevel.VERY_LOW]:
            adjustments["simple_styles"] = [HumorStyle.GENTLE, HumorStyle.OBSERVATIONAL]

        # Agreeableness affects humor appropriateness
        agreeableness = personality_profile.get_trait_level(
            PersonalityTrait.AGREEABLENESS
        )
        if agreeableness in [TraitLevel.HIGH, TraitLevel.VERY_HIGH]:
            adjustments["appropriateness_boost"] = 0.1
            adjustments["gentle_styles"] = [
                HumorStyle.GENTLE,
                HumorStyle.SELF_DEPRECATING,
            ]

        # Neuroticism affects humor sensitivity
        neuroticism = personality_profile.get_trait_level(PersonalityTrait.NEUROTICISM)
        if neuroticism in [TraitLevel.HIGH, TraitLevel.VERY_HIGH]:
            adjustments["sensitivity_increase"] = 0.2
            adjustments["avoid_styles"] = [HumorStyle.DRY]

        return adjustments

    def _apply_personality_adjustments(
        self, preference: HumorPreference, adjustments: Dict[str, any]
    ) -> HumorPreference:
        """Apply personality-based adjustments to humor preferences."""
        # Adjust frequency
        if "frequency_boost" in adjustments:
            preference.humor_frequency = min(
                preference.humor_frequency + adjustments["frequency_boost"], 1.0
            )
        if "frequency_reduction" in adjustments:
            preference.humor_frequency = max(
                preference.humor_frequency - adjustments["frequency_reduction"], 0.0
            )

        # Adjust appropriateness threshold
        if "appropriateness_boost" in adjustments:
            preference.appropriateness_threshold = min(
                preference.appropriateness_threshold
                + adjustments["appropriateness_boost"],
                1.0,
            )
        if "sensitivity_increase" in adjustments:
            preference.appropriateness_threshold = min(
                preference.appropriateness_threshold
                + adjustments["sensitivity_increase"],
                1.0,
            )

        # Add preferred styles
        for key in [
            "preferred_styles",
            "creative_styles",
            "gentle_styles",
            "simple_styles",
        ]:
            if key in adjustments:
                for style in adjustments[key]:
                    if style not in preference.preferred_styles:
                        preference.preferred_styles.append(style)

        # Add avoided styles
        if "avoid_styles" in adjustments:
            for style in adjustments["avoid_styles"]:
                if style not in preference.avoided_styles:
                    preference.avoided_styles.append(style)

        return preference

    def _analyze_timing_preferences(
        self, conversation_history: List[str], user_reactions: List[str]
    ) -> List[HumorTiming]:
        """Analyze when user prefers humor to be used."""
        # Simple heuristic based on conversation patterns
        timing_preferences = []

        # If user uses humor in greetings, they like opening humor
        opening_humor_count = sum(
            1
            for msg in conversation_history[:5]  # First 5 messages
            if any(
                indicator in msg.lower()
                for indicators in self.HUMOR_INDICATORS.values()
                for indicator in indicators
            )
        )

        if opening_humor_count > 0:
            timing_preferences.append(HumorTiming.OPENING)

        # If positive reactions, they generally like humor
        positive_reactions = sum(
            1
            for reaction in user_reactions
            if any(
                re.search(pattern, reaction.lower())
                for pattern in self.POSITIVE_REACTIONS
            )
        )

        if positive_reactions > len(user_reactions) * 0.5:
            timing_preferences.extend([HumorTiming.TRANSITION, HumorTiming.EXPLANATION])

        # Default to some timing if none detected
        if not timing_preferences:
            timing_preferences.append(HumorTiming.TRANSITION)

        return timing_preferences

    def _calculate_preference_confidence(
        self,
        message_count: int,
        reaction_count: int,
        personality_profile: Optional[PersonalityProfile],
    ) -> float:
        """Calculate confidence in humor preference analysis."""
        confidence = 0.0

        # Base confidence on data availability
        if message_count > 10:
            confidence += 0.3
        elif message_count > 5:
            confidence += 0.2
        elif message_count > 0:
            confidence += 0.1

        if reaction_count > 5:
            confidence += 0.3
        elif reaction_count > 2:
            confidence += 0.2
        elif reaction_count > 0:
            confidence += 0.1

        # Boost confidence if personality profile available
        if personality_profile and personality_profile.overall_confidence > 0.5:
            confidence += 0.4 * personality_profile.overall_confidence

        return min(confidence, 1.0)

    def detect_humor_in_message(self, message: str) -> Optional[HumorStyle]:
        """Detect if a message contains humor and what style."""
        import re

        message_lower = message.lower()

        for style, patterns in self.HUMOR_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return style

        return None

    def evaluate_humor_success(
        self, humor_attempt: HumorAttempt, user_response: str
    ) -> float:
        """Evaluate the success of a humor attempt based on user response."""
        import re

        response_lower = user_response.lower()
        success_score = 0.0

        # Check for positive indicators
        for pattern in self.POSITIVE_REACTIONS:
            if re.search(pattern, response_lower):
                success_score += 0.3

        # Check for negative indicators
        for pattern in self.NEGATIVE_REACTIONS:
            if re.search(pattern, response_lower):
                success_score -= 0.5

        # Neutral response gets moderate score
        if success_score == 0.0:
            success_score = 0.5

        return max(0.0, min(1.0, success_score))


class HumorGenerator:
    """
    Generates appropriate humorous content based on user preferences.

    Creates contextually appropriate humor that matches user preferences,
    personality traits, and current conversation context.
    """

    # Humor templates by style
    HUMOR_TEMPLATES = {
        HumorStyle.WITTY: [
            "I'd make a joke about {topic}, but I'm afraid it might be too {adjective}.",
            "They say {topic} is {adjective}, but I think it's more like {comparison}.",
            "I was going to explain {topic}, but then I realized you're probably {assumption}.",
        ],
        HumorStyle.PLAYFUL: [
            "Ooh, {topic}! That's like my favorite thing after {comparison}! 😄",
            "You know what's fun about {topic}? {observation}!",
            "I'm so excited about {topic}, I might just {exaggeration}! 🎉",
        ],
        HumorStyle.DRY: [
            "Ah yes, {topic}. Fascinating.",
            "I suppose {topic} is... adequate.",
            "Well, that's certainly one way to think about {topic}.",
        ],
        HumorStyle.SELF_DEPRECATING: [
            "I'd help you with {topic}, but I'm just an AI who {limitation}.",
            "My understanding of {topic} is about as good as my {comparison}.",
            "I'm probably the wrong AI to ask about {topic}, but here goes nothing...",
        ],
        HumorStyle.OBSERVATIONAL: [
            "Isn't it funny how {topic} always seems to {observation}?",
            "Have you ever noticed that {topic} is like {comparison}?",
            "Why is it that whenever someone mentions {topic}, {reaction}?",
        ],
        HumorStyle.PUNNY: [
            "I guess you could say {topic} is quite {pun}!",
            "That {topic} joke was {pun_adjective}, wasn't it?",
            "I'm {pun_verb} to make more jokes about {topic}!",
        ],
        HumorStyle.GENTLE: [
            "You know, {topic} reminds me of {gentle_comparison} in the nicest way.",
            "I find {topic} rather charming, like {comparison}.",
            "There's something wonderfully {adjective} about {topic}, don't you think?",
        ],
    }

    def __init__(self):
        """Initialize humor generator."""
        logger.info("Humor generator initialized")

    def generate_humor(
        self,
        context: str,
        humor_preference: HumorPreference,
        timing: HumorTiming,
        topic: Optional[str] = None,
    ) -> Optional[HumorSuggestion]:
        """
        Generate appropriate humor for the given context.

        Args:
            context: Current conversation context
            humor_preference: User's humor preferences
            timing: When the humor will be used
            topic: Optional specific topic for humor

        Returns:
            HumorSuggestion or None if no appropriate humor found
        """
        logger.debug("Generating humor for context: %s", context[:50])

        # Check if humor is appropriate for this timing
        if timing not in humor_preference.preferred_timing:
            return None

        # Select appropriate humor style
        selected_style = self._select_humor_style(humor_preference, context)
        if not selected_style:
            return None

        # Generate humor content
        humor_content = self._generate_humor_content(selected_style, context, topic)
        if not humor_content:
            return None

        # Calculate appropriateness score
        appropriateness_score = self._calculate_appropriateness(
            humor_content, context, humor_preference
        )

        # Check if it meets the threshold
        if appropriateness_score < humor_preference.appropriateness_threshold:
            return None

        # Generate reasoning
        reasoning = self._generate_humor_reasoning(
            selected_style, timing, appropriateness_score
        )

        suggestion = HumorSuggestion(
            suggestion_id=str(uuid.uuid4()),
            humor_style=selected_style,
            content=humor_content,
            appropriateness_score=appropriateness_score,
            timing=timing,
            reasoning=reasoning,
            context_factors=[context[:100]],  # Truncate for storage
        )

        logger.debug(
            "Generated %s humor with appropriateness %.2f",
            selected_style.value,
            appropriateness_score,
        )

        return suggestion

    def _select_humor_style(
        self, humor_preference: HumorPreference, context: str
    ) -> Optional[HumorStyle]:
        """Select appropriate humor style for context."""
        # Filter preferred styles that aren't avoided
        available_styles = [
            style
            for style in humor_preference.preferred_styles
            if style not in humor_preference.avoided_styles
        ]

        if not available_styles:
            # Fallback to gentle humor if no preferences
            return HumorStyle.GENTLE

        # Simple selection - could be enhanced with context analysis
        return available_styles[0]

    def _generate_humor_content(
        self, style: HumorStyle, context: str, topic: Optional[str]
    ) -> Optional[str]:
        """Generate humor content for the given style."""
        if style not in self.HUMOR_TEMPLATES:
            return None

        templates = self.HUMOR_TEMPLATES[style]
        if not templates:
            return None

        # Select a template (simple random selection)
        import random

        template = random.choice(templates)

        # Fill in template variables
        filled_template = self._fill_template(template, context, topic)

        return filled_template

    def _fill_template(self, template: str, context: str, topic: Optional[str]) -> str:
        """Fill in template variables with context-appropriate content."""
        # Simple template filling - could be enhanced with NLP
        variables = {
            "topic": topic or "this",
            "adjective": "interesting",
            "comparison": "a puzzle",
            "assumption": "smart enough to figure it out",
            "observation": "it's always more complex than it seems",
            "exaggeration": "do a little happy dance",
            "limitation": "sometimes gets confused by simple things",
            "reaction": "everyone suddenly becomes an expert",
            "pun": "pun-derful",
            "pun_adjective": "pun-believable",
            "pun_verb": "pun-dering",
            "gentle_comparison": "a warm cup of tea",
        }

        try:
            return template.format(**variables)
        except KeyError:
            # If template has variables we don't handle, return as-is
            return template

    def _calculate_appropriateness(
        self, humor_content: str, context: str, humor_preference: HumorPreference
    ) -> float:
        """Calculate appropriateness score for humor content."""
        # Simple appropriateness calculation
        base_score = 0.7

        # Boost score if content matches user preferences
        if any(
            style_word in humor_content.lower()
            for style in humor_preference.preferred_styles
            for style_word in [style.value]
        ):
            base_score += 0.2

        # Reduce score if content might be inappropriate
        inappropriate_indicators = ["controversial", "sensitive", "personal"]
        if any(indicator in context.lower() for indicator in inappropriate_indicators):
            base_score -= 0.3

        return max(0.0, min(1.0, base_score))

    def _generate_humor_reasoning(
        self, style: HumorStyle, timing: HumorTiming, appropriateness_score: float
    ) -> str:
        """Generate reasoning for humor suggestion."""
        return (
            f"Selected {style.value} humor for {timing.value} timing "
            f"with appropriateness score {appropriateness_score:.2f}"
        )
