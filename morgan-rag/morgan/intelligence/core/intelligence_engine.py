"""
Emotional Intelligence Engine for Morgan RAG.

Provides real-time emotion detection, mood pattern analysis, empathetic response
generation, and personal preference learning to create meaningful companion
relationships with users.

Semantic-First Architecture:
============================
This engine now uses semantic-first processing throughout:
1. Emotion detection uses LLM semantic analysis as PRIMARY method
2. Pattern matching validates and boosts confidence (SECONDARY)
3. Response generation considers hidden emotions and emotional masking
4. All components understand meaning, not just keywords
"""

import hashlib
import json
import re
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from morgan.config import get_settings
# Use relative import to avoid circular dependency via intelligence/__init__.py
from ..constants import (
    EMOTION_PATTERNS,
    INTENSITY_MODIFIERS,
)
from morgan.intelligence.core.models import (
    CompanionProfile,
    ConversationContext,
    EmotionalState,
    EmotionType,
    EmpatheticResponse,
    InteractionData,
    MilestoneType,
    MoodPattern,
    RelationshipMilestone,
    ResponseLength,
    UserPreferences,
)
from morgan.services.llm import get_llm_service
from morgan.utils.cache import FileCache
from morgan.utils.llm_parsing import parse_llm_json
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class EmotionalIntelligenceEngine:
    """
    Emotional Intelligence Engine for companion AI.

    Semantic-First Architecture:
    ============================
    - Emotion detection uses LLM semantic analysis as PRIMARY method
    - Understands hidden emotions, sarcasm, and emotional masking
    - Pattern matching validates and boosts confidence (SECONDARY)
    - Empathetic responses consider actual emotional needs

    Features:
    - Real-time semantic emotion detection from text analysis
    - Detection of hidden emotions and emotional masking
    - Mood pattern analysis and tracking over time
    - Empathetic response generation with context awareness
    - Personal preference learning and adaptation
    - Relationship milestone detection and tracking

    Note: Emotion patterns and modifiers are imported from
    morgan.intelligence.constants for validation purposes.
    """

    def __init__(self):
        """Initialize emotional intelligence engine."""
        self.settings = get_settings()
        self.llm_service = get_llm_service()

        # Setup cache for emotional analysis
        cache_dir = self.settings.morgan_data_dir / "cache" / "emotional"
        self.cache = FileCache(cache_dir)

        # Setup cache for semantic analysis (longer TTL)
        semantic_cache_dir = self.settings.morgan_data_dir / "cache" / "semantic"
        self.semantic_cache = FileCache(semantic_cache_dir)

        # In-memory storage for user profiles and patterns
        self.user_profiles: Dict[str, CompanionProfile] = {}
        self.mood_patterns: Dict[str, List[EmotionalState]] = defaultdict(list)

        logger.info("Emotional Intelligence Engine initialized (semantic-first mode)")

    def analyze_emotion(
        self, text: str, context: ConversationContext, use_semantic: bool = True
    ) -> EmotionalState:
        """
        Analyze emotion from text using semantic-first approach.

        Semantic-First Architecture:
        1. PRIMARY: LLM semantic analysis for deep understanding
        2. SECONDARY: Pattern validation for confidence calibration
        3. Result: Combined understanding with calibrated confidence

        Args:
            text: Text to analyze for emotion
            context: Conversation context for better analysis
            use_semantic: Whether to use semantic analysis (default True)

        Returns:
            Detected emotional state with hidden emotion awareness
        """
        # Check cache first
        cache_key = self._get_emotion_cache_key(text, context.user_id)
        cached_emotion = self.cache.get(cache_key)
        if cached_emotion:
            logger.debug(f"Emotion cache hit for user {context.user_id}")
            return EmotionalState(**cached_emotion)

        if use_semantic:
            # SEMANTIC-FIRST APPROACH
            # Primary: LLM semantic analysis
            semantic_result = self._analyze_emotion_semantically(text, context)

            # Secondary: Pattern validation
            pattern_validation = self._validate_emotion_with_patterns(
                text, semantic_result
            )

            # Combine results
            final_emotion = self._finalize_emotion_result(
                semantic_result, pattern_validation
            )
        else:
            # LEGACY APPROACH (backwards compatibility)
            rule_emotions = self._detect_emotions_rule_based(text)
            llm_emotion = None
            if not rule_emotions or max(score for _, score in rule_emotions) < 0.6:
                llm_emotion = self._detect_emotions_llm(text, context)
            final_emotion = self._combine_emotion_results(rule_emotions, llm_emotion, text)

        # Store in cache
        timestamp_str = (
            final_emotion.timestamp.isoformat()
            if isinstance(final_emotion.timestamp, datetime)
            else final_emotion.timestamp
        )
        emotion_dict = {
            "primary_emotion": final_emotion.primary_emotion,
            "intensity": final_emotion.intensity,
            "confidence": final_emotion.confidence,
            "secondary_emotions": final_emotion.secondary_emotions,
            "emotional_indicators": final_emotion.emotional_indicators,
            "timestamp": timestamp_str,
        }
        self.cache.set(cache_key, emotion_dict)

        # Update mood patterns
        self.mood_patterns[context.user_id].append(final_emotion)

        logger.debug(
            f"Detected emotion for user {context.user_id}: "
            f"{final_emotion.primary_emotion.value} "
            f"(intensity: {final_emotion.intensity:.2f}, "
            f"confidence: {final_emotion.confidence:.2f})"
        )

        return final_emotion

    def _analyze_emotion_semantically(
        self, text: str, context: ConversationContext
    ) -> Optional[EmotionalState]:
        """
        Semantic emotion analysis using LLM.

        Understands:
        - Surface emotions (explicit)
        - Hidden emotions (implicit)
        - Sarcasm and irony
        - Emotional masking

        Args:
            text: Text to analyze
            context: Conversation context

        Returns:
            Semantic emotional understanding
        """
        # Check semantic cache
        cache_key = f"semantic_{self._get_emotion_cache_key(text, context.user_id)}"
        cached = self.semantic_cache.get(cache_key)
        if cached:
            return EmotionalState(**cached)

        try:
            context_info = ""
            if context.previous_messages:
                recent = context.previous_messages[-3:]
                context_info = f"\nRecent context:\n" + "\n".join(f"- {m}" for m in recent)

            prompt = f"""Analyze the emotional content with deep semantic understanding.

Message: "{text}"{context_info}

Consider:
1. SURFACE EMOTION: What's explicitly expressed?
2. HIDDEN EMOTION: What might be underneath?
3. SARCASM/IRONY: Is the meaning opposite to literal words?
4. EMOTIONAL MASKING: Is the person hiding their true feelings?

Respond with JSON ONLY:
{{
    "surface_emotion": "joy|sadness|anger|fear|surprise|disgust|neutral",
    "hidden_emotion": "joy|sadness|anger|fear|surprise|disgust|neutral|none",
    "is_sarcastic": true|false,
    "is_masking": true|false,
    "primary_emotion": "the TRUE emotion (hidden if masking, real if sarcastic)",
    "intensity": 0.0-1.0,
    "confidence": 0.0-1.0,
    "secondary_emotions": ["emotion1"],
    "indicators": ["evidence"]
}}"""

            response = self.llm_service.generate(
                prompt=prompt, temperature=0.2, max_tokens=500
            )

            # Parse JSON using utility that handles reasoning blocks
            data = parse_llm_json(response.content)
            if data is None:
                logger.warning("Failed to parse semantic emotion response")
                return None

            indicators = data.get("indicators", [])
            if data.get("is_sarcastic"):
                indicators.append("sarcasm_detected")
            if data.get("is_masking"):
                indicators.append("emotional_masking_detected")
            if data.get("hidden_emotion") != "none":
                indicators.append(f"hidden_emotion:{data['hidden_emotion']}")

            result = EmotionalState(
                primary_emotion=EmotionType(data["primary_emotion"]),
                intensity=float(data["intensity"]),
                confidence=float(data["confidence"]),
                secondary_emotions=[
                    EmotionType(e) for e in data.get("secondary_emotions", [])
                    if e in [et.value for et in EmotionType]
                ],
                emotional_indicators=indicators,
            )

            # Cache result
            timestamp_str = result.timestamp.isoformat() if isinstance(result.timestamp, datetime) else result.timestamp
            self.semantic_cache.set(cache_key, {
                "primary_emotion": result.primary_emotion,
                "intensity": result.intensity,
                "confidence": result.confidence,
                "secondary_emotions": result.secondary_emotions,
                "emotional_indicators": result.emotional_indicators,
                "timestamp": timestamp_str,
            })

            return result

        except Exception as e:
            logger.warning(f"Semantic emotion analysis failed: {e}")
            return None

    def _validate_emotion_with_patterns(
        self, text: str, semantic_result: Optional[EmotionalState]
    ) -> Dict[str, Any]:
        """Validate semantic result using pattern matching."""
        text_lower = text.lower()
        pattern_emotions = {}

        for emotion_type, patterns in EMOTION_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    score += len(matches) * 0.3

            if score > 0:
                for modifier, multiplier in INTENSITY_MODIFIERS.items():
                    if modifier in text_lower:
                        score *= multiplier
                        break
                score = min(1.0, score)
                if score > 0.1:
                    pattern_emotions[emotion_type] = score

        validation = {
            "pattern_emotions": pattern_emotions,
            "agrees_with_semantic": False,
            "confidence_adjustment": 0.0,
            "fallback_emotion": None,
        }

        if pattern_emotions:
            strongest = max(pattern_emotions.items(), key=lambda x: x[1])
            validation["fallback_emotion"] = strongest[0]
            validation["fallback_score"] = strongest[1]

            if semantic_result:
                if semantic_result.primary_emotion == strongest[0]:
                    validation["agrees_with_semantic"] = True
                    validation["confidence_adjustment"] = 0.15
                elif semantic_result.primary_emotion in pattern_emotions:
                    validation["agrees_with_semantic"] = True
                    validation["confidence_adjustment"] = 0.1

        return validation

    def _finalize_emotion_result(
        self,
        semantic_result: Optional[EmotionalState],
        pattern_validation: Dict[str, Any],
    ) -> EmotionalState:
        """Finalize emotion result from semantic and pattern analysis."""
        if semantic_result:
            final_confidence = min(
                1.0,
                semantic_result.confidence + pattern_validation["confidence_adjustment"],
            )

            indicators = list(semantic_result.emotional_indicators)
            if pattern_validation["agrees_with_semantic"]:
                indicators.append("pattern_validated")
            else:
                indicators.append("semantic_only")

            return EmotionalState(
                primary_emotion=semantic_result.primary_emotion,
                intensity=semantic_result.intensity,
                confidence=final_confidence,
                secondary_emotions=semantic_result.secondary_emotions,
                emotional_indicators=indicators,
            )

        if pattern_validation.get("fallback_emotion"):
            return EmotionalState(
                primary_emotion=pattern_validation["fallback_emotion"],
                intensity=pattern_validation.get("fallback_score", 0.5),
                confidence=0.6,
                secondary_emotions=[],
                emotional_indicators=["pattern_fallback"],
            )

        return EmotionalState(
            primary_emotion=EmotionType.NEUTRAL,
            intensity=0.5,
            confidence=0.3,
            emotional_indicators=["no_clear_indicators"],
        )

    def track_mood_patterns(self, user_id: str, timeframe: str = "30d") -> MoodPattern:
        """
        Analyze user's mood patterns over time.

        Args:
            user_id: User identifier
            timeframe: Time period to analyze (e.g., "7d", "30d")

        Returns:
            Mood pattern analysis
        """
        # Parse timeframe
        days = int(timeframe.rstrip("d"))
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        # Get recent emotional states
        recent_emotions = [
            emotion
            for emotion in self.mood_patterns[user_id]
            if emotion.timestamp >= cutoff_date
        ]

        if not recent_emotions:
            # Return neutral pattern for new users
            return MoodPattern(
                user_id=user_id,
                timeframe=timeframe,
                dominant_emotions=[EmotionType.NEUTRAL],
                average_intensity=0.5,
                mood_stability=1.0,
                emotional_trends={},
                pattern_confidence=0.0,
            )

        # Analyze patterns
        emotion_counts = Counter(e.primary_emotion for e in recent_emotions)
        dominant_emotions = [emotion for emotion, _ in emotion_counts.most_common(3)]

        # Calculate average intensity
        avg_intensity = sum(e.intensity for e in recent_emotions) / len(recent_emotions)

        # Calculate mood stability (lower variance = higher stability)
        intensities = [e.intensity for e in recent_emotions]
        variance = sum((x - avg_intensity) ** 2 for x in intensities) / len(intensities)
        mood_stability = max(0.0, 1.0 - variance)

        # Analyze trends
        emotional_trends = self._analyze_emotional_trends(recent_emotions)

        # Calculate confidence based on data points
        pattern_confidence = min(
            1.0, len(recent_emotions) / 20.0
        )  # Full confidence at 20+ data points

        return MoodPattern(
            user_id=user_id,
            timeframe=timeframe,
            dominant_emotions=dominant_emotions,
            average_intensity=avg_intensity,
            mood_stability=mood_stability,
            emotional_trends=emotional_trends,
            pattern_confidence=pattern_confidence,
        )

    def detect_relationship_milestones(
        self, conversation_history: List[ConversationContext]
    ) -> List[RelationshipMilestone]:
        """
        Detect significant relationship milestones from conversation history.

        Args:
            conversation_history: List of conversation contexts

        Returns:
            List of detected milestones
        """
        if not conversation_history:
            return []

        milestones = []
        user_id = conversation_history[0].user_id

        # First conversation milestone - always create for the first conversation
        # Sort conversations by timestamp to find the actual first one
        sorted_conversations = sorted(conversation_history, key=lambda c: c.timestamp)
        first_conversation = sorted_conversations[0]

        milestone = RelationshipMilestone(
            milestone_id=f"{user_id}_first_conversation",
            milestone_type=MilestoneType.FIRST_CONVERSATION,
            description="First conversation with Morgan",
            timestamp=first_conversation.timestamp,
            emotional_significance=0.8,
        )
        milestones.append(milestone)

        # Detect other milestones based on patterns
        milestones.extend(self._detect_breakthrough_moments(conversation_history))
        milestones.extend(self._detect_learning_milestones(conversation_history))
        milestones.extend(self._detect_trust_building_moments(conversation_history))

        return milestones

    def generate_empathetic_response(
        self, user_emotion: EmotionalState, context: str
    ) -> EmpatheticResponse:
        """
        Generate empathetic response based on user's emotional state.

        Args:
            user_emotion: User's current emotional state
            context: Context for the response

        Returns:
            Empathetic response with emotional awareness
        """
        # Determine empathy level based on emotion intensity
        empathy_level = min(1.0, user_emotion.intensity * 1.2)

        # Select appropriate emotional tone
        emotional_tone = self._select_emotional_tone(user_emotion)

        # Generate response using LLM with emotional awareness
        response_text = self._generate_empathetic_text(
            user_emotion, context, emotional_tone
        )

        # Identify personalization elements
        personalization_elements = self._identify_personalization_elements(
            user_emotion, context
        )

        # Calculate confidence score
        confidence_score = (
            user_emotion.confidence * 0.8
        )  # Slightly lower than emotion confidence

        return EmpatheticResponse(
            response_text=response_text,
            emotional_tone=emotional_tone,
            empathy_level=empathy_level,
            personalization_elements=personalization_elements,
            relationship_context=context,
            confidence_score=confidence_score,
        )

    def update_user_profile(
        self, user_id: str, interaction_data: InteractionData
    ) -> CompanionProfile:
        """
        Update user profile based on interaction data.

        Args:
            user_id: User identifier
            interaction_data: Data from the interaction

        Returns:
            Updated companion profile
        """
        # Get or create profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = self._create_initial_profile(user_id)

        profile = self.user_profiles[user_id]

        # Update interaction count and last interaction
        profile.interaction_count += 1
        profile.last_interaction = interaction_data.conversation_context.timestamp

        # Update relationship duration
        profile.relationship_duration = (
            profile.last_interaction - profile.profile_created
        )

        # Update emotional patterns
        self._update_emotional_patterns(profile, interaction_data.emotional_state)

        # Update communication preferences
        self._update_communication_preferences(profile, interaction_data)

        # Update trust and engagement scores
        self._update_relationship_scores(profile, interaction_data)

        logger.debug(f"Updated profile for user {user_id}")
        return profile

    def _detect_emotions_rule_based(self, text: str) -> List[Tuple[EmotionType, float]]:
        """Detect emotions using rule-based patterns."""
        text_lower = text.lower()
        emotion_scores = {}

        for emotion_type, patterns in EMOTION_PATTERNS.items():
            score = 0.0
            matches = []

            for pattern in patterns:
                matches_found = re.findall(pattern, text_lower)
                if matches_found:
                    matches.extend(matches_found)
                    score += len(matches_found) * 0.3

            # Apply intensity modifiers
            for modifier, multiplier in INTENSITY_MODIFIERS.items():
                if modifier in text_lower:
                    score *= multiplier

            # Normalize score
            score = min(1.0, score)

            if score > 0.1:  # Minimum threshold
                emotion_scores[emotion_type] = score

        # Return sorted by score
        return sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)

    def _detect_emotions_llm(
        self, text: str, context: ConversationContext
    ) -> Optional[EmotionalState]:
        """Use LLM for complex emotion detection."""
        try:
            prompt = f"""
            Analyze the emotional state in this text. Consider the conversation context.

            Text: "{text}"

            Previous messages: {context.previous_messages[-3:] if context.previous_messages else "None"}

            Respond with JSON format:
            {{
                "primary_emotion": "joy|sadness|anger|fear|surprise|disgust|neutral",
                "intensity": 0.0-1.0,
                "confidence": 0.0-1.0,
                "secondary_emotions": ["emotion1", "emotion2"],
                "indicators": ["text pattern that indicated emotion"]
            }}

            Focus on subtle emotional cues, context, and implied feelings.
            """

            response = self.llm_service.generate(
                prompt=prompt, temperature=0.3, max_tokens=400
            )

            # Parse JSON response using utility that handles reasoning blocks
            emotion_data = parse_llm_json(response.content)
            if emotion_data is None:
                logger.warning("Failed to parse LLM emotion detection response")
                return None

            return EmotionalState(
                primary_emotion=EmotionType(emotion_data["primary_emotion"]),
                intensity=float(emotion_data["intensity"]),
                confidence=float(emotion_data["confidence"]),
                secondary_emotions=[
                    EmotionType(e) for e in emotion_data.get("secondary_emotions", [])
                ],
                emotional_indicators=emotion_data.get("indicators", []),
            )

        except Exception as e:
            logger.warning(f"LLM emotion detection failed: {e}")
            return None

    def _combine_emotion_results(
        self,
        rule_emotions: List[Tuple[EmotionType, float]],
        llm_emotion: Optional[EmotionalState],
        text: str,
    ) -> EmotionalState:
        """Combine rule-based and LLM emotion results."""
        if not rule_emotions and not llm_emotion:
            # Default to neutral
            return EmotionalState(
                primary_emotion=EmotionType.NEUTRAL,
                intensity=0.5,
                confidence=0.3,
                emotional_indicators=["no clear emotional indicators"],
            )

        if rule_emotions and not llm_emotion:
            # Use rule-based result
            primary_emotion, intensity = rule_emotions[0]
            secondary_emotions = [emotion for emotion, _ in rule_emotions[1:3]]

            return EmotionalState(
                primary_emotion=primary_emotion,
                intensity=intensity,
                confidence=0.7,  # Good confidence for rule-based
                secondary_emotions=secondary_emotions,
                emotional_indicators=["pattern-based detection"],
            )

        if llm_emotion and not rule_emotions:
            # Use LLM result
            return llm_emotion

        # Combine both results
        rule_primary, rule_intensity = rule_emotions[0]

        # If they agree on primary emotion, boost confidence
        if rule_primary == llm_emotion.primary_emotion:
            combined_intensity = (rule_intensity + llm_emotion.intensity) / 2
            combined_confidence = min(1.0, llm_emotion.confidence + 0.2)
        else:
            # Use LLM result but lower confidence
            combined_intensity = llm_emotion.intensity
            combined_confidence = llm_emotion.confidence * 0.8
            rule_primary = llm_emotion.primary_emotion

        return EmotionalState(
            primary_emotion=rule_primary,
            intensity=combined_intensity,
            confidence=combined_confidence,
            secondary_emotions=llm_emotion.secondary_emotions,
            emotional_indicators=llm_emotion.emotional_indicators
            + ["hybrid detection"],
        )

    def _analyze_emotional_trends(
        self, emotions: List[EmotionalState]
    ) -> Dict[str, Any]:
        """Analyze emotional trends over time."""
        if len(emotions) < 3:
            return {}

        # Sort by timestamp
        emotions.sort(key=lambda e: e.timestamp)

        # Analyze intensity trend
        intensities = [e.intensity for e in emotions]
        recent_avg = sum(intensities[-5:]) / min(5, len(intensities))
        older_avg = sum(intensities[:-5]) / max(1, len(intensities) - 5)

        intensity_trend = "stable"
        if recent_avg > older_avg + 0.1:
            intensity_trend = "increasing"
        elif recent_avg < older_avg - 0.1:
            intensity_trend = "decreasing"

        # Analyze emotion diversity
        emotion_types = [e.primary_emotion for e in emotions]
        unique_emotions = len(set(emotion_types))
        emotion_diversity = unique_emotions / len(EmotionType)

        return {
            "intensity_trend": intensity_trend,
            "recent_average_intensity": recent_avg,
            "emotion_diversity": emotion_diversity,
            "most_recent_emotion": emotions[-1].primary_emotion.value,
        }

    def _detect_breakthrough_moments(
        self, conversations: List[ConversationContext]
    ) -> List[RelationshipMilestone]:
        """Detect breakthrough moments in conversations."""
        milestones = []

        # Look for patterns indicating breakthroughs
        breakthrough_patterns = [
            r"\b(understand|get it|makes sense|clarity|breakthrough)\b",
            r"\b(thank you|grateful|appreciate|helped me)\b",
            r"\b(never thought|new perspective|different way)\b",
        ]

        for i, conv in enumerate(conversations):
            text_lower = conv.message_text.lower()
            breakthrough_score = 0.0

            for pattern in breakthrough_patterns:
                matches = len(re.findall(pattern, text_lower))
                breakthrough_score += matches * 0.3

            # High feedback rating also indicates breakthrough
            if conv.user_feedback and conv.user_feedback >= 4:
                breakthrough_score += 0.4

            if breakthrough_score >= 0.6:
                milestone = RelationshipMilestone(
                    milestone_id=f"{conv.user_id}_breakthrough_{i}",
                    milestone_type=MilestoneType.BREAKTHROUGH_MOMENT,
                    description="User had a breakthrough or insight",
                    timestamp=conv.timestamp,
                    emotional_significance=min(1.0, breakthrough_score),
                )
                milestones.append(milestone)

        return milestones

    def _detect_learning_milestones(
        self, conversations: List[ConversationContext]
    ) -> List[RelationshipMilestone]:
        """Detect learning milestones."""
        milestones = []

        learning_patterns = [
            r"\b(learned|discovered|found out|realized)\b",
            r"\b(now I know|understand how|figured out)\b",
            r"\b(skill|knowledge|technique|method)\b",
        ]

        for i, conv in enumerate(conversations):
            text_lower = conv.message_text.lower()
            learning_score = 0.0

            for pattern in learning_patterns:
                matches = len(re.findall(pattern, text_lower))
                learning_score += matches * 0.25

            if learning_score >= 0.5:
                milestone = RelationshipMilestone(
                    milestone_id=f"{conv.user_id}_learning_{i}",
                    milestone_type=MilestoneType.LEARNING_MILESTONE,
                    description="User achieved a learning milestone",
                    timestamp=conv.timestamp,
                    emotional_significance=min(1.0, learning_score),
                )
                milestones.append(milestone)

        return milestones

    def _detect_trust_building_moments(
        self, conversations: List[ConversationContext]
    ) -> List[RelationshipMilestone]:
        """Detect trust building moments."""
        milestones = []

        trust_patterns = [
            r"\b(trust|rely on|depend on|count on)\b",
            r"\b(personal|private|confidential|share)\b",
            r"\b(comfortable|safe|secure)\b",
        ]

        for i, conv in enumerate(conversations):
            text_lower = conv.message_text.lower()
            trust_score = 0.0

            for pattern in trust_patterns:
                matches = len(re.findall(pattern, text_lower))
                trust_score += matches * 0.3

            # Longer messages often indicate more trust/sharing
            if len(conv.message_text) > 200:
                trust_score += 0.2

            if trust_score >= 0.6:
                milestone = RelationshipMilestone(
                    milestone_id=f"{conv.user_id}_trust_{i}",
                    milestone_type=MilestoneType.TRUST_BUILDING,
                    description="User showed increased trust and openness",
                    timestamp=conv.timestamp,
                    emotional_significance=min(1.0, trust_score),
                )
                milestones.append(milestone)

        return milestones

    def _select_emotional_tone(self, user_emotion: EmotionalState) -> str:
        """Select appropriate emotional tone for response."""
        emotion_tones = {
            EmotionType.JOY: "warm and celebratory",
            EmotionType.SADNESS: "gentle and supportive",
            EmotionType.ANGER: "calm and understanding",
            EmotionType.FEAR: "reassuring and confident",
            EmotionType.SURPRISE: "curious and engaging",
            EmotionType.DISGUST: "respectful and neutral",
            EmotionType.NEUTRAL: "friendly and helpful",
        }

        return emotion_tones.get(
            user_emotion.primary_emotion, "empathetic and supportive"
        )

    def _generate_empathetic_text(
        self, user_emotion: EmotionalState, context: str, emotional_tone: str
    ) -> str:
        """Generate empathetic response text using LLM."""
        try:
            prompt = f"""
            Generate an empathetic response for a user experiencing {user_emotion.primary_emotion.value}
            with intensity {user_emotion.intensity:.1f}.

            Context: {context}
            Emotional tone to use: {emotional_tone}

            Guidelines:
            - Be genuinely empathetic and understanding
            - Match the emotional tone appropriately
            - Keep response natural and conversational
            - Show emotional awareness without being clinical
            - Offer support or encouragement as appropriate

            Response:
            """

            response = self.llm_service.generate(
                prompt=prompt, temperature=0.7, max_tokens=150
            )

            return response.content.strip()

        except Exception as e:
            logger.warning(f"Failed to generate empathetic response: {e}")
            # Fallback to simple empathetic response
            return self._get_fallback_empathetic_response(user_emotion)

    def _get_fallback_empathetic_response(self, user_emotion: EmotionalState) -> str:
        """Get fallback empathetic response."""
        fallback_responses = {
            EmotionType.JOY: "I'm so happy to hear that! Your excitement is wonderful to see.",
            EmotionType.SADNESS: "I can sense you're going through a difficult time. I'm here to support you.",
            EmotionType.ANGER: "I understand you're feeling frustrated. Let's work through this together.",
            EmotionType.FEAR: "It's natural to feel worried sometimes. You're not alone in this.",
            EmotionType.SURPRISE: "That sounds unexpected! I'd love to hear more about what happened.",
            EmotionType.DISGUST: "I can see this situation is really bothering you.",
            EmotionType.NEUTRAL: "I'm here to help you with whatever you need.",
        }

        return fallback_responses.get(
            user_emotion.primary_emotion,
            "I appreciate you sharing this with me. How can I best support you?",
        )

    def _identify_personalization_elements(
        self, user_emotion: EmotionalState, context: str
    ) -> List[str]:
        """Identify elements that make the response more personal."""
        elements = []

        # Emotional acknowledgment
        elements.append(f"emotion_acknowledgment:{user_emotion.primary_emotion.value}")

        # Intensity matching
        if user_emotion.intensity > 0.7:
            elements.append("high_intensity_matching")
        elif user_emotion.intensity < 0.3:
            elements.append("gentle_approach")

        # Context awareness
        if len(context) > 100:
            elements.append("detailed_context_awareness")

        return elements

    def _create_initial_profile(self, user_id: str) -> CompanionProfile:
        """Create initial companion profile for new user."""
        return CompanionProfile(
            user_id=user_id,
            relationship_duration=timedelta(0),
            interaction_count=0,
            preferred_name="friend",  # Default until we learn their preference
            communication_preferences=UserPreferences(),
            trust_level=0.1,  # Start with minimal trust
            engagement_score=0.5,  # Neutral starting engagement
        )

    def _update_emotional_patterns(
        self, profile: CompanionProfile, emotional_state: EmotionalState
    ):
        """Update emotional patterns in user profile."""
        if "recent_emotions" not in profile.emotional_patterns:
            profile.emotional_patterns["recent_emotions"] = []

        # Keep last 20 emotional states
        timestamp_str = (
            emotional_state.timestamp.isoformat()
            if isinstance(emotional_state.timestamp, datetime)
            else emotional_state.timestamp
        )
        profile.emotional_patterns["recent_emotions"].append(
            {
                "emotion": emotional_state.primary_emotion.value,
                "intensity": emotional_state.intensity,
                "timestamp": timestamp_str,
            }
        )

        if len(profile.emotional_patterns["recent_emotions"]) > 20:
            profile.emotional_patterns["recent_emotions"] = profile.emotional_patterns[
                "recent_emotions"
            ][-20:]

    def _update_communication_preferences(
        self, profile: CompanionProfile, interaction_data: InteractionData
    ):
        """Update communication preferences based on interaction."""
        # Analyze message length preference
        msg_length = len(interaction_data.conversation_context.message_text)

        if msg_length > 200:
            # User prefers detailed communication
            profile.communication_preferences.preferred_response_length = (
                ResponseLength.DETAILED
            )
        elif msg_length < 50:
            # User prefers brief communication
            profile.communication_preferences.preferred_response_length = (
                ResponseLength.BRIEF
            )

        # Update topics of interest
        for topic in interaction_data.topics_discussed:
            if topic not in profile.communication_preferences.topics_of_interest:
                profile.communication_preferences.topics_of_interest.append(topic)

        # Keep only recent topics (max 20)
        if len(profile.communication_preferences.topics_of_interest) > 20:
            profile.communication_preferences.topics_of_interest = (
                profile.communication_preferences.topics_of_interest[-20:]
            )

    def _update_relationship_scores(
        self, profile: CompanionProfile, interaction_data: InteractionData
    ):
        """Update trust and engagement scores."""
        # Update trust based on interaction quality
        if interaction_data.user_satisfaction:
            trust_delta = (interaction_data.user_satisfaction - 0.5) * 0.1
            profile.trust_level = max(0.0, min(1.0, profile.trust_level + trust_delta))

        # Update engagement based on conversation length and feedback
        engagement_factors = []

        # Message length indicates engagement
        msg_length = len(interaction_data.conversation_context.message_text)
        engagement_factors.append(min(1.0, msg_length / 200.0))

        # Positive emotions indicate engagement
        if interaction_data.emotional_state.primary_emotion == EmotionType.JOY:
            engagement_factors.append(0.8)

        # User feedback indicates engagement
        if interaction_data.conversation_context.user_feedback:
            engagement_factors.append(
                interaction_data.conversation_context.user_feedback / 5.0
            )

        if engagement_factors:
            avg_engagement = sum(engagement_factors) / len(engagement_factors)
            # Smooth update (weighted average)
            profile.engagement_score = (profile.engagement_score * 0.8) + (
                avg_engagement * 0.2
            )

    def _get_emotion_cache_key(self, text: str, user_id: str) -> str:
        """Generate cache key for emotion analysis."""
        cache_input = f"{user_id}:{text}"
        return hashlib.sha256(cache_input.encode()).hexdigest()


# Singleton instance
_emotional_intelligence_instance = None
_emotional_intelligence_lock = threading.Lock()


def get_emotional_intelligence_engine() -> EmotionalIntelligenceEngine:
    """
    Get singleton emotional intelligence engine instance.

    Returns:
        Shared EmotionalIntelligenceEngine instance
    """
    global _emotional_intelligence_instance

    if _emotional_intelligence_instance is None:
        with _emotional_intelligence_lock:
            if _emotional_intelligence_instance is None:
                _emotional_intelligence_instance = EmotionalIntelligenceEngine()

    return _emotional_intelligence_instance
