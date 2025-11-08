"""
Emotional Intelligence Engine for Morgan RAG.

Provides real-time emotion detection, mood pattern analysis, empathetic response
generation, and personal preference learning to create meaningful companion
relationships with users.
"""

import hashlib
import json
import re
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from morgan.config import get_settings
from morgan.emotional.models import (
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
from morgan.services.llm_service import get_llm_service
from morgan.utils.cache import FileCache
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class EmotionalIntelligenceEngine:
    """
    Emotional Intelligence Engine for companion AI.

    Features:
    - Real-time emotion detection from text analysis
    - Mood pattern analysis and tracking over time
    - Empathetic response generation with emotional awareness
    - Personal preference learning and adaptation
    - Relationship milestone detection and tracking
    """

    # Emotion detection patterns (rule-based + LLM hybrid approach)
    EMOTION_PATTERNS = {
        EmotionType.JOY: [
            r"\b(happy|joy|excited|thrilled|delighted|pleased|glad|cheerful|elated)\b",
            r"\b(awesome|amazing|fantastic|wonderful|great|excellent|perfect)\b",
            r"\b(love|adore|enjoy)\b",
            r"[!]{2,}",  # Multiple exclamation marks
            r":\)|:D|😊|😄|😃|🎉|❤️",  # Emoticons and emojis
        ],
        EmotionType.SADNESS: [
            r"\b(sad|depressed|down|upset|disappointed|heartbroken|miserable)\b",
            r"\b(cry|crying|tears|weep)\b",
            r"\b(lonely|alone|isolated)\b",
            r":\(|😢|😭|💔",  # Sad emoticons
        ],
        EmotionType.ANGER: [
            r"\b(angry|mad|furious|irritated|annoyed|frustrated|pissed)\b",
            r"\b(hate|despise|can\'t stand)\b",
            r"\b(stupid|idiotic|ridiculous|absurd)\b",
            r"[!]{3,}",  # Many exclamation marks (anger indicator)
            r"😠|😡|🤬",  # Angry emojis
        ],
        EmotionType.FEAR: [
            r"\b(scared|afraid|terrified|worried|anxious|nervous|panic)\b",
            r"\b(fear|phobia|dread)\b",
            r"\b(what if|concerned about)\b",
            r"😰|😨|😱",  # Fear emojis
        ],
        EmotionType.SURPRISE: [
            r"\b(surprised|shocked|amazed|astonished|wow|whoa)\b",
            r"\b(unexpected|sudden|didn\'t expect)\b",
            r"😲|😮|🤯",  # Surprise emojis
        ],
        EmotionType.DISGUST: [
            r"\b(disgusting|gross|revolting|sick|nauseating)\b",
            r"\b(ugh|eww|yuck)\b",
            r"🤢|🤮|😷",  # Disgust emojis
        ],
    }

    # Intensity modifiers
    INTENSITY_MODIFIERS = {
        "very": 1.3,
        "extremely": 1.5,
        "really": 1.2,
        "so": 1.2,
        "quite": 1.1,
        "somewhat": 0.8,
        "a bit": 0.7,
        "slightly": 0.6,
        "not very": 0.4,
        "barely": 0.3,
    }

    def __init__(self):
        """Initialize emotional intelligence engine."""
        self.settings = get_settings()
        self.llm_service = get_llm_service()

        # Setup cache for emotional analysis
        cache_dir = self.settings.morgan_data_dir / "cache" / "emotional"
        self.cache = FileCache(cache_dir)

        # In-memory storage for user profiles and patterns
        self.user_profiles: Dict[str, CompanionProfile] = {}
        self.mood_patterns: Dict[str, List[EmotionalState]] = defaultdict(list)

        logger.info("Emotional Intelligence Engine initialized")

    def analyze_emotion(
        self, text: str, context: ConversationContext
    ) -> EmotionalState:
        """
        Analyze emotion from text using hybrid rule-based + LLM approach.

        Args:
            text: Text to analyze for emotion
            context: Conversation context for better analysis

        Returns:
            Detected emotional state
        """
        # Check cache first
        cache_key = self._get_emotion_cache_key(text, context.user_id)
        cached_emotion = self.cache.get(cache_key)
        if cached_emotion:
            logger.debug(f"Emotion cache hit for user {context.user_id}")
            return EmotionalState(**cached_emotion)

        # Rule-based emotion detection
        rule_emotions = self._detect_emotions_rule_based(text)

        # LLM-enhanced emotion analysis for complex cases
        llm_emotion = None
        if not rule_emotions or max(score for _, score in rule_emotions) < 0.6:
            llm_emotion = self._detect_emotions_llm(text, context)

        # Combine results
        final_emotion = self._combine_emotion_results(rule_emotions, llm_emotion, text)

        # Store in cache
        emotion_dict = {
            "primary_emotion": final_emotion.primary_emotion,
            "intensity": final_emotion.intensity,
            "confidence": final_emotion.confidence,
            "secondary_emotions": final_emotion.secondary_emotions,
            "emotional_indicators": final_emotion.emotional_indicators,
            "timestamp": final_emotion.timestamp.isoformat(),
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
        cutoff_date = datetime.utcnow() - timedelta(days=days)

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

        for emotion_type, patterns in self.EMOTION_PATTERNS.items():
            score = 0.0
            matches = []

            for pattern in patterns:
                matches_found = re.findall(pattern, text_lower)
                if matches_found:
                    matches.extend(matches_found)
                    score += len(matches_found) * 0.3

            # Apply intensity modifiers
            for modifier, multiplier in self.INTENSITY_MODIFIERS.items():
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
                prompt=prompt, temperature=0.3, max_tokens=200
            )

            # Parse JSON response
            emotion_data = json.loads(response.content)

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
        profile.emotional_patterns["recent_emotions"].append(
            {
                "emotion": emotional_state.primary_emotion.value,
                "intensity": emotional_state.intensity,
                "timestamp": emotional_state.timestamp.isoformat(),
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
