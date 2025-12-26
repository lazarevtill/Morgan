"""
Real-time emotion detection module.

Provides focused emotion detection from text using hybrid rule-based and LLM approaches
with caching for performance optimization.
"""

import hashlib
import json
import re
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from morgan.config import get_settings
from morgan.intelligence.core.models import (
    ConversationContext,
    EmotionalState,
    EmotionType,
)
from morgan.services.llm_service import get_llm_service
from morgan.utils.cache import FileCache
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class EmotionDetector:
    """
    Real-time emotion detection from text analysis.

    Features:
    - Hybrid rule-based and LLM emotion detection
    - Performance-optimized caching
    - Multi-pattern emotion recognition
    - Confidence scoring and validation
    """

    # Core emotion detection patterns
    EMOTION_PATTERNS = {
        EmotionType.JOY: [
            r"\b(happy|joy|excited|thrilled|delighted|pleased|glad|cheerful|elated|wonderful)\b",
            r"\b(awesome|amazing|fantastic|great|excellent|perfect|brilliant|outstanding)\b",
            r"\b(love|adore|enjoy|celebrate|rejoice)\b",
            r"[!]{2,}",  # Multiple exclamation marks
            r":\)|:D|😊|😄|😃|🎉|❤️|🥳|✨",  # Joy emoticons and emojis
        ],
        EmotionType.SADNESS: [
            r"\b(sad|depressed|down|upset|disappointed|heartbroken|miserable|devastated)\b",
            r"\b(cry|crying|tears|weep|sob|mourn)\b",
            r"\b(lonely|alone|isolated|empty|hopeless)\b",
            r"\b(loss|grief|sorrow|despair)\b",
            r":\(|😢|😭|💔|😞|😔",  # Sad emoticons
        ],
        EmotionType.ANGER: [
            r"\b(angry|mad|furious|irritated|annoyed|frustrated|pissed|enraged)\b",
            r"\b(hate|despise|can\'t stand|loathe|detest)\b",
            r"\b(stupid|idiotic|ridiculous|absurd|outrageous)\b",
            r"\b(damn|hell|shit|fuck)\b",  # Strong language indicators
            r"[!]{3,}",  # Many exclamation marks (anger indicator)
            r"😠|😡|🤬|👿|💢",  # Angry emojis
        ],
        EmotionType.FEAR: [
            r"\b(scared|afraid|terrified|worried|anxious|nervous|panic|frightened)\b",
            r"\b(fear|phobia|dread|terror|horror)\b",
            r"\b(what if|concerned about|worried that|afraid that)\b",
            r"\b(stress|stressed|overwhelmed|helpless)\b",
            r"😰|😨|😱|😟|😧",  # Fear emojis
        ],
        EmotionType.SURPRISE: [
            r"\b(surprised|shocked|amazed|astonished|wow|whoa|incredible|unbelievable)\b",
            r"\b(unexpected|sudden|didn\'t expect|never thought)\b",
            r"\b(blown away|mind blown|can\'t believe)\b",
            r"😲|😮|🤯|😯|🙀",  # Surprise emojis
        ],
        EmotionType.DISGUST: [
            r"\b(disgusting|gross|revolting|sick|nauseating|repulsive|vile)\b",
            r"\b(ugh|eww|yuck|blech|nasty)\b",
            r"\b(horrible|awful|terrible|dreadful)\b",
            r"🤢|🤮|😷|🤧|😖",  # Disgust emojis
        ],
    }

    # Intensity modifiers that affect emotion strength
    INTENSITY_MODIFIERS = {
        "extremely": 1.5,
        "very": 1.3,
        "really": 1.2,
        "so": 1.2,
        "quite": 1.1,
        "pretty": 1.1,
        "somewhat": 0.8,
        "a bit": 0.7,
        "slightly": 0.6,
        "barely": 0.4,
        "not very": 0.4,
        "hardly": 0.3,
    }

    # Negation patterns that can flip emotion polarity
    NEGATION_PATTERNS = [
        r"\b(not|no|never|nothing|nobody|nowhere|neither|nor)\b",
        r"\b(don\'t|doesn\'t|didn\'t|won\'t|wouldn\'t|can\'t|couldn\'t)\b",
        r"\b(isn\'t|aren\'t|wasn\'t|weren\'t|haven\'t|hasn\'t|hadn\'t)\b",
    ]

    def __init__(self):
        """Initialize emotion detector with caching and LLM service."""
        self.settings = get_settings()
        self.llm_service = get_llm_service()

        # Setup cache for emotion detection results
        cache_dir = self.settings.morgan_data_dir / "cache" / "emotions"
        self.cache = FileCache(cache_dir)

        logger.info("Emotion Detector initialized")

    def detect_emotion(
        self,
        text: str,
        context: Optional[ConversationContext] = None,
        use_cache: bool = True,
    ) -> EmotionalState:
        """
        Detect emotion from text using hybrid approach.

        Args:
            text: Text to analyze for emotion
            context: Optional conversation context for better analysis
            use_cache: Whether to use cached results

        Returns:
            Detected emotional state with confidence scores
        """
        if not text or not text.strip():
            return self._create_neutral_emotion("empty text")

        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(text, context)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug("Emotion detection cache hit")
                return self._deserialize_emotion(cached_result)

        # Perform rule-based detection
        rule_emotions = self._detect_emotions_rule_based(text)

        # Use LLM for complex cases or low confidence results
        llm_emotion = None
        if self._should_use_llm(rule_emotions, text):
            llm_emotion = self._detect_emotions_llm(text, context)

        # Combine and validate results
        final_emotion = self._combine_emotion_results(rule_emotions, llm_emotion, text)

        # Cache the result if caching is enabled
        if use_cache:
            cache_key = self._get_cache_key(text, context)
            self.cache.set(cache_key, self._serialize_emotion(final_emotion))

        logger.debug(
            f"Detected emotion: {final_emotion.primary_emotion.value} "
            f"(intensity: {final_emotion.intensity:.2f}, "
            f"confidence: {final_emotion.confidence:.2f})"
        )

        return final_emotion

    def detect_emotions_batch(
        self, texts: List[str], contexts: Optional[List[ConversationContext]] = None
    ) -> List[EmotionalState]:
        """
        Detect emotions for multiple texts efficiently.

        Args:
            texts: List of texts to analyze
            contexts: Optional list of conversation contexts

        Returns:
            List of detected emotional states
        """
        if not texts:
            return []

        contexts = contexts or [None] * len(texts)
        results = []

        for i, text in enumerate(texts):
            context = contexts[i] if i < len(contexts) else None
            emotion = self.detect_emotion(text, context)
            results.append(emotion)

        logger.info(f"Batch emotion detection completed for {len(texts)} texts")
        return results

    def _detect_emotions_rule_based(self, text: str) -> List[Tuple[EmotionType, float]]:
        """
        Detect emotions using rule-based pattern matching.

        Args:
            text: Text to analyze

        Returns:
            List of (emotion_type, score) tuples sorted by score
        """
        text_lower = text.lower()
        emotion_scores = {}

        # Check for negation context
        has_negation = any(
            re.search(pattern, text_lower) for pattern in self.NEGATION_PATTERNS
        )

        for emotion_type, patterns in self.EMOTION_PATTERNS.items():
            score = 0.0
            matched_patterns = []

            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    matched_patterns.extend(matches)
                    # Base score for each match
                    score += len(matches) * 0.3

            if score > 0:
                # Apply intensity modifiers
                score = self._apply_intensity_modifiers(score, text_lower)

                # Apply negation effects (reduce positive emotions, boost negative ones)
                if has_negation:
                    if emotion_type in [EmotionType.JOY, EmotionType.SURPRISE]:
                        score *= 0.5  # Reduce positive emotions
                    elif emotion_type in [
                        EmotionType.SADNESS,
                        EmotionType.ANGER,
                        EmotionType.FEAR,
                    ]:
                        score *= 1.2  # Boost negative emotions slightly

                # Normalize score to 0-1 range
                score = min(1.0, score)

                if score > 0.1:  # Minimum threshold
                    emotion_scores[emotion_type] = score

        # Return sorted by score (highest first)
        return sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)

    def _detect_emotions_llm(
        self, text: str, context: Optional[ConversationContext]
    ) -> Optional[EmotionalState]:
        """
        Use LLM for sophisticated emotion detection.

        Args:
            text: Text to analyze
            context: Optional conversation context

        Returns:
            LLM-detected emotional state or None if failed
        """
        try:
            # Build context-aware prompt
            context_info = ""
            if context and context.previous_messages:
                recent_messages = context.previous_messages[-2:]  # Last 2 messages
                context_info = f"Previous context: {' | '.join(recent_messages)}"

            prompt = f"""
            Analyze the emotional state in this text. Consider subtle emotional cues, implied feelings, and context.

            Text to analyze: "{text}"
            {context_info}

            Respond with JSON format only:
            {{
                "primary_emotion": "joy|sadness|anger|fear|surprise|disgust|neutral",
                "intensity": 0.0-1.0,
                "confidence": 0.0-1.0,
                "secondary_emotions": ["emotion1", "emotion2"],
                "indicators": ["specific text pattern or phrase that indicated the emotion"]
            }}

            Focus on:
            - Subtle emotional language and tone
            - Contextual emotional cues
            - Implied feelings beyond explicit words
            - Cultural and conversational nuances
            """

            response = self.llm_service.generate(
                prompt=prompt,
                temperature=0.2,  # Lower temperature for more consistent results
                max_tokens=200,
            )

            # Parse JSON response
            emotion_data = json.loads(response.content.strip())

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

    def _should_use_llm(
        self, rule_emotions: List[Tuple[EmotionType, float]], text: str
    ) -> bool:
        """
        Determine if LLM analysis is needed.

        Args:
            rule_emotions: Results from rule-based detection
            text: Original text

        Returns:
            True if LLM analysis should be used
        """
        # Use LLM if no strong rule-based results
        if not rule_emotions or max(score for _, score in rule_emotions) < 0.6:
            return True

        # Use LLM for longer, complex texts
        if len(text) > 100:
            return True

        # Use LLM if multiple emotions detected (ambiguous case)
        if len(rule_emotions) > 2:
            return True

        return False

    def _combine_emotion_results(
        self,
        rule_emotions: List[Tuple[EmotionType, float]],
        llm_emotion: Optional[EmotionalState],
        text: str,
    ) -> EmotionalState:
        """
        Combine rule-based and LLM emotion detection results.

        Args:
            rule_emotions: Rule-based detection results
            llm_emotion: LLM detection result
            text: Original text

        Returns:
            Combined emotional state
        """
        if not rule_emotions and not llm_emotion:
            return self._create_neutral_emotion("no clear emotional indicators")

        if rule_emotions and not llm_emotion:
            # Use rule-based result
            primary_emotion, intensity = rule_emotions[0]
            secondary_emotions = [emotion for emotion, _ in rule_emotions[1:3]]

            return EmotionalState(
                primary_emotion=primary_emotion,
                intensity=intensity,
                confidence=0.75,  # Good confidence for clear rule-based detection
                secondary_emotions=secondary_emotions,
                emotional_indicators=["pattern-based detection"],
            )

        if llm_emotion and not rule_emotions:
            # Use LLM result with slightly reduced confidence
            llm_emotion.confidence *= 0.9
            return llm_emotion

        # Combine both results
        rule_primary, rule_intensity = rule_emotions[0]

        # If they agree on primary emotion, boost confidence and average intensity
        if rule_primary == llm_emotion.primary_emotion:
            combined_intensity = (rule_intensity + llm_emotion.intensity) / 2
            combined_confidence = min(1.0, llm_emotion.confidence + 0.15)

            return EmotionalState(
                primary_emotion=rule_primary,
                intensity=combined_intensity,
                confidence=combined_confidence,
                secondary_emotions=llm_emotion.secondary_emotions,
                emotional_indicators=llm_emotion.emotional_indicators
                + ["hybrid detection"],
            )
        else:
            # They disagree - use LLM result but with lower confidence
            llm_emotion.confidence *= 0.7
            llm_emotion.emotional_indicators.append("conflicting rule-based detection")
            return llm_emotion

    def _apply_intensity_modifiers(self, base_score: float, text: str) -> float:
        """
        Apply intensity modifiers to emotion score.

        Args:
            base_score: Base emotion score
            text: Text to check for modifiers

        Returns:
            Modified score
        """
        modified_score = base_score

        for modifier, multiplier in self.INTENSITY_MODIFIERS.items():
            if modifier in text:
                modified_score *= multiplier
                break  # Apply only the first modifier found

        return modified_score

    def _create_neutral_emotion(self, reason: str) -> EmotionalState:
        """
        Create a neutral emotional state.

        Args:
            reason: Reason for neutral classification

        Returns:
            Neutral emotional state
        """
        return EmotionalState(
            primary_emotion=EmotionType.NEUTRAL,
            intensity=0.5,
            confidence=0.8,
            emotional_indicators=[reason],
        )

    def _get_cache_key(self, text: str, context: Optional[ConversationContext]) -> str:
        """
        Generate cache key for emotion detection.

        Args:
            text: Text being analyzed
            context: Optional conversation context

        Returns:
            Cache key string
        """
        cache_input = text
        if context:
            cache_input += f":{context.user_id}"

        return hashlib.sha256(cache_input.encode()).hexdigest()

    def _serialize_emotion(self, emotion: EmotionalState) -> Dict[str, Any]:
        """Serialize emotional state for caching."""
        return {
            "primary_emotion": emotion.primary_emotion.value,
            "intensity": emotion.intensity,
            "confidence": emotion.confidence,
            "secondary_emotions": [e.value for e in emotion.secondary_emotions],
            "emotional_indicators": emotion.emotional_indicators,
            "timestamp": emotion.timestamp.isoformat(),
        }

    def _deserialize_emotion(self, data: Dict[str, Any]) -> EmotionalState:
        """Deserialize emotional state from cache."""
        return EmotionalState(
            primary_emotion=EmotionType(data["primary_emotion"]),
            intensity=data["intensity"],
            confidence=data["confidence"],
            secondary_emotions=[
                EmotionType(e) for e in data.get("secondary_emotions", [])
            ],
            emotional_indicators=data.get("emotional_indicators", []),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


# Singleton instance
_detector_instance = None
_detector_lock = threading.Lock()


def get_emotion_detector() -> EmotionDetector:
    """
    Get singleton emotion detector instance.

    Returns:
        Shared EmotionDetector instance
    """
    global _detector_instance

    if _detector_instance is None:
        with _detector_lock:
            if _detector_instance is None:
                _detector_instance = EmotionDetector()

    return _detector_instance
