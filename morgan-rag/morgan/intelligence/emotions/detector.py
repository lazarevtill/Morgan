"""
Real-time emotion detection module.

Provides focused emotion detection from text using LLM-based semantic understanding
as the PRIMARY method, with pattern matching for validation and confidence boosting.
Includes caching for performance optimization.

Semantic-First Architecture:
- LLM analysis is the primary detection method
- Pattern matching validates and boosts confidence
- Detects hidden emotions, sarcasm, and context-dependent meanings
"""

import hashlib
import json
import re
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from morgan.config import get_settings
from morgan.intelligence.constants import (
    EMOTION_PATTERNS,
    INTENSITY_MODIFIERS,
    NEGATION_PATTERNS,
)
from morgan.intelligence.core.models import (
    ConversationContext,
    EmotionalState,
    EmotionType,
)
from morgan.services.llm import get_llm_service
from morgan.utils.cache import FileCache
from morgan.utils.llm_parsing import parse_llm_json
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class EmotionDetector:
    """
    Real-time emotion detection from text analysis.

    Features:
    - LLM-first semantic emotion detection (PRIMARY method)
    - Pattern validation for confidence boosting (SECONDARY)
    - Detection of hidden emotions, sarcasm, and masked feelings
    - Context-aware emotional understanding
    - Performance-optimized caching

    Architecture:
    1. Primary: LLM semantic analysis - understands meaning, not just keywords
    2. Secondary: Pattern matching - validates and boosts confidence
    3. Result: Combined understanding with calibrated confidence

    Note: Emotion patterns and modifiers are imported from
    morgan.intelligence.constants for validation purposes.
    """

    def __init__(self):
        """Initialize emotion detector with caching and LLM service."""
        self.settings = get_settings()
        self.llm_service = get_llm_service()

        # Setup cache for emotion detection results
        cache_dir = self.settings.morgan_data_dir / "cache" / "emotions"
        self.cache = FileCache(cache_dir)

        # Separate cache for semantic analysis results (longer TTL)
        semantic_cache_dir = self.settings.morgan_data_dir / "cache" / "semantic_emotions"
        self.semantic_cache = FileCache(semantic_cache_dir)

        logger.info("Emotion Detector initialized (semantic-first mode)")

    def detect_emotion(
        self,
        text: str,
        context: Optional[ConversationContext] = None,
        use_cache: bool = True,
        use_semantic: bool = True,
    ) -> EmotionalState:
        """
        Detect emotion from text using semantic-first approach.

        The detection follows a semantic-first architecture:
        1. Primary: LLM semantic analysis for deep understanding
        2. Secondary: Pattern validation for confidence calibration
        3. Result: Combined understanding with calibrated confidence

        Args:
            text: Text to analyze for emotion
            context: Optional conversation context for better analysis
            use_cache: Whether to use cached results
            use_semantic: Whether to use LLM semantic analysis (default True).
                         Set to False to use legacy pattern-first behavior.

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

        if use_semantic:
            # SEMANTIC-FIRST APPROACH (new default)
            # Primary: LLM semantic analysis
            semantic_understanding = self._analyze_semantically(text, context)

            # Secondary: Pattern validation for confidence boost
            pattern_validation = self._validate_with_patterns(
                text, semantic_understanding
            )

            # Combine results with pattern validation
            final_emotion = self._finalize_emotion(
                semantic_understanding, pattern_validation
            )
        else:
            # LEGACY APPROACH (for backwards compatibility)
            rule_emotions = self._detect_emotions_rule_based(text)
            llm_emotion = None
            if self._should_use_llm_legacy(rule_emotions, text):
                llm_emotion = self._detect_emotions_llm(text, context)
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

    def _analyze_semantically(
        self,
        text: str,
        context: Optional[ConversationContext],
    ) -> Optional[EmotionalState]:
        """
        Primary semantic analysis using LLM.

        This is the CORE of the semantic-first approach. It analyzes:
        - Surface emotions (what's explicitly stated)
        - Hidden emotions (what's implied or masked)
        - Sarcasm and irony detection
        - Context-dependent emotional meanings
        - Emotional masking (saying "I'm fine" when not fine)

        Args:
            text: Text to analyze
            context: Conversation context

        Returns:
            Semantic emotional understanding, or None if analysis fails
        """
        # Check semantic cache first
        cache_key = self._get_semantic_cache_key(text, context)
        cached_semantic = self.semantic_cache.get(cache_key)
        if cached_semantic:
            logger.debug("Semantic analysis cache hit")
            return self._deserialize_emotion(cached_semantic)

        try:
            # Build context-aware prompt
            context_info = ""
            if context and context.previous_messages:
                recent_messages = context.previous_messages[-3:]
                context_info = f"\nRecent conversation context:\n" + "\n".join(
                    f"- {msg}" for msg in recent_messages
                )

            prompt = f"""Analyze the emotional content in this message with deep semantic understanding.

Message: "{text}"{context_info}

Consider:
1. SURFACE EMOTION: What emotion is explicitly expressed?
2. HIDDEN EMOTION: What emotion might be underneath the surface?
   - "Great, another meeting" = frustration, not joy
   - "I'm fine" (after bad news) = sadness being masked
   - Excessive enthusiasm might mask anxiety
3. SARCASM/IRONY: Is the literal meaning opposite to the intended meaning?
   - "Oh wonderful" with eye-roll energy = frustration
   - "Thanks a lot" sarcastically = anger/frustration
4. EMOTIONAL MASKING: Is the person hiding their true feelings?
   - Deflecting with humor when hurt
   - Being overly positive when struggling
5. CONTEXT SIGNALS: How does the context change the emotional meaning?

Respond with JSON ONLY:
{{
    "surface_emotion": "joy|sadness|anger|fear|surprise|disgust|neutral",
    "hidden_emotion": "joy|sadness|anger|fear|surprise|disgust|neutral|none",
    "is_sarcastic": true|false,
    "is_masking": true|false,
    "primary_emotion": "joy|sadness|anger|fear|surprise|disgust|neutral",
    "intensity": 0.0-1.0,
    "confidence": 0.0-1.0,
    "secondary_emotions": ["emotion1", "emotion2"],
    "indicators": ["specific textual or contextual evidence"],
    "reasoning": "brief explanation of the emotional analysis"
}}

The primary_emotion should be the TRUE emotion (hidden if masking, real if sarcastic)."""

            response = self.llm_service.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=500,  # Increased for reasoning models
            )

            # Parse JSON response using utility that handles reasoning blocks
            emotion_data = parse_llm_json(response.content)
            if emotion_data is None:
                logger.warning("Failed to parse semantic analysis response")
                return None

            # Build indicators list with semantic insights
            indicators = emotion_data.get("indicators", [])
            if emotion_data.get("is_sarcastic"):
                indicators.append("sarcasm_detected")
            if emotion_data.get("is_masking"):
                indicators.append("emotional_masking_detected")
            if emotion_data.get("hidden_emotion") != "none":
                indicators.append(f"hidden_emotion:{emotion_data['hidden_emotion']}")

            emotional_state = EmotionalState(
                primary_emotion=EmotionType(emotion_data["primary_emotion"]),
                intensity=float(emotion_data["intensity"]),
                confidence=float(emotion_data["confidence"]),
                secondary_emotions=[
                    EmotionType(e)
                    for e in emotion_data.get("secondary_emotions", [])
                    if e in [et.value for et in EmotionType]
                ],
                emotional_indicators=indicators,
            )

            # Cache the semantic result
            self.semantic_cache.set(cache_key, self._serialize_emotion(emotional_state))

            logger.debug(
                f"Semantic analysis: surface={emotion_data.get('surface_emotion')}, "
                f"hidden={emotion_data.get('hidden_emotion')}, "
                f"sarcasm={emotion_data.get('is_sarcastic')}, "
                f"masking={emotion_data.get('is_masking')}"
            )

            return emotional_state

        except Exception as e:
            logger.warning(f"Semantic emotion analysis failed: {e}")
            return None

    def _validate_with_patterns(
        self,
        text: str,
        semantic_result: Optional[EmotionalState],
    ) -> Dict[str, Any]:
        """
        Validate semantic understanding using pattern matching.

        This is the SECONDARY step - patterns are used to:
        1. Validate LLM results
        2. Boost confidence when patterns agree
        3. Flag potential disagreements for lower confidence
        4. Provide fallback if semantic analysis fails

        Args:
            text: Original text
            semantic_result: Result from semantic analysis

        Returns:
            Validation results with pattern matches and confidence adjustments
        """
        text_lower = text.lower()
        pattern_emotions = {}

        # Check for negation context
        has_negation = any(
            re.search(pattern, text_lower) for pattern in NEGATION_PATTERNS
        )

        # Match emotion patterns
        for emotion_type, patterns in EMOTION_PATTERNS.items():
            score = 0.0
            matched_patterns = []

            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    matched_patterns.extend(matches)
                    score += len(matches) * 0.3

            if score > 0:
                # Apply intensity modifiers
                score = self._apply_intensity_modifiers(score, text_lower)

                # Apply negation effects
                if has_negation:
                    if emotion_type in [EmotionType.JOY, EmotionType.SURPRISE]:
                        score *= 0.5
                    elif emotion_type in [
                        EmotionType.SADNESS,
                        EmotionType.ANGER,
                        EmotionType.FEAR,
                    ]:
                        score *= 1.2

                score = min(1.0, score)
                if score > 0.1:
                    pattern_emotions[emotion_type] = {
                        "score": score,
                        "matches": matched_patterns,
                    }

        # Determine validation result
        validation = {
            "pattern_emotions": pattern_emotions,
            "has_negation": has_negation,
            "agrees_with_semantic": False,
            "confidence_adjustment": 0.0,
            "fallback_emotion": None,
        }

        if pattern_emotions:
            # Find strongest pattern emotion
            strongest_pattern = max(
                pattern_emotions.items(), key=lambda x: x[1]["score"]
            )
            validation["fallback_emotion"] = strongest_pattern[0]
            validation["fallback_score"] = strongest_pattern[1]["score"]

            # Check agreement with semantic result
            if semantic_result:
                if semantic_result.primary_emotion == strongest_pattern[0]:
                    validation["agrees_with_semantic"] = True
                    validation["confidence_adjustment"] = 0.15  # Boost confidence
                elif semantic_result.primary_emotion in pattern_emotions:
                    validation["agrees_with_semantic"] = True
                    validation["confidence_adjustment"] = 0.1  # Partial boost
                else:
                    # Disagreement - semantic result might be catching sarcasm/masking
                    validation["confidence_adjustment"] = -0.05

        return validation

    def _finalize_emotion(
        self,
        semantic_result: Optional[EmotionalState],
        pattern_validation: Dict[str, Any],
    ) -> EmotionalState:
        """
        Finalize emotion result by combining semantic understanding with pattern validation.

        Priority:
        1. Semantic result (if available) - it understands meaning
        2. Pattern fallback (if semantic fails) - it catches explicit emotions

        Args:
            semantic_result: Result from semantic analysis
            pattern_validation: Results from pattern validation

        Returns:
            Final emotional state with calibrated confidence
        """
        if semantic_result:
            # Use semantic result as primary, adjust confidence based on validation
            final_confidence = min(
                1.0,
                semantic_result.confidence + pattern_validation["confidence_adjustment"],
            )

            # Add validation info to indicators
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

        # Fallback to pattern-based result
        if pattern_validation.get("fallback_emotion"):
            return EmotionalState(
                primary_emotion=pattern_validation["fallback_emotion"],
                intensity=pattern_validation.get("fallback_score", 0.5),
                confidence=0.6,  # Lower confidence for pattern-only detection
                secondary_emotions=[],
                emotional_indicators=["pattern_fallback"],
            )

        # Ultimate fallback
        return self._create_neutral_emotion("no clear emotional indicators")

    def _get_semantic_cache_key(
        self, text: str, context: Optional[ConversationContext]
    ) -> str:
        """Generate cache key for semantic analysis."""
        cache_input = text
        if context:
            cache_input += f":{context.user_id}"
            if context.previous_messages:
                cache_input += ":" + "|".join(context.previous_messages[-2:])
        return f"semantic_{hashlib.sha256(cache_input.encode()).hexdigest()}"

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
            re.search(pattern, text_lower) for pattern in NEGATION_PATTERNS
        )

        for emotion_type, patterns in EMOTION_PATTERNS.items():
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

    def _should_use_llm_legacy(
        self, rule_emotions: List[Tuple[EmotionType, float]], text: str
    ) -> bool:
        """
        Determine if LLM analysis is needed (LEGACY method for backwards compatibility).

        This method is only used when use_semantic=False.
        The new semantic-first approach always uses LLM as primary.

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

        for modifier, multiplier in INTENSITY_MODIFIERS.items():
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
