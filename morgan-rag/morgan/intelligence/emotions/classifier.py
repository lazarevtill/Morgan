"""
Emotion categorization and classification module.

Provides advanced emotion classification with multi-dimensional analysis,
emotion mapping, and contextual classification refinement.
"""

import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from morgan.config import get_settings
from morgan.intelligence.core.models import ConversationContext, EmotionalState, EmotionType
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class EmotionClassifier:
    """
    Advanced emotion classification and categorization system.

    Features:
    - Multi-dimensional emotion mapping
    - Contextual classification refinement
    - Emotion intensity categorization
    - Complex emotion detection (mixed emotions)
    - Cultural and linguistic emotion patterns
    """

    # Emotion dimensions for multi-dimensional analysis
    EMOTION_DIMENSIONS = {
        "valence": {  # Positive/Negative
            EmotionType.JOY: 1.0,
            EmotionType.SURPRISE: 0.2,
            EmotionType.NEUTRAL: 0.0,
            EmotionType.FEAR: -0.4,
            EmotionType.DISGUST: -0.6,
            EmotionType.SADNESS: -0.8,
            EmotionType.ANGER: -0.9,
        },
        "arousal": {  # High/Low energy
            EmotionType.ANGER: 1.0,
            EmotionType.FEAR: 0.9,
            EmotionType.JOY: 0.8,
            EmotionType.SURPRISE: 0.7,
            EmotionType.DISGUST: 0.4,
            EmotionType.SADNESS: 0.2,
            EmotionType.NEUTRAL: 0.0,
        },
        "dominance": {  # Control/Submission
            EmotionType.ANGER: 0.9,
            EmotionType.JOY: 0.6,
            EmotionType.DISGUST: 0.4,
            EmotionType.NEUTRAL: 0.0,
            EmotionType.SURPRISE: -0.2,
            EmotionType.SADNESS: -0.6,
            EmotionType.FEAR: -0.8,
        },
    }

    # Intensity categories
    INTENSITY_CATEGORIES = {
        "minimal": (0.0, 0.2),
        "low": (0.2, 0.4),
        "moderate": (0.4, 0.6),
        "high": (0.6, 0.8),
        "extreme": (0.8, 1.0),
    }

    # Complex emotion patterns (combinations that commonly occur together)
    COMPLEX_EMOTION_PATTERNS = {
        "bittersweet": [EmotionType.JOY, EmotionType.SADNESS],
        "anxious_excitement": [EmotionType.JOY, EmotionType.FEAR],
        "frustrated_sadness": [EmotionType.ANGER, EmotionType.SADNESS],
        "surprised_delight": [EmotionType.SURPRISE, EmotionType.JOY],
        "fearful_anger": [EmotionType.FEAR, EmotionType.ANGER],
        "disgusted_anger": [EmotionType.DISGUST, EmotionType.ANGER],
    }

    # Contextual emotion modifiers
    CONTEXTUAL_MODIFIERS = {
        "work": {
            "stress_indicators": ["deadline", "meeting", "boss", "project", "work"],
            "emotion_boost": {EmotionType.FEAR: 0.2, EmotionType.ANGER: 0.1},
        },
        "relationships": {
            "stress_indicators": [
                "partner",
                "family",
                "friend",
                "relationship",
                "love",
            ],
            "emotion_boost": {EmotionType.SADNESS: 0.2, EmotionType.JOY: 0.1},
        },
        "health": {
            "stress_indicators": ["sick", "pain", "doctor", "hospital", "health"],
            "emotion_boost": {EmotionType.FEAR: 0.3, EmotionType.SADNESS: 0.2},
        },
    }

    def __init__(self):
        """Initialize emotion classifier."""
        self.settings = get_settings()
        logger.info("Emotion Classifier initialized")

    def classify_emotion(
        self,
        emotional_state: EmotionalState,
        context: Optional[ConversationContext] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive emotion classification.

        Args:
            emotional_state: Emotional state to classify
            context: Optional conversation context

        Returns:
            Comprehensive classification results
        """
        classification = {
            "primary_classification": self._classify_primary_emotion(emotional_state),
            "intensity_classification": self._classify_intensity(emotional_state),
            "dimensional_analysis": self._analyze_dimensions(emotional_state),
            "complex_emotions": self._detect_complex_emotions(emotional_state),
            "contextual_factors": self._analyze_contextual_factors(
                emotional_state, context
            ),
            "confidence_assessment": self._assess_classification_confidence(
                emotional_state
            ),
        }

        # Add overall classification summary
        classification["summary"] = self._generate_classification_summary(
            classification
        )

        logger.debug(
            f"Classified emotion: {emotional_state.primary_emotion.value} "
            f"-> {classification['summary']['category']}"
        )

        return classification

    def classify_emotion_batch(
        self,
        emotional_states: List[EmotionalState],
        contexts: Optional[List[ConversationContext]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple emotions efficiently.

        Args:
            emotional_states: List of emotional states to classify
            contexts: Optional list of conversation contexts

        Returns:
            List of classification results
        """
        contexts = contexts or [None] * len(emotional_states)
        results = []

        for i, emotional_state in enumerate(emotional_states):
            context = contexts[i] if i < len(contexts) else None
            classification = self.classify_emotion(emotional_state, context)
            results.append(classification)

        logger.info(f"Batch classified {len(emotional_states)} emotions")
        return results

    def get_emotion_categories(self) -> Dict[str, List[str]]:
        """
        Get available emotion categories and their members.

        Returns:
            Dictionary of emotion categories
        """
        return {
            "positive": [EmotionType.JOY.value],
            "negative": [
                EmotionType.SADNESS.value,
                EmotionType.ANGER.value,
                EmotionType.FEAR.value,
                EmotionType.DISGUST.value,
            ],
            "neutral": [EmotionType.NEUTRAL.value, EmotionType.SURPRISE.value],
            "high_arousal": [
                EmotionType.ANGER.value,
                EmotionType.FEAR.value,
                EmotionType.JOY.value,
                EmotionType.SURPRISE.value,
            ],
            "low_arousal": [
                EmotionType.SADNESS.value,
                EmotionType.DISGUST.value,
                EmotionType.NEUTRAL.value,
            ],
        }

    def find_similar_emotions(
        self, target_emotion: EmotionType, similarity_threshold: float = 0.7
    ) -> List[Tuple[EmotionType, float]]:
        """
        Find emotions similar to the target emotion based on dimensional analysis.

        Args:
            target_emotion: Emotion to find similarities for
            similarity_threshold: Minimum similarity score

        Returns:
            List of (emotion, similarity_score) tuples
        """
        target_dims = self._get_emotion_dimensions(target_emotion)
        similarities = []

        for emotion in EmotionType:
            if emotion == target_emotion:
                continue

            emotion_dims = self._get_emotion_dimensions(emotion)
            similarity = self._calculate_dimensional_similarity(
                target_dims, emotion_dims
            )

            if similarity >= similarity_threshold:
                similarities.append((emotion, similarity))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

    def analyze_emotion_transitions(
        self, emotion_sequence: List[EmotionalState]
    ) -> Dict[str, Any]:
        """
        Analyze transitions between emotions in a sequence.

        Args:
            emotion_sequence: Sequence of emotional states

        Returns:
            Transition analysis results
        """
        if len(emotion_sequence) < 2:
            return {"transitions": [], "patterns": {}}

        transitions = []
        transition_counts = defaultdict(int)

        for i in range(len(emotion_sequence) - 1):
            current_emotion = emotion_sequence[i].primary_emotion
            next_emotion = emotion_sequence[i + 1].primary_emotion

            transition = {
                "from": current_emotion.value,
                "to": next_emotion.value,
                "time_gap": (
                    emotion_sequence[i + 1].timestamp - emotion_sequence[i].timestamp
                ).total_seconds(),
                "intensity_change": emotion_sequence[i + 1].intensity
                - emotion_sequence[i].intensity,
            }

            transitions.append(transition)
            transition_counts[(current_emotion, next_emotion)] += 1

        # Find most common transitions
        common_transitions = sorted(
            transition_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "transitions": transitions,
            "patterns": {
                "most_common": [
                    {"from": from_emotion.value, "to": to_emotion.value, "count": count}
                    for (from_emotion, to_emotion), count in common_transitions
                ],
                "total_transitions": len(transitions),
                "unique_transitions": len(transition_counts),
            },
        }

    def _classify_primary_emotion(
        self, emotional_state: EmotionalState
    ) -> Dict[str, Any]:
        """Classify the primary emotion with detailed analysis."""
        primary = emotional_state.primary_emotion

        # Get emotion family
        emotion_family = self._get_emotion_family(primary)

        # Assess emotion purity (how much it's mixed with others)
        purity_score = 1.0
        if emotional_state.secondary_emotions:
            purity_score = max(
                0.3, 1.0 - (len(emotional_state.secondary_emotions) * 0.2)
            )

        return {
            "emotion": primary.value,
            "emotion_family": emotion_family,
            "purity_score": purity_score,
            "has_secondary_emotions": bool(emotional_state.secondary_emotions),
            "secondary_count": len(emotional_state.secondary_emotions),
        }

    def _classify_intensity(self, emotional_state: EmotionalState) -> Dict[str, Any]:
        """Classify emotion intensity into categories."""
        intensity = emotional_state.intensity

        # Find intensity category
        intensity_category = "moderate"  # default
        for category, (min_val, max_val) in self.INTENSITY_CATEGORIES.items():
            if min_val <= intensity < max_val:
                intensity_category = category
                break

        # Assess intensity appropriateness (based on confidence)
        intensity_confidence = emotional_state.confidence

        return {
            "intensity_value": intensity,
            "intensity_category": intensity_category,
            "intensity_confidence": intensity_confidence,
            "is_extreme": intensity >= 0.8,
            "is_minimal": intensity <= 0.2,
        }

    def _analyze_dimensions(self, emotional_state: EmotionalState) -> Dict[str, Any]:
        """Analyze emotion across multiple dimensions."""
        primary_emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity

        dimensions = {}
        for dimension_name, emotion_values in self.EMOTION_DIMENSIONS.items():
            base_value = emotion_values.get(primary_emotion, 0.0)
            # Scale by intensity
            scaled_value = base_value * intensity
            dimensions[dimension_name] = {
                "value": scaled_value,
                "category": self._categorize_dimension_value(scaled_value),
            }

        return {
            "dimensions": dimensions,
            "dominant_dimension": max(
                dimensions.keys(), key=lambda d: abs(dimensions[d]["value"])
            ),
        }

    def _detect_complex_emotions(
        self, emotional_state: EmotionalState
    ) -> Dict[str, Any]:
        """Detect complex emotion patterns."""
        if not emotional_state.secondary_emotions:
            return {"has_complex_emotions": False, "patterns": []}

        # Check for known complex patterns
        detected_patterns = []
        emotions_set = {emotional_state.primary_emotion} | set(
            emotional_state.secondary_emotions
        )

        for pattern_name, pattern_emotions in self.COMPLEX_EMOTION_PATTERNS.items():
            if set(pattern_emotions).issubset(emotions_set):
                detected_patterns.append(
                    {
                        "pattern": pattern_name,
                        "emotions": [e.value for e in pattern_emotions],
                        "confidence": 0.8,  # Base confidence for known patterns
                    }
                )

        # Analyze emotional conflict (opposing emotions)
        conflict_score = self._calculate_emotional_conflict(emotions_set)

        return {
            "has_complex_emotions": len(emotional_state.secondary_emotions) > 0,
            "patterns": detected_patterns,
            "emotional_conflict": conflict_score,
            "complexity_score": len(emotional_state.secondary_emotions)
            / 3.0,  # Normalize to 0-1
        }

    def _analyze_contextual_factors(
        self, emotional_state: EmotionalState, context: Optional[ConversationContext]
    ) -> Dict[str, Any]:
        """Analyze contextual factors that might influence emotion classification."""
        if not context:
            return {"has_context": False}

        contextual_analysis = {"has_context": True, "factors": []}

        # Analyze message content for contextual clues
        message_text = context.message_text.lower()

        for context_type, config in self.CONTEXTUAL_MODIFIERS.items():
            stress_indicators = config["stress_indicators"]

            # Check if any stress indicators are present
            indicator_matches = [
                indicator
                for indicator in stress_indicators
                if indicator in message_text
            ]

            if indicator_matches:
                contextual_analysis["factors"].append(
                    {
                        "context_type": context_type,
                        "indicators_found": indicator_matches,
                        "potential_emotion_boost": config.get("emotion_boost", {}),
                    }
                )

        # Analyze conversation history for emotional context
        if context.previous_messages:
            contextual_analysis["conversation_context"] = {
                "has_history": True,
                "message_count": len(context.previous_messages),
                "conversation_length": len(" ".join(context.previous_messages)),
            }

        return contextual_analysis

    def _assess_classification_confidence(
        self, emotional_state: EmotionalState
    ) -> Dict[str, Any]:
        """Assess confidence in the emotion classification."""
        base_confidence = emotional_state.confidence

        # Factors that affect classification confidence
        confidence_factors = {
            "base_emotion_confidence": base_confidence,
            "intensity_clarity": 1.0
            - abs(emotional_state.intensity - 0.5) * 2,  # Higher for extreme values
            "indicator_quality": len(emotional_state.emotional_indicators)
            / 5.0,  # More indicators = higher confidence
            "complexity_penalty": max(
                0.0, 1.0 - len(emotional_state.secondary_emotions) * 0.2
            ),
        }

        # Calculate overall confidence
        overall_confidence = sum(confidence_factors.values()) / len(confidence_factors)
        overall_confidence = min(1.0, max(0.0, overall_confidence))

        return {
            "overall_confidence": overall_confidence,
            "confidence_factors": confidence_factors,
            "reliability": (
                "high"
                if overall_confidence > 0.7
                else "medium" if overall_confidence > 0.4 else "low"
            ),
        }

    def _generate_classification_summary(
        self, classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a summary of the classification results."""
        primary = classification["primary_classification"]
        intensity = classification["intensity_classification"]
        complex_emotions = classification["complex_emotions"]
        confidence = classification["confidence_assessment"]

        # Determine overall category
        if complex_emotions["has_complex_emotions"]:
            category = "complex_emotion"
        elif intensity["is_extreme"]:
            category = "intense_emotion"
        elif intensity["is_minimal"]:
            category = "subtle_emotion"
        else:
            category = "standard_emotion"

        return {
            "category": category,
            "primary_emotion": primary["emotion"],
            "intensity_level": intensity["intensity_category"],
            "complexity": (
                "high" if complex_emotions.get("complexity_score", 0.0) > 0.6 else "low"
            ),
            "reliability": confidence["reliability"],
            "key_characteristics": self._extract_key_characteristics(classification),
        }

    def _get_emotion_family(self, emotion: EmotionType) -> str:
        """Get the emotion family for an emotion."""
        families = {
            "positive": [EmotionType.JOY],
            "negative": [
                EmotionType.SADNESS,
                EmotionType.ANGER,
                EmotionType.FEAR,
                EmotionType.DISGUST,
            ],
            "neutral": [EmotionType.NEUTRAL, EmotionType.SURPRISE],
        }

        for family, emotions in families.items():
            if emotion in emotions:
                return family

        return "unknown"

    def _get_emotion_dimensions(self, emotion: EmotionType) -> Dict[str, float]:
        """Get dimensional values for an emotion."""
        return {
            dimension: values.get(emotion, 0.0)
            for dimension, values in self.EMOTION_DIMENSIONS.items()
        }

    def _calculate_dimensional_similarity(
        self, dims1: Dict[str, float], dims2: Dict[str, float]
    ) -> float:
        """Calculate similarity between two emotion dimension profiles."""
        total_diff = 0.0
        dimension_count = len(dims1)

        for dimension in dims1:
            diff = abs(dims1[dimension] - dims2.get(dimension, 0.0))
            total_diff += diff

        # Convert difference to similarity (0-1 scale)
        max_possible_diff = dimension_count * 2.0  # Max diff per dimension is 2.0
        similarity = 1.0 - (total_diff / max_possible_diff)

        return max(0.0, min(1.0, similarity))

    def _categorize_dimension_value(self, value: float) -> str:
        """Categorize a dimension value."""
        if value > 0.6:
            return "high"
        elif value > 0.2:
            return "moderate"
        elif value > -0.2:
            return "neutral"
        elif value > -0.6:
            return "low"
        else:
            return "very_low"

    def _calculate_emotional_conflict(self, emotions: Set[EmotionType]) -> float:
        """Calculate emotional conflict score based on opposing emotions."""
        conflict_pairs = [
            (EmotionType.JOY, EmotionType.SADNESS),
            (EmotionType.JOY, EmotionType.ANGER),
            (EmotionType.FEAR, EmotionType.ANGER),
            (EmotionType.SURPRISE, EmotionType.NEUTRAL),
        ]

        conflict_score = 0.0
        for emotion1, emotion2 in conflict_pairs:
            if emotion1 in emotions and emotion2 in emotions:
                conflict_score += 0.5

        return min(1.0, conflict_score)

    def _extract_key_characteristics(self, classification: Dict[str, Any]) -> List[str]:
        """Extract key characteristics from classification."""
        characteristics = []

        # Intensity characteristics
        intensity = classification["intensity_classification"]
        if intensity["is_extreme"]:
            characteristics.append("high_intensity")
        elif intensity["is_minimal"]:
            characteristics.append("low_intensity")

        # Complexity characteristics
        complex_emotions = classification["complex_emotions"]
        if complex_emotions["has_complex_emotions"]:
            characteristics.append("mixed_emotions")

        if complex_emotions.get("emotional_conflict", 0.0) > 0.3:
            characteristics.append("emotional_conflict")

        # Dimensional characteristics
        dimensions = classification["dimensional_analysis"]["dimensions"]
        for dim_name, dim_data in dimensions.items():
            if dim_data["category"] in ["high", "very_low"]:
                characteristics.append(f"{dim_name}_{dim_data['category']}")

        return characteristics[:5]  # Limit to top 5 characteristics


# Singleton instance
_classifier_instance = None
_classifier_lock = threading.Lock()


def get_emotion_classifier() -> EmotionClassifier:
    """
    Get singleton emotion classifier instance.

    Returns:
        Shared EmotionClassifier instance
    """
    global _classifier_instance

    if _classifier_instance is None:
        with _classifier_lock:
            if _classifier_instance is None:
                _classifier_instance = EmotionClassifier()

    return _classifier_instance
