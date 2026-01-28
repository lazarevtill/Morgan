"""
Emotional intensity measurement module.

Provides precise measurement and analysis of emotional intensity with
calibration, normalization, and contextual adjustment capabilities.
"""

import statistics
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.intelligence.core.models import (
    ConversationContext,
    EmotionalState,
    EmotionType,
)
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class IntensityMeasurer:
    """
    Precise emotional intensity measurement and analysis.

    Features:
    - Multi-factor intensity calculation
    - Contextual intensity adjustment
    - Intensity calibration and normalization
    - Temporal intensity analysis
    - Comparative intensity assessment
    """

    # Base intensity weights for different emotion types
    EMOTION_INTENSITY_WEIGHTS = {
        EmotionType.ANGER: 1.2,  # Anger tends to be more intense
        EmotionType.FEAR: 1.1,  # Fear can be very intense
        EmotionType.JOY: 1.0,  # Joy baseline
        EmotionType.SADNESS: 0.9,  # Sadness often more subdued
        EmotionType.SURPRISE: 0.8,  # Surprise often brief/moderate
        EmotionType.DISGUST: 0.9,  # Disgust moderate intensity
        EmotionType.NEUTRAL: 0.5,  # Neutral low intensity
    }

    # Intensity modifiers based on text patterns
    INTENSITY_MODIFIERS = {
        "amplifiers": {
            "extremely": 1.5,
            "incredibly": 1.4,
            "absolutely": 1.3,
            "totally": 1.3,
            "completely": 1.3,
            "very": 1.2,
            "really": 1.2,
            "so": 1.1,
            "quite": 1.1,
        },
        "diminishers": {
            "slightly": 0.6,
            "somewhat": 0.7,
            "a bit": 0.7,
            "kind of": 0.8,
            "sort of": 0.8,
            "rather": 0.9,
            "fairly": 0.9,
        },
        "negators": {
            "not very": 0.4,
            "not really": 0.4,
            "barely": 0.3,
            "hardly": 0.3,
            "scarcely": 0.3,
        },
    }

    # Contextual intensity adjustments
    CONTEXTUAL_ADJUSTMENTS = {
        "time_of_day": {
            "morning": 0.9,  # Generally lower intensity in morning
            "afternoon": 1.0,  # Baseline
            "evening": 1.1,  # Slightly higher in evening
            "night": 1.2,  # Higher intensity at night
        },
        "conversation_length": {
            "short": 1.0,  # Brief messages baseline
            "medium": 1.1,  # Medium messages slightly higher
            "long": 1.2,  # Long messages higher intensity
        },
        "response_time": {
            "immediate": 1.2,  # Quick responses often more intense
            "normal": 1.0,  # Normal response time baseline
            "delayed": 0.9,  # Delayed responses often less intense
        },
    }

    def __init__(self):
        """Initialize intensity measurer."""
        self.settings = get_settings()

        # Calibration data for intensity normalization
        self.calibration_data = defaultdict(list)

        logger.info("Intensity Measurer initialized")

    def measure_intensity(
        self,
        emotional_state: EmotionalState,
        context: Optional[ConversationContext] = None,
        calibrate: bool = True,
    ) -> Dict[str, Any]:
        """
        Measure emotional intensity with comprehensive analysis.

        Args:
            emotional_state: Emotional state to measure
            context: Optional conversation context
            calibrate: Whether to apply calibration adjustments

        Returns:
            Comprehensive intensity measurement results
        """
        # Base intensity from emotional state
        base_intensity = emotional_state.intensity

        # Apply emotion-specific weight
        emotion_weight = self.EMOTION_INTENSITY_WEIGHTS.get(
            emotional_state.primary_emotion, 1.0
        )
        weighted_intensity = base_intensity * emotion_weight

        # Apply text-based modifiers
        text_modified_intensity = self._apply_text_modifiers(
            weighted_intensity, emotional_state.emotional_indicators
        )

        # Apply contextual adjustments
        context_adjusted_intensity = self._apply_contextual_adjustments(
            text_modified_intensity, context
        )

        # Apply calibration if enabled
        final_intensity = context_adjusted_intensity
        if calibrate:
            final_intensity = self._apply_calibration(
                context_adjusted_intensity, emotional_state.primary_emotion
            )

        # Ensure intensity stays in valid range
        final_intensity = max(0.0, min(1.0, final_intensity))

        # Calculate intensity confidence
        intensity_confidence = self._calculate_intensity_confidence(
            emotional_state, context, final_intensity
        )

        # Analyze intensity characteristics
        intensity_analysis = self._analyze_intensity_characteristics(
            final_intensity, emotional_state
        )

        result = {
            "raw_intensity": base_intensity,
            "weighted_intensity": weighted_intensity,
            "text_modified_intensity": text_modified_intensity,
            "context_adjusted_intensity": context_adjusted_intensity,
            "final_intensity": final_intensity,
            "intensity_confidence": intensity_confidence,
            "intensity_analysis": intensity_analysis,
            "adjustment_factors": {
                "emotion_weight": emotion_weight,
                "text_modifiers": self._get_applied_modifiers(
                    emotional_state.emotional_indicators
                ),
                "contextual_adjustments": self._get_contextual_factors(context),
            },
        }

        # Store for calibration
        if calibrate:
            self._update_calibration_data(
                emotional_state.primary_emotion, final_intensity
            )

        logger.debug(
            f"Measured intensity for {emotional_state.primary_emotion.value}: "
            f"{base_intensity:.2f} -> {final_intensity:.2f}"
        )

        return result

    def measure_intensity_batch(
        self,
        emotional_states: List[EmotionalState],
        contexts: Optional[List[ConversationContext]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Measure intensity for multiple emotional states efficiently.

        Args:
            emotional_states: List of emotional states to measure
            contexts: Optional list of conversation contexts

        Returns:
            List of intensity measurement results
        """
        contexts = contexts or [None] * len(emotional_states)
        results = []

        for i, emotional_state in enumerate(emotional_states):
            context = contexts[i] if i < len(contexts) else None
            intensity_result = self.measure_intensity(emotional_state, context)
            results.append(intensity_result)

        logger.info(f"Batch measured intensity for {len(emotional_states)} emotions")
        return results

    def analyze_intensity_patterns(
        self, emotional_states: List[EmotionalState], timeframe: str = "7d"
    ) -> Dict[str, Any]:
        """
        Analyze intensity patterns over time.

        Args:
            emotional_states: List of emotional states to analyze
            timeframe: Analysis timeframe

        Returns:
            Intensity pattern analysis
        """
        if not emotional_states:
            return {"pattern": "no_data"}

        # Filter by timeframe
        days = int(timeframe.rstrip("d"))
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_emotions = [e for e in emotional_states if e.timestamp >= cutoff_date]

        if len(recent_emotions) < 3:
            return {"pattern": "insufficient_data", "count": len(recent_emotions)}

        # Measure intensities
        intensities = [
            self.measure_intensity(e)["final_intensity"] for e in recent_emotions
        ]

        # Calculate statistics
        intensity_stats = {
            "mean": statistics.mean(intensities),
            "median": statistics.median(intensities),
            "std_dev": statistics.stdev(intensities) if len(intensities) > 1 else 0.0,
            "min": min(intensities),
            "max": max(intensities),
            "range": max(intensities) - min(intensities),
        }

        # Analyze trends
        trend_analysis = self._analyze_intensity_trend(intensities, recent_emotions)

        # Analyze volatility
        volatility_analysis = self._analyze_intensity_volatility(intensities)

        # Analyze by emotion type
        emotion_intensity_analysis = self._analyze_intensity_by_emotion(recent_emotions)

        return {
            "pattern": "analyzed",
            "timeframe": timeframe,
            "sample_count": len(recent_emotions),
            "statistics": intensity_stats,
            "trend": trend_analysis,
            "volatility": volatility_analysis,
            "by_emotion": emotion_intensity_analysis,
        }

    def compare_intensity_periods(
        self,
        current_emotions: List[EmotionalState],
        previous_emotions: List[EmotionalState],
    ) -> Dict[str, Any]:
        """
        Compare intensity between two time periods.

        Args:
            current_emotions: Recent emotional states
            previous_emotions: Previous period emotional states

        Returns:
            Intensity comparison analysis
        """
        if not current_emotions or not previous_emotions:
            return {"comparison": "insufficient_data"}

        # Measure intensities for both periods
        current_intensities = [
            self.measure_intensity(e)["final_intensity"] for e in current_emotions
        ]
        previous_intensities = [
            self.measure_intensity(e)["final_intensity"] for e in previous_emotions
        ]

        # Calculate period statistics
        current_mean = statistics.mean(current_intensities)
        previous_mean = statistics.mean(previous_intensities)

        current_volatility = (
            statistics.stdev(current_intensities)
            if len(current_intensities) > 1
            else 0.0
        )
        previous_volatility = (
            statistics.stdev(previous_intensities)
            if len(previous_intensities) > 1
            else 0.0
        )

        # Calculate changes
        intensity_change = current_mean - previous_mean
        volatility_change = current_volatility - previous_volatility

        # Determine significance
        change_significance = self._assess_change_significance(
            intensity_change, current_mean, previous_mean
        )

        return {
            "comparison": "completed",
            "current_period": {
                "mean_intensity": current_mean,
                "volatility": current_volatility,
                "sample_count": len(current_emotions),
            },
            "previous_period": {
                "mean_intensity": previous_mean,
                "volatility": previous_volatility,
                "sample_count": len(previous_emotions),
            },
            "changes": {
                "intensity_change": intensity_change,
                "volatility_change": volatility_change,
                "intensity_change_percent": (
                    (intensity_change / previous_mean) * 100 if previous_mean > 0 else 0
                ),
                "significance": change_significance,
            },
        }

    def calibrate_intensity_scale(
        self,
        emotional_states: List[EmotionalState],
        target_distribution: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Calibrate intensity scale based on historical data.

        Args:
            emotional_states: Historical emotional states for calibration
            target_distribution: Optional target intensity distribution

        Returns:
            Calibration results and adjustments
        """
        if len(emotional_states) < 20:
            return {"calibration": "insufficient_data"}

        # Default target distribution (normal-like)
        if target_distribution is None:
            target_distribution = {
                "low": 0.2,  # 20% low intensity (0.0-0.3)
                "medium": 0.6,  # 60% medium intensity (0.3-0.7)
                "high": 0.2,  # 20% high intensity (0.7-1.0)
            }

        # Analyze current distribution
        intensities = [e.intensity for e in emotional_states]
        current_distribution = self._calculate_intensity_distribution(intensities)

        # Calculate calibration adjustments
        calibration_adjustments = {}
        for emotion_type in EmotionType:
            emotion_intensities = [
                e.intensity
                for e in emotional_states
                if e.primary_emotion == emotion_type
            ]

            if len(emotion_intensities) >= 5:
                adjustment = self._calculate_calibration_adjustment(
                    emotion_intensities, target_distribution
                )
                calibration_adjustments[emotion_type.value] = adjustment

        # Update calibration data
        for emotion_type, adjustment in calibration_adjustments.items():
            self.calibration_data[emotion_type] = adjustment

        return {
            "calibration": "completed",
            "sample_count": len(emotional_states),
            "current_distribution": current_distribution,
            "target_distribution": target_distribution,
            "adjustments": calibration_adjustments,
        }

    def _apply_text_modifiers(
        self, base_intensity: float, indicators: List[str]
    ) -> float:
        """Apply text-based intensity modifiers."""
        modified_intensity = base_intensity

        # Combine all indicators into a single text
        text = " ".join(indicators).lower()

        # Apply modifiers in order of strength
        for modifier_type in ["amplifiers", "diminishers", "negators"]:
            modifiers = self.INTENSITY_MODIFIERS[modifier_type]

            for modifier, multiplier in modifiers.items():
                if modifier in text:
                    modified_intensity *= multiplier
                    break  # Apply only the first modifier found in each category

        return modified_intensity

    def _apply_contextual_adjustments(
        self, base_intensity: float, context: Optional[ConversationContext]
    ) -> float:
        """Apply contextual intensity adjustments."""
        if not context:
            return base_intensity

        adjusted_intensity = base_intensity

        # Time of day adjustment
        hour = context.timestamp.hour
        if 5 <= hour < 12:
            time_period = "morning"
        elif 12 <= hour < 17:
            time_period = "afternoon"
        elif 17 <= hour < 22:
            time_period = "evening"
        else:
            time_period = "night"

        time_adjustment = self.CONTEXTUAL_ADJUSTMENTS["time_of_day"][time_period]
        adjusted_intensity *= time_adjustment

        # Message length adjustment
        message_length = len(context.message_text)
        if message_length < 50:
            length_category = "short"
        elif message_length < 200:
            length_category = "medium"
        else:
            length_category = "long"

        length_adjustment = self.CONTEXTUAL_ADJUSTMENTS["conversation_length"][
            length_category
        ]
        adjusted_intensity *= length_adjustment

        return adjusted_intensity

    def _apply_calibration(
        self, base_intensity: float, emotion_type: EmotionType
    ) -> float:
        """Apply calibration adjustments."""
        emotion_key = emotion_type.value

        if emotion_key in self.calibration_data and isinstance(
            self.calibration_data[emotion_key], (int, float)
        ):
            calibration_factor = self.calibration_data[emotion_key]
            return base_intensity * calibration_factor

        return base_intensity

    def _calculate_intensity_confidence(
        self,
        emotional_state: EmotionalState,
        context: Optional[ConversationContext],
        final_intensity: float,
    ) -> float:
        """Calculate confidence in intensity measurement."""
        confidence_factors = []

        # Base emotion confidence
        confidence_factors.append(emotional_state.confidence)

        # Indicator quality (more indicators = higher confidence)
        indicator_confidence = min(1.0, len(emotional_state.emotional_indicators) / 3.0)
        confidence_factors.append(indicator_confidence)

        # Context availability
        context_confidence = 0.8 if context else 0.5
        confidence_factors.append(context_confidence)

        # Intensity extremeness (extreme values often more reliable)
        extremeness = abs(final_intensity - 0.5) * 2  # 0 at center, 1 at extremes
        extremeness_confidence = 0.5 + (extremeness * 0.3)
        confidence_factors.append(extremeness_confidence)

        return sum(confidence_factors) / len(confidence_factors)

    def _analyze_intensity_characteristics(
        self, intensity: float, emotional_state: EmotionalState
    ) -> Dict[str, Any]:
        """Analyze characteristics of the measured intensity."""
        characteristics = {
            "level": self._categorize_intensity_level(intensity),
            "is_extreme": intensity >= 0.8 or intensity <= 0.2,
            "is_moderate": 0.3 <= intensity <= 0.7,
            "deviation_from_emotion_baseline": self._calculate_baseline_deviation(
                intensity, emotional_state.primary_emotion
            ),
        }

        # Add descriptive labels
        if intensity >= 0.9:
            characteristics["description"] = "overwhelming"
        elif intensity >= 0.8:
            characteristics["description"] = "intense"
        elif intensity >= 0.6:
            characteristics["description"] = "strong"
        elif intensity >= 0.4:
            characteristics["description"] = "moderate"
        elif intensity >= 0.2:
            characteristics["description"] = "mild"
        else:
            characteristics["description"] = "subtle"

        return characteristics

    def _get_applied_modifiers(self, indicators: List[str]) -> List[str]:
        """Get list of text modifiers that were applied."""
        applied = []
        text = " ".join(indicators).lower()

        for modifier_type, modifiers in self.INTENSITY_MODIFIERS.items():
            for modifier in modifiers:
                if modifier in text:
                    applied.append(f"{modifier_type}:{modifier}")

        return applied

    def _get_contextual_factors(
        self, context: Optional[ConversationContext]
    ) -> Dict[str, Any]:
        """Get contextual factors that influenced intensity."""
        if not context:
            return {}

        factors = {}

        # Time factor
        hour = context.timestamp.hour
        if 5 <= hour < 12:
            factors["time_of_day"] = "morning"
        elif 12 <= hour < 17:
            factors["time_of_day"] = "afternoon"
        elif 17 <= hour < 22:
            factors["time_of_day"] = "evening"
        else:
            factors["time_of_day"] = "night"

        # Message length factor
        message_length = len(context.message_text)
        if message_length < 50:
            factors["message_length"] = "short"
        elif message_length < 200:
            factors["message_length"] = "medium"
        else:
            factors["message_length"] = "long"

        return factors

    def _analyze_intensity_trend(
        self, intensities: List[float], emotions: List[EmotionalState]
    ) -> Dict[str, Any]:
        """Analyze intensity trend over time."""
        if len(intensities) < 3:
            return {"trend": "insufficient_data"}

        # Calculate simple linear trend
        n = len(intensities)
        x_values = list(range(n))

        # Linear regression slope
        slope = (
            n * sum(i * intensity for i, intensity in enumerate(intensities))
            - sum(x_values) * sum(intensities)
        ) / (n * sum(x * x for x in x_values) - sum(x_values) ** 2)

        # Determine trend direction
        if slope > 0.02:
            trend_direction = "increasing"
        elif slope < -0.02:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"

        # Calculate recent vs older average
        recent_avg = statistics.mean(intensities[-n // 3 :])
        older_avg = statistics.mean(intensities[: n // 3])

        return {
            "trend": trend_direction,
            "slope": slope,
            "recent_average": recent_avg,
            "older_average": older_avg,
            "change": recent_avg - older_avg,
        }

    def _analyze_intensity_volatility(self, intensities: List[float]) -> Dict[str, Any]:
        """Analyze intensity volatility."""
        if len(intensities) < 2:
            return {"volatility": "insufficient_data"}

        # Calculate standard deviation as volatility measure
        volatility = statistics.stdev(intensities)

        # Categorize volatility
        if volatility > 0.3:
            volatility_level = "high"
        elif volatility > 0.15:
            volatility_level = "medium"
        else:
            volatility_level = "low"

        # Calculate coefficient of variation
        mean_intensity = statistics.mean(intensities)
        cv = volatility / mean_intensity if mean_intensity > 0 else 0

        return {
            "volatility": volatility_level,
            "standard_deviation": volatility,
            "coefficient_of_variation": cv,
            "stability": "stable" if volatility < 0.15 else "unstable",
        }

    def _analyze_intensity_by_emotion(
        self, emotions: List[EmotionalState]
    ) -> Dict[str, Any]:
        """Analyze intensity patterns by emotion type."""
        emotion_intensities = defaultdict(list)

        for emotion in emotions:
            intensity_result = self.measure_intensity(emotion)
            emotion_intensities[emotion.primary_emotion.value].append(
                intensity_result["final_intensity"]
            )

        analysis = {}
        for emotion_type, intensities in emotion_intensities.items():
            if len(intensities) >= 2:
                analysis[emotion_type] = {
                    "count": len(intensities),
                    "mean": statistics.mean(intensities),
                    "std_dev": statistics.stdev(intensities),
                    "min": min(intensities),
                    "max": max(intensities),
                }

        return analysis

    def _calculate_intensity_distribution(
        self, intensities: List[float]
    ) -> Dict[str, float]:
        """Calculate intensity distribution."""
        low_count = sum(1 for i in intensities if i < 0.3)
        medium_count = sum(1 for i in intensities if 0.3 <= i < 0.7)
        high_count = sum(1 for i in intensities if i >= 0.7)

        total = len(intensities)

        return {
            "low": low_count / total,
            "medium": medium_count / total,
            "high": high_count / total,
        }

    def _calculate_calibration_adjustment(
        self, intensities: List[float], target_distribution: Dict[str, float]
    ) -> float:
        """Calculate calibration adjustment factor."""
        current_mean = statistics.mean(intensities)
        target_mean = 0.5  # Target mean intensity

        # Simple linear adjustment
        if current_mean > 0:
            adjustment = target_mean / current_mean
        else:
            adjustment = 1.0

        # Limit adjustment to reasonable range
        return max(0.5, min(2.0, adjustment))

    def _categorize_intensity_level(self, intensity: float) -> str:
        """Categorize intensity into levels."""
        if intensity >= 0.8:
            return "very_high"
        elif intensity >= 0.6:
            return "high"
        elif intensity >= 0.4:
            return "medium"
        elif intensity >= 0.2:
            return "low"
        else:
            return "very_low"

    def _calculate_baseline_deviation(
        self, intensity: float, emotion_type: EmotionType
    ) -> float:
        """Calculate deviation from emotion baseline."""
        baseline = 0.5  # Default baseline

        # Adjust baseline based on emotion type
        if emotion_type in [EmotionType.ANGER, EmotionType.FEAR]:
            baseline = 0.6  # These emotions tend to be more intense
        elif emotion_type in [EmotionType.SADNESS, EmotionType.NEUTRAL]:
            baseline = 0.4  # These tend to be less intense

        return intensity - baseline

    def _assess_change_significance(
        self, change: float, current_mean: float, previous_mean: float
    ) -> str:
        """Assess significance of intensity change."""
        if abs(change) < 0.05:
            return "negligible"
        elif abs(change) < 0.15:
            return "small"
        elif abs(change) < 0.25:
            return "moderate"
        else:
            return "large"

    def _update_calibration_data(self, emotion_type: EmotionType, intensity: float):
        """Update calibration data with new measurement."""
        emotion_key = emotion_type.value

        # Keep a rolling window of recent intensities for calibration
        if emotion_key not in self.calibration_data:
            self.calibration_data[emotion_key] = []

        self.calibration_data[emotion_key].append(intensity)

        # Keep only recent data (last 100 measurements)
        if len(self.calibration_data[emotion_key]) > 100:
            self.calibration_data[emotion_key] = self.calibration_data[emotion_key][
                -100:
            ]


# Singleton instance
_intensity_measurer_instance = None
_intensity_measurer_lock = threading.Lock()


def get_intensity_measurer() -> IntensityMeasurer:
    """
    Get singleton intensity measurer instance.

    Returns:
        Shared IntensityMeasurer instance
    """
    global _intensity_measurer_instance

    if _intensity_measurer_instance is None:
        with _intensity_measurer_lock:
            if _intensity_measurer_instance is None:
                _intensity_measurer_instance = IntensityMeasurer()

    return _intensity_measurer_instance
