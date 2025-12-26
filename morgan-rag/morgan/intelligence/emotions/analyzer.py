"""
Mood pattern analysis module.

Provides comprehensive mood pattern analysis, trend detection, and emotional
stability assessment over time periods.
"""

import statistics
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.intelligence.core.models import EmotionalState, EmotionType, MoodPattern
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class MoodAnalyzer:
    """
    Mood pattern analysis and trend detection.

    Features:
    - Multi-timeframe mood analysis (daily, weekly, monthly)
    - Emotional trend detection and prediction
    - Mood stability assessment
    - Pattern confidence scoring
    - Comparative mood analysis
    """

    # Timeframe configurations
    TIMEFRAME_CONFIGS = {
        "1d": {"days": 1, "min_samples": 3},
        "3d": {"days": 3, "min_samples": 5},
        "7d": {"days": 7, "min_samples": 10},
        "14d": {"days": 14, "min_samples": 15},
        "30d": {"days": 30, "min_samples": 20},
        "90d": {"days": 90, "min_samples": 30},
    }

    # Emotion valence mapping for trend analysis
    EMOTION_VALENCE = {
        EmotionType.JOY: 1.0,
        EmotionType.SURPRISE: 0.3,
        EmotionType.NEUTRAL: 0.0,
        EmotionType.FEAR: -0.4,
        EmotionType.DISGUST: -0.6,
        EmotionType.SADNESS: -0.8,
        EmotionType.ANGER: -0.9,
    }

    def __init__(self):
        """Initialize mood analyzer."""
        self.settings = get_settings()
        logger.info("Mood Analyzer initialized")

    def analyze_mood_patterns(
        self,
        user_id: str,
        emotional_states: List[EmotionalState],
        timeframe: str = "30d",
    ) -> MoodPattern:
        """
        Analyze mood patterns for a user over a specified timeframe.

        Args:
            user_id: User identifier
            emotional_states: List of emotional states to analyze
            timeframe: Analysis timeframe (e.g., "7d", "30d")

        Returns:
            Comprehensive mood pattern analysis
        """
        if timeframe not in self.TIMEFRAME_CONFIGS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        config = self.TIMEFRAME_CONFIGS[timeframe]
        cutoff_date = datetime.utcnow() - timedelta(days=config["days"])

        # Filter emotions within timeframe
        recent_emotions = [
            emotion for emotion in emotional_states if emotion.timestamp >= cutoff_date
        ]

        if len(recent_emotions) < config["min_samples"]:
            return self._create_insufficient_data_pattern(
                user_id, timeframe, len(recent_emotions)
            )

        # Analyze dominant emotions
        dominant_emotions = self._analyze_dominant_emotions(recent_emotions)

        # Calculate average intensity
        average_intensity = self._calculate_average_intensity(recent_emotions)

        # Assess mood stability
        mood_stability = self._assess_mood_stability(recent_emotions)

        # Analyze emotional trends
        emotional_trends = self._analyze_emotional_trends(recent_emotions, timeframe)

        # Calculate pattern confidence
        pattern_confidence = self._calculate_pattern_confidence(recent_emotions, config)

        return MoodPattern(
            user_id=user_id,
            timeframe=timeframe,
            dominant_emotions=dominant_emotions,
            average_intensity=average_intensity,
            mood_stability=mood_stability,
            emotional_trends=emotional_trends,
            pattern_confidence=pattern_confidence,
        )

    def compare_mood_periods(
        self,
        user_id: str,
        current_emotions: List[EmotionalState],
        previous_emotions: List[EmotionalState],
        timeframe: str = "7d",
    ) -> Dict[str, Any]:
        """
        Compare mood patterns between two time periods.

        Args:
            user_id: User identifier
            current_emotions: Recent emotional states
            previous_emotions: Previous period emotional states
            timeframe: Comparison timeframe

        Returns:
            Mood comparison analysis
        """
        current_pattern = self.analyze_mood_patterns(
            user_id, current_emotions, timeframe
        )
        previous_pattern = self.analyze_mood_patterns(
            user_id, previous_emotions, timeframe
        )

        # Calculate changes
        intensity_change = (
            current_pattern.average_intensity - previous_pattern.average_intensity
        )
        stability_change = (
            current_pattern.mood_stability - previous_pattern.mood_stability
        )

        # Analyze emotion shifts
        emotion_shifts = self._analyze_emotion_shifts(
            previous_pattern.dominant_emotions, current_pattern.dominant_emotions
        )

        # Determine overall trend
        overall_trend = self._determine_overall_trend(
            intensity_change, stability_change, emotion_shifts
        )

        return {
            "current_period": current_pattern,
            "previous_period": previous_pattern,
            "intensity_change": intensity_change,
            "stability_change": stability_change,
            "emotion_shifts": emotion_shifts,
            "overall_trend": overall_trend,
            "comparison_confidence": min(
                current_pattern.pattern_confidence, previous_pattern.pattern_confidence
            ),
        }

    def detect_mood_anomalies(
        self,
        user_id: str,
        emotional_states: List[EmotionalState],
        baseline_days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Detect unusual mood patterns or emotional anomalies.

        Args:
            user_id: User identifier
            emotional_states: Emotional states to analyze
            baseline_days: Days to use for baseline calculation

        Returns:
            List of detected anomalies
        """
        if len(emotional_states) < 10:
            return []  # Need sufficient data for anomaly detection

        # Sort by timestamp
        sorted_emotions = sorted(emotional_states, key=lambda e: e.timestamp)

        # Calculate baseline statistics
        baseline_cutoff = datetime.utcnow() - timedelta(days=baseline_days)
        baseline_emotions = [
            e for e in sorted_emotions if e.timestamp >= baseline_cutoff
        ]

        if len(baseline_emotions) < 5:
            return []

        baseline_stats = self._calculate_baseline_statistics(baseline_emotions)

        # Detect anomalies
        anomalies = []

        # Check recent emotions against baseline
        recent_emotions = sorted_emotions[-5:]  # Last 5 emotions

        for emotion in recent_emotions:
            anomaly = self._check_emotion_anomaly(emotion, baseline_stats)
            if anomaly:
                anomalies.append(anomaly)

        # Check for pattern anomalies
        pattern_anomalies = self._detect_pattern_anomalies(
            recent_emotions, baseline_stats
        )
        anomalies.extend(pattern_anomalies)

        logger.debug(f"Detected {len(anomalies)} mood anomalies for user {user_id}")
        return anomalies

    def predict_mood_trajectory(
        self,
        user_id: str,
        emotional_states: List[EmotionalState],
        prediction_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Predict likely mood trajectory based on historical patterns.

        Args:
            user_id: User identifier
            emotional_states: Historical emotional states
            prediction_days: Days to predict ahead

        Returns:
            Mood trajectory prediction
        """
        if len(emotional_states) < 14:
            return {"prediction": "insufficient_data", "confidence": 0.0}

        # Sort by timestamp
        sorted_emotions = sorted(emotional_states, key=lambda e: e.timestamp)

        # Analyze recent trends
        recent_trend = self._calculate_recent_trend(sorted_emotions[-14:])

        # Calculate cyclical patterns
        cyclical_patterns = self._detect_cyclical_patterns(sorted_emotions)

        # Generate prediction
        prediction = self._generate_mood_prediction(
            recent_trend, cyclical_patterns, prediction_days
        )

        return prediction

    def _analyze_dominant_emotions(
        self, emotions: List[EmotionalState]
    ) -> List[EmotionType]:
        """Identify dominant emotions in the dataset."""
        emotion_counts = Counter(e.primary_emotion for e in emotions)

        # Weight by intensity
        weighted_scores = defaultdict(float)
        for emotion in emotions:
            weighted_scores[emotion.primary_emotion] += emotion.intensity

        # Combine frequency and intensity
        combined_scores = {}
        total_emotions = len(emotions)

        for emotion_type in emotion_counts:
            frequency_score = emotion_counts[emotion_type] / total_emotions
            intensity_score = (
                weighted_scores[emotion_type] / emotion_counts[emotion_type]
            )
            combined_scores[emotion_type] = frequency_score * intensity_score

        # Return top 3 emotions
        sorted_emotions = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )

        return [emotion for emotion, _ in sorted_emotions[:3]]

    def _calculate_average_intensity(self, emotions: List[EmotionalState]) -> float:
        """Calculate average emotional intensity."""
        if not emotions:
            return 0.5

        return sum(e.intensity for e in emotions) / len(emotions)

    def _assess_mood_stability(self, emotions: List[EmotionalState]) -> float:
        """
        Assess mood stability based on intensity variance and emotion consistency.

        Returns:
            Stability score from 0.0 (very unstable) to 1.0 (very stable)
        """
        if len(emotions) < 2:
            return 0.5

        # Calculate intensity variance
        intensities = [e.intensity for e in emotions]
        intensity_variance = statistics.variance(intensities)

        # Calculate emotion consistency (how often the same emotion appears)
        emotion_counts = Counter(e.primary_emotion for e in emotions)
        max_emotion_frequency = max(emotion_counts.values()) / len(emotions)

        # Combine metrics (lower variance and higher consistency = more stable)
        intensity_stability = max(0.0, 1.0 - (intensity_variance * 2))
        emotion_stability = max_emotion_frequency

        # Weighted combination
        overall_stability = (intensity_stability * 0.6) + (emotion_stability * 0.4)

        return min(1.0, max(0.0, overall_stability))

    def _analyze_emotional_trends(
        self, emotions: List[EmotionalState], timeframe: str
    ) -> Dict[str, Any]:
        """Analyze emotional trends over the timeframe."""
        if len(emotions) < 3:
            return {"trend": "insufficient_data"}

        # Sort by timestamp
        sorted_emotions = sorted(emotions, key=lambda e: e.timestamp)

        # Calculate valence trend
        valence_scores = [
            self.EMOTION_VALENCE.get(e.primary_emotion, 0.0) * e.intensity
            for e in sorted_emotions
        ]

        # Simple linear trend calculation
        if len(valence_scores) >= 3:
            recent_avg = statistics.mean(valence_scores[-len(valence_scores) // 3 :])
            older_avg = statistics.mean(valence_scores[: len(valence_scores) // 3])

            trend_direction = "stable"
            if recent_avg > older_avg + 0.1:
                trend_direction = "improving"
            elif recent_avg < older_avg - 0.1:
                trend_direction = "declining"
        else:
            trend_direction = "stable"

        # Calculate intensity trend
        intensities = [e.intensity for e in sorted_emotions]
        recent_intensity = statistics.mean(intensities[-len(intensities) // 3 :])
        older_intensity = statistics.mean(intensities[: len(intensities) // 3])

        intensity_trend = "stable"
        if recent_intensity > older_intensity + 0.1:
            intensity_trend = "increasing"
        elif recent_intensity < older_intensity - 0.1:
            intensity_trend = "decreasing"

        return {
            "valence_trend": trend_direction,
            "intensity_trend": intensity_trend,
            "recent_valence": recent_avg if len(valence_scores) >= 3 else 0.0,
            "valence_change": (
                recent_avg - older_avg if len(valence_scores) >= 3 else 0.0
            ),
            "most_recent_emotions": [
                e.primary_emotion.value for e in sorted_emotions[-3:]
            ],
        }

    def _calculate_pattern_confidence(
        self, emotions: List[EmotionalState], config: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the pattern analysis."""
        sample_count = len(emotions)
        min_samples = config["min_samples"]

        # Base confidence on sample size
        sample_confidence = min(1.0, sample_count / (min_samples * 2))

        # Adjust for emotion confidence scores
        avg_emotion_confidence = sum(e.confidence for e in emotions) / len(emotions)

        # Combine factors
        overall_confidence = (sample_confidence * 0.7) + (avg_emotion_confidence * 0.3)

        return min(1.0, max(0.0, overall_confidence))

    def _create_insufficient_data_pattern(
        self, user_id: str, timeframe: str, sample_count: int
    ) -> MoodPattern:
        """Create pattern for insufficient data cases."""
        return MoodPattern(
            user_id=user_id,
            timeframe=timeframe,
            dominant_emotions=[EmotionType.NEUTRAL],
            average_intensity=0.5,
            mood_stability=0.5,
            emotional_trends={
                "trend": "insufficient_data",
                "sample_count": sample_count,
            },
            pattern_confidence=0.0,
        )

    def _analyze_emotion_shifts(
        self, previous_emotions: List[EmotionType], current_emotions: List[EmotionType]
    ) -> Dict[str, Any]:
        """Analyze shifts between emotion sets."""
        prev_set = set(previous_emotions)
        curr_set = set(current_emotions)

        new_emotions = curr_set - prev_set
        lost_emotions = prev_set - curr_set
        stable_emotions = prev_set & curr_set

        return {
            "new_emotions": [e.value for e in new_emotions],
            "lost_emotions": [e.value for e in lost_emotions],
            "stable_emotions": [e.value for e in stable_emotions],
            "shift_magnitude": len(new_emotions) + len(lost_emotions),
        }

    def _determine_overall_trend(
        self,
        intensity_change: float,
        stability_change: float,
        emotion_shifts: Dict[str, Any],
    ) -> str:
        """Determine overall mood trend."""
        # Positive indicators
        positive_score = 0
        if intensity_change > 0.1:
            positive_score += 1
        if stability_change > 0.1:
            positive_score += 1
        if EmotionType.JOY.value in emotion_shifts.get("new_emotions", []):
            positive_score += 1

        # Negative indicators
        negative_score = 0
        if intensity_change < -0.1:
            negative_score += 1
        if stability_change < -0.1:
            negative_score += 1
        if any(
            e in emotion_shifts.get("new_emotions", [])
            for e in [EmotionType.SADNESS.value, EmotionType.ANGER.value]
        ):
            negative_score += 1

        if positive_score > negative_score:
            return "improving"
        elif negative_score > positive_score:
            return "declining"
        else:
            return "stable"

    def _calculate_baseline_statistics(
        self, emotions: List[EmotionalState]
    ) -> Dict[str, Any]:
        """Calculate baseline statistics for anomaly detection."""
        intensities = [e.intensity for e in emotions]
        emotion_counts = Counter(e.primary_emotion for e in emotions)

        return {
            "mean_intensity": statistics.mean(intensities),
            "intensity_stdev": (
                statistics.stdev(intensities) if len(intensities) > 1 else 0.1
            ),
            "common_emotions": set(emotion_counts.keys()),
            "emotion_frequencies": dict(emotion_counts),
            "total_samples": len(emotions),
        }

    def _check_emotion_anomaly(
        self, emotion: EmotionalState, baseline_stats: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check if an emotion is anomalous compared to baseline."""
        # Check intensity anomaly
        intensity_z_score = abs(
            (emotion.intensity - baseline_stats["mean_intensity"])
            / baseline_stats["intensity_stdev"]
        )

        # Check emotion type anomaly
        is_unusual_emotion = (
            emotion.primary_emotion not in baseline_stats["common_emotions"]
        )

        # Determine if anomalous
        if intensity_z_score > 2.0 or is_unusual_emotion:
            return {
                "type": "emotion_anomaly",
                "emotion": emotion.primary_emotion.value,
                "intensity": emotion.intensity,
                "timestamp": emotion.timestamp,
                "anomaly_score": max(
                    intensity_z_score / 3.0, 0.8 if is_unusual_emotion else 0.0
                ),
                "reason": (
                    "unusual_intensity"
                    if intensity_z_score > 2.0
                    else "unusual_emotion"
                ),
            }

        return None

    def _detect_pattern_anomalies(
        self, recent_emotions: List[EmotionalState], baseline_stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect pattern-level anomalies."""
        anomalies = []

        # Check for emotional volatility
        if len(recent_emotions) >= 3:
            recent_intensities = [e.intensity for e in recent_emotions]
            recent_variance = statistics.variance(recent_intensities)

            if recent_variance > 0.3:  # High volatility threshold
                anomalies.append(
                    {
                        "type": "volatility_anomaly",
                        "variance": recent_variance,
                        "timestamp": recent_emotions[-1].timestamp,
                        "anomaly_score": min(1.0, recent_variance / 0.5),
                        "reason": "high_emotional_volatility",
                    }
                )

        return anomalies

    def _calculate_recent_trend(self, emotions: List[EmotionalState]) -> Dict[str, Any]:
        """Calculate recent emotional trend."""
        if len(emotions) < 3:
            return {"trend": "insufficient_data"}

        # Calculate valence trend over time
        valence_scores = [
            self.EMOTION_VALENCE.get(e.primary_emotion, 0.0) * e.intensity
            for e in emotions
        ]

        # Simple linear regression slope
        n = len(valence_scores)
        x_values = list(range(n))

        slope = (
            n * sum(i * v for i, v in enumerate(valence_scores))
            - sum(x_values) * sum(valence_scores)
        ) / (n * sum(x * x for x in x_values) - sum(x_values) ** 2)

        return {
            "slope": slope,
            "direction": (
                "improving"
                if slope > 0.01
                else "declining" if slope < -0.01 else "stable"
            ),
            "recent_valence": statistics.mean(valence_scores[-3:]),
            "trend_strength": abs(slope),
        }

    def _detect_cyclical_patterns(
        self, emotions: List[EmotionalState]
    ) -> Dict[str, Any]:
        """Detect cyclical patterns in emotions."""
        # Simplified cyclical detection - could be enhanced with FFT analysis
        if len(emotions) < 14:
            return {"patterns": "insufficient_data"}

        # Group by day of week
        day_patterns = defaultdict(list)
        for emotion in emotions:
            day_of_week = emotion.timestamp.weekday()
            valence = self.EMOTION_VALENCE.get(emotion.primary_emotion, 0.0)
            day_patterns[day_of_week].append(valence * emotion.intensity)

        # Calculate average valence by day
        day_averages = {}
        for day, valences in day_patterns.items():
            if len(valences) >= 2:
                day_averages[day] = statistics.mean(valences)

        return {
            "weekly_pattern": day_averages,
            "pattern_detected": len(day_averages) >= 5,
        }

    def _generate_mood_prediction(
        self,
        recent_trend: Dict[str, Any],
        cyclical_patterns: Dict[str, Any],
        prediction_days: int,
    ) -> Dict[str, Any]:
        """Generate mood trajectory prediction."""
        if recent_trend.get("trend") == "insufficient_data":
            return {"prediction": "insufficient_data", "confidence": 0.0}

        # Base prediction on recent trend
        trend_direction = recent_trend.get("direction", "stable")
        trend_strength = recent_trend.get("trend_strength", 0.0)

        # Adjust confidence based on trend strength and data quality
        confidence = min(0.8, trend_strength * 2.0)

        # Generate prediction
        if trend_direction == "improving":
            prediction = "likely_improvement"
        elif trend_direction == "declining":
            prediction = "potential_decline"
        else:
            prediction = "stable_continuation"

        return {
            "prediction": prediction,
            "confidence": confidence,
            "trend_direction": trend_direction,
            "prediction_days": prediction_days,
            "factors": {
                "recent_trend": recent_trend,
                "cyclical_influence": cyclical_patterns.get("pattern_detected", False),
            },
        }


# Singleton instance
_analyzer_instance = None
_analyzer_lock = threading.Lock()


def get_mood_analyzer() -> MoodAnalyzer:
    """
    Get singleton mood analyzer instance.

    Returns:
        Shared MoodAnalyzer instance
    """
    global _analyzer_instance

    if _analyzer_instance is None:
        with _analyzer_lock:
            if _analyzer_instance is None:
                _analyzer_instance = MoodAnalyzer()

    return _analyzer_instance
