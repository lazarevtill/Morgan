"""
Emotional Processing Optimizer for Morgan RAG.

Optimizes emotional intelligence processing for real-time companion interactions:
- Fast emotion detection and analysis
- Cached emotional state management
- Optimized empathy response generation
- Real-time mood pattern analysis

Key Features:
- Sub-100ms emotion detection for real-time interactions
- Intelligent caching of emotional states and patterns
- Batch processing for historical emotional analysis
- Memory-efficient emotional data structures
"""

import hashlib
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from morgan.config import get_settings
from morgan.utils.cache import FileCache
from morgan.utils.error_decorators import (
    RetryConfig,
    handle_embedding_errors,
    monitor_performance,
)
from morgan.utils.error_handling import EmotionalProcessingError, ErrorSeverity
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EmotionalState:
    """Optimized emotional state representation."""

    primary_emotion: str
    intensity: float
    confidence: float
    secondary_emotions: List[str]
    emotional_indicators: List[str]
    timestamp: datetime
    user_id: str

    def to_cache_key(self) -> str:
        """Generate cache key for this emotional state."""
        content = f"{self.user_id}:{self.primary_emotion}:{self.intensity:.2f}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class EmotionalPattern:
    """Emotional pattern analysis result."""

    user_id: str
    dominant_emotions: List[Tuple[str, float]]  # (emotion, frequency)
    mood_trend: str  # "improving", "declining", "stable"
    emotional_volatility: float  # 0.0 to 1.0
    pattern_confidence: float
    analysis_period: timedelta
    last_updated: datetime


@dataclass
class OptimizationMetrics:
    """Performance metrics for emotional processing."""

    avg_detection_time: float
    cache_hit_rate: float
    total_detections: int
    batch_processing_speedup: float
    memory_usage_mb: float


class EmotionalProcessingOptimizer:
    """
    High-performance optimizer for emotional intelligence processing.

    Provides:
    - Real-time emotion detection (target: <100ms)
    - Intelligent caching of emotional states and patterns
    - Batch processing for historical analysis
    - Memory-efficient data structures
    """

    def __init__(self):
        """Initialize emotional processing optimizer."""
        self.settings = get_settings()

        # Caching system
        cache_dir = self.settings.morgan_data_dir / "cache" / "emotional"
        self.emotion_cache = FileCache(cache_dir / "emotions")
        self.pattern_cache = FileCache(cache_dir / "patterns")

        # In-memory caches for real-time processing
        self.recent_emotions = defaultdict(
            lambda: deque(maxlen=100)
        )  # Last 100 emotions per user
        self.emotion_keywords = self._load_emotion_keywords()
        self.pattern_cache_memory = {}  # User patterns in memory

        # Performance tracking
        self.metrics = {
            "detection_times": deque(maxlen=1000),
            "cache_hits": 0,
            "cache_misses": 0,
            "total_detections": 0,
            "batch_operations": 0,
        }
        self.metrics_lock = threading.Lock()

        # Optimization settings
        self.cache_ttl = 300  # 5 minutes for emotional states
        self.pattern_cache_ttl = 3600  # 1 hour for patterns
        self.batch_size = 50

        logger.info("EmotionalProcessingOptimizer initialized")

    @monitor_performance("detect_emotion_fast", "emotional_optimizer")
    def detect_emotion_fast(
        self,
        text: str,
        user_id: str,
        use_cache: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> EmotionalState:
        """
        Fast emotion detection optimized for real-time interactions.

        Target: <100ms response time for typical inputs.

        Args:
            text: Text to analyze for emotions
            user_id: User identifier for personalization
            use_cache: Whether to use caching (default: True)
            context: Optional context for enhanced detection

        Returns:
            EmotionalState with detected emotions
        """
        start_time = time.time()

        try:
            # Generate cache key
            cache_key = self._generate_emotion_cache_key(text, user_id, context)

            # Check cache first
            if use_cache:
                cached_emotion = self.emotion_cache.get(cache_key)
                if cached_emotion:
                    with self.metrics_lock:
                        self.metrics["cache_hits"] += 1
                        self.metrics["total_detections"] += 1

                    detection_time = time.time() - start_time
                    logger.debug(f"Emotion cache hit in {detection_time*1000:.1f}ms")
                    return cached_emotion

            # Fast keyword-based detection
            emotional_state = self._detect_emotion_keywords(text, user_id, context)

            # Cache the result
            if use_cache:
                self.emotion_cache.set(cache_key, emotional_state, ttl=self.cache_ttl)

            # Update recent emotions for pattern analysis
            self.recent_emotions[user_id].append(emotional_state)

            # Update metrics
            detection_time = time.time() - start_time
            with self.metrics_lock:
                self.metrics["detection_times"].append(detection_time)
                self.metrics["cache_misses"] += 1
                self.metrics["total_detections"] += 1

            logger.debug(
                f"Fast emotion detection completed in {detection_time*1000:.1f}ms"
            )
            return emotional_state

        except Exception as e:
            detection_time = time.time() - start_time
            raise EmotionalProcessingError(
                f"Fast emotion detection failed: {e}",
                operation="detect_emotion_fast",
                component="emotional_optimizer",
                severity=ErrorSeverity.MEDIUM,
                metadata={
                    "text_length": len(text),
                    "user_id": user_id,
                    "detection_time": detection_time,
                    "error_type": type(e).__name__,
                },
            ) from e

    @handle_embedding_errors(
        "analyze_emotional_patterns_batch",
        "emotional_optimizer",
        RetryConfig(max_attempts=2, base_delay=1.0),
    )
    @monitor_performance("analyze_emotional_patterns_batch", "emotional_optimizer")
    def analyze_emotional_patterns_batch(
        self,
        user_emotions: Dict[str, List[EmotionalState]],
        analysis_period: timedelta = timedelta(days=30),
    ) -> Dict[str, EmotionalPattern]:
        """
        Batch analyze emotional patterns for multiple users.

        Optimized for processing large amounts of historical emotional data.

        Args:
            user_emotions: Dictionary of user_id -> list of emotional states
            analysis_period: Time period for pattern analysis

        Returns:
            Dictionary of user_id -> emotional pattern
        """
        start_time = time.time()
        patterns = {}

        try:
            logger.info(f"Analyzing emotional patterns for {len(user_emotions)} users")

            # Process users in batches for memory efficiency
            user_ids = list(user_emotions.keys())

            for i in range(0, len(user_ids), self.batch_size):
                batch_users = user_ids[i : i + self.batch_size]

                for user_id in batch_users:
                    emotions = user_emotions[user_id]

                    # Check pattern cache first
                    pattern_cache_key = f"{user_id}:{analysis_period.days}d"
                    cached_pattern = self.pattern_cache.get(pattern_cache_key)

                    if cached_pattern and self._is_pattern_fresh(cached_pattern):
                        patterns[user_id] = cached_pattern
                        continue

                    # Analyze pattern
                    pattern = self._analyze_user_emotional_pattern(
                        user_id, emotions, analysis_period
                    )
                    patterns[user_id] = pattern

                    # Cache the pattern
                    self.pattern_cache.set(
                        pattern_cache_key, pattern, ttl=self.pattern_cache_ttl
                    )

                    # Store in memory cache for fast access
                    self.pattern_cache_memory[user_id] = pattern

            processing_time = time.time() - start_time

            with self.metrics_lock:
                self.metrics["batch_operations"] += 1

            logger.info(
                f"Batch emotional pattern analysis completed: {len(patterns)} patterns "
                f"in {processing_time:.2f}s ({len(patterns)/processing_time:.1f} patterns/sec)"
            )

            return patterns

        except Exception as e:
            processing_time = time.time() - start_time
            raise EmotionalProcessingError(
                f"Batch emotional pattern analysis failed: {e}",
                operation="analyze_emotional_patterns_batch",
                component="emotional_optimizer",
                severity=ErrorSeverity.HIGH,
                metadata={
                    "user_count": len(user_emotions),
                    "processing_time": processing_time,
                    "error_type": type(e).__name__,
                },
            ) from e

    def get_user_emotional_pattern(
        self, user_id: str, force_refresh: bool = False
    ) -> Optional[EmotionalPattern]:
        """
        Get emotional pattern for a user with caching.

        Args:
            user_id: User identifier
            force_refresh: Force refresh from recent emotions

        Returns:
            EmotionalPattern if available
        """
        # Check memory cache first
        if not force_refresh and user_id in self.pattern_cache_memory:
            pattern = self.pattern_cache_memory[user_id]
            if self._is_pattern_fresh(pattern):
                return pattern

        # Check file cache
        pattern_cache_key = f"{user_id}:30d"  # Default 30-day analysis
        cached_pattern = self.pattern_cache.get(pattern_cache_key)

        if (
            not force_refresh
            and cached_pattern
            and self._is_pattern_fresh(cached_pattern)
        ):
            self.pattern_cache_memory[user_id] = cached_pattern
            return cached_pattern

        # Generate new pattern from recent emotions
        if user_id in self.recent_emotions:
            emotions = list(self.recent_emotions[user_id])
            if emotions:
                pattern = self._analyze_user_emotional_pattern(
                    user_id, emotions, timedelta(days=30)
                )

                # Cache the new pattern
                self.pattern_cache.set(
                    pattern_cache_key, pattern, ttl=self.pattern_cache_ttl
                )
                self.pattern_cache_memory[user_id] = pattern

                return pattern

        return None

    def optimize_emotional_response_generation(
        self,
        user_emotion: EmotionalState,
        response_context: str,
        user_pattern: Optional[EmotionalPattern] = None,
    ) -> Dict[str, Any]:
        """
        Generate optimized emotional response parameters.

        Args:
            user_emotion: Current user emotional state
            response_context: Context for response generation
            user_pattern: Optional user emotional pattern for personalization

        Returns:
            Dictionary with optimized response parameters
        """
        try:
            # Base empathy level based on emotion intensity
            empathy_level = min(1.0, user_emotion.intensity * 1.2)

            # Adjust based on emotion type
            emotion_adjustments = {
                "sadness": {"empathy_boost": 0.3, "tone": "supportive"},
                "anger": {"empathy_boost": 0.2, "tone": "calming"},
                "fear": {"empathy_boost": 0.4, "tone": "reassuring"},
                "joy": {"empathy_boost": 0.1, "tone": "celebratory"},
                "surprise": {"empathy_boost": 0.1, "tone": "curious"},
                "disgust": {"empathy_boost": 0.2, "tone": "understanding"},
            }

            adjustment = emotion_adjustments.get(user_emotion.primary_emotion, {})
            empathy_level += adjustment.get("empathy_boost", 0.0)
            empathy_level = min(1.0, empathy_level)

            # Personalization based on user pattern
            personalization_factors = []
            if user_pattern:
                # Adjust for emotional volatility
                if user_pattern.emotional_volatility > 0.7:
                    empathy_level += 0.1  # More empathy for volatile users
                    personalization_factors.append("high_volatility_adjustment")

                # Adjust for mood trend
                if user_pattern.mood_trend == "declining":
                    empathy_level += 0.15
                    personalization_factors.append("declining_mood_support")

            # Response optimization parameters
            response_params = {
                "empathy_level": min(1.0, empathy_level),
                "emotional_tone": adjustment.get("tone", "neutral"),
                "response_length": (
                    "detailed" if user_emotion.intensity > 0.7 else "concise"
                ),
                "personalization_factors": personalization_factors,
                "confidence": user_emotion.confidence,
                "processing_time": time.time(),  # For performance tracking
            }

            logger.debug(
                f"Optimized emotional response: empathy={empathy_level:.2f}, "
                f"tone={response_params['emotional_tone']}"
            )

            return response_params

        except Exception as e:
            logger.error(f"Failed to optimize emotional response: {e}")
            # Return safe defaults
            return {
                "empathy_level": 0.5,
                "emotional_tone": "neutral",
                "response_length": "concise",
                "personalization_factors": [],
                "confidence": 0.5,
                "processing_time": time.time(),
            }

    def _detect_emotion_keywords(
        self, text: str, user_id: str, context: Optional[Dict[str, Any]] = None
    ) -> EmotionalState:
        """Fast keyword-based emotion detection."""
        text_lower = text.lower()
        emotion_scores = defaultdict(float)
        emotional_indicators = []

        # Score emotions based on keyword matches
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Weight by keyword strength and frequency
                    weight = keywords[keyword] if isinstance(keywords, dict) else 1.0
                    count = text_lower.count(keyword)
                    emotion_scores[emotion] += weight * count
                    emotional_indicators.append(keyword)

        # Determine primary emotion
        if emotion_scores:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            intensity = min(1.0, emotion_scores[primary_emotion] / 3.0)  # Normalize
            confidence = min(
                1.0, len(emotional_indicators) / 5.0
            )  # Based on indicator count

            # Get secondary emotions
            sorted_emotions = sorted(
                emotion_scores.items(), key=lambda x: x[1], reverse=True
            )
            secondary_emotions = [
                emotion for emotion, score in sorted_emotions[1:3] if score > 0
            ]
        else:
            # Default neutral state
            primary_emotion = "neutral"
            intensity = 0.1
            confidence = 0.3
            secondary_emotions = []

        return EmotionalState(
            primary_emotion=primary_emotion,
            intensity=intensity,
            confidence=confidence,
            secondary_emotions=secondary_emotions,
            emotional_indicators=emotional_indicators,
            timestamp=datetime.now(),
            user_id=user_id,
        )

    def _analyze_user_emotional_pattern(
        self, user_id: str, emotions: List[EmotionalState], analysis_period: timedelta
    ) -> EmotionalPattern:
        """Analyze emotional pattern for a user."""
        if not emotions:
            return EmotionalPattern(
                user_id=user_id,
                dominant_emotions=[("neutral", 1.0)],
                mood_trend="stable",
                emotional_volatility=0.0,
                pattern_confidence=0.0,
                analysis_period=analysis_period,
                last_updated=datetime.now(),
            )

        # Filter emotions within analysis period
        cutoff_time = datetime.now() - analysis_period
        recent_emotions = [e for e in emotions if e.timestamp >= cutoff_time]

        if not recent_emotions:
            recent_emotions = emotions[-10:]  # Use last 10 if none in period

        # Calculate emotion frequencies
        emotion_counts = defaultdict(int)
        total_intensity = 0.0
        intensities = []

        for emotion in recent_emotions:
            emotion_counts[emotion.primary_emotion] += 1
            total_intensity += emotion.intensity
            intensities.append(emotion.intensity)

        # Calculate dominant emotions
        total_emotions = len(recent_emotions)
        dominant_emotions = [
            (emotion, count / total_emotions)
            for emotion, count in sorted(
                emotion_counts.items(), key=lambda x: x[1], reverse=True
            )
        ]

        # Calculate mood trend (simplified)
        if len(recent_emotions) >= 5:
            first_half = recent_emotions[: len(recent_emotions) // 2]
            second_half = recent_emotions[len(recent_emotions) // 2 :]

            first_avg_intensity = sum(e.intensity for e in first_half) / len(first_half)
            second_avg_intensity = sum(e.intensity for e in second_half) / len(
                second_half
            )

            if second_avg_intensity > first_avg_intensity + 0.1:
                mood_trend = "improving"
            elif second_avg_intensity < first_avg_intensity - 0.1:
                mood_trend = "declining"
            else:
                mood_trend = "stable"
        else:
            mood_trend = "stable"

        # Calculate emotional volatility (standard deviation of intensities)
        if len(intensities) > 1:
            mean_intensity = sum(intensities) / len(intensities)
            variance = sum((x - mean_intensity) ** 2 for x in intensities) / len(
                intensities
            )
            emotional_volatility = min(1.0, (variance**0.5) * 2)  # Normalize to 0-1
        else:
            emotional_volatility = 0.0

        # Pattern confidence based on data quality
        pattern_confidence = min(
            1.0, len(recent_emotions) / 20.0
        )  # Full confidence at 20+ emotions

        return EmotionalPattern(
            user_id=user_id,
            dominant_emotions=dominant_emotions[:5],  # Top 5 emotions
            mood_trend=mood_trend,
            emotional_volatility=emotional_volatility,
            pattern_confidence=pattern_confidence,
            analysis_period=analysis_period,
            last_updated=datetime.now(),
        )

    def _load_emotion_keywords(self) -> Dict[str, Dict[str, float]]:
        """Load emotion keywords with weights for fast detection."""
        return {
            "joy": {
                "happy": 1.0,
                "excited": 1.0,
                "great": 0.8,
                "awesome": 1.0,
                "wonderful": 1.0,
                "amazing": 1.0,
                "fantastic": 1.0,
                "love": 0.9,
                "perfect": 0.8,
                "excellent": 0.9,
                "brilliant": 0.9,
                "thrilled": 1.0,
            },
            "sadness": {
                "sad": 1.0,
                "depressed": 1.0,
                "disappointed": 0.9,
                "upset": 0.8,
                "down": 0.7,
                "unhappy": 0.9,
                "miserable": 1.0,
                "heartbroken": 1.0,
                "devastated": 1.0,
                "crying": 0.9,
                "tears": 0.8,
                "lonely": 0.8,
            },
            "anger": {
                "angry": 1.0,
                "furious": 1.0,
                "mad": 0.9,
                "irritated": 0.8,
                "annoyed": 0.7,
                "frustrated": 0.9,
                "outraged": 1.0,
                "livid": 1.0,
                "hate": 0.9,
                "disgusted": 0.8,
                "pissed": 1.0,
                "enraged": 1.0,
            },
            "fear": {
                "scared": 1.0,
                "afraid": 1.0,
                "terrified": 1.0,
                "worried": 0.8,
                "anxious": 0.9,
                "nervous": 0.7,
                "panic": 1.0,
                "frightened": 1.0,
                "concerned": 0.6,
                "stressed": 0.8,
                "overwhelmed": 0.9,
                "helpless": 0.9,
            },
            "surprise": {
                "surprised": 1.0,
                "shocked": 1.0,
                "amazed": 0.9,
                "astonished": 1.0,
                "stunned": 1.0,
                "wow": 0.8,
                "incredible": 0.8,
                "unbelievable": 0.9,
                "unexpected": 0.7,
                "sudden": 0.6,
                "startled": 0.8,
                "blown away": 1.0,
            },
            "disgust": {
                "disgusted": 1.0,
                "revolted": 1.0,
                "sick": 0.8,
                "gross": 0.9,
                "awful": 0.8,
                "terrible": 0.8,
                "horrible": 0.9,
                "repulsive": 1.0,
                "nasty": 0.8,
                "yuck": 0.9,
                "ew": 0.7,
                "appalled": 1.0,
            },
        }

    def _generate_emotion_cache_key(
        self, text: str, user_id: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key for emotion detection."""
        # Use first 200 chars of text for cache key to balance uniqueness and performance
        text_sample = text[:200].lower().strip()
        context_str = str(sorted(context.items())) if context else ""

        cache_input = f"{user_id}:{text_sample}:{context_str}"
        return hashlib.sha256(cache_input.encode()).hexdigest()

    def _is_pattern_fresh(self, pattern: EmotionalPattern) -> bool:
        """Check if emotional pattern is still fresh."""
        age = datetime.now() - pattern.last_updated
        return age.total_seconds() < self.pattern_cache_ttl

    def get_optimization_metrics(self) -> OptimizationMetrics:
        """Get performance metrics for emotional processing."""
        with self.metrics_lock:
            detection_times = list(self.metrics["detection_times"])
            total_cache_requests = (
                self.metrics["cache_hits"] + self.metrics["cache_misses"]
            )

            avg_detection_time = (
                sum(detection_times) / len(detection_times) if detection_times else 0.0
            )
            cache_hit_rate = (
                (self.metrics["cache_hits"] / total_cache_requests * 100)
                if total_cache_requests > 0
                else 0.0
            )

            # Estimate batch speedup (simplified)
            batch_speedup = (
                self.batch_size * 0.7
            )  # Assume 70% efficiency gain from batching

        return OptimizationMetrics(
            avg_detection_time=avg_detection_time,
            cache_hit_rate=cache_hit_rate,
            total_detections=self.metrics["total_detections"],
            batch_processing_speedup=batch_speedup,
            memory_usage_mb=0.0,  # Would need system monitoring
        )

    def clear_caches(self):
        """Clear all emotional processing caches."""
        self.emotion_cache.clear()
        self.pattern_cache.clear()
        self.pattern_cache_memory.clear()
        self.recent_emotions.clear()

        logger.info("Emotional processing caches cleared")


# Singleton instance
_emotional_optimizer_instance = None
_emotional_optimizer_lock = threading.Lock()


def get_emotional_optimizer() -> EmotionalProcessingOptimizer:
    """
    Get singleton emotional processing optimizer instance (thread-safe).

    Returns:
        Shared EmotionalProcessingOptimizer instance
    """
    global _emotional_optimizer_instance

    if _emotional_optimizer_instance is None:
        with _emotional_optimizer_lock:
            if _emotional_optimizer_instance is None:
                _emotional_optimizer_instance = EmotionalProcessingOptimizer()

    return _emotional_optimizer_instance
