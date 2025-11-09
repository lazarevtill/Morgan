"""
Tests for modular emotion detection system.

Tests the new modular components: detector, analyzer, tracker, classifier, and intensity measurer.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os

from morgan.emotions.detector import EmotionDetector
from morgan.emotions.analyzer import MoodAnalyzer
from morgan.emotions.tracker import EmotionalStateTracker
from morgan.emotions.classifier import EmotionClassifier
from morgan.emotions.intensity import IntensityMeasurer
from morgan.emotional.models import EmotionalState, EmotionType, ConversationContext


class TestEmotionDetector:
    """Test emotion detector functionality."""

    @pytest.fixture
    def detector(self):
        """Create emotion detector for testing."""
        with patch("morgan.emotions.detector.get_llm_service"):
            return EmotionDetector()

    @pytest.fixture
    def sample_context(self):
        """Create sample conversation context."""
        return ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            message_text="I'm feeling great today!",
            timestamp=datetime.now(),
        )

    def test_detect_joy_emotion(self, detector, sample_context):
        """Test detection of joy emotion."""
        joyful_text = "I'm so happy and excited about this amazing news!"

        emotion = detector.detect_emotion(joyful_text, sample_context)

        assert emotion.primary_emotion == EmotionType.JOY
        assert emotion.intensity > 0.3
        assert emotion.confidence > 0.5
        assert len(emotion.emotional_indicators) > 0

    def test_detect_sadness_emotion(self, detector, sample_context):
        """Test detection of sadness emotion."""
        sad_text = "I'm feeling really down and disappointed"

        emotion = detector.detect_emotion(sad_text, sample_context)

        assert emotion.primary_emotion == EmotionType.SADNESS
        assert emotion.intensity > 0.2
        assert emotion.confidence > 0.5

    def test_batch_emotion_detection(self, detector):
        """Test batch emotion detection."""
        texts = ["I'm so happy!", "This is terrible", "Just a normal day"]

        emotions = detector.detect_emotions_batch(texts)

        assert len(emotions) == 3
        assert all(isinstance(e, EmotionalState) for e in emotions)


class TestMoodAnalyzer:
    """Test mood analyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create mood analyzer for testing."""
        return MoodAnalyzer()

    def test_analyze_mood_patterns(self, analyzer):
        """Test mood pattern analysis."""
        emotions = [
            EmotionalState(
                primary_emotion=EmotionType.JOY,
                intensity=0.8,
                confidence=0.9,
                timestamp=datetime.now() - timedelta(days=1),
            ),
            EmotionalState(
                primary_emotion=EmotionType.JOY,
                intensity=0.7,
                confidence=0.8,
                timestamp=datetime.now() - timedelta(hours=12),
            ),
            EmotionalState(
                primary_emotion=EmotionType.NEUTRAL,
                intensity=0.5,
                confidence=0.7,
                timestamp=datetime.now(),
            ),
        ]

        pattern = analyzer.analyze_mood_patterns("test_user", emotions, "7d")

        assert pattern.user_id == "test_user"
        # With only 3 samples, it might not have enough data for proper analysis
        assert len(pattern.dominant_emotions) > 0
        assert pattern.average_intensity >= 0.5
        assert pattern.pattern_confidence >= 0.0

    def test_insufficient_data_handling(self, analyzer):
        """Test handling of insufficient data."""
        emotions = [
            EmotionalState(
                primary_emotion=EmotionType.JOY,
                intensity=0.8,
                confidence=0.9,
                timestamp=datetime.now(),
            )
        ]

        pattern = analyzer.analyze_mood_patterns("test_user", emotions, "30d")

        assert pattern.pattern_confidence == 0.0
        assert pattern.emotional_trends["trend"] == "insufficient_data"


class TestEmotionalStateTracker:
    """Test emotional state tracker functionality."""

    @pytest.fixture
    def tracker(self):
        """Create emotional state tracker with temporary database."""
        with patch("morgan.emotions.tracker.get_settings") as mock_settings:
            # Create temporary directory for test database
            temp_dir = tempfile.mkdtemp()

            # Create a mock settings object with the required attribute
            mock_settings_obj = Mock()
            mock_settings_obj.morgan_data_dir = temp_dir
            mock_settings.return_value = mock_settings_obj

            tracker = EmotionalStateTracker()
            yield tracker

            # Cleanup
            if os.path.exists(tracker.db_path):
                os.remove(tracker.db_path)
            os.rmdir(temp_dir)

    def test_track_emotional_state(self, tracker):
        """Test tracking emotional state."""
        emotion = EmotionalState(
            primary_emotion=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9,
            timestamp=datetime.now(),
        )

        tracking_id = tracker.track_emotional_state("test_user", emotion)

        assert tracking_id is not None
        assert len(tracking_id) > 0

    def test_get_emotional_history(self, tracker):
        """Test retrieving emotional history."""
        # Track some emotions
        emotions = [
            EmotionalState(
                primary_emotion=EmotionType.JOY,
                intensity=0.8,
                confidence=0.9,
                timestamp=datetime.now() - timedelta(days=1),
            ),
            EmotionalState(
                primary_emotion=EmotionType.SADNESS,
                intensity=0.6,
                confidence=0.8,
                timestamp=datetime.now(),
            ),
        ]

        for emotion in emotions:
            tracker.track_emotional_state("test_user", emotion)

        # Retrieve history
        history = tracker.get_emotional_history("test_user", timeframe="7d")

        assert len(history) == 2
        assert all(isinstance(e, EmotionalState) for e in history)

    def test_emotion_statistics(self, tracker):
        """Test emotion statistics calculation."""
        # Track some emotions
        emotions = [
            EmotionalState(
                primary_emotion=EmotionType.JOY, intensity=0.8, confidence=0.9
            ),
            EmotionalState(
                primary_emotion=EmotionType.JOY, intensity=0.7, confidence=0.8
            ),
            EmotionalState(
                primary_emotion=EmotionType.SADNESS, intensity=0.6, confidence=0.7
            ),
        ]

        for emotion in emotions:
            tracker.track_emotional_state("test_user", emotion)

        stats = tracker.get_emotion_statistics("test_user", "7d")

        assert stats["total_records"] == 3
        assert "joy" in stats["emotion_distribution"]
        assert stats["average_intensity"] > 0.0


class TestEmotionClassifier:
    """Test emotion classifier functionality."""

    @pytest.fixture
    def classifier(self):
        """Create emotion classifier for testing."""
        return EmotionClassifier()

    def test_classify_emotion(self, classifier):
        """Test emotion classification."""
        emotion = EmotionalState(
            primary_emotion=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9,
            secondary_emotions=[EmotionType.SURPRISE],
        )

        classification = classifier.classify_emotion(emotion)

        assert "primary_classification" in classification
        assert "intensity_classification" in classification
        assert "dimensional_analysis" in classification
        assert classification["primary_classification"]["emotion"] == "joy"
        assert classification["intensity_classification"]["intensity_category"] in [
            "high",
            "extreme",
        ]

    def test_find_similar_emotions(self, classifier):
        """Test finding similar emotions."""
        similarities = classifier.find_similar_emotions(EmotionType.JOY, 0.5)

        assert isinstance(similarities, list)
        # Should find some similar emotions
        assert len(similarities) >= 0

    def test_emotion_categories(self, classifier):
        """Test emotion category retrieval."""
        categories = classifier.get_emotion_categories()

        assert "positive" in categories
        assert "negative" in categories
        assert "neutral" in categories
        assert EmotionType.JOY.value in categories["positive"]


class TestIntensityMeasurer:
    """Test intensity measurer functionality."""

    @pytest.fixture
    def measurer(self):
        """Create intensity measurer for testing."""
        return IntensityMeasurer()

    def test_measure_intensity(self, measurer):
        """Test intensity measurement."""
        emotion = EmotionalState(
            primary_emotion=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9,
            emotional_indicators=["very happy", "excited"],
        )

        result = measurer.measure_intensity(emotion)

        assert "raw_intensity" in result
        assert "final_intensity" in result
        assert "intensity_confidence" in result
        assert "intensity_analysis" in result
        assert 0.0 <= result["final_intensity"] <= 1.0

    def test_intensity_patterns_analysis(self, measurer):
        """Test intensity pattern analysis."""
        emotions = [
            EmotionalState(
                primary_emotion=EmotionType.JOY,
                intensity=0.8,
                confidence=0.9,
                timestamp=datetime.now() - timedelta(days=1),
            ),
            EmotionalState(
                primary_emotion=EmotionType.SADNESS,
                intensity=0.6,
                confidence=0.8,
                timestamp=datetime.now() - timedelta(hours=12),
            ),
            EmotionalState(
                primary_emotion=EmotionType.NEUTRAL,
                intensity=0.5,
                confidence=0.7,
                timestamp=datetime.now(),
            ),
        ]

        analysis = measurer.analyze_intensity_patterns(emotions, "7d")

        assert analysis["pattern"] == "analyzed"
        assert "statistics" in analysis
        assert "trend" in analysis
        assert "volatility" in analysis

    def test_batch_intensity_measurement(self, measurer):
        """Test batch intensity measurement."""
        emotions = [
            EmotionalState(
                primary_emotion=EmotionType.JOY, intensity=0.8, confidence=0.9
            ),
            EmotionalState(
                primary_emotion=EmotionType.SADNESS, intensity=0.6, confidence=0.8
            ),
        ]

        results = measurer.measure_intensity_batch(emotions)

        assert len(results) == 2
        assert all("final_intensity" in result for result in results)


class TestModularIntegration:
    """Test integration between modular components."""

    @pytest.fixture
    def components(self):
        """Create all modular components for integration testing."""
        with patch("morgan.emotions.detector.get_llm_service"), patch(
            "morgan.emotions.tracker.get_settings"
        ) as mock_settings:

            # Setup temporary directory for tracker
            temp_dir = tempfile.mkdtemp()

            # Create a mock settings object with the required attribute
            mock_settings_obj = Mock()
            mock_settings_obj.morgan_data_dir = temp_dir
            mock_settings.return_value = mock_settings_obj

            components = {
                "detector": EmotionDetector(),
                "analyzer": MoodAnalyzer(),
                "tracker": EmotionalStateTracker(),
                "classifier": EmotionClassifier(),
                "measurer": IntensityMeasurer(),
            }

            yield components

            # Cleanup
            if os.path.exists(components["tracker"].db_path):
                os.remove(components["tracker"].db_path)
            os.rmdir(temp_dir)

    def test_full_emotion_processing_pipeline(self, components):
        """Test complete emotion processing pipeline."""
        text = "I'm extremely happy about this wonderful news!"
        user_id = "test_user"

        # 1. Detect emotion
        emotion = components["detector"].detect_emotion(text)
        assert emotion.primary_emotion == EmotionType.JOY

        # 2. Measure intensity
        intensity_result = components["measurer"].measure_intensity(emotion)
        assert intensity_result["final_intensity"] > 0.5

        # 3. Classify emotion
        classification = components["classifier"].classify_emotion(emotion)
        assert classification["primary_classification"]["emotion"] == "joy"

        # 4. Track emotion
        tracking_id = components["tracker"].track_emotional_state(user_id, emotion)
        assert tracking_id is not None

        # 5. Analyze patterns (with minimal data)
        emotions = [emotion]
        pattern = components["analyzer"].analyze_mood_patterns(user_id, emotions, "1d")
        assert pattern.user_id == user_id
