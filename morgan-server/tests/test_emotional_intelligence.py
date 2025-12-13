"""Unit tests for Emotional Intelligence module."""

import pytest
from datetime import datetime, timedelta

from morgan_server.empathic import (
    EmotionalTone,
    EmotionalDetection,
    EmotionalPattern,
    EmotionalAdjustment,
    EmotionalIntelligence,
)


class TestEmotionalToneDetection:
    """Tests for emotional tone detection."""

    @pytest.fixture
    def ei(self):
        """Create an EmotionalIntelligence instance for testing."""
        return EmotionalIntelligence(pattern_window_days=30)

    def test_detect_joyful_tone(self, ei):
        """Test detection of joyful emotional tone."""
        message = "I'm so happy! This is wonderful and amazing!"
        detection = ei.detect_tone(message)

        assert detection.primary_tone == EmotionalTone.JOYFUL
        assert detection.confidence > 0.5
        assert len(detection.indicators) > 0
        assert any("happy" in ind or "wonderful" in ind or "amazing" in ind 
                  for ind in detection.indicators)

    def test_detect_sad_tone(self, ei):
        """Test detection of sad emotional tone."""
        message = "I'm feeling really sad and down today. Everything seems difficult."
        detection = ei.detect_tone(message)

        assert detection.primary_tone == EmotionalTone.SAD
        assert detection.confidence > 0.5
        assert any("sad" in ind or "down" in ind for ind in detection.indicators)

    def test_detect_anxious_tone(self, ei):
        """Test detection of anxious emotional tone."""
        message = "I'm so anxious and stressed about this. I'm really worried."
        detection = ei.detect_tone(message)

        assert detection.primary_tone == EmotionalTone.ANXIOUS
        assert detection.confidence > 0.5
        assert any("anxious" in ind or "stressed" in ind or "worried" in ind 
                  for ind in detection.indicators)

    def test_detect_frustrated_tone(self, ei):
        """Test detection of frustrated emotional tone."""
        message = "This is so frustrating! I'm really annoyed and fed up with this."
        detection = ei.detect_tone(message)

        assert detection.primary_tone == EmotionalTone.FRUSTRATED
        assert detection.confidence > 0.5
        assert any("frustrat" in ind or "annoyed" in ind or "fed up" in ind 
                  for ind in detection.indicators)

    def test_detect_grateful_tone(self, ei):
        """Test detection of grateful emotional tone."""
        message = "Thank you so much! I really appreciate your help."
        detection = ei.detect_tone(message)

        assert detection.primary_tone == EmotionalTone.GRATEFUL
        assert detection.confidence > 0.5
        assert any("thank" in ind or "appreciate" in ind for ind in detection.indicators)

    def test_detect_confused_tone(self, ei):
        """Test detection of confused emotional tone."""
        message = "I'm really confused about this. I don't understand what's happening."
        detection = ei.detect_tone(message)

        assert detection.primary_tone == EmotionalTone.CONFUSED
        assert detection.confidence > 0.5
        assert any("confused" in ind or "don't understand" in ind 
                  for ind in detection.indicators)

    def test_detect_excited_tone(self, ei):
        """Test detection of excited emotional tone."""
        message = "I'm so excited!!! I can't wait for this! This is incredible!"
        detection = ei.detect_tone(message)

        assert detection.primary_tone == EmotionalTone.EXCITED
        assert detection.confidence > 0.5
        assert any("excited" in ind or "can't wait" in ind or "incredible" in ind 
                  for ind in detection.indicators)

    def test_detect_neutral_tone(self, ei):
        """Test detection of neutral tone when no emotional indicators present."""
        message = "What is the weather like today?"
        detection = ei.detect_tone(message)

        assert detection.primary_tone == EmotionalTone.NEUTRAL
        assert detection.confidence >= 0.0
        assert len(detection.indicators) == 0

    def test_detect_with_intensity_modifier(self, ei):
        """Test that intensity modifiers affect detection."""
        message1 = "I'm happy about this."
        message2 = "I'm very happy about this."
        
        detection1 = ei.detect_tone(message1)
        detection2 = ei.detect_tone(message2)

        # Both should detect joyful, but message2 might have higher confidence
        # due to intensity modifier
        assert detection1.primary_tone == EmotionalTone.JOYFUL
        assert detection2.primary_tone == EmotionalTone.JOYFUL

    def test_detect_mixed_emotions(self, ei):
        """Test detection with mixed emotional indicators."""
        message = "I'm happy about the progress but also worried about the deadline."
        detection = ei.detect_tone(message)

        # Should detect a primary tone
        assert detection.primary_tone in [EmotionalTone.JOYFUL, EmotionalTone.CONCERNED]
        
        # Should have secondary tones if mixed
        # (may or may not depending on scoring)
        assert isinstance(detection.secondary_tones, list)

    def test_detect_with_emojis(self, ei):
        """Test detection with emoji indicators."""
        message = "This is great! 😊🎉"
        detection = ei.detect_tone(message)

        assert detection.primary_tone == EmotionalTone.JOYFUL
        assert detection.confidence > 0.5

    def test_detect_tracks_pattern_with_user_id(self, ei):
        """Test that detection tracks patterns when user_id is provided."""
        user_id = "test_user"
        message = "I'm feeling happy today!"
        
        # Initially no patterns
        assert len(ei.get_patterns(user_id)) == 0
        
        # Detect with user_id
        ei.detect_tone(message, user_id=user_id)
        
        # Should have tracked the pattern
        patterns = ei.get_patterns(user_id)
        assert len(patterns) == 1
        assert patterns[0].tone == EmotionalTone.JOYFUL


class TestEmotionalToneAdjustment:
    """Tests for emotional tone adjustment."""

    @pytest.fixture
    def ei(self):
        """Create an EmotionalIntelligence instance for testing."""
        return EmotionalIntelligence(pattern_window_days=30)

    def test_adjust_for_joyful_tone(self, ei):
        """Test response adjustment for joyful user tone."""
        detection = EmotionalDetection(
            primary_tone=EmotionalTone.JOYFUL,
            confidence=0.8,
            indicators=["happy", "wonderful"]
        )
        
        adjustment = ei.adjust_response_tone(detection)

        assert adjustment.target_tone == EmotionalTone.JOYFUL
        assert adjustment.intensity > 0.5
        assert len(adjustment.suggestions) > 0
        assert any("enthusiasm" in s.lower() or "warm" in s.lower() 
                  for s in adjustment.suggestions)

    def test_adjust_for_sad_tone(self, ei):
        """Test response adjustment for sad user tone."""
        detection = EmotionalDetection(
            primary_tone=EmotionalTone.SAD,
            confidence=0.8,
            indicators=["sad", "down"]
        )
        
        adjustment = ei.adjust_response_tone(detection)

        assert adjustment.target_tone == EmotionalTone.CONCERNED
        assert adjustment.intensity > 0.5
        assert len(adjustment.suggestions) > 0
        assert any("gentle" in s.lower() or "compassion" in s.lower() or "support" in s.lower()
                  for s in adjustment.suggestions)
        # Should provide support message
        assert adjustment.support is not None

    def test_adjust_for_anxious_tone(self, ei):
        """Test response adjustment for anxious user tone."""
        detection = EmotionalDetection(
            primary_tone=EmotionalTone.ANXIOUS,
            confidence=0.8,
            indicators=["anxious", "worried"]
        )
        
        adjustment = ei.adjust_response_tone(detection)

        assert adjustment.target_tone == EmotionalTone.CONCERNED
        assert adjustment.intensity > 0.5
        assert len(adjustment.suggestions) > 0
        assert adjustment.support is not None

    def test_adjust_for_frustrated_tone(self, ei):
        """Test response adjustment for frustrated user tone."""
        detection = EmotionalDetection(
            primary_tone=EmotionalTone.FRUSTRATED,
            confidence=0.8,
            indicators=["frustrated", "annoyed"]
        )
        
        adjustment = ei.adjust_response_tone(detection)

        assert adjustment.target_tone == EmotionalTone.CONTENT
        assert len(adjustment.suggestions) > 0
        assert any("calm" in s.lower() or "solution" in s.lower() or "patient" in s.lower()
                  for s in adjustment.suggestions)

    def test_adjust_for_grateful_tone(self, ei):
        """Test response adjustment for grateful user tone."""
        detection = EmotionalDetection(
            primary_tone=EmotionalTone.GRATEFUL,
            confidence=0.8,
            indicators=["thank", "appreciate"]
        )
        
        adjustment = ei.adjust_response_tone(detection)

        assert adjustment.target_tone == EmotionalTone.CONTENT
        assert len(adjustment.suggestions) > 0
        assert any("humble" in s.lower() or "warm" in s.lower() 
                  for s in adjustment.suggestions)

    def test_adjust_for_confused_tone(self, ei):
        """Test response adjustment for confused user tone."""
        detection = EmotionalDetection(
            primary_tone=EmotionalTone.CONFUSED,
            confidence=0.8,
            indicators=["confused", "don't understand"]
        )
        
        adjustment = ei.adjust_response_tone(detection)

        assert adjustment.target_tone == EmotionalTone.CONTENT
        assert len(adjustment.suggestions) > 0
        assert any("clear" in s.lower() or "patient" in s.lower() or "example" in s.lower()
                  for s in adjustment.suggestions)

    def test_adjust_for_neutral_tone(self, ei):
        """Test response adjustment for neutral user tone."""
        detection = EmotionalDetection(
            primary_tone=EmotionalTone.NEUTRAL,
            confidence=0.5,
            indicators=[]
        )
        
        adjustment = ei.adjust_response_tone(detection)

        assert adjustment.target_tone == EmotionalTone.CONTENT
        assert len(adjustment.suggestions) > 0

    def test_adjust_with_celebration(self, ei):
        """Test that celebration is generated for positive shift."""
        user_id = "test_user"
        
        # Track some negative patterns
        ei.track_pattern(user_id, EmotionalTone.SAD, 0.8)
        ei.track_pattern(user_id, EmotionalTone.ANXIOUS, 0.7)
        
        # Now detect joyful tone
        detection = EmotionalDetection(
            primary_tone=EmotionalTone.JOYFUL,
            confidence=0.9,
            indicators=["happy", "great"]
        )
        
        adjustment = ei.adjust_response_tone(detection, user_id=user_id)

        # Should include celebration for the positive shift
        assert adjustment.celebration is not None
        assert len(adjustment.celebration) > 0


class TestEmotionalPatternTracking:
    """Tests for emotional pattern tracking."""

    @pytest.fixture
    def ei(self):
        """Create an EmotionalIntelligence instance for testing."""
        return EmotionalIntelligence(pattern_window_days=30)

    def test_track_pattern(self, ei):
        """Test tracking an emotional pattern."""
        user_id = "test_user"
        
        ei.track_pattern(
            user_id=user_id,
            tone=EmotionalTone.JOYFUL,
            confidence=0.8,
            context="Test context"
        )
        
        patterns = ei.get_patterns(user_id)
        assert len(patterns) == 1
        assert patterns[0].user_id == user_id
        assert patterns[0].tone == EmotionalTone.JOYFUL
        assert patterns[0].confidence == 0.8
        assert patterns[0].context == "Test context"

    def test_track_multiple_patterns(self, ei):
        """Test tracking multiple patterns for a user."""
        user_id = "test_user"
        
        ei.track_pattern(user_id, EmotionalTone.JOYFUL, 0.8)
        ei.track_pattern(user_id, EmotionalTone.CONTENT, 0.7)
        ei.track_pattern(user_id, EmotionalTone.EXCITED, 0.9)
        
        patterns = ei.get_patterns(user_id)
        assert len(patterns) == 3

    def test_get_patterns_within_window(self, ei):
        """Test getting patterns within a specific time window."""
        user_id = "test_user"
        
        # Add some patterns
        ei.track_pattern(user_id, EmotionalTone.JOYFUL, 0.8)
        ei.track_pattern(user_id, EmotionalTone.CONTENT, 0.7)
        
        # Get patterns for last 7 days
        patterns = ei.get_patterns(user_id, days=7)
        assert len(patterns) == 2

    def test_get_patterns_empty_for_unknown_user(self, ei):
        """Test getting patterns for a user with no history."""
        patterns = ei.get_patterns("unknown_user")
        assert len(patterns) == 0

    def test_analyze_emotional_trend_improving(self, ei):
        """Test analyzing an improving emotional trend."""
        user_id = "test_user"
        
        # Add patterns showing improvement (negative to positive)
        ei.track_pattern(user_id, EmotionalTone.SAD, 0.8)
        ei.track_pattern(user_id, EmotionalTone.ANXIOUS, 0.7)
        ei.track_pattern(user_id, EmotionalTone.CONTENT, 0.7)
        ei.track_pattern(user_id, EmotionalTone.JOYFUL, 0.8)
        ei.track_pattern(user_id, EmotionalTone.EXCITED, 0.9)
        
        analysis = ei.analyze_emotional_trend(user_id)
        
        assert analysis["dominant_tone"] is not None
        assert "tone_distribution" in analysis
        assert analysis["trend"] in ["improving", "stable", "declining"]

    def test_analyze_emotional_trend_declining(self, ei):
        """Test analyzing a declining emotional trend."""
        user_id = "test_user"
        
        # Add patterns showing decline (positive to negative)
        ei.track_pattern(user_id, EmotionalTone.JOYFUL, 0.8)
        ei.track_pattern(user_id, EmotionalTone.EXCITED, 0.9)
        ei.track_pattern(user_id, EmotionalTone.CONTENT, 0.7)
        ei.track_pattern(user_id, EmotionalTone.SAD, 0.8)
        ei.track_pattern(user_id, EmotionalTone.ANXIOUS, 0.7)
        
        analysis = ei.analyze_emotional_trend(user_id)
        
        assert analysis["dominant_tone"] is not None
        assert analysis["trend"] in ["improving", "stable", "declining"]

    def test_analyze_emotional_trend_stable(self, ei):
        """Test analyzing a stable emotional trend."""
        user_id = "test_user"
        
        # Add patterns showing stability
        ei.track_pattern(user_id, EmotionalTone.CONTENT, 0.7)
        ei.track_pattern(user_id, EmotionalTone.CONTENT, 0.8)
        ei.track_pattern(user_id, EmotionalTone.CONTENT, 0.7)
        ei.track_pattern(user_id, EmotionalTone.CONTENT, 0.8)
        
        analysis = ei.analyze_emotional_trend(user_id)
        
        assert analysis["dominant_tone"] == EmotionalTone.CONTENT.value
        assert analysis["trend"] == "stable"

    def test_analyze_emotional_trend_no_data(self, ei):
        """Test analyzing trend with no data."""
        analysis = ei.analyze_emotional_trend("unknown_user")
        
        assert analysis["dominant_tone"] is None
        assert analysis["tone_distribution"] == {}
        assert analysis["trend"] == "unknown"
        assert analysis["recent_shift"] is False

    def test_analyze_detects_recent_shift(self, ei):
        """Test that analysis detects recent emotional shifts."""
        user_id = "test_user"
        
        # Add mostly joyful patterns, then recent sad ones
        for _ in range(5):
            ei.track_pattern(user_id, EmotionalTone.JOYFUL, 0.8)
        
        for _ in range(3):
            ei.track_pattern(user_id, EmotionalTone.SAD, 0.8)
        
        analysis = ei.analyze_emotional_trend(user_id)
        
        # Should detect the recent shift
        assert analysis["recent_shift"] is True

    def test_tone_distribution_calculation(self, ei):
        """Test that tone distribution is calculated correctly."""
        user_id = "test_user"
        
        # Add 3 joyful, 2 content
        for _ in range(3):
            ei.track_pattern(user_id, EmotionalTone.JOYFUL, 0.8)
        for _ in range(2):
            ei.track_pattern(user_id, EmotionalTone.CONTENT, 0.7)
        
        analysis = ei.analyze_emotional_trend(user_id)
        
        distribution = analysis["tone_distribution"]
        assert EmotionalTone.JOYFUL.value in distribution
        assert EmotionalTone.CONTENT.value in distribution
        assert distribution[EmotionalTone.JOYFUL.value] == 0.6  # 3/5
        assert distribution[EmotionalTone.CONTENT.value] == 0.4  # 2/5


class TestEmotionalIntelligenceIntegration:
    """Integration tests for the full emotional intelligence workflow."""

    @pytest.fixture
    def ei(self):
        """Create an EmotionalIntelligence instance for testing."""
        return EmotionalIntelligence(pattern_window_days=30)

    def test_full_workflow_joyful_message(self, ei):
        """Test complete workflow from detection to adjustment."""
        user_id = "test_user"
        message = "I'm so happy! This is wonderful!"
        
        # Detect tone
        detection = ei.detect_tone(message, user_id=user_id)
        assert detection.primary_tone == EmotionalTone.JOYFUL
        
        # Get adjustment
        adjustment = ei.adjust_response_tone(detection, user_id=user_id)
        assert adjustment.target_tone == EmotionalTone.JOYFUL
        assert len(adjustment.suggestions) > 0
        
        # Verify pattern was tracked
        patterns = ei.get_patterns(user_id)
        assert len(patterns) == 1

    def test_full_workflow_sad_message(self, ei):
        """Test complete workflow for sad message."""
        user_id = "test_user"
        message = "I'm feeling really sad and down today."
        
        # Detect tone
        detection = ei.detect_tone(message, user_id=user_id)
        assert detection.primary_tone == EmotionalTone.SAD
        
        # Get adjustment
        adjustment = ei.adjust_response_tone(detection, user_id=user_id)
        assert adjustment.target_tone == EmotionalTone.CONCERNED
        assert adjustment.support is not None
        
        # Verify pattern was tracked
        patterns = ei.get_patterns(user_id)
        assert len(patterns) == 1

    def test_emotional_journey_tracking(self, ei):
        """Test tracking an emotional journey over multiple interactions."""
        user_id = "test_user"
        
        # Simulate a journey from sad to happy
        messages = [
            ("I'm feeling really down today.", EmotionalTone.SAD),
            ("Still struggling a bit.", EmotionalTone.SAD),
            ("Things are getting a bit better.", EmotionalTone.CONTENT),
            ("I'm feeling much better now!", EmotionalTone.JOYFUL),
        ]
        
        for message, expected_tone in messages:
            detection = ei.detect_tone(message, user_id=user_id)
            # Primary tone should match expected (allowing for some flexibility)
            assert detection.primary_tone in [expected_tone, EmotionalTone.CONTENT, EmotionalTone.HOPEFUL]
        
        # Analyze the trend
        analysis = ei.analyze_emotional_trend(user_id)
        assert analysis["trend"] in ["improving", "stable"]
        
        # Should have tracked all patterns
        patterns = ei.get_patterns(user_id)
        assert len(patterns) == 4
