"""
Tests for interaction pattern analysis module.

Tests communication pattern analysis, topic pattern analysis,
timing pattern analysis, and behavioral pattern detection.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from morgan.learning.patterns import (
    InteractionPatternAnalyzer,
    PatternConfidence,
    CommunicationPattern,
    TopicPattern,
    TimingPattern,
    BehavioralPattern,
)
from morgan.emotional.models import (
    InteractionData,
    ConversationContext,
    EmotionalState,
    EmotionType,
    CommunicationStyle,
)


class TestInteractionPatternAnalyzer:
    """Test interaction pattern analyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create interaction pattern analyzer."""
        return InteractionPatternAnalyzer()

    @pytest.fixture
    def sample_interactions(self):
        """Create sample interactions for testing."""
        interactions = []
        messages = [
            "Please help me understand this technical concept",
            "Thank you for the detailed explanation",
            "Could you provide more examples?",
            "I appreciate your help with this",
            "Can you explain the algorithm in detail?",
        ]

        for i, msg in enumerate(messages):
            context = ConversationContext(
                user_id="test_user",
                conversation_id="test_conv",
                message_text=msg,
                timestamp=datetime.utcnow() - timedelta(hours=i),
            )
            interaction = InteractionData(
                interaction_id=f"test_{i}",
                user_id="test_user",
                conversation_context=context,
                emotional_state=EmotionalState(
                    primary_emotion=EmotionType.NEUTRAL,
                    intensity=0.5,
                    confidence=0.7,
                    timestamp=datetime.utcnow(),
                ),
                timestamp=datetime.utcnow() - timedelta(hours=i),
            )
            interactions.append(interaction)

        return interactions

    def test_analyze_patterns_sufficient_data(self, analyzer, sample_interactions):
        """Test analyzing patterns with sufficient data."""
        patterns = analyzer.analyze_patterns("test_user", sample_interactions * 2)

        assert patterns is not None
        assert patterns.user_id == "test_user"
        assert len(patterns.communication_patterns) > 0

    def test_analyze_patterns_insufficient_data(self, analyzer):
        """Test analyzing patterns with insufficient interactions."""
        few_interactions = [
            InteractionData(
                interaction_id="test_1",
                user_id="test_user",
                conversation_context=ConversationContext(
                    user_id="test_user",
                    conversation_id="test_conv",
                    message_text="Test",
                    timestamp=datetime.utcnow(),
                ),
                emotional_state=None,
                timestamp=datetime.utcnow(),
            )
        ]

        patterns = analyzer.analyze_patterns("test_user", few_interactions)

        assert patterns is not None
        assert len(patterns.communication_patterns) == 0

    def test_analyze_communication_patterns(self, analyzer, sample_interactions):
        """Test analyzing communication patterns."""
        patterns = analyzer._analyze_communication_patterns(
            "test_user", sample_interactions
        )

        assert isinstance(patterns, list)
        if patterns:
            assert isinstance(patterns[0], CommunicationPattern)
            assert patterns[0].user_id == "test_user"

    def test_calculate_formality_level_formal(self, analyzer):
        """Test calculating formality level - formal."""
        formal_messages = [
            "Please provide assistance with this matter",
            "Thank you for your help",
            "I would appreciate your guidance",
        ]

        formality = analyzer._calculate_formality_level(formal_messages)

        assert formality > 0.5

    def test_calculate_formality_level_casual(self, analyzer):
        """Test calculating formality level - casual."""
        casual_messages = [
            "Hey! Can you help?",
            "Yeah that's cool",
            "Awesome, thanks!!",
        ]

        formality = analyzer._calculate_formality_level(casual_messages)

        assert formality < 0.5

    def test_calculate_technical_depth_high(self, analyzer):
        """Test calculating technical depth - high."""
        technical_messages = [
            "Can you explain the algorithm implementation?",
            "What's the API architecture?",
            "How does the database optimization work?",
        ]

        depth = analyzer._calculate_technical_depth(technical_messages)

        assert depth > 0.2

    def test_calculate_technical_depth_low(self, analyzer):
        """Test calculating technical depth - low."""
        simple_messages = [
            "How are you today?",
            "Can you help me?",
            "Thank you for the information",
        ]

        depth = analyzer._calculate_technical_depth(simple_messages)

        assert depth < 0.3

    def test_determine_communication_style_formal(self, analyzer):
        """Test determining communication style - formal."""
        style = analyzer._determine_communication_style(
            formality_level=0.8, technical_depth=0.3
        )

        assert style == CommunicationStyle.FORMAL

    def test_determine_communication_style_technical(self, analyzer):
        """Test determining communication style - technical."""
        style = analyzer._determine_communication_style(
            formality_level=0.5, technical_depth=0.7
        )

        assert style == CommunicationStyle.TECHNICAL

    def test_determine_communication_style_casual(self, analyzer):
        """Test determining communication style - casual."""
        style = analyzer._determine_communication_style(
            formality_level=0.2, technical_depth=0.3
        )

        assert style == CommunicationStyle.CASUAL

    def test_extract_question_types(self, analyzer):
        """Test extracting question types from messages."""
        messages = [
            "How do I solve this problem?",
            "What is machine learning?",
            "Why does this happen?",
            "When should I use this?",
            "Where can I find more information?",
        ]

        question_types = analyzer._extract_question_types(messages)

        assert isinstance(question_types, list)
        assert "how-to" in question_types
        assert "definition" in question_types

    def test_determine_vocabulary_level_basic(self, analyzer):
        """Test determining vocabulary level - basic."""
        basic_messages = ["I am ok", "Can you help", "This is good"]

        level = analyzer._determine_vocabulary_level(basic_messages)

        assert level == "basic"

    def test_determine_vocabulary_level_advanced(self, analyzer):
        """Test determining vocabulary level - advanced."""
        advanced_messages = [
            "The implementation demonstrates sophisticated architecture",
            "Comprehensive understanding requires contextual analysis",
        ]

        level = analyzer._determine_vocabulary_level(advanced_messages)

        assert level in ["intermediate", "advanced"]

    def test_calculate_confidence_high(self, analyzer):
        """Test calculating confidence level - high."""
        confidence = analyzer._calculate_confidence(25)

        assert confidence == PatternConfidence.HIGH

    def test_calculate_confidence_medium(self, analyzer):
        """Test calculating confidence level - medium."""
        confidence = analyzer._calculate_confidence(15)

        assert confidence == PatternConfidence.MEDIUM

    def test_calculate_confidence_low(self, analyzer):
        """Test calculating confidence level - low."""
        confidence = analyzer._calculate_confidence(5)

        assert confidence == PatternConfidence.LOW

    def test_analyze_topic_patterns(self, analyzer, sample_interactions):
        """Test analyzing topic patterns."""
        patterns = analyzer._analyze_topic_patterns(
            "test_user", sample_interactions, timedelta(days=30)
        )

        assert isinstance(patterns, list)

    def test_analyze_timing_patterns(self, analyzer, sample_interactions):
        """Test analyzing timing patterns."""
        patterns = analyzer._analyze_timing_patterns("test_user", sample_interactions)

        assert isinstance(patterns, list)

    def test_analyze_behavioral_patterns(self, analyzer, sample_interactions):
        """Test analyzing behavioral patterns."""
        patterns = analyzer._analyze_behavioral_patterns(
            "test_user", sample_interactions
        )

        assert isinstance(patterns, list)
        if patterns:
            assert isinstance(patterns[0], BehavioralPattern)

    def test_analyze_active_hours(self, analyzer):
        """Test analyzing active hours from timestamps."""
        timestamps = [
            datetime(2024, 1, 1, 9, 0),
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 9, 30),
            datetime(2024, 1, 1, 14, 0),
            datetime(2024, 1, 1, 14, 15),
        ]

        active_hours = analyzer._analyze_active_hours(timestamps)

        assert isinstance(active_hours, list)
        assert len(active_hours) <= 3
        assert 9 in active_hours or 14 in active_hours

    def test_determine_interaction_frequency_daily(self, analyzer):
        """Test determining interaction frequency - daily."""
        timestamps = [datetime.utcnow() - timedelta(hours=i * 12) for i in range(5)]

        frequency = analyzer._determine_interaction_frequency(timestamps)

        assert frequency == "daily"

    def test_determine_interaction_frequency_weekly(self, analyzer):
        """Test determining interaction frequency - weekly."""
        timestamps = [datetime.utcnow() - timedelta(days=i * 3) for i in range(5)]

        frequency = analyzer._determine_interaction_frequency(timestamps)

        assert frequency == "weekly"

    def test_calculate_overall_confidence(self, analyzer, sample_interactions):
        """Test calculating overall confidence across patterns."""
        comm_patterns = analyzer._analyze_communication_patterns(
            "test_user", sample_interactions
        )
        topic_patterns = analyzer._analyze_topic_patterns(
            "test_user", sample_interactions, timedelta(days=30)
        )

        confidence = analyzer._calculate_overall_confidence(
            [comm_patterns, topic_patterns]
        )

        assert confidence >= 0.0
        assert confidence <= 1.0

    def test_create_empty_patterns(self, analyzer):
        """Test creating empty patterns structure."""
        patterns = analyzer._create_empty_patterns("test_user", timedelta(days=30))

        assert patterns.user_id == "test_user"
        assert len(patterns.communication_patterns) == 0
        assert patterns.overall_confidence == 0.0
