"""
Tests for enhanced memory processor with emotional awareness.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock

from morgan.memory.memory_processor import (
    MemoryProcessor,
    Memory,
    MemoryExtractionResult,
    get_memory_processor,
)
from morgan.core.memory import ConversationTurn
from morgan.intelligence.core.models import (
    EmotionalState,
    EmotionType,
    ConversationContext,
    UserPreferences,
    CommunicationStyle,
    ResponseLength,
)


class TestMemoryProcessor:
    """Test cases for MemoryProcessor."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies."""
        with patch(
            "morgan.memory.memory_processor.get_settings"
        ) as mock_settings, patch(
            "morgan.memory.memory_processor.get_embedding_service"
        ) as mock_embedding, patch(
            "morgan.memory.memory_processor.VectorDBClient"
        ) as mock_vector_db, patch(
            "morgan.memory.memory_processor.get_emotional_intelligence_engine"
        ) as mock_emotional:

            # Setup mock settings
            mock_settings.return_value.morgan_data_dir = "/tmp/morgan"

            # Setup mock embedding service
            mock_embedding.return_value.get_embedding_dimension.return_value = 384
            mock_embedding.return_value.encode.return_value = [0.1] * 384

            # Setup mock vector database
            mock_vector_db.return_value.collection_exists.return_value = True
            mock_vector_db.return_value.upsert_points.return_value = True

            # Setup mock emotional intelligence
            mock_emotional.return_value.analyze_emotion.return_value = EmotionalState(
                primary_emotion=EmotionType.JOY,
                intensity=0.7,
                confidence=0.8,
                emotional_indicators=["happy", "excited"],
            )

            yield {
                "settings": mock_settings.return_value,
                "embedding": mock_embedding.return_value,
                "vector_db": mock_vector_db.return_value,
                "emotional": mock_emotional.return_value,
            }

    @pytest.fixture
    def memory_processor(self, mock_dependencies):
        """Create memory processor instance with mocked dependencies."""
        return MemoryProcessor()

    @pytest.fixture
    def sample_conversation_turn(self):
        """Create sample conversation turn."""
        return ConversationTurn(
            turn_id="turn_123",
            conversation_id="conv_456",
            timestamp=datetime.now(timezone.utc).isoformat(),
            question="I'm really excited about learning Python! It's been my goal for months.",
            answer="That's wonderful! Python is a great language to start with. Let me help you get started.",
            sources=["python-tutorial.md"],
            feedback_rating=5,
            feedback_comment="Very helpful!",
        )

    @pytest.fixture
    def sample_emotional_state(self):
        """Create sample emotional state."""
        return EmotionalState(
            primary_emotion=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9,
            secondary_emotions=[EmotionType.SURPRISE],
            emotional_indicators=["excited", "goal", "wonderful"],
        )

    def test_extract_memories_basic(
        self, memory_processor, sample_conversation_turn, sample_emotional_state
    ):
        """Test basic memory extraction."""
        result = memory_processor.extract_memories(
            sample_conversation_turn,
            feedback_rating=5,
            emotional_context=sample_emotional_state,
        )

        assert isinstance(result, MemoryExtractionResult)
        assert len(result.memories) > 0
        assert len(result.emotional_insights) > 0
        assert isinstance(result.preference_updates, list)
        assert isinstance(result.relationship_indicators, list)

        # Check memory properties
        memory = result.memories[0]
        assert memory.importance_score > 0
        assert memory.conversation_id == sample_conversation_turn.conversation_id
        assert memory.turn_id == sample_conversation_turn.turn_id
        assert memory.emotional_context is not None
        assert memory.user_mood == EmotionType.JOY.value

    def test_score_importance_with_emotional_weight(self, memory_processor):
        """Test importance scoring with emotional weighting."""
        content = "This is really important for my career goals!"
        context = {
            "feedback_rating": 5,
            "is_user_message": True,
            "emotional_context": EmotionalState(
                primary_emotion=EmotionType.JOY, intensity=0.8, confidence=0.9
            ),
        }

        # Test with emotional weight
        score_with_emotion = memory_processor.score_importance(
            content, context, emotional_weight=1.5
        )

        # Test without emotional weight
        score_without_emotion = memory_processor.score_importance(
            content, context, emotional_weight=1.0
        )

        assert score_with_emotion > score_without_emotion
        assert 0.0 <= score_with_emotion <= 1.0
        assert 0.0 <= score_without_emotion <= 1.0

    def test_detect_entities(self, memory_processor):
        """Test entity detection."""
        content = "I work at Google using Python and Docker for my projects."
        entities = memory_processor.detect_entities(content)

        assert len(entities) > 0
        # Should detect technology entities
        tech_entities = [e for e in entities if e.startswith("technology:")]
        assert len(tech_entities) > 0

    def test_extract_personal_preferences(self, memory_processor):
        """Test personal preference extraction."""
        conversation_history = [
            ConversationTurn(
                turn_id="1",
                conversation_id="conv1",
                timestamp=datetime.now(timezone.utc).isoformat(),
                question="I love working with Python and machine learning",
                answer="Great choice!",
                sources=[],
            ),
            ConversationTurn(
                turn_id="2",
                conversation_id="conv1",
                timestamp=datetime.now(timezone.utc).isoformat(),
                question="I want to learn more about Docker containers",
                answer="Docker is very useful!",
                sources=[],
            ),
        ]

        preferences = memory_processor.extract_personal_preferences(
            conversation_history
        )

        assert isinstance(preferences, UserPreferences)
        assert len(preferences.topics_of_interest) > 0
        assert len(preferences.learning_goals) > 0
        assert preferences.communication_style in [
            style.value for style in CommunicationStyle
        ]
        assert preferences.preferred_response_length in [
            length.value for length in ResponseLength
        ]

    def test_store_memory(self, memory_processor, mock_dependencies):
        """Test memory storage."""
        memory = Memory(
            memory_id="mem_123",
            content="Important information about Python",
            importance_score=0.8,
            entities=["technology:Python"],
            concepts=["programming", "learning"],
            conversation_id="conv_456",
            turn_id="turn_123",
            timestamp=datetime.now(timezone.utc),
            feedback_rating=5,
            emotional_context={
                "primary_emotion": "joy",
                "intensity": 0.8,
                "confidence": 0.9,
            },
            user_mood="joy",
            relationship_significance=0.7,
            personal_preferences=["python", "programming"],
        )

        result = memory_processor.store_memory(memory)

        assert result is True
        mock_dependencies["vector_db"].upsert_points.assert_called_once()

    def test_emotional_weight_calculation(self, memory_processor):
        """Test emotional weight calculation."""
        # High intensity joy should have high weight
        high_joy = EmotionalState(
            primary_emotion=EmotionType.JOY, intensity=0.9, confidence=0.8
        )

        weight_high_joy = memory_processor._calculate_emotional_weight(high_joy, True)

        # Low intensity neutral should have lower weight
        low_neutral = EmotionalState(
            primary_emotion=EmotionType.NEUTRAL, intensity=0.2, confidence=0.8
        )

        weight_low_neutral = memory_processor._calculate_emotional_weight(
            low_neutral, False
        )

        assert weight_high_joy > weight_low_neutral
        assert weight_high_joy > 1.0  # Should be amplified

    def test_relationship_significance_calculation(self, memory_processor):
        """Test relationship significance calculation."""
        # Text with relationship indicators
        relationship_text = (
            "I really trust you and feel comfortable sharing this personal information"
        )

        emotional_state = EmotionalState(
            primary_emotion=EmotionType.JOY, intensity=0.7, confidence=0.8
        )

        significance = memory_processor._calculate_relationship_significance(
            relationship_text, emotional_state
        )

        assert 0.0 <= significance <= 1.0
        assert significance > 0.5  # Should be high due to trust indicators

    def test_memory_deduplication(self, memory_processor):
        """Test memory deduplication."""
        memories = [
            Memory(
                memory_id="1",
                content="I love Python programming",
                importance_score=0.8,
                entities=[],
                concepts=[],
                conversation_id="conv1",
                turn_id="turn1",
                timestamp=datetime.now(timezone.utc),
            ),
            Memory(
                memory_id="2",
                content="I love Python programming",  # Duplicate
                importance_score=0.7,
                entities=[],
                concepts=[],
                conversation_id="conv1",
                turn_id="turn1",
                timestamp=datetime.now(timezone.utc),
            ),
            Memory(
                memory_id="3",
                content="Docker is useful for deployment",
                importance_score=0.6,
                entities=[],
                concepts=[],
                conversation_id="conv1",
                turn_id="turn1",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        unique_memories = memory_processor._deduplicate_memories(memories)

        assert len(unique_memories) == 2  # Should remove one duplicate
        contents = [m.content for m in unique_memories]
        assert "I love Python programming" in contents
        assert "Docker is useful for deployment" in contents

    def test_extract_emotional_insights(
        self, memory_processor, sample_conversation_turn, sample_emotional_state
    ):
        """Test emotional insights extraction."""
        insights = memory_processor._extract_emotional_insights(
            sample_conversation_turn, sample_emotional_state
        )

        assert "primary_emotion" in insights
        assert "emotional_intensity" in insights
        assert "emotional_confidence" in insights
        assert "emotional_indicators" in insights
        assert "conversation_emotional_tone" in insights

        assert insights["primary_emotion"] == EmotionType.JOY.value
        assert insights["emotional_intensity"] == sample_emotional_state.intensity

    def test_extract_preference_updates(
        self, memory_processor, sample_conversation_turn, sample_emotional_state
    ):
        """Test preference updates extraction."""
        updates = memory_processor._extract_preference_updates(
            sample_conversation_turn, sample_emotional_state
        )

        assert isinstance(updates, list)
        # Should extract topics from the question about Python
        topic_updates = [u for u in updates if u.startswith("topic:")]
        assert len(topic_updates) > 0

    def test_extract_relationship_indicators(
        self, memory_processor, sample_emotional_state
    ):
        """Test relationship indicators extraction."""
        # Create conversation turn with relationship indicators
        turn_with_trust = ConversationTurn(
            turn_id="turn_123",
            conversation_id="conv_456",
            timestamp=datetime.now(timezone.utc).isoformat(),
            question="I feel comfortable sharing this personal information with you because I trust your advice and guidance. This is something very important to me and I really appreciate having someone I can rely on for help with these complex technical challenges.",
            answer="Thank you for trusting me with this.",
            sources=[],
            feedback_rating=5,
        )

        indicators = memory_processor._extract_relationship_indicators(
            turn_with_trust, sample_emotional_state
        )

        assert isinstance(indicators, list)
        assert "trust_building" in indicators
        assert "emotional_sharing" in indicators  # High intensity emotion
        assert "positive_feedback" in indicators  # Rating of 5
        assert "high_engagement" in indicators  # Long message

    def test_communication_style_determination(self, memory_processor):
        """Test communication style determination."""
        # Formal conversation history
        formal_history = [
            ConversationTurn(
                turn_id="1",
                conversation_id="conv1",
                timestamp=datetime.now(timezone.utc).isoformat(),
                question="Could you please help me understand this concept?",
                answer="Certainly!",
                sources=[],
            ),
            ConversationTurn(
                turn_id="2",
                conversation_id="conv1",
                timestamp=datetime.now(timezone.utc).isoformat(),
                question="Would you mind explaining the technical details?",
                answer="Of course!",
                sources=[],
            ),
        ]

        style = memory_processor._determine_communication_style(formal_history)
        assert style == "formal"

        # Casual conversation history
        casual_history = [
            ConversationTurn(
                turn_id="1",
                conversation_id="conv1",
                timestamp=datetime.now(timezone.utc).isoformat(),
                question="Hey, can you help me with this?",
                answer="Sure!",
                sources=[],
            ),
            ConversationTurn(
                turn_id="2",
                conversation_id="conv1",
                timestamp=datetime.now(timezone.utc).isoformat(),
                question="Thanks! That's really cool.",
                answer="Glad to help!",
                sources=[],
            ),
        ]

        style = memory_processor._determine_communication_style(casual_history)
        assert style == "casual"

    def test_response_length_preference_determination(self, memory_processor):
        """Test response length preference determination."""
        # Short questions indicate brief preference
        brief_history = [
            ConversationTurn(
                turn_id="1",
                conversation_id="conv1",
                timestamp=datetime.now(timezone.utc).isoformat(),
                question="What is Python?",
                answer="A programming language.",
                sources=[],
            )
        ]

        preference = memory_processor._determine_response_length_preference(
            brief_history
        )
        assert preference == "brief"

        # Long questions indicate detailed preference
        detailed_history = [
            ConversationTurn(
                turn_id="1",
                conversation_id="conv1",
                timestamp=datetime.now(timezone.utc).isoformat(),
                question="I'm trying to understand the differences between Python and JavaScript for web development, and I'd like to know about their performance characteristics, ecosystem, and learning curve.",
                answer="Great question! Let me explain...",
                sources=[],
            )
        ]

        preference = memory_processor._determine_response_length_preference(
            detailed_history
        )
        assert preference == "comprehensive"

    def test_singleton_instance(self):
        """Test singleton pattern for memory processor."""
        processor1 = get_memory_processor()
        processor2 = get_memory_processor()

        assert processor1 is processor2  # Should be same instance

    def test_memory_extraction_with_no_emotional_context(
        self, memory_processor, sample_conversation_turn, mock_dependencies
    ):
        """Test memory extraction when no emotional context is provided."""
        result = memory_processor.extract_memories(
            sample_conversation_turn,
            feedback_rating=4,
            # No emotional_context provided
        )

        assert isinstance(result, MemoryExtractionResult)
        # Should still work by analyzing emotion from text
        mock_dependencies["emotional"].analyze_emotion.assert_called_once()

    def test_importance_scoring_edge_cases(self, memory_processor):
        """Test importance scoring with edge cases."""
        # Empty content
        empty_score = memory_processor.score_importance("", {})
        assert empty_score == 0.0

        # Very short content
        short_score = memory_processor.score_importance("Hi", {})
        assert 0.0 <= short_score <= 1.0

        # Content with multiple importance indicators
        important_content = "This is really important and crucial for my goals. I need to remember this key information."
        context = {"feedback_rating": 5}
        important_score = memory_processor.score_importance(important_content, context)
        assert important_score > 0.5

    def test_entity_detection_edge_cases(self, memory_processor):
        """Test entity detection with edge cases."""
        # Empty content
        empty_entities = memory_processor.detect_entities("")
        assert empty_entities == []

        # Content with no entities
        no_entities = memory_processor.detect_entities("this is just text")
        assert len(no_entities) == 0

        # Content with mixed case entities
        mixed_case = memory_processor.detect_entities("I use PYTHON and JavaScript")
        tech_entities = [e for e in mixed_case if "Python" in e or "JavaScript" in e]
        assert len(tech_entities) > 0


class TestMemoryDataStructures:
    """Test memory data structures."""

    def test_memory_creation(self):
        """Test Memory dataclass creation."""
        memory = Memory(
            memory_id="test_123",
            content="Test content",
            importance_score=0.8,
            entities=["person:John"],
            concepts=["learning"],
            conversation_id="conv_456",
            turn_id="turn_789",
            timestamp=datetime.now(timezone.utc),
            feedback_rating=5,
            emotional_context={"emotion": "joy"},
            user_mood="happy",
            relationship_significance=0.7,
        )

        assert memory.memory_id == "test_123"
        assert memory.importance_score == 0.8
        assert memory.personal_preferences == []  # Default value

    def test_memory_extraction_result_creation(self):
        """Test MemoryExtractionResult dataclass creation."""
        result = MemoryExtractionResult(
            memories=[],
            emotional_insights={"emotion": "joy"},
            preference_updates=["python"],
            relationship_indicators=["trust"],
        )

        assert isinstance(result.memories, list)
        assert isinstance(result.emotional_insights, dict)
        assert isinstance(result.preference_updates, list)
        assert isinstance(result.relationship_indicators, list)


if __name__ == "__main__":
    pytest.main([__file__])
