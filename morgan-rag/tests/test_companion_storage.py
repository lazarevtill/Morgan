"""
Tests for companion storage and database schema.

Tests the companion data models, storage operations, and vector database
schema for emotional intelligence and relationship management.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from morgan.companion.storage import CompanionStorage
from morgan.companion.schema import (
    CompanionDatabaseSchema,
    validate_companion_payload,
    validate_emotion_payload
)
from morgan.emotional.models import (
    CompanionProfile, EmotionalState, UserPreferences, RelationshipMilestone,
    EmotionType, CommunicationStyle, ResponseLength, MilestoneType
)


class TestCompanionStorage:
    """Test companion storage operations."""
    
    @pytest.fixture
    def mock_vector_client(self):
        """Mock vector database client."""
        mock_client = Mock()
        mock_client.collection_exists.return_value = True
        mock_client.create_collection.return_value = True
        mock_client.upsert_points.return_value = True
        mock_client.search_with_filter.return_value = []
        mock_client.search.return_value = []
        mock_client.delete_points.return_value = True
        mock_client.get_collection_info.return_value = {
            "name": "test_collection",
            "points_count": 0
        }
        return mock_client
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        mock_service = Mock()
        mock_service.embed_text.return_value = [0.1] * 1536  # Mock embedding
        return mock_service
    
    @pytest.fixture
    def companion_storage(self, mock_vector_client, mock_embedding_service):
        """Create companion storage with mocked dependencies."""
        return CompanionStorage(
            vector_client=mock_vector_client,
            embedding_service=mock_embedding_service
        )
    
    @pytest.fixture
    def sample_companion_profile(self):
        """Create sample companion profile for testing."""
        preferences = UserPreferences(
            topics_of_interest=["AI", "programming"],
            communication_style=CommunicationStyle.FRIENDLY,
            preferred_response_length=ResponseLength.DETAILED,
            learning_goals=["Learn Python", "Understand ML"]
        )
        
        return CompanionProfile(
            user_id="test_user_123",
            preferred_name="Alice",
            relationship_duration=timedelta(days=30),
            interaction_count=50,
            communication_preferences=preferences,
            trust_level=0.8,
            engagement_score=0.9
        )
    
    @pytest.fixture
    def sample_emotional_state(self):
        """Create sample emotional state for testing."""
        return EmotionalState(
            primary_emotion=EmotionType.JOY,
            intensity=0.8,
            confidence=0.9,
            secondary_emotions=[EmotionType.SURPRISE],
            emotional_indicators=["excited", "happy", "enthusiastic"]
        )
    
    @pytest.fixture
    def sample_milestone(self):
        """Create sample relationship milestone for testing."""
        return RelationshipMilestone(
            milestone_id="milestone_123",
            milestone_type=MilestoneType.BREAKTHROUGH_MOMENT,
            description="User successfully completed first AI project",
            timestamp=datetime.utcnow(),
            emotional_significance=0.9,
            related_memories=["memory_1", "memory_2"]
        )
    
    def test_store_companion_profile(self, companion_storage, sample_companion_profile):
        """Test storing a companion profile."""
        result = companion_storage.store_companion_profile(sample_companion_profile)
        
        assert result is True
        companion_storage.vector_client.upsert_points.assert_called_once()
        
        # Verify the call arguments
        call_args = companion_storage.vector_client.upsert_points.call_args
        assert call_args[1]["collection_name"] == CompanionStorage.COMPANIONS_COLLECTION
        
        points = call_args[1]["points"]
        assert len(points) == 1
        assert points[0]["id"] == "test_user_123"
        assert "payload" in points[0]
        assert points[0]["payload"]["preferred_name"] == "Alice"
    
    def test_store_emotional_state(self, companion_storage, sample_emotional_state):
        """Test storing an emotional state."""
        result = companion_storage.store_emotional_state("test_user", sample_emotional_state)
        
        assert result is True
        companion_storage.vector_client.upsert_points.assert_called_once()
        
        # Verify the call arguments
        call_args = companion_storage.vector_client.upsert_points.call_args
        assert call_args[1]["collection_name"] == CompanionStorage.EMOTIONS_COLLECTION
        
        points = call_args[1]["points"]
        assert len(points) == 1
        assert points[0]["payload"]["primary_emotion"] == "joy"
        assert points[0]["payload"]["intensity"] == 0.8
    
    def test_store_relationship_milestone(self, companion_storage, sample_milestone):
        """Test storing a relationship milestone."""
        result = companion_storage.store_relationship_milestone("test_user", sample_milestone)
        
        assert result is True
        companion_storage.vector_client.upsert_points.assert_called_once()
        
        # Verify the call arguments
        call_args = companion_storage.vector_client.upsert_points.call_args
        assert call_args[1]["collection_name"] == CompanionStorage.MILESTONES_COLLECTION
        
        points = call_args[1]["points"]
        assert len(points) == 1
        assert points[0]["payload"]["milestone_type"] == "breakthrough_moment"
        assert points[0]["payload"]["emotional_significance"] == 0.9
    
    def test_get_companion_profile_not_found(self, companion_storage):
        """Test getting a companion profile that doesn't exist."""
        companion_storage.vector_client.search_with_filter.return_value = []
        
        result = companion_storage.get_companion_profile("nonexistent_user")
        
        assert result is None
    
    def test_get_storage_stats(self, companion_storage):
        """Test getting storage statistics."""
        stats = companion_storage.get_storage_stats()
        
        assert isinstance(stats, dict)
        assert CompanionStorage.COMPANIONS_COLLECTION in stats
        assert CompanionStorage.EMOTIONS_COLLECTION in stats
        assert CompanionStorage.MILESTONES_COLLECTION in stats


class TestCompanionDatabaseSchema:
    """Test companion database schema definitions."""
    
    def test_get_companions_schema(self):
        """Test companions collection schema."""
        schema = CompanionDatabaseSchema.get_companions_schema()
        
        assert schema.name == "morgan_companions"
        assert schema.vector_size == 1536
        assert schema.distance_metric == "cosine"
        assert "user_id" in schema.payload_schema
        assert "preferred_name" in schema.payload_schema
        assert "trust_level" in schema.payload_schema
    
    def test_get_emotions_schema(self):
        """Test emotions collection schema."""
        schema = CompanionDatabaseSchema.get_emotions_schema()
        
        assert schema.name == "morgan_emotions"
        assert schema.vector_size == 768
        assert "primary_emotion" in schema.payload_schema
        assert "intensity" in schema.payload_schema
        assert "confidence" in schema.payload_schema
    
    def test_get_milestones_schema(self):
        """Test milestones collection schema."""
        schema = CompanionDatabaseSchema.get_milestones_schema()
        
        assert schema.name == "morgan_milestones"
        assert schema.vector_size == 512
        assert "milestone_type" in schema.payload_schema
        assert "emotional_significance" in schema.payload_schema
    
    def test_get_all_schemas(self):
        """Test getting all schemas."""
        schemas = CompanionDatabaseSchema.get_all_schemas()
        
        assert len(schemas) == 4
        schema_names = [s.name for s in schemas]
        assert "morgan_companions" in schema_names
        assert "morgan_memories" in schema_names
        assert "morgan_emotions" in schema_names
        assert "morgan_milestones" in schema_names
    
    def test_get_schema_summary(self):
        """Test schema summary generation."""
        summary = CompanionDatabaseSchema.get_schema_summary()
        
        assert "total_collections" in summary
        assert "collections" in summary
        assert summary["total_collections"] == 4
        assert "schema_version" in summary


class TestSchemaValidation:
    """Test schema validation functions."""
    
    def test_validate_companion_payload_valid(self):
        """Test validation of valid companion payload."""
        payload = {
            "user_id": "test_user",
            "preferred_name": "Alice",
            "communication_style": "friendly",
            "trust_level": 0.8,
            "engagement_score": 0.9
        }
        
        errors = validate_companion_payload(payload)
        assert len(errors) == 0
    
    def test_validate_companion_payload_missing_fields(self):
        """Test validation with missing required fields."""
        payload = {
            "preferred_name": "Alice"
            # Missing user_id and communication_style
        }
        
        errors = validate_companion_payload(payload)
        assert len(errors) >= 2
        assert any("user_id" in error for error in errors)
        assert any("communication_style" in error for error in errors)
    
    def test_validate_companion_payload_invalid_ranges(self):
        """Test validation with invalid value ranges."""
        payload = {
            "user_id": "test_user",
            "preferred_name": "Alice",
            "communication_style": "friendly",
            "trust_level": 1.5,  # Invalid: > 1.0
            "engagement_score": -0.1  # Invalid: < 0.0
        }
        
        errors = validate_companion_payload(payload)
        assert len(errors) >= 2
        assert any("trust_level" in error for error in errors)
        assert any("engagement_score" in error for error in errors)
    
    def test_validate_emotion_payload_valid(self):
        """Test validation of valid emotion payload."""
        payload = {
            "user_id": "test_user",
            "primary_emotion": "joy",
            "intensity": 0.8,
            "confidence": 0.9
        }
        
        errors = validate_emotion_payload(payload)
        assert len(errors) == 0
    
    def test_validate_emotion_payload_invalid_emotion(self):
        """Test validation with invalid emotion type."""
        payload = {
            "user_id": "test_user",
            "primary_emotion": "invalid_emotion",
            "intensity": 0.8,
            "confidence": 0.9
        }
        
        errors = validate_emotion_payload(payload)
        assert len(errors) >= 1
        assert any("Invalid primary_emotion" in error for error in errors)
    
    def test_validate_emotion_payload_invalid_ranges(self):
        """Test validation with invalid intensity/confidence ranges."""
        payload = {
            "user_id": "test_user",
            "primary_emotion": "joy",
            "intensity": 1.5,  # Invalid: > 1.0
            "confidence": -0.1  # Invalid: < 0.0
        }
        
        errors = validate_emotion_payload(payload)
        assert len(errors) >= 2
        assert any("intensity" in error for error in errors)
        assert any("confidence" in error for error in errors)