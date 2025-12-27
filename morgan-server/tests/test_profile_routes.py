"""
Unit tests for Profile API routes.

Tests cover:
- Profile retrieval
- Profile updates
- Timeline retrieval
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from morgan_server.api.routes.profile import router
from morgan_server.api.routes.chat import set_assistant
from morgan.intelligence.core.models import (
    CompanionProfile,
    UserPreferences,
    CommunicationStyle,
    ResponseLength,
    RelationshipMilestone,
    MilestoneType,
)


@pytest.fixture
def mock_assistant():
    """Create a mock MorganAssistant for testing."""
    assistant = MagicMock()

    # Mock get_user_profile to return a CompanionProfile
    def mock_get_profile(user_id: str):
        profile = CompanionProfile(
            user_id=user_id,
            relationship_duration=timedelta(days=0),
            interaction_count=0,
            preferred_name=None,
            communication_preferences=UserPreferences(
                communication_style=CommunicationStyle.FRIENDLY,
                preferred_response_length=ResponseLength.DETAILED,
                topics_of_interest=[],
            ),
        )
        profile.profile_created = datetime.now(timezone.utc)
        profile.relationship_milestones = []
        profile.trust_level = 0.0
        profile.engagement_score = 0.0
        return profile

    assistant.get_user_profile = MagicMock(side_effect=mock_get_profile)

    # Mock update_user_profile to return updated profile
    def mock_update_profile(user_id: str, **updates):
        profile = mock_get_profile(user_id)
        if "preferred_name" in updates:
            profile.preferred_name = updates["preferred_name"]
        if "communication_style" in updates:
            profile.communication_preferences.communication_style = updates[
                "communication_style"
            ]
        if "response_length" in updates:
            profile.communication_preferences.preferred_response_length = updates[
                "response_length"
            ]
        if "topics_of_interest" in updates:
            profile.communication_preferences.topics_of_interest = updates[
                "topics_of_interest"
            ]
        return profile

    assistant.update_user_profile = MagicMock(side_effect=mock_update_profile)

    return assistant


@pytest.fixture
def app(mock_assistant):
    """Create a FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router)
    set_assistant(mock_assistant)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestGetUserProfile:
    """Test GET /api/profile/{user_id} endpoint."""

    def test_get_profile_new_user(self, client, mock_assistant):
        """Test getting profile for new user creates profile."""
        response = client.get("/api/profile/user123")

        assert response.status_code == 200
        data = response.json()

        assert data["user_id"] == "user123"
        assert data["preferred_name"] is None
        assert data["relationship_age_days"] == 0
        assert data["trust_level"] == 0.0
        assert data["engagement_score"] == 0.0
        assert data["communication_style"] == "friendly"
        assert data["response_length"] == "detailed"
        assert data["topics_of_interest"] == []

    def test_get_profile_returns_all_fields(self, client, mock_assistant):
        """Test that profile response includes all required fields."""
        response = client.get("/api/profile/user123")

        assert response.status_code == 200
        data = response.json()

        # Check all required fields are present
        required_fields = [
            "user_id",
            "preferred_name",
            "relationship_age_days",
            "interaction_count",
            "trust_level",
            "engagement_score",
            "communication_style",
            "response_length",
            "topics_of_interest",
        ]

        for field in required_fields:
            assert field in data


class TestUpdateUserProfile:
    """Test PUT /api/profile/{user_id} endpoint."""

    def test_update_preferred_name(self, client, mock_assistant):
        """Test updating preferred name."""
        response = client.put("/api/profile/user123", json={"preferred_name": "Alicia"})

        assert response.status_code == 200
        data = response.json()

        assert data["preferred_name"] == "Alicia"

    def test_update_communication_style(self, client, mock_assistant):
        """Test updating communication style."""
        response = client.put(
            "/api/profile/user123", json={"communication_style": "professional"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["communication_style"] == "professional"

    def test_update_response_length(self, client, mock_assistant):
        """Test updating response length."""
        response = client.put("/api/profile/user123", json={"response_length": "brief"})

        assert response.status_code == 200
        data = response.json()

        assert data["response_length"] == "brief"

    def test_update_topics_of_interest(self, client, mock_assistant):
        """Test updating topics of interest."""
        response = client.put(
            "/api/profile/user123",
            json={"topics_of_interest": ["AI", "programming", "music"]},
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["topics_of_interest"]) == 3
        assert "AI" in data["topics_of_interest"]
        assert "programming" in data["topics_of_interest"]
        assert "music" in data["topics_of_interest"]

    def test_update_multiple_fields(self, client, mock_assistant):
        """Test updating multiple fields at once."""
        response = client.put(
            "/api/profile/user123",
            json={
                "preferred_name": "Alice",
                "communication_style": "casual",
                "response_length": "detailed",
                "topics_of_interest": ["AI", "music"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["preferred_name"] == "Alice"
        assert data["communication_style"] == "casual"
        assert data["response_length"] == "detailed"
        assert "AI" in data["topics_of_interest"]
        assert "music" in data["topics_of_interest"]

    def test_update_invalid_communication_style(self, client, mock_assistant):
        """Test updating with invalid communication style returns 422."""
        response = client.put(
            "/api/profile/user123", json={"communication_style": "invalid_style"}
        )

        assert response.status_code == 422  # Pydantic validation error

    def test_update_invalid_response_length(self, client, mock_assistant):
        """Test updating with invalid response length returns 422."""
        response = client.put(
            "/api/profile/user123", json={"response_length": "invalid_length"}
        )

        assert response.status_code == 422  # Pydantic validation error


class TestGetUserTimeline:
    """Test GET /api/timeline/{user_id} endpoint."""

    def test_get_timeline_new_user(self, client, mock_assistant):
        """Test getting timeline for new user."""
        response = client.get("/api/timeline/user123")

        assert response.status_code == 200
        data = response.json()

        assert data["user_id"] == "user123"
        assert data["total_events"] >= 1
        assert len(data["events"]) >= 1

        # Should have at least profile creation event
        events = data["events"]
        assert any(e["event_type"] == "profile_created" for e in events)

    def test_get_timeline_with_milestones(self, client, mock_assistant):
        """Test timeline includes relationship milestones."""

        # Mock profile with milestones
        def mock_get_profile_with_milestones(user_id: str):
            profile = CompanionProfile(
                user_id=user_id,
                relationship_duration=timedelta(days=10),
                interaction_count=0,
                preferred_name=None,
                communication_preferences=UserPreferences(),
            )
            profile.profile_created = datetime.now(timezone.utc) - timedelta(days=10)
            profile.relationship_milestones = [
                RelationshipMilestone(
                    milestone_id="milestone_1",
                    milestone_type=MilestoneType.FIRST_CONVERSATION,
                    timestamp=datetime.now(timezone.utc) - timedelta(days=10),
                    description="First conversation",
                    emotional_significance=0.8,
                ),
                RelationshipMilestone(
                    milestone_id="milestone_2",
                    milestone_type=MilestoneType.EMOTIONAL_BREAKTHROUGH,
                    timestamp=datetime.now(timezone.utc) - timedelta(days=5),
                    description="Trust established",
                    emotional_significance=0.9,
                ),
            ]
            return profile

        mock_assistant.get_user_profile = MagicMock(
            side_effect=mock_get_profile_with_milestones
        )

        response = client.get("/api/timeline/user123")

        assert response.status_code == 200
        data = response.json()

        events = data["events"]

        # Should have milestone events
        assert len(events) >= 3  # profile_created + 2 milestones

    def test_get_timeline_event_structure(self, client, mock_assistant):
        """Test that timeline events have correct structure."""
        response = client.get("/api/timeline/user123")

        assert response.status_code == 200
        data = response.json()

        events = data["events"]
        assert len(events) > 0

        # Check first event structure
        event = events[0]
        assert "event_type" in event
        assert "timestamp" in event
        assert "description" in event
        assert "metadata" in event

    def test_get_timeline_total_events_matches_list(self, client, mock_assistant):
        """Test that total_events matches the length of events list."""
        response = client.get("/api/timeline/user123")

        assert response.status_code == 200
        data = response.json()

        assert data["total_events"] == len(data["events"])
