"""
Unit tests for Profile API routes.

Tests cover:
- Profile retrieval
- Profile updates
- Timeline retrieval
"""

import pytest
import tempfile
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.testclient import TestClient

from morgan_server.api.routes.profile import router, set_profile_manager
from morgan_server.personalization.profile import (
    ProfileManager,
    CommunicationStyle,
    ResponseLength,
)


@pytest.fixture
def profile_manager():
    """Create a profile manager for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ProfileManager(storage_dir=tmpdir)
        yield manager


@pytest.fixture
def app(profile_manager):
    """Create a FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router)
    set_profile_manager(profile_manager)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestGetUserProfile:
    """Test GET /api/profile/{user_id} endpoint."""
    
    def test_get_profile_new_user(self, client, profile_manager):
        """Test getting profile for new user creates profile."""
        response = client.get("/api/profile/user123")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["user_id"] == "user123"
        assert data["preferred_name"] is None
        assert data["relationship_age_days"] == 0
        assert data["interaction_count"] == 0
        assert data["trust_level"] == 0.0
        assert data["engagement_score"] == 0.0
        assert data["communication_style"] == "friendly"
        assert data["response_length"] == "moderate"
        assert data["topics_of_interest"] == []
    
    def test_get_profile_existing_user(self, client, profile_manager):
        """Test getting profile for existing user."""
        # Create profile
        profile_manager.create_profile(
            "user123",
            preferred_name="Alice",
            communication_style=CommunicationStyle.PROFESSIONAL,
            topics_of_interest=["AI", "programming"]
        )
        profile_manager.update_metrics(
            "user123",
            trust_level=0.75,
            engagement_score=0.85,
            interaction_count=50
        )
        
        response = client.get("/api/profile/user123")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["user_id"] == "user123"
        assert data["preferred_name"] == "Alice"
        assert data["interaction_count"] == 50
        assert data["trust_level"] == 0.75
        assert data["engagement_score"] == 0.85
        assert data["communication_style"] == "professional"
        assert "AI" in data["topics_of_interest"]
        assert "programming" in data["topics_of_interest"]
    
    def test_get_profile_empty_user_id(self, client):
        """Test getting profile with empty user_id returns 400."""
        response = client.get("/api/profile/ ")
        
        # FastAPI will return 404 for empty path parameter
        assert response.status_code in [400, 404]
    
    def test_get_profile_calculates_relationship_age(self, client, profile_manager):
        """Test that profile includes calculated relationship age."""
        # Create profile
        profile = profile_manager.create_profile("user123")
        
        # Manually set created_at to 10 days ago
        profile.created_at = datetime.now() - timedelta(days=10)
        profile_manager.profiles["user123"] = profile
        
        response = client.get("/api/profile/user123")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should be approximately 10 days
        assert 9 <= data["relationship_age_days"] <= 11
    
    def test_get_profile_returns_all_fields(self, client, profile_manager):
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
    
    def test_update_preferred_name(self, client, profile_manager):
        """Test updating preferred name."""
        # Create profile
        profile_manager.create_profile("user123", preferred_name="Alice")
        
        # Update
        response = client.put(
            "/api/profile/user123",
            json={"preferred_name": "Alicia"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["preferred_name"] == "Alicia"
    
    def test_update_communication_style(self, client, profile_manager):
        """Test updating communication style."""
        # Create profile
        profile_manager.create_profile("user123")
        
        # Update
        response = client.put(
            "/api/profile/user123",
            json={"communication_style": "professional"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["communication_style"] == "professional"
    
    def test_update_response_length(self, client, profile_manager):
        """Test updating response length."""
        # Create profile
        profile_manager.create_profile("user123")
        
        # Update
        response = client.put(
            "/api/profile/user123",
            json={"response_length": "brief"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["response_length"] == "brief"
    
    def test_update_topics_of_interest(self, client, profile_manager):
        """Test updating topics of interest."""
        # Create profile
        profile_manager.create_profile("user123", topics_of_interest=["AI"])
        
        # Update
        response = client.put(
            "/api/profile/user123",
            json={"topics_of_interest": ["AI", "programming", "music"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["topics_of_interest"]) == 3
        assert "AI" in data["topics_of_interest"]
        assert "programming" in data["topics_of_interest"]
        assert "music" in data["topics_of_interest"]
    
    def test_update_multiple_fields(self, client, profile_manager):
        """Test updating multiple fields at once."""
        # Create profile
        profile_manager.create_profile("user123")
        
        # Update
        response = client.put(
            "/api/profile/user123",
            json={
                "preferred_name": "Alice",
                "communication_style": "casual",
                "response_length": "detailed",
                "topics_of_interest": ["AI", "music"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["preferred_name"] == "Alice"
        assert data["communication_style"] == "casual"
        assert data["response_length"] == "detailed"
        assert "AI" in data["topics_of_interest"]
        assert "music" in data["topics_of_interest"]
    
    def test_update_invalid_communication_style(self, client, profile_manager):
        """Test updating with invalid communication style returns 400."""
        # Create profile
        profile_manager.create_profile("user123")
        
        # Update with invalid style
        response = client.put(
            "/api/profile/user123",
            json={"communication_style": "invalid_style"}
        )
        
        assert response.status_code == 422  # Pydantic validation error
    
    def test_update_invalid_response_length(self, client, profile_manager):
        """Test updating with invalid response length returns 400."""
        # Create profile
        profile_manager.create_profile("user123")
        
        # Update with invalid length
        response = client.put(
            "/api/profile/user123",
            json={"response_length": "invalid_length"}
        )
        
        assert response.status_code == 422  # Pydantic validation error
    
    def test_update_empty_user_id(self, client):
        """Test updating with empty user_id returns 400."""
        response = client.put(
            "/api/profile/ ",
            json={"preferred_name": "Alice"}
        )
        
        # FastAPI will return 404 for empty path parameter
        assert response.status_code in [400, 404]
    
    def test_update_partial_preserves_other_fields(self, client, profile_manager):
        """Test that partial update preserves other fields."""
        # Create profile with multiple fields
        profile_manager.create_profile(
            "user123",
            preferred_name="Alice",
            communication_style=CommunicationStyle.PROFESSIONAL,
            topics_of_interest=["AI"]
        )
        
        # Update only preferred name
        response = client.put(
            "/api/profile/user123",
            json={"preferred_name": "Alicia"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Other fields should be preserved
        assert data["preferred_name"] == "Alicia"
        assert data["communication_style"] == "professional"
        assert "AI" in data["topics_of_interest"]
    
    def test_update_new_user_creates_profile(self, client, profile_manager):
        """Test updating non-existent user creates profile."""
        response = client.put(
            "/api/profile/newuser",
            json={"preferred_name": "Bob"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["user_id"] == "newuser"
        assert data["preferred_name"] == "Bob"
    
    def test_update_empty_body(self, client, profile_manager):
        """Test updating with empty body."""
        # Create profile
        profile_manager.create_profile("user123", preferred_name="Alice")
        
        # Update with empty body
        response = client.put(
            "/api/profile/user123",
            json={}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Profile should remain unchanged
        assert data["preferred_name"] == "Alice"
    
    def test_update_returns_all_fields(self, client, profile_manager):
        """Test that update response includes all required fields."""
        # Create profile
        profile_manager.create_profile("user123")
        
        # Update
        response = client.put(
            "/api/profile/user123",
            json={"preferred_name": "Alice"}
        )
        
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


class TestGetUserTimeline:
    """Test GET /api/timeline/{user_id} endpoint."""
    
    def test_get_timeline_new_user(self, client, profile_manager):
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
    
    def test_get_timeline_with_trust_milestones(self, client, profile_manager):
        """Test timeline includes trust milestones."""
        # Create profile with trust level
        profile_manager.create_profile("user123")
        profile_manager.update_metrics("user123", trust_level=0.75)
        
        response = client.get("/api/timeline/user123")
        
        assert response.status_code == 200
        data = response.json()
        
        events = data["events"]
        
        # Should have trust milestone events
        trust_events = [e for e in events if e["event_type"] == "trust_milestone"]
        assert len(trust_events) >= 1
    
    def test_get_timeline_with_interaction_milestones(self, client, profile_manager):
        """Test timeline includes interaction milestones."""
        # Create profile with interactions
        profile_manager.create_profile("user123")
        profile_manager.update_metrics("user123", interaction_count=50)
        
        response = client.get("/api/timeline/user123")
        
        assert response.status_code == 200
        data = response.json()
        
        events = data["events"]
        
        # Should have interaction milestone events
        interaction_events = [e for e in events if e["event_type"] == "interaction_milestone"]
        assert len(interaction_events) >= 1
    
    def test_get_timeline_events_sorted_by_timestamp(self, client, profile_manager):
        """Test that timeline events are sorted by timestamp."""
        # Create profile with multiple milestones
        profile_manager.create_profile("user123")
        profile_manager.update_metrics(
            "user123",
            trust_level=0.75,
            interaction_count=100
        )
        
        response = client.get("/api/timeline/user123")
        
        assert response.status_code == 200
        data = response.json()
        
        events = data["events"]
        
        # Check events are sorted
        timestamps = [datetime.fromisoformat(e["timestamp"]) for e in events]
        assert timestamps == sorted(timestamps)
    
    def test_get_timeline_empty_user_id(self, client):
        """Test getting timeline with empty user_id returns 400."""
        response = client.get("/api/timeline/ ")
        
        # FastAPI will return 404 for empty path parameter
        assert response.status_code in [400, 404]
    
    def test_get_timeline_event_structure(self, client, profile_manager):
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
    
    def test_get_timeline_total_events_matches_list(self, client, profile_manager):
        """Test that total_events matches the length of events list."""
        response = client.get("/api/timeline/user123")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_events"] == len(data["events"])
    
    def test_get_timeline_profile_created_event_details(self, client, profile_manager):
        """Test profile_created event has correct details."""
        response = client.get("/api/timeline/user123")
        
        assert response.status_code == 200
        data = response.json()
        
        events = data["events"]
        created_events = [e for e in events if e["event_type"] == "profile_created"]
        
        assert len(created_events) == 1
        event = created_events[0]
        
        assert event["description"] == "First interaction with Morgan"
        assert event["metadata"]["user_id"] == "user123"
    
    def test_get_timeline_trust_milestone_25(self, client, profile_manager):
        """Test 25% trust milestone appears in timeline."""
        profile_manager.create_profile("user123")
        profile_manager.update_metrics("user123", trust_level=0.25)
        
        response = client.get("/api/timeline/user123")
        
        assert response.status_code == 200
        data = response.json()
        
        events = data["events"]
        trust_25_events = [
            e for e in events
            if e["event_type"] == "trust_milestone"
            and e["metadata"].get("trust_level") == 0.25
        ]
        
        assert len(trust_25_events) == 1
    
    def test_get_timeline_trust_milestone_50(self, client, profile_manager):
        """Test 50% trust milestone appears in timeline."""
        profile_manager.create_profile("user123")
        profile_manager.update_metrics("user123", trust_level=0.5)
        
        response = client.get("/api/timeline/user123")
        
        assert response.status_code == 200
        data = response.json()
        
        events = data["events"]
        trust_50_events = [
            e for e in events
            if e["event_type"] == "trust_milestone"
            and e["metadata"].get("trust_level") == 0.5
        ]
        
        assert len(trust_50_events) == 1
    
    def test_get_timeline_trust_milestone_75(self, client, profile_manager):
        """Test 75% trust milestone appears in timeline."""
        profile_manager.create_profile("user123")
        profile_manager.update_metrics("user123", trust_level=0.75)
        
        response = client.get("/api/timeline/user123")
        
        assert response.status_code == 200
        data = response.json()
        
        events = data["events"]
        trust_75_events = [
            e for e in events
            if e["event_type"] == "trust_milestone"
            and e["metadata"].get("trust_level") == 0.75
        ]
        
        assert len(trust_75_events) == 1
    
    def test_get_timeline_interaction_milestone_10(self, client, profile_manager):
        """Test 10 interactions milestone appears in timeline."""
        profile_manager.create_profile("user123")
        profile_manager.update_metrics("user123", interaction_count=10)
        
        response = client.get("/api/timeline/user123")
        
        assert response.status_code == 200
        data = response.json()
        
        events = data["events"]
        interaction_10_events = [
            e for e in events
            if e["event_type"] == "interaction_milestone"
            and e["metadata"].get("interaction_count") == 10
        ]
        
        assert len(interaction_10_events) == 1
    
    def test_get_timeline_interaction_milestone_50(self, client, profile_manager):
        """Test 50 interactions milestone appears in timeline."""
        profile_manager.create_profile("user123")
        profile_manager.update_metrics("user123", interaction_count=50)
        
        response = client.get("/api/timeline/user123")
        
        assert response.status_code == 200
        data = response.json()
        
        events = data["events"]
        interaction_50_events = [
            e for e in events
            if e["event_type"] == "interaction_milestone"
            and e["metadata"].get("interaction_count") == 50
        ]
        
        assert len(interaction_50_events) == 1
    
    def test_get_timeline_interaction_milestone_100(self, client, profile_manager):
        """Test 100 interactions milestone appears in timeline."""
        profile_manager.create_profile("user123")
        profile_manager.update_metrics("user123", interaction_count=100)
        
        response = client.get("/api/timeline/user123")
        
        assert response.status_code == 200
        data = response.json()
        
        events = data["events"]
        interaction_100_events = [
            e for e in events
            if e["event_type"] == "interaction_milestone"
            and e["metadata"].get("interaction_count") == 100
        ]
        
        assert len(interaction_100_events) == 1


class TestProfileRoutesErrorHandling:
    """Test error handling in profile routes."""
    
    def test_get_profile_without_manager(self, client):
        """Test getting profile when manager is not initialized."""
        # Reset manager
        set_profile_manager(None)
        
        response = client.get("/api/profile/user123")
        
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()
    
    def test_update_profile_without_manager(self, client):
        """Test updating profile when manager is not initialized."""
        # Reset manager
        set_profile_manager(None)
        
        response = client.put(
            "/api/profile/user123",
            json={"preferred_name": "Alice"}
        )
        
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()
    
    def test_get_timeline_without_manager(self, client):
        """Test getting timeline when manager is not initialized."""
        # Reset manager
        set_profile_manager(None)
        
        response = client.get("/api/timeline/user123")
        
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()
