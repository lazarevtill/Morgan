"""
Tests for the new feature module API endpoints.

Tests the /api/suggestions, /api/wellness, /api/habits, and /api/quality endpoints.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from morgan_server.api.routes.features import router, set_assistant, get_assistant
from morgan_server.api.models import (
    SuggestionsResponse,
    WellnessResponse,
    HabitsResponse,
    ConversationQualityResponse,
)
from morgan_server.assistant import MorganAssistant


@pytest.fixture
def mock_assistant():
    """Create a mock MorganAssistant."""
    assistant = MagicMock(spec=MorganAssistant)
    return assistant


@pytest.fixture
def app(mock_assistant):
    """Create a FastAPI app with feature routes."""
    app = FastAPI()
    app.include_router(router)
    set_assistant(mock_assistant)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


# ============================================================================
# Suggestions endpoint
# ============================================================================


class TestSuggestionsEndpoint:
    """Tests for GET /api/suggestions/{user_id}."""

    def test_get_suggestions_success(self, client, mock_assistant):
        """Should return suggestions for a user."""
        mock_assistant.get_proactive_suggestions = AsyncMock(
            return_value=[
                {"text": "Check your calendar", "type": "reminder"},
                {"text": "Review pending tasks", "type": "task"},
            ]
        )
        response = client.get("/api/suggestions/user_123")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user_123"
        assert len(data["suggestions"]) == 2

    def test_get_suggestions_empty(self, client, mock_assistant):
        """Should return empty list when no suggestions."""
        mock_assistant.get_proactive_suggestions = AsyncMock(return_value=[])
        response = client.get("/api/suggestions/user_123")
        assert response.status_code == 200
        data = response.json()
        assert data["suggestions"] == []

    def test_get_suggestions_non_list_return(self, client, mock_assistant):
        """Should handle non-list return gracefully."""
        mock_assistant.get_proactive_suggestions = AsyncMock(return_value=None)
        response = client.get("/api/suggestions/user_123")
        assert response.status_code == 200
        data = response.json()
        assert data["suggestions"] == []

    def test_get_suggestions_error(self, client, mock_assistant):
        """Should return 500 on internal error."""
        mock_assistant.get_proactive_suggestions = AsyncMock(
            side_effect=RuntimeError("Service unavailable")
        )
        response = client.get("/api/suggestions/user_123")
        assert response.status_code == 500


# ============================================================================
# Wellness endpoint
# ============================================================================


class TestWellnessEndpoint:
    """Tests for GET /api/wellness/{user_id}."""

    def test_get_wellness_success(self, client, mock_assistant):
        """Should return wellness insights."""
        mock_assistant.get_wellness_insights = AsyncMock(
            return_value={
                "insights": [
                    {
                        "category": "sleep",
                        "score": 0.8,
                        "trend": "improving",
                        "recommendations": ["Keep consistent schedule"],
                    }
                ],
                "overall_score": 0.75,
            }
        )
        response = client.get("/api/wellness/user_123")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user_123"
        assert len(data["insights"]) == 1
        assert data["insights"][0]["category"] == "sleep"
        assert data["overall_score"] == 0.75

    def test_get_wellness_empty(self, client, mock_assistant):
        """Should handle empty wellness data."""
        mock_assistant.get_wellness_insights = AsyncMock(return_value={})
        response = client.get("/api/wellness/user_123")
        assert response.status_code == 200
        data = response.json()
        assert data["insights"] == []

    def test_get_wellness_error(self, client, mock_assistant):
        """Should return 500 on internal error."""
        mock_assistant.get_wellness_insights = AsyncMock(
            side_effect=RuntimeError("Service unavailable")
        )
        response = client.get("/api/wellness/user_123")
        assert response.status_code == 500


# ============================================================================
# Habits endpoint
# ============================================================================


class TestHabitsEndpoint:
    """Tests for GET /api/habits/{user_id}."""

    def test_get_habits_success(self, client, mock_assistant):
        """Should return habit patterns."""
        mock_assistant.get_habit_patterns = AsyncMock(
            return_value={
                "habits": [
                    {
                        "name": "Morning check-in",
                        "type": "routine",
                        "frequency": "daily",
                        "confidence": "high",
                        "consistency": 0.9,
                    }
                ],
                "total_interactions": 150,
            }
        )
        response = client.get("/api/habits/user_123")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user_123"
        assert len(data["habits"]) == 1
        assert data["habits"][0]["name"] == "Morning check-in"
        assert data["total_interactions"] == 150

    def test_get_habits_empty(self, client, mock_assistant):
        """Should handle empty habits data."""
        mock_assistant.get_habit_patterns = AsyncMock(return_value={})
        response = client.get("/api/habits/user_123")
        assert response.status_code == 200
        data = response.json()
        assert data["habits"] == []
        assert data["total_interactions"] == 0

    def test_get_habits_error(self, client, mock_assistant):
        """Should return 500 on internal error."""
        mock_assistant.get_habit_patterns = AsyncMock(
            side_effect=RuntimeError("Service unavailable")
        )
        response = client.get("/api/habits/user_123")
        assert response.status_code == 500


# ============================================================================
# Quality endpoint
# ============================================================================


class TestQualityEndpoint:
    """Tests for GET /api/quality/{conversation_id}."""

    def test_get_quality_success(self, client, mock_assistant):
        """Should return conversation quality assessment."""
        mock_assistant.get_conversation_quality = AsyncMock(
            return_value={
                "overall_score": 0.85,
                "overall_level": "high",
                "dimensions": [
                    {
                        "dimension": "coherence",
                        "score": 0.9,
                        "level": "high",
                    },
                    {
                        "dimension": "empathy",
                        "score": 0.8,
                        "level": "high",
                    },
                ],
                "strengths": ["Good topic transitions", "Empathetic responses"],
                "improvements": ["Could be more concise"],
            }
        )
        response = client.get("/api/quality/conv_456")
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "conv_456"
        assert data["overall_score"] == 0.85
        assert data["overall_level"] == "high"
        assert len(data["dimensions"]) == 2
        assert len(data["strengths"]) == 2
        assert len(data["improvements"]) == 1

    def test_get_quality_empty(self, client, mock_assistant):
        """Should handle empty quality data."""
        mock_assistant.get_conversation_quality = AsyncMock(return_value={})
        response = client.get("/api/quality/conv_456")
        assert response.status_code == 200
        data = response.json()
        assert data["dimensions"] == []

    def test_get_quality_error(self, client, mock_assistant):
        """Should return 500 on internal error."""
        mock_assistant.get_conversation_quality = AsyncMock(
            side_effect=RuntimeError("Service unavailable")
        )
        response = client.get("/api/quality/conv_456")
        assert response.status_code == 500


# ============================================================================
# Assistant not initialized
# ============================================================================


class TestAssistantNotInitialized:
    """Tests that endpoints return 503 when assistant is not set."""

    def test_no_assistant_raises_503(self):
        """All endpoints should return 503 when assistant not initialized."""
        from morgan_server.api.routes.features import _assistant

        app = FastAPI()
        app.include_router(router)

        # Reset the global assistant to None
        import morgan_server.api.routes.features as features_mod
        original = features_mod._assistant
        features_mod._assistant = None

        try:
            client = TestClient(app)
            response = client.get("/api/suggestions/user_123")
            assert response.status_code == 503

            response = client.get("/api/wellness/user_123")
            assert response.status_code == 503

            response = client.get("/api/habits/user_123")
            assert response.status_code == 503

            response = client.get("/api/quality/conv_456")
            assert response.status_code == 503
        finally:
            features_mod._assistant = original
