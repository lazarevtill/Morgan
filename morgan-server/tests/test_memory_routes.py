"""Unit tests for Memory API routes."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from morgan_server.api.routes.memory import (
    router,
    set_memory_manager,
    get_memory_manager,
)
from morgan_server.personalization.memory import (
    MemoryManager,
    Conversation,
    Message,
    MessageRole,
)


@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager for testing."""
    manager = MagicMock(spec=MemoryManager)
    
    # Mock get_memory_stats
    manager.get_memory_stats.return_value = {
        "total_conversations": 5,
        "total_messages": 50,
        "oldest_conversation": "2024-01-01T10:00:00",
        "newest_conversation": "2024-12-08T10:00:00",
    }
    
    # Mock search_conversations
    test_conversation = Conversation(
        conversation_id="conv_123",
        user_id="user_123",
    )
    test_message = Message(
        role=MessageRole.USER,
        content="Test user message",
        timestamp=datetime(2024, 12, 8, 10, 0, 0),
    )
    test_response = Message(
        role=MessageRole.ASSISTANT,
        content="Test assistant response",
        timestamp=datetime(2024, 12, 8, 10, 0, 1),
    )
    test_conversation.messages = [test_message, test_response]
    
    manager.search_conversations.return_value = [
        (test_conversation, test_message, 0.85)
    ]
    
    # Mock cleanup_old_conversations
    manager.cleanup_old_conversations.return_value = 3
    
    return manager


@pytest.fixture
def app(mock_memory_manager):
    """Create a FastAPI app with the memory router for testing."""
    app = FastAPI()
    app.include_router(router)
    
    # Set the mock memory manager
    set_memory_manager(mock_memory_manager)
    
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestMemoryStatsEndpoint:
    """Tests for GET /api/memory/stats endpoint."""

    def test_get_stats_success(self, client, mock_memory_manager):
        """Test successful stats retrieval."""
        response = client.get("/api/memory/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "total_conversations" in data
        assert "active_conversations" in data
        assert "total_messages" in data
        assert "oldest_conversation" in data
        assert "newest_conversation" in data
        
        # Verify values
        assert data["total_conversations"] == 5
        assert data["active_conversations"] == 5
        assert data["total_messages"] == 50
        
        # Verify memory manager was called
        mock_memory_manager.get_memory_stats.assert_called_once_with(user_id=None)

    def test_get_stats_with_user_id(self, client, mock_memory_manager):
        """Test stats retrieval filtered by user_id."""
        response = client.get("/api/memory/stats?user_id=test_user_123")
        
        assert response.status_code == 200
        
        # Verify memory manager was called with user_id
        mock_memory_manager.get_memory_stats.assert_called_once_with(
            user_id="test_user_123"
        )

    def test_get_stats_empty_memory(self, client, mock_memory_manager):
        """Test stats retrieval when no conversations exist."""
        # Mock empty stats
        mock_memory_manager.get_memory_stats.return_value = {
            "total_conversations": 0,
            "total_messages": 0,
            "oldest_conversation": None,
            "newest_conversation": None,
        }
        
        response = client.get("/api/memory/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_conversations"] == 0
        assert data["total_messages"] == 0
        assert data["oldest_conversation"] is None
        assert data["newest_conversation"] is None

    def test_get_stats_manager_error(self, client, mock_memory_manager):
        """Test stats retrieval when memory manager raises an error."""
        # Make memory manager raise an error
        mock_memory_manager.get_memory_stats.side_effect = Exception("Test error")
        
        response = client.get("/api/memory/stats")
        
        # Should return 500 error
        assert response.status_code == 500
        assert "Failed to retrieve memory stats" in response.json()["detail"]

    def test_get_stats_manager_not_initialized(self, client):
        """Test stats retrieval when memory manager is not initialized."""
        # Clear the global memory manager
        set_memory_manager(None)
        
        response = client.get("/api/memory/stats")
        
        # Should return 503 error
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()


class TestMemorySearchEndpoint:
    """Tests for GET /api/memory/search endpoint."""

    def test_search_success(self, client, mock_memory_manager):
        """Test successful memory search."""
        response = client.get(
            "/api/memory/search?query=test&user_id=user_123&limit=10"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response is a list
        assert isinstance(data, list)
        assert len(data) == 1
        
        # Verify result structure
        result = data[0]
        assert "conversation_id" in result
        assert "timestamp" in result
        assert "message" in result
        assert "response" in result
        assert "relevance_score" in result
        
        # Verify values
        assert result["conversation_id"] == "conv_123"
        assert result["message"] == "Test user message"
        assert result["response"] == "Test assistant response"
        assert result["relevance_score"] == 0.85
        
        # Verify memory manager was called correctly
        mock_memory_manager.search_conversations.assert_called_once_with(
            user_id="user_123",
            query="test",
            limit=10
        )

    def test_search_without_user_id(self, client):
        """Test search without user_id (should fail)."""
        response = client.get("/api/memory/search?query=test")
        
        # Should return 400 error
        assert response.status_code == 400
        assert "user_id is required" in response.json()["detail"]

    def test_search_empty_query(self, client):
        """Test search with empty query."""
        response = client.get("/api/memory/search?query=&user_id=user_123")
        
        # Should fail validation
        assert response.status_code == 422

    def test_search_whitespace_query(self, client):
        """Test search with whitespace-only query."""
        response = client.get("/api/memory/search?query=   &user_id=user_123")
        
        # Should return 400 error
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_search_missing_query(self, client):
        """Test search without query parameter."""
        response = client.get("/api/memory/search?user_id=user_123")
        
        # Should fail validation
        assert response.status_code == 422

    def test_search_custom_limit(self, client, mock_memory_manager):
        """Test search with custom limit."""
        response = client.get(
            "/api/memory/search?query=test&user_id=user_123&limit=5"
        )
        
        assert response.status_code == 200
        
        # Verify limit was passed correctly
        mock_memory_manager.search_conversations.assert_called_once_with(
            user_id="user_123",
            query="test",
            limit=5
        )

    def test_search_limit_too_high(self, client):
        """Test search with limit exceeding maximum."""
        response = client.get(
            "/api/memory/search?query=test&user_id=user_123&limit=200"
        )
        
        # Should fail validation (max is 100)
        assert response.status_code == 422

    def test_search_limit_too_low(self, client):
        """Test search with limit below minimum."""
        response = client.get(
            "/api/memory/search?query=test&user_id=user_123&limit=0"
        )
        
        # Should fail validation (min is 1)
        assert response.status_code == 422

    def test_search_no_results(self, client, mock_memory_manager):
        """Test search with no matching results."""
        # Mock empty results
        mock_memory_manager.search_conversations.return_value = []
        
        response = client.get(
            "/api/memory/search?query=nonexistent&user_id=user_123"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) == 0

    def test_search_message_without_response(self, client, mock_memory_manager):
        """Test search result where message has no assistant response."""
        # Create conversation with only user message
        test_conversation = Conversation(
            conversation_id="conv_456",
            user_id="user_123",
        )
        test_message = Message(
            role=MessageRole.USER,
            content="User message without response",
            timestamp=datetime(2024, 12, 8, 10, 0, 0),
        )
        test_conversation.messages = [test_message]
        
        mock_memory_manager.search_conversations.return_value = [
            (test_conversation, test_message, 0.75)
        ]
        
        response = client.get(
            "/api/memory/search?query=test&user_id=user_123"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data) == 1
        assert data[0]["response"] == ""  # No response found

    def test_search_manager_error(self, client, mock_memory_manager):
        """Test search when memory manager raises an error."""
        # Make memory manager raise an error
        mock_memory_manager.search_conversations.side_effect = Exception("Test error")
        
        response = client.get(
            "/api/memory/search?query=test&user_id=user_123"
        )
        
        # Should return 500 error
        assert response.status_code == 500
        assert "Failed to search memory" in response.json()["detail"]

    def test_search_manager_not_initialized(self, client):
        """Test search when memory manager is not initialized."""
        # Clear the global memory manager
        set_memory_manager(None)
        
        response = client.get(
            "/api/memory/search?query=test&user_id=user_123"
        )
        
        # Should return 503 error
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()


class TestMemoryCleanupEndpoint:
    """Tests for DELETE /api/memory/cleanup endpoint."""

    def test_cleanup_success(self, client, mock_memory_manager):
        """Test successful memory cleanup."""
        response = client.delete(
            "/api/memory/cleanup?user_id=user_123&keep_recent=10"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "status" in data
        assert "deleted_count" in data
        assert "kept_count" in data
        assert "message" in data
        
        # Verify values
        assert data["status"] == "success"
        assert data["deleted_count"] == 3
        assert data["kept_count"] == 10
        
        # Verify memory manager was called correctly
        mock_memory_manager.cleanup_old_conversations.assert_called_once_with(
            user_id="user_123",
            keep_recent=10
        )

    def test_cleanup_without_user_id(self, client):
        """Test cleanup without user_id (should fail)."""
        response = client.delete("/api/memory/cleanup?keep_recent=10")
        
        # Should fail validation
        assert response.status_code == 422

    def test_cleanup_empty_user_id(self, client):
        """Test cleanup with empty user_id."""
        response = client.delete("/api/memory/cleanup?user_id=&keep_recent=10")
        
        # Should return 400 error
        assert response.status_code == 400
        assert "user_id is required" in response.json()["detail"]

    def test_cleanup_whitespace_user_id(self, client):
        """Test cleanup with whitespace-only user_id."""
        response = client.delete("/api/memory/cleanup?user_id=   &keep_recent=10")
        
        # Should return 400 error
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"]

    def test_cleanup_custom_keep_recent(self, client, mock_memory_manager):
        """Test cleanup with custom keep_recent value."""
        response = client.delete(
            "/api/memory/cleanup?user_id=user_123&keep_recent=5"
        )
        
        assert response.status_code == 200
        
        # Verify keep_recent was passed correctly
        mock_memory_manager.cleanup_old_conversations.assert_called_once_with(
            user_id="user_123",
            keep_recent=5
        )

    def test_cleanup_default_keep_recent(self, client, mock_memory_manager):
        """Test cleanup with default keep_recent value."""
        response = client.delete("/api/memory/cleanup?user_id=user_123")
        
        assert response.status_code == 200
        
        # Verify default keep_recent (10) was used
        mock_memory_manager.cleanup_old_conversations.assert_called_once_with(
            user_id="user_123",
            keep_recent=10
        )

    def test_cleanup_keep_recent_too_high(self, client):
        """Test cleanup with keep_recent exceeding maximum."""
        response = client.delete(
            "/api/memory/cleanup?user_id=user_123&keep_recent=200"
        )
        
        # Should fail validation (max is 100)
        assert response.status_code == 422

    def test_cleanup_keep_recent_too_low(self, client):
        """Test cleanup with keep_recent below minimum."""
        response = client.delete(
            "/api/memory/cleanup?user_id=user_123&keep_recent=0"
        )
        
        # Should fail validation (min is 1)
        assert response.status_code == 422

    def test_cleanup_no_conversations_deleted(self, client, mock_memory_manager):
        """Test cleanup when no conversations need to be deleted."""
        # Mock no deletions
        mock_memory_manager.cleanup_old_conversations.return_value = 0
        
        response = client.delete(
            "/api/memory/cleanup?user_id=user_123&keep_recent=10"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["deleted_count"] == 0
        assert data["status"] == "success"

    def test_cleanup_manager_error(self, client, mock_memory_manager):
        """Test cleanup when memory manager raises an error."""
        # Make memory manager raise an error
        mock_memory_manager.cleanup_old_conversations.side_effect = Exception(
            "Test error"
        )
        
        response = client.delete(
            "/api/memory/cleanup?user_id=user_123&keep_recent=10"
        )
        
        # Should return 500 error
        assert response.status_code == 500
        assert "Failed to clean up memory" in response.json()["detail"]

    def test_cleanup_manager_not_initialized(self, client):
        """Test cleanup when memory manager is not initialized."""
        # Clear the global memory manager
        set_memory_manager(None)
        
        response = client.delete(
            "/api/memory/cleanup?user_id=user_123&keep_recent=10"
        )
        
        # Should return 503 error
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_set_and_get_memory_manager(self, mock_memory_manager):
        """Test setting and getting memory manager."""
        set_memory_manager(mock_memory_manager)
        
        retrieved = get_memory_manager()
        
        assert retrieved == mock_memory_manager

    def test_get_memory_manager_not_initialized(self):
        """Test get_memory_manager when not initialized."""
        # Clear the global memory manager
        set_memory_manager(None)
        
        with pytest.raises(Exception) as exc_info:
            get_memory_manager()
        
        # Should raise HTTPException with 503 status
        assert "503" in str(exc_info.value) or "not initialized" in str(
            exc_info.value
        ).lower()
