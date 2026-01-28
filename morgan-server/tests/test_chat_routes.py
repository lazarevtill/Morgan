"""Unit tests for Chat API routes."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

from morgan_server.api.routes.chat import (
    router,
    set_assistant,
    get_assistant,
    _convert_assistant_response,
    ConnectionManager,
)
from morgan_server.api.models import ChatRequest, ChatResponse
from morgan_server.assistant import AssistantResponse, MorganAssistant


@pytest.fixture
def mock_assistant():
    """Create a mock MorganAssistant for testing."""
    assistant = MagicMock(spec=MorganAssistant)

    # Mock the chat method to return a proper AssistantResponse
    async def mock_chat(*args, **kwargs):
        return AssistantResponse(
            answer="This is a test response",
            conversation_id="test_conv_123",
            emotional_tone="neutral",
            empathy_level=0.7,
            personalization_elements=["trait:friendly", "communication_style:casual"],
            milestone_celebration=None,
            confidence=0.95,
            sources=[
                {
                    "content": "Test source content",
                    "document_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.85,
                    "metadata": {"type": "test"},
                }
            ],
            metadata={"test": "metadata"},
        )

    assistant.chat = AsyncMock(side_effect=mock_chat)
    return assistant


@pytest.fixture
def app(mock_assistant):
    """Create a FastAPI app with the chat router for testing."""
    app = FastAPI()
    app.include_router(router)

    # Set the mock assistant
    set_assistant(mock_assistant)

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestChatEndpoint:
    """Tests for POST /api/chat endpoint."""

    def test_chat_success(self, client, mock_assistant):
        """Test successful chat request."""
        request_data = {
            "message": "Hello, Morgan!",
            "user_id": "test_user_123",
            "conversation_id": "test_conv_123",
            "metadata": {"test": "data"},
        }

        response = client.post("/api/chat", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "answer" in data
        assert "conversation_id" in data
        assert data["answer"] == "This is a test response"
        assert data["conversation_id"] == "test_conv_123"
        assert data["emotional_tone"] == "neutral"
        assert data["empathy_level"] == 0.7
        assert len(data["personalization_elements"]) == 2
        assert data["confidence"] == 0.95
        assert len(data["sources"]) == 1

        # Verify assistant was called correctly
        mock_assistant.chat.assert_called_once()
        call_kwargs = mock_assistant.chat.call_args[1]
        assert call_kwargs["message"] == "Hello, Morgan!"
        assert call_kwargs["user_id"] == "test_user_123"
        assert call_kwargs["conversation_id"] == "test_conv_123"

    def test_chat_without_user_id(self, client, mock_assistant):
        """Test chat request without user_id (should generate one)."""
        request_data = {
            "message": "Hello!",
        }

        response = client.post("/api/chat", json=request_data)

        assert response.status_code == 200

        # Verify assistant was called with a generated user_id
        mock_assistant.chat.assert_called_once()
        call_kwargs = mock_assistant.chat.call_args[1]
        assert call_kwargs["user_id"] is not None
        assert len(call_kwargs["user_id"]) > 0

    def test_chat_empty_message(self, client):
        """Test chat request with empty message."""
        request_data = {
            "message": "",
        }

        response = client.post("/api/chat", json=request_data)

        # Should fail validation
        assert response.status_code == 422

    def test_chat_whitespace_only_message(self, client):
        """Test chat request with whitespace-only message."""
        request_data = {
            "message": "   ",
        }

        response = client.post("/api/chat", json=request_data)

        # Should fail validation
        assert response.status_code == 422

    def test_chat_missing_message(self, client):
        """Test chat request without message field."""
        request_data = {
            "user_id": "test_user",
        }

        response = client.post("/api/chat", json=request_data)

        # Should fail validation
        assert response.status_code == 422

    def test_chat_assistant_error(self, client, mock_assistant):
        """Test chat request when assistant raises an error."""
        # Make assistant raise an error
        mock_assistant.chat.side_effect = Exception("Test error")

        request_data = {
            "message": "Hello!",
        }

        response = client.post("/api/chat", json=request_data)

        # Should return 500 error
        assert response.status_code == 500
        assert "Failed to process message" in response.json()["detail"]

    def test_chat_with_milestone_celebration(self, client, mock_assistant):
        """Test chat response with milestone celebration."""

        # Mock response with milestone
        async def mock_chat_with_milestone(*args, **kwargs):
            return AssistantResponse(
                answer="Congratulations!",
                conversation_id="test_conv",
                milestone_celebration="You've reached 100 conversations!",
                confidence=1.0,
            )

        mock_assistant.chat = AsyncMock(side_effect=mock_chat_with_milestone)

        request_data = {
            "message": "Hello!",
        }

        response = client.post("/api/chat", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["milestone_celebration"] is not None
        assert "message" in data["milestone_celebration"]


class TestWebSocketEndpoint:
    """Tests for WebSocket /ws/{user_id} endpoint."""

    def test_websocket_connection(self, client, mock_assistant):
        """Test WebSocket connection establishment."""
        with client.websocket_connect("/api/ws/test_user") as websocket:
            # Connection should be established
            assert websocket is not None

    def test_websocket_message_exchange(self, client, mock_assistant):
        """Test sending and receiving messages via WebSocket."""
        with client.websocket_connect("/api/ws/test_user") as websocket:
            # Send a message
            websocket.send_json(
                {
                    "message": "Hello via WebSocket!",
                    "conversation_id": "ws_conv_123",
                }
            )

            # Receive response
            response = websocket.receive_json()

            assert response["type"] == "response"
            assert "answer" in response
            assert response["answer"] == "This is a test response"
            assert response["conversation_id"] == "test_conv_123"

            # Verify assistant was called
            mock_assistant.chat.assert_called_once()

    def test_websocket_empty_message(self, client, mock_assistant):
        """Test WebSocket with empty message."""
        with client.websocket_connect("/api/ws/test_user") as websocket:
            # Send empty message
            websocket.send_json(
                {
                    "message": "",
                }
            )

            # Should receive error response
            response = websocket.receive_json()

            assert response["type"] == "error"
            assert response["error"] == "INVALID_REQUEST"
            assert "empty" in response["message"].lower()

    def test_websocket_missing_message_field(self, client, mock_assistant):
        """Test WebSocket without message field."""
        with client.websocket_connect("/api/ws/test_user") as websocket:
            # Send without message field
            websocket.send_json(
                {
                    "conversation_id": "test_conv",
                }
            )

            # Should receive error response
            response = websocket.receive_json()

            assert response["type"] == "error"
            assert response["error"] == "INVALID_REQUEST"
            assert "required" in response["message"].lower()

    def test_websocket_assistant_error(self, client, mock_assistant):
        """Test WebSocket when assistant raises an error."""
        # Make assistant raise an error
        mock_assistant.chat.side_effect = Exception("Test error")

        with client.websocket_connect("/api/ws/test_user") as websocket:
            # Send a message
            websocket.send_json(
                {
                    "message": "Hello!",
                }
            )

            # Should receive error response
            response = websocket.receive_json()

            assert response["type"] == "error"
            assert response["error"] == "PROCESSING_ERROR"

    def test_websocket_multiple_messages(self, client, mock_assistant):
        """Test sending multiple messages in one WebSocket session."""
        with client.websocket_connect("/api/ws/test_user") as websocket:
            # Send first message
            websocket.send_json({"message": "First message"})
            response1 = websocket.receive_json()
            assert response1["type"] == "response"

            # Send second message
            websocket.send_json({"message": "Second message"})
            response2 = websocket.receive_json()
            assert response2["type"] == "response"

            # Verify assistant was called twice
            assert mock_assistant.chat.call_count == 2


class TestConnectionManager:
    """Tests for ConnectionManager class."""

    @pytest.fixture
    def manager(self):
        """Create a ConnectionManager for testing."""
        return ConnectionManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        ws = MagicMock(spec=WebSocket)
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()
        ws.send_json = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_connect(self, manager, mock_websocket):
        """Test connecting a WebSocket."""
        await manager.connect("user_1", mock_websocket)

        assert "user_1" in manager.active_connections
        assert manager.active_connections["user_1"] == mock_websocket
        mock_websocket.accept.assert_called_once()

    def test_disconnect(self, manager, mock_websocket):
        """Test disconnecting a WebSocket."""
        manager.active_connections["user_1"] = mock_websocket

        manager.disconnect("user_1")

        assert "user_1" not in manager.active_connections

    def test_disconnect_nonexistent(self, manager):
        """Test disconnecting a non-existent connection."""
        # Should not raise an error
        manager.disconnect("nonexistent_user")

    @pytest.mark.asyncio
    async def test_send_message(self, manager, mock_websocket):
        """Test sending a text message."""
        manager.active_connections["user_1"] = mock_websocket

        await manager.send_message("user_1", "Test message")

        mock_websocket.send_text.assert_called_once_with("Test message")

    @pytest.mark.asyncio
    async def test_send_json(self, manager, mock_websocket):
        """Test sending JSON data."""
        manager.active_connections["user_1"] = mock_websocket

        test_data = {"key": "value"}
        await manager.send_json("user_1", test_data)

        mock_websocket.send_json.assert_called_once_with(test_data)

    @pytest.mark.asyncio
    async def test_send_to_nonexistent_user(self, manager):
        """Test sending to a non-existent user."""
        # Should not raise an error
        await manager.send_message("nonexistent_user", "Test")
        await manager.send_json("nonexistent_user", {})


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_convert_assistant_response(self):
        """Test converting AssistantResponse to ChatResponse."""
        assistant_response = AssistantResponse(
            answer="Test answer",
            conversation_id="conv_123",
            emotional_tone="happy",
            empathy_level=0.8,
            personalization_elements=["element1", "element2"],
            milestone_celebration="Milestone reached!",
            confidence=0.9,
            sources=[
                {
                    "content": "Source 1",
                    "document_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.95,
                    "metadata": {"key": "value"},
                }
            ],
            metadata={"meta": "data"},
        )

        chat_response = _convert_assistant_response(assistant_response)

        assert isinstance(chat_response, ChatResponse)
        assert chat_response.answer == "Test answer"
        assert chat_response.conversation_id == "conv_123"
        assert chat_response.emotional_tone == "happy"
        assert chat_response.empathy_level == 0.8
        assert len(chat_response.personalization_elements) == 2
        assert chat_response.milestone_celebration is not None
        assert chat_response.milestone_celebration.message == "Milestone reached!"
        assert chat_response.confidence == 0.9
        assert len(chat_response.sources) == 1
        assert chat_response.sources[0].content == "Source 1"

    def test_convert_assistant_response_no_milestone(self):
        """Test converting AssistantResponse without milestone."""
        assistant_response = AssistantResponse(
            answer="Test answer",
            conversation_id="conv_123",
            milestone_celebration=None,
            confidence=1.0,
        )

        chat_response = _convert_assistant_response(assistant_response)

        assert chat_response.milestone_celebration is None

    def test_get_assistant_not_initialized(self):
        """Test get_assistant when not initialized."""
        # Clear the global assistant
        set_assistant(None)

        with pytest.raises(Exception) as exc_info:
            get_assistant()

        # Should raise HTTPException with 503 status
        assert (
            "503" in str(exc_info.value)
            or "not initialized" in str(exc_info.value).lower()
        )

    def test_set_and_get_assistant(self, mock_assistant):
        """Test setting and getting assistant."""
        set_assistant(mock_assistant)

        retrieved = get_assistant()

        assert retrieved == mock_assistant
