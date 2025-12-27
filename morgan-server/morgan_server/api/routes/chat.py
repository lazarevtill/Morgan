"""
Chat API Routes

This module implements the chat endpoints for Morgan server:
- POST /api/chat: Send a message and get a response
- WebSocket /ws/{user_id}: Real-time chat via WebSocket
"""

import uuid
from typing import Dict, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.responses import JSONResponse

from morgan_server.api.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    Source,
    MilestoneCelebration,
)
from morgan_server.assistant import MorganAssistant, AssistantResponse


router = APIRouter(prefix="/api", tags=["chat"])


# Global assistant instance (will be injected via dependency injection)
_assistant: Optional[MorganAssistant] = None


def set_assistant(assistant: MorganAssistant) -> None:
    """
    Set the global assistant instance.

    This should be called during application startup.

    Args:
        assistant: MorganAssistant instance
    """
    global _assistant
    _assistant = assistant


def get_assistant() -> MorganAssistant:
    """
    Get the global assistant instance.

    Returns:
        MorganAssistant instance

    Raises:
        HTTPException: If assistant is not initialized
    """
    if _assistant is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant not initialized",
        )
    return _assistant


def _convert_assistant_response(response: AssistantResponse) -> ChatResponse:
    """
    Convert AssistantResponse to ChatResponse API model.

    Args:
        response: AssistantResponse from assistant

    Returns:
        ChatResponse API model
    """
    # Convert sources to Source models
    sources = []
    for source in response.sources:
        sources.append(
            Source(
                content=source.get("content", ""),
                document_id=source.get("document_id"),
                chunk_id=source.get("chunk_id"),
                score=source.get("score", 0.0),
                metadata=source.get("metadata", {}),
            )
        )

    # Convert milestone celebration if present
    milestone = None
    if response.milestone_celebration:
        milestone = MilestoneCelebration(
            milestone_type="general",
            message=response.milestone_celebration,
        )

    return ChatResponse(
        answer=response.answer,
        conversation_id=response.conversation_id,
        emotional_tone=response.emotional_tone,
        empathy_level=response.empathy_level,
        personalization_elements=response.personalization_elements,
        milestone_celebration=milestone,
        confidence=response.confidence,
        sources=sources,
        metadata=response.metadata,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message to Morgan and get a response.

    This endpoint processes a user message through all of Morgan's engines:
    - Empathic Engine: Emotional intelligence and personality
    - Knowledge Engine: RAG and semantic search
    - Personalization Layer: User preferences and memory

    Args:
        request: ChatRequest with message and optional user/conversation IDs

    Returns:
        ChatResponse with Morgan's answer and metadata

    Raises:
        HTTPException: If assistant is not initialized or processing fails
    """
    try:
        assistant = get_assistant()

        # Generate user_id if not provided
        user_id = request.user_id or str(uuid.uuid4())

        # Process message through assistant
        response = await assistant.chat(
            message=request.message,
            user_id=user_id,
            conversation_id=request.conversation_id,
            use_knowledge=True,
            use_memory=True,
        )

        # Convert to API response model
        return _convert_assistant_response(response)

    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Error processing chat request: {e}")

        # Return structured error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}",
        )


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time chat."""

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, user_id: str, websocket: WebSocket) -> None:
        """
        Accept and register a WebSocket connection.

        Args:
            user_id: User identifier
            websocket: WebSocket connection
        """
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str) -> None:
        """
        Remove a WebSocket connection.

        Args:
            user_id: User identifier
        """
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_message(self, user_id: str, message: str) -> None:
        """
        Send a message to a specific user.

        Args:
            user_id: User identifier
            message: Message to send
        """
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

    async def send_json(self, user_id: str, data: dict) -> None:
        """
        Send JSON data to a specific user.

        Args:
            user_id: User identifier
            data: Data to send as JSON
        """
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(data)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time chat.

    This endpoint provides a persistent connection for real-time messaging.
    Messages are sent as JSON with the following format:

    Client -> Server:
    {
        "message": "user message",
        "conversation_id": "optional conversation id",
        "metadata": {}
    }

    Server -> Client:
    {
        "type": "response",
        "answer": "assistant response",
        "conversation_id": "conversation id",
        "emotional_tone": "detected tone",
        "metadata": {}
    }

    Or for errors:
    {
        "type": "error",
        "error": "ERROR_CODE",
        "message": "error message"
    }

    Args:
        websocket: WebSocket connection
        user_id: User identifier from path
    """
    await manager.connect(user_id, websocket)

    try:
        assistant = get_assistant()

        while True:
            # Receive message from client
            data = await websocket.receive_json()

            # Validate message
            if "message" not in data:
                await manager.send_json(
                    user_id,
                    {
                        "type": "error",
                        "error": "INVALID_REQUEST",
                        "message": "Message field is required",
                    },
                )
                continue

            message = data.get("message", "").strip()
            if not message:
                await manager.send_json(
                    user_id,
                    {
                        "type": "error",
                        "error": "INVALID_REQUEST",
                        "message": "Message cannot be empty",
                    },
                )
                continue

            conversation_id = data.get("conversation_id")
            metadata = data.get("metadata", {})

            try:
                # Process message through assistant
                response = await assistant.chat(
                    message=message,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    use_knowledge=True,
                    use_memory=True,
                )

                # Send response to client
                await manager.send_json(
                    user_id,
                    {
                        "type": "response",
                        "answer": response.answer,
                        "conversation_id": response.conversation_id,
                        "emotional_tone": response.emotional_tone,
                        "empathy_level": response.empathy_level,
                        "personalization_elements": response.personalization_elements,
                        "milestone_celebration": response.milestone_celebration,
                        "confidence": response.confidence,
                        "sources": [
                            {
                                "content": s.get("content", ""),
                                "score": s.get("score", 0.0),
                                "metadata": s.get("metadata", {}),
                            }
                            for s in response.sources
                        ],
                        "metadata": response.metadata,
                    },
                )

            except Exception as e:
                # Send error response
                await manager.send_json(
                    user_id,
                    {
                        "type": "error",
                        "error": "PROCESSING_ERROR",
                        "message": f"Failed to process message: {str(e)}",
                    },
                )

    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        # Log error
        print(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(user_id)
