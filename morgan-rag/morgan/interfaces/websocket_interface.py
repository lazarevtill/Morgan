"""
Morgan WebSocket Interface.

Provides real-time bidirectional communication for:
- Interactive chat with streaming
- Real-time emotion updates
- Live system status
- Connection management
- Session persistence

Full async/await with WebSocket support.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Set

from pydantic import BaseModel, Field

try:
    from fastapi import WebSocket, WebSocketDisconnect, status
except ImportError:
    raise ImportError(
        "FastAPI not installed. Install with: pip install fastapi uvicorn websockets"
    )

from morgan.core.assistant import MorganAssistant
from morgan.core.types import MessageRole

logger = logging.getLogger(__name__)


# ==================== Message Types ====================


class MessageType(str, Enum):
    """WebSocket message types."""

    # Client -> Server
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    CHAT = "chat"
    FEEDBACK = "feedback"
    TYPING = "typing"
    PING = "ping"

    # Server -> Client
    CONNECTED = "connected"
    CHAT_START = "chat_start"
    CHAT_CHUNK = "chat_chunk"
    CHAT_COMPLETE = "chat_complete"
    EMOTION = "emotion"
    SOURCES = "sources"
    METRICS = "metrics"
    ERROR = "error"
    PONG = "pong"
    STATUS = "status"


# ==================== Message Models ====================


class WSMessage(BaseModel):
    """Base WebSocket message."""

    type: MessageType
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """Chat message from client."""

    message: str
    session_id: Optional[str] = None
    include_emotion: bool = True
    include_sources: bool = True
    include_metrics: bool = False


class FeedbackMessage(BaseModel):
    """Feedback message from client."""

    response_id: str
    rating: float = Field(..., ge=0.0, le=1.0)
    comment: Optional[str] = None


class TypingMessage(BaseModel):
    """Typing indicator message."""

    is_typing: bool


# ==================== Connection Manager ====================


class ConnectionManager:
    """
    Manages WebSocket connections.

    Features:
    - Connection tracking
    - User session mapping
    - Broadcasting
    - Connection cleanup
    """

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        user_id: str,
    ) -> None:
        """
        Accept and register new connection.

        Args:
            websocket: WebSocket connection
            connection_id: Unique connection ID
            user_id: User ID
        """
        await websocket.accept()

        async with self._lock:
            self.active_connections[connection_id] = websocket

            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(connection_id)

        logger.info(
            f"WebSocket connected: connection_id={connection_id}, user_id={user_id}"
        )

    async def disconnect(self, connection_id: str, user_id: str) -> None:
        """
        Disconnect and cleanup.

        Args:
            connection_id: Connection ID to disconnect
            user_id: User ID
        """
        async with self._lock:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

            if user_id in self.user_sessions:
                self.user_sessions[user_id].discard(connection_id)
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]

        logger.info(
            f"WebSocket disconnected: connection_id={connection_id}, user_id={user_id}"
        )

    async def send_message(
        self,
        connection_id: str,
        message: WSMessage,
    ) -> bool:
        """
        Send message to specific connection.

        Args:
            connection_id: Target connection ID
            message: Message to send

        Returns:
            True if sent successfully, False otherwise
        """
        websocket = self.active_connections.get(connection_id)
        if not websocket:
            return False

        try:
            await websocket.send_json(message.dict())
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            return False

    async def send_to_user(
        self,
        user_id: str,
        message: WSMessage,
    ) -> int:
        """
        Send message to all connections for a user.

        Args:
            user_id: Target user ID
            message: Message to send

        Returns:
            Number of connections message was sent to
        """
        connection_ids = self.user_sessions.get(user_id, set())
        sent_count = 0

        for connection_id in connection_ids:
            if await self.send_message(connection_id, message):
                sent_count += 1

        return sent_count

    async def broadcast(self, message: WSMessage, exclude: Optional[Set[str]] = None) -> int:
        """
        Broadcast message to all connections.

        Args:
            message: Message to broadcast
            exclude: Set of connection IDs to exclude

        Returns:
            Number of connections message was sent to
        """
        exclude = exclude or set()
        sent_count = 0

        for connection_id in list(self.active_connections.keys()):
            if connection_id not in exclude:
                if await self.send_message(connection_id, message):
                    sent_count += 1

        return sent_count

    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)

    @property
    def user_count(self) -> int:
        """Get number of connected users."""
        return len(self.user_sessions)


# ==================== WebSocket Handler ====================


class WebSocketHandler:
    """
    Handles WebSocket connections and message routing.

    Features:
    - Connection lifecycle management
    - Message routing
    - Chat streaming
    - Error handling
    - Heartbeat/keepalive
    """

    def __init__(
        self,
        assistant: MorganAssistant,
        connection_manager: Optional[ConnectionManager] = None,
        heartbeat_interval: float = 30.0,
    ):
        """
        Initialize WebSocket handler.

        Args:
            assistant: Morgan assistant instance
            connection_manager: Connection manager (creates new if None)
            heartbeat_interval: Heartbeat interval in seconds
        """
        self.assistant = assistant
        self.connection_manager = connection_manager or ConnectionManager()
        self.heartbeat_interval = heartbeat_interval

    async def handle_connection(
        self,
        websocket: WebSocket,
        user_id: str,
    ) -> None:
        """
        Handle WebSocket connection lifecycle.

        Args:
            websocket: WebSocket connection
            user_id: User ID
        """
        # Generate connection ID
        connection_id = str(uuid.uuid4())

        # Connect
        await self.connection_manager.connect(websocket, connection_id, user_id)

        # Send connected message
        await self._send_message(
            connection_id,
            MessageType.CONNECTED,
            {
                "connection_id": connection_id,
                "user_id": user_id,
                "message": "Connected to Morgan AI",
            },
        )

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(connection_id)
        )

        try:
            # Message loop
            while True:
                # Receive message
                try:
                    data = await websocket.receive_json()
                except Exception as e:
                    logger.error(f"Failed to receive message: {e}")
                    break

                # Process message
                await self._process_message(
                    connection_id,
                    user_id,
                    data,
                )

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")

        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)
            # Send error message
            await self._send_message(
                connection_id,
                MessageType.ERROR,
                {
                    "error": "Internal error",
                    "message": str(e),
                },
            )

        finally:
            # Cancel heartbeat
            heartbeat_task.cancel()

            # Disconnect
            await self.connection_manager.disconnect(connection_id, user_id)

    async def _process_message(
        self,
        connection_id: str,
        user_id: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Process incoming message.

        Args:
            connection_id: Connection ID
            user_id: User ID
            data: Message data
        """
        try:
            # Parse message type
            message_type = MessageType(data.get("type", ""))

            # Route to handler
            if message_type == MessageType.CHAT:
                await self._handle_chat(connection_id, user_id, data)

            elif message_type == MessageType.FEEDBACK:
                await self._handle_feedback(connection_id, user_id, data)

            elif message_type == MessageType.TYPING:
                await self._handle_typing(connection_id, user_id, data)

            elif message_type == MessageType.PING:
                await self._handle_ping(connection_id)

            else:
                logger.warning(f"Unknown message type: {message_type}")

        except ValueError as e:
            logger.error(f"Invalid message type: {e}")
            await self._send_message(
                connection_id,
                MessageType.ERROR,
                {
                    "error": "Invalid message type",
                    "message": str(e),
                },
            )

        except Exception as e:
            logger.error(f"Message processing error: {e}", exc_info=True)
            await self._send_message(
                connection_id,
                MessageType.ERROR,
                {
                    "error": "Processing error",
                    "message": str(e),
                },
            )

    async def _handle_chat(
        self,
        connection_id: str,
        user_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Handle chat message."""
        try:
            # Parse chat message
            chat_data = data.get("data", {})
            message = chat_data.get("message", "")
            session_id = chat_data.get("session_id") or str(uuid.uuid4())
            include_emotion = chat_data.get("include_emotion", True)
            include_sources = chat_data.get("include_sources", True)
            include_metrics = chat_data.get("include_metrics", False)

            if not message:
                await self._send_message(
                    connection_id,
                    MessageType.ERROR,
                    {"error": "Empty message"},
                )
                return

            # Send chat start
            response_id = str(uuid.uuid4())
            await self._send_message(
                connection_id,
                MessageType.CHAT_START,
                {
                    "response_id": response_id,
                    "session_id": session_id,
                },
            )

            # Stream response
            full_response = []
            async for chunk in self.assistant.stream_response(
                user_id=user_id,
                message=message,
                session_id=session_id,
            ):
                full_response.append(chunk)

                # Send chunk
                await self._send_message(
                    connection_id,
                    MessageType.CHAT_CHUNK,
                    {
                        "response_id": response_id,
                        "content": chunk,
                    },
                )

            # Get complete response for metadata
            # Note: In streaming mode, we don't have emotion/sources
            # This would require a follow-up call or modified streaming

            # Send completion
            await self._send_message(
                connection_id,
                MessageType.CHAT_COMPLETE,
                {
                    "response_id": response_id,
                    "session_id": session_id,
                    "content": "".join(full_response),
                },
            )

        except Exception as e:
            logger.error(f"Chat handling error: {e}", exc_info=True)
            await self._send_message(
                connection_id,
                MessageType.ERROR,
                {
                    "error": "Chat processing failed",
                    "message": str(e),
                },
            )

    async def _handle_feedback(
        self,
        connection_id: str,
        user_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Handle feedback message."""
        try:
            feedback_data = data.get("data", {})
            response_id = feedback_data.get("response_id")
            rating = feedback_data.get("rating")

            if not response_id or rating is None:
                await self._send_message(
                    connection_id,
                    MessageType.ERROR,
                    {"error": "Invalid feedback data"},
                )
                return

            # Process feedback
            if self.assistant.learning_engine:
                from morgan.learning.types import FeedbackSignal, FeedbackType

                feedback = FeedbackSignal(
                    feedback_type=FeedbackType.EXPLICIT,
                    signal_value=float(rating),
                    timestamp=datetime.now(),
                    response_id=response_id,
                    user_id=user_id,
                    session_id=feedback_data.get("session_id", "unknown"),
                    context_data={"comment": feedback_data.get("comment")},
                )

                await self.assistant.learning_engine.process_feedback(feedback)

                # Send confirmation
                await self._send_message(
                    connection_id,
                    MessageType.STATUS,
                    {
                        "status": "feedback_received",
                        "message": "Thank you for your feedback!",
                    },
                )

        except Exception as e:
            logger.error(f"Feedback handling error: {e}", exc_info=True)
            await self._send_message(
                connection_id,
                MessageType.ERROR,
                {
                    "error": "Feedback processing failed",
                    "message": str(e),
                },
            )

    async def _handle_typing(
        self,
        connection_id: str,
        user_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Handle typing indicator."""
        # Could broadcast to other users in same session
        # For now, just acknowledge
        pass

    async def _handle_ping(self, connection_id: str) -> None:
        """Handle ping message."""
        await self._send_message(
            connection_id,
            MessageType.PONG,
            {"timestamp": datetime.now().isoformat()},
        )

    async def _heartbeat_loop(self, connection_id: str) -> None:
        """
        Send periodic heartbeat messages.

        Args:
            connection_id: Connection ID
        """
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval)

                # Send ping
                success = await self._send_message(
                    connection_id,
                    MessageType.PING,
                    {"timestamp": datetime.now().isoformat()},
                )

                if not success:
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")

    async def _send_message(
        self,
        connection_id: str,
        message_type: MessageType,
        data: Dict[str, Any],
    ) -> bool:
        """
        Send message to connection.

        Args:
            connection_id: Target connection ID
            message_type: Message type
            data: Message data

        Returns:
            True if sent successfully
        """
        message = WSMessage(
            type=message_type,
            timestamp=datetime.now(),
            data=data,
        )

        return await self.connection_manager.send_message(connection_id, message)


# ==================== FastAPI Integration ====================


def add_websocket_routes(
    app,
    assistant: MorganAssistant,
    path: str = "/ws",
) -> None:
    """
    Add WebSocket routes to FastAPI app.

    Args:
        app: FastAPI application
        assistant: Morgan assistant instance
        path: WebSocket endpoint path
    """
    # Create handler
    handler = WebSocketHandler(assistant)

    @app.websocket(path)
    async def websocket_endpoint(
        websocket: WebSocket,
        user_id: str = "anonymous",
    ):
        """
        WebSocket endpoint for real-time chat.

        Query parameters:
            user_id: User identifier (default: "anonymous")

        Message format:
            {
                "type": "chat|feedback|typing|ping",
                "data": {
                    // Type-specific data
                }
            }

        Example client usage:
            const ws = new WebSocket('ws://localhost:8000/ws?user_id=user123');

            ws.onopen = () => {
                ws.send(JSON.stringify({
                    type: 'chat',
                    data: {
                        message: 'Hello Morgan!',
                        session_id: 'session123'
                    }
                }));
            };

            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                console.log('Received:', message);
            };
        """
        await handler.handle_connection(websocket, user_id)

    # Add status endpoint
    @app.get("/ws/status")
    async def websocket_status():
        """Get WebSocket connection statistics."""
        return {
            "active_connections": handler.connection_manager.connection_count,
            "connected_users": handler.connection_manager.user_count,
            "timestamp": datetime.now().isoformat(),
        }


# ==================== Standalone Server ====================


if __name__ == "__main__":
    """Run standalone WebSocket server for testing."""
    import asyncio
    from pathlib import Path

    from fastapi import FastAPI
    import uvicorn

    from morgan.core.assistant import MorganAssistant

    async def create_test_app():
        """Create test application with WebSocket support."""
        # Create assistant
        assistant = MorganAssistant(
            storage_path=Path.home() / ".morgan" / "test",
        )
        await assistant.initialize()

        # Create FastAPI app
        app = FastAPI(title="Morgan WebSocket Test Server")

        # Add WebSocket routes
        add_websocket_routes(app, assistant)

        return app, assistant

    # Run server
    app, assistant = asyncio.run(create_test_app())

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            log_level="info",
        )
    finally:
        asyncio.run(assistant.cleanup())
