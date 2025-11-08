"""
WebSocket handler for real-time Morgan chat interface.

Handles WebSocket connections, message processing, and real-time communication
following KISS principles - focused solely on WebSocket logic.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from ..core.assistant import MorganAssistant
from ..utils.logger import get_logger

logger = get_logger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time chat.

    KISS: Single responsibility - handle WebSocket connections and messaging.
    """

    def __init__(self, morgan_assistant: Optional[MorganAssistant] = None):
        """Initialize WebSocket manager."""
        self.morgan = morgan_assistant or MorganAssistant()
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        logger.info("WebSocket manager initialized")

    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept and manage a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[user_id] = websocket

        # Initialize user session
        if user_id not in self.user_sessions:
            conversation_id = self.morgan.start_conversation(user_id=user_id)
            self.user_sessions[user_id] = {
                "conversation_id": conversation_id,
                "connected_at": datetime.utcnow(),
            }

        logger.info(f"WebSocket connected for user {user_id}")

        # Send welcome message
        await self._send_welcome_message(websocket, user_id)

        # Send conversation suggestions
        await self._send_conversation_suggestions(websocket, user_id)

    def disconnect(self, user_id: str):
        """Handle WebSocket disconnection."""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        logger.info(f"WebSocket disconnected for user {user_id}")

    async def handle_message(
        self, websocket: WebSocket, user_id: str, data: Dict[str, Any]
    ):
        """Handle incoming WebSocket message."""
        try:
            message_type = data.get("type")

            if message_type == "message":
                await self._handle_chat_message(websocket, user_id, data)
            elif message_type == "feedback":
                await self._handle_feedback(websocket, user_id, data)
            elif message_type == "typing":
                await self._handle_typing_indicator(websocket, user_id, data)
            else:
                logger.warning(f"Unknown message type: {message_type}")

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self._send_error_message(websocket, str(e))

    async def _send_welcome_message(self, websocket: WebSocket, user_id: str):
        """Send personalized welcome message."""
        try:
            # Get user profile for personalization
            user_profile = self.morgan.emotional_processor.get_or_create_user_profile(
                user_id
            )

            if user_profile.interaction_count > 0:
                greeting_obj = (
                    self.morgan.emotional_processor.generate_personalized_greeting(
                        user_profile
                    )
                )
                welcome_msg = (
                    greeting_obj.greeting_text
                    if hasattr(greeting_obj, "greeting_text")
                    else f"Welcome back, {user_profile.preferred_name}!"
                )
            else:
                welcome_msg = "Hello! I'm Morgan, your emotionally intelligent AI companion. How can I help you today?"

            await websocket.send_json(
                {
                    "type": "welcome",
                    "message": welcome_msg,
                    "conversation_id": self.user_sessions[user_id]["conversation_id"],
                    "user_profile": {
                        "preferred_name": user_profile.preferred_name,
                        "interaction_count": user_profile.interaction_count,
                        "relationship_age_days": user_profile.get_relationship_age_days(),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error sending welcome message: {e}")

    async def _send_conversation_suggestions(self, websocket: WebSocket, user_id: str):
        """Send conversation topic suggestions."""
        try:
            suggestions = self.morgan.suggest_conversation_topics(user_id)
            if suggestions:
                await websocket.send_json(
                    {"type": "suggestions", "suggestions": suggestions[:3]}
                )

        except Exception as e:
            logger.error(f"Error sending suggestions: {e}")

    async def _handle_chat_message(
        self, websocket: WebSocket, user_id: str, data: Dict[str, Any]
    ):
        """Handle chat message from user."""
        try:
            message = data.get("message", "").strip()
            if not message:
                return

            conversation_id = self.user_sessions[user_id]["conversation_id"]

            # Process message with Morgan
            response = self.morgan.ask(
                question=message, conversation_id=conversation_id, user_id=user_id
            )

            # Prepare response data
            response_data = {
                "type": "response",
                "message": response.answer,
                "emotional_tone": response.emotional_tone,
                "empathy_level": response.empathy_level,
                "personalization_elements": response.personalization_elements,
                "confidence": response.confidence,
                "sources": response.sources,
                "suggestions": response.suggestions,
            }

            # Include milestone celebration if any
            if response.milestone_celebration:
                celebration_msg = (
                    self.morgan.milestone_tracker.generate_celebration_message(
                        response.milestone_celebration
                    )
                )
                response_data["milestone"] = {
                    "type": response.milestone_celebration.milestone_type.value,
                    "description": response.milestone_celebration.description,
                    "significance": response.milestone_celebration.emotional_significance,
                    "celebration_message": celebration_msg,
                }

            await websocket.send_json(response_data)

        except Exception as e:
            logger.error(f"Error handling chat message: {e}")
            await self._send_error_message(
                websocket,
                "I encountered an error processing your message. Please try again.",
            )

    async def _handle_feedback(
        self, websocket: WebSocket, user_id: str, data: Dict[str, Any]
    ):
        """Handle feedback submission."""
        try:
            conversation_id = self.user_sessions[user_id]["conversation_id"]
            rating = data.get("rating")
            comment = data.get("comment")

            if rating is None:
                await self._send_error_message(
                    websocket, "Rating is required for feedback"
                )
                return

            success = self.morgan.provide_feedback(
                conversation_id=conversation_id,
                rating=rating,
                comment=comment,
                user_id=user_id,
            )

            if success:
                await websocket.send_json(
                    {
                        "type": "feedback_received",
                        "message": "Thank you for your feedback! 😊",
                    }
                )
            else:
                await self._send_error_message(websocket, "Failed to record feedback")

        except Exception as e:
            logger.error(f"Error handling feedback: {e}")
            await self._send_error_message(websocket, "Error processing feedback")

    async def _handle_typing_indicator(
        self, websocket: WebSocket, user_id: str, data: Dict[str, Any]
    ):
        """Handle typing indicator (for future use)."""
        # For now, just acknowledge
        is_typing = data.get("is_typing", False)
        logger.debug(f"User {user_id} typing: {is_typing}")

    async def _send_error_message(self, websocket: WebSocket, error_message: str):
        """Send error message to client."""
        try:
            await websocket.send_json({"type": "error", "message": error_message})
        except Exception as e:
            logger.error(f"Error sending error message: {e}")

    async def broadcast_to_user(self, user_id: str, message: Dict[str, Any]):
        """Broadcast message to specific user if connected."""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to user {user_id}: {e}")
                # Remove stale connection
                self.disconnect(user_id)

    def get_active_users(self) -> List[str]:
        """Get list of currently connected users."""
        return list(self.active_connections.keys())

    def get_user_session_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get session information for a user."""
        return self.user_sessions.get(user_id)


async def websocket_endpoint(
    websocket: WebSocket, user_id: str, ws_manager: WebSocketManager
):
    """
    WebSocket endpoint handler.

    Args:
        websocket: WebSocket connection
        user_id: User identifier
        ws_manager: WebSocket manager instance
    """
    await ws_manager.connect(websocket, user_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            await ws_manager.handle_message(websocket, user_id, data)

    except WebSocketDisconnect:
        ws_manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        ws_manager.disconnect(user_id)
