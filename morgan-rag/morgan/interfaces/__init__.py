"""
Morgan Interfaces Package.

Provides network interfaces for:
- REST API (FastAPI)
- WebSocket real-time communication
- Streaming responses
- Health monitoring
"""

from morgan.interfaces.web_interface import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MorganWebApp,
    create_app,
)
from morgan.interfaces.websocket_interface import (
    ConnectionManager,
    MessageType,
    WebSocketHandler,
    WSMessage,
    add_websocket_routes,
)

__all__ = [
    # Web Interface
    "MorganWebApp",
    "create_app",
    "ChatRequest",
    "ChatResponse",
    "ErrorResponse",
    "FeedbackRequest",
    "FeedbackResponse",
    "HealthResponse",
    # WebSocket Interface
    "WebSocketHandler",
    "ConnectionManager",
    "MessageType",
    "WSMessage",
    "add_websocket_routes",
]
