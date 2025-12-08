"""API Routes."""

from morgan_server.api.routes.chat import router as chat_router
from morgan_server.api.routes.memory import router as memory_router
from morgan_server.api.routes.knowledge import router as knowledge_router
from morgan_server.api.routes.health import router as health_router

__all__ = [
    "chat_router",
    "memory_router",
    "knowledge_router",
    "health_router",
]
