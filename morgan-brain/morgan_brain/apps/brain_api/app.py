"""FastAPI app factory for brain-api. Phase 1: /api/chat drives the cognitive loop."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from morgan_brain import __version__
from morgan_brain.composition import build_orchestrator
from morgan_brain.config import get_settings


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    user_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    model_used: str


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="morgan-brain", version=__version__)
    orchestrator = build_orchestrator(settings)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__, "event_bus": settings.event_bus}

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        user_id = req.user_id or settings.owner_user_id
        result = await orchestrator.handle_turn(
            user_id=user_id, text=req.message, session_id=req.session_id
        )
        return ChatResponse(response=result.text, model_used=result.model_used)

    return app


app = create_app()
