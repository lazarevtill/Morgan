"""FastAPI app factory for brain-api.

Phase 0: health + a /api/chat route shape. The route returns 501 until the modules are wired
in Phase 1 — the contract and the loop ordering exist; the implementations don't yet.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from morgan_brain import __version__
from morgan_brain.config import get_settings


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="morgan-brain", version=__version__)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__, "event_bus": settings.event_bus}

    @app.post("/api/chat", status_code=501)
    async def chat(_: ChatRequest) -> dict[str, str]:
        # Wired in Phase 1: build Orchestrator(perception, personalizer, memory, skills,
        # reasoner, learner, bus) and return orchestrator.handle_turn(...).
        return {"detail": "not implemented — Phase 1 wires the cognitive loop"}

    return app


app = create_app()
