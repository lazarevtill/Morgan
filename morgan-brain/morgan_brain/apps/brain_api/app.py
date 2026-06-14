"""FastAPI app factory for brain-api.

Phase 1: /api/chat (blocking) drives the cognitive loop.
Phase 5: /api/chat/stream (SSE) streams token deltas with a terminal [DONE] sentinel.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from morgan_brain import __version__
from morgan_brain.apps.brain_api.auth import require_api_key
from morgan_brain.composition import _load_champion_override, build_app_context
from morgan_brain.config import get_settings
from morgan_brain.learning.history import session_key


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    user_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    model_used: str
    turn_id: str


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="morgan-brain", version=__version__)
    ctx = build_app_context(settings)
    orchestrator = ctx.orchestrator
    _auth = Depends(require_api_key(settings))

    # Read the current champion preprompt once at startup (best-effort: "" if none).
    # The champion is written by the learning-worker and is the gated, eval-validated
    # system prompt candidate — zero inference-time overhead (just a string prepend).
    _champion_override: str = _load_champion_override(ctx.prompt_registry)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__, "event_bus": settings.event_bus}

    @app.post("/api/chat", response_model=ChatResponse, dependencies=[_auth])
    async def chat(req: ChatRequest) -> ChatResponse:
        user_id = req.user_id or settings.owner_user_id
        hkey = session_key(user_id, req.session_id)
        history = ctx.history_store.recent(hkey) if ctx.history_store else []
        result, turn_id = await orchestrator.handle_turn_with_id(
            user_id=user_id,
            text=req.message,
            session_id=req.session_id,
            history=history,
            system_override=_champion_override,
        )
        return ChatResponse(response=result.text, model_used=result.model_used, turn_id=turn_id)

    @app.post("/api/chat/stream", dependencies=[_auth])
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        """SSE stream of token deltas.

        Each token is emitted as ``data: <json>\\n\\n`` where the JSON object is
        ``{"delta": "<text>"}``.  The stream ends with ``data: [DONE]\\n\\n``.
        Clients should treat ``[DONE]`` as the end-of-stream sentinel (OpenAI convention).
        """
        user_id = req.user_id or settings.owner_user_id

        hkey = session_key(user_id, req.session_id)

        async def _event_stream() -> AsyncIterator[str]:
            history = ctx.history_store.recent(hkey) if ctx.history_store else []
            async for delta in orchestrator.stream_turn(
                user_id=user_id,
                text=req.message,
                session_id=req.session_id,
                history=history,
                system_override=_champion_override,
            ):
                payload = json.dumps({"delta": delta})
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    # Mount the read/feedback router built from the app context
    from morgan_brain.apps.brain_api.routes import build_router

    app.include_router(
        build_router(
            orchestrator=orchestrator,
            signal_recorder=ctx.signal_recorder,
            executor=ctx.executor,
            skills=ctx.skills,
            learner=ctx.learner,
            settings=settings,
        )
    )

    return app


app = create_app()
