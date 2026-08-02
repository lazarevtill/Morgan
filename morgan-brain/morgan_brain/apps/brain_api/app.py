"""FastAPI app factory for brain-api.

Phase 1: /api/chat (blocking) drives the cognitive loop.
Phase 5: /api/chat/stream (SSE) streams token deltas with a terminal [DONE] sentinel.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from morgan_brain import __version__
from morgan_brain.apps.brain_api.auth import require_api_key
from morgan_brain.composition import AppContext, ChampionCache, build_app_context
from morgan_brain.config import get_settings
from morgan_brain.learning.history import session_key


class ChatRequest(BaseModel):
    message: str
    project: str
    session_id: str | None = None
    user_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    model_used: str
    turn_id: str


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the event bus on entry, stop it on exit.

    Before this, nothing called ``bus.start()`` anywhere in the API process, so queued
    cold-path work (consolidation, signal mining) would never run even though every
    subscriber was registered.
    """
    ctx: AppContext = app.state.ctx
    await ctx.bus.start()
    try:
        yield
    finally:
        await ctx.bus.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    ctx = build_app_context(settings)
    app = FastAPI(title="morgan-brain", version=__version__, lifespan=_lifespan)
    app.state.ctx = ctx
    orchestrator = ctx.orchestrator
    _auth = Depends(require_api_key(settings))

    # Live champion preprompt: the learning-worker promotes the gated, eval-validated
    # system prompt into the shared registry; the cache refreshes on a short TTL so a
    # promotion reaches live traffic without a brain-api restart (zero inference-time
    # overhead — just a string prepend, read at most once per TTL).
    _champion = ChampionCache(ctx.prompt_registry)

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
            project=req.project,
            text=req.message,
            session_id=req.session_id,
            history=history,
            system_override=await _champion.body(),
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
            champion = await _champion.body()
            async for delta in orchestrator.stream_turn(
                user_id=user_id,
                project=req.project,
                text=req.message,
                session_id=req.session_id,
                history=history,
                system_override=champion,
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
