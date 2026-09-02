"""FastAPI app factory for brain-api.

Phase 1: /api/chat (blocking) drives the cognitive loop.
Phase 5: /api/chat/stream (SSE) streams token deltas with a terminal [DONE] sentinel.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from morgan_brain import __version__
from morgan_brain.apps.brain_api.auth import require_api_key
from morgan_brain.composition import AppContext, ChampionCache, build_app_context
from morgan_brain.config import Settings, get_settings
from morgan_brain.interfaces.llm import ProviderUnreachable
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


def create_app(settings: Settings | None = None, ctx: AppContext | None = None) -> FastAPI:
    """Build the gateway over *ctx*, or over the production wiring when none is given.

    Accepting a context is what lets a test run the real routes -- the exception mapping,
    the SSE framing, the champion cache -- over fakes, instead of a copy of them.
    """
    settings = settings or get_settings()
    ctx = ctx or build_app_context(settings)
    app = FastAPI(title="morgan-brain", version=__version__, lifespan=_lifespan)
    app.state.ctx = ctx
    orchestrator = ctx.orchestrator
    _auth = Depends(require_api_key(settings))

    @app.exception_handler(ProviderUnreachable)
    async def _provider_unreachable(_: Request, exc: ProviderUnreachable) -> JSONResponse:
        # The model server is a separate upstream; its absence is a bad gateway, not an
        # internal error, and the body says which endpoint so the owner can go look.
        return JSONResponse(
            status_code=502,
            content={"error": str(exc), "endpoint": exc.endpoint},
        )

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
        history = ctx.history_store.recent(hkey, project=req.project) if ctx.history_store else []
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
        history = ctx.history_store.recent(hkey, project=req.project) if ctx.history_store else []
        deltas = orchestrator.stream_turn(
            user_id=user_id,
            project=req.project,
            text=req.message,
            session_id=req.session_id,
            history=history,
            system_override=await _champion.body(),
        )
        # Pull the first delta before committing to a 200: the pre-reasoning pipeline and the
        # connection to the model server both happen here, and a model server that is down
        # must be a 502 the client can act on, not an empty stream that looks like an answer
        # with no words in it. Once bytes are out, the status is fixed -- a failure after
        # that is reported in-band, as an error event before the terminal sentinel.
        first = await anext(deltas, None)

        async def _event_stream() -> AsyncIterator[str]:
            if first is not None:
                yield f"data: {json.dumps({'delta': first})}\n\n"
                try:
                    async for delta in deltas:
                        yield f"data: {json.dumps({'delta': delta})}\n\n"
                except ProviderUnreachable as exc:
                    yield f"data: {json.dumps({'error': str(exc), 'endpoint': exc.endpoint})}\n\n"
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
