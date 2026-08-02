"""Feedback + read API router (commit 3).

Exposes:
  POST /api/feedback     — thumb / edit / retry signal accumulation
  GET  /api/tools        — list registered tools
  POST /api/tools/{name} — execute a tool
  GET  /api/skills       — list loaded skills
  POST /api/skills/{name} — fetch a skill body
  GET  /api/profile      — get user profile (UserModel)

All routes share the same ``require_api_key`` dependency as /api/chat.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from morgan_brain.apps.brain_api.auth import require_api_key
from morgan_brain.config import Settings
from morgan_brain.learning.recorder import SignalRecorder
from morgan_brain.learning.signals import Thumb
from morgan_brain.modules.tools.executor import ToolExecutorImpl
from morgan_brain.modules.skills.registry import SkillRegistry


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    turn_id: str
    project: str
    user_id: str | None = None
    kind: str  # "edit" | "retry" | "thumb"
    edited_reply: str | None = None
    thumb: str | None = None  # "up" | "down"


class FeedbackResponse(BaseModel):
    ok: bool


_log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def build_router(
    *,
    orchestrator: Any,
    signal_recorder: SignalRecorder,
    executor: ToolExecutorImpl,
    skills: SkillRegistry,
    learner: Any,
    settings: Settings,
) -> APIRouter:
    """Return an APIRouter with feedback + read endpoints wired to the given handles."""
    router = APIRouter()
    _auth = Depends(require_api_key(settings))

    @router.post("/api/feedback", response_model=FeedbackResponse, dependencies=[_auth])
    async def feedback(req: FeedbackRequest) -> FeedbackResponse:
        user_id = req.user_id or settings.owner_user_id
        kind = req.kind.lower()
        if kind == "edit":
            if req.edited_reply is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="edited_reply required for kind='edit'",
                )
            await signal_recorder.add_edit(
                turn_id=req.turn_id, user_id=user_id, edited_reply=req.edited_reply
            )
            # Cold-path best-effort: learn a durable preference from this edit so future
            # turns reflect it (CIPHER → agent-inferred comm_* facts). Learning must never
            # fail the feedback request.
            try:
                original = await signal_recorder.original_reply_for(
                    user_id=user_id, turn_id=req.turn_id
                )
                if original:
                    await learner.learn_from_edit(
                        user_id=user_id,
                        project=req.project,
                        original=original,
                        edited=req.edited_reply,
                    )
            except Exception as exc:  # noqa: BLE001 — best-effort learning hook
                _log.warning("learn_from_edit_failed", turn_id=req.turn_id, error=str(exc))
        elif kind == "retry":
            await signal_recorder.add_retry(turn_id=req.turn_id, user_id=user_id)
        elif kind == "thumb":
            if req.thumb is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="thumb required for kind='thumb'",
                )
            try:
                thumb_val = Thumb(req.thumb.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"Invalid thumb value '{req.thumb}'; must be 'up' or 'down'.",
                )
            await signal_recorder.add_thumb(turn_id=req.turn_id, user_id=user_id, thumb=thumb_val)
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Unknown feedback kind '{req.kind}'; must be 'edit', 'retry', or 'thumb'.",
            )
        return FeedbackResponse(ok=True)

    @router.get("/api/tools", dependencies=[_auth])
    async def list_tools() -> list[dict[str, Any]]:
        return executor.list()

    @router.post("/api/tools/{name}", dependencies=[_auth])
    async def run_tool(name: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = body or {}
        result = await executor.execute(name, user_id=settings.owner_user_id, **kwargs)
        return {"ok": result.ok, "output": result.output, "error": result.error}

    @router.get("/api/skills", dependencies=[_auth])
    async def list_skills() -> list[dict[str, Any]]:
        return [
            {"name": s.name, "triggers": s.triggers, "version": s.version}
            for s in skills.list_skills()
        ]

    @router.post("/api/skills/{name}", dependencies=[_auth])
    async def get_skill(name: str) -> dict[str, Any]:
        skill = await skills.get(name)
        if skill is None:
            raise HTTPException(status_code=404, detail=f"Skill '{name}' not found.")
        return {"name": skill.name, "triggers": skill.triggers, "body": skill.body}

    @router.get("/api/profile", dependencies=[_auth])
    async def get_profile(user_id: str | None = None) -> dict[str, Any]:
        uid = user_id or settings.owner_user_id
        model = await learner.user_model(uid)
        result: dict[str, Any] = model.model_dump()
        return result

    return router
