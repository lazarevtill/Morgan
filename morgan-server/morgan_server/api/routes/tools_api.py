"""
API routes for the new Morgan modules (tools, skills, tasks, channels, workspace).

Task 13: Integration Wiring — exposes the 12 new modules via REST endpoints.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Modules"])


# ============================================================================
# Request / response models
# ============================================================================


class ToolExecuteRequest(BaseModel):
    """Request body for POST /api/tools/{name}."""

    input: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None


class SkillExecuteRequest(BaseModel):
    """Request body for POST /api/skills/{name}."""

    variables: Dict[str, str] = Field(default_factory=dict)
    prompt_context: Optional[str] = None


class ScheduleAddRequest(BaseModel):
    """Request body for POST /api/schedule."""

    schedule: str
    prompt: str
    channel: str = "system"
    model: str = "default"
    isolated: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Module-level caches so we don't rebuild on every request.
_cached_tool_executor = None
_cached_skill_registry = None


def _build_tool_executor():
    global _cached_tool_executor
    if _cached_tool_executor is not None:
        return _cached_tool_executor

    from morgan.tools import ToolExecutor
    from morgan.tools.builtin import ALL_BUILTIN_TOOLS

    executor = ToolExecutor()
    for tool in ALL_BUILTIN_TOOLS:
        executor.register(tool)
    _cached_tool_executor = executor
    return executor


def _build_skill_registry(request: Optional[Request] = None):
    global _cached_skill_registry
    if _cached_skill_registry is not None:
        return _cached_skill_registry

    from morgan.skills import SkillRegistry, load_skills_from_dir

    registry = SkillRegistry()

    bundled_dir = (
        Path(__file__).resolve().parents[4]
        / "morgan-rag"
        / "morgan"
        / "skills"
        / "bundled"
    )
    if bundled_dir.exists():
        for skill in load_skills_from_dir(str(bundled_dir)):
            registry.register(skill)

    if request is not None:
        ws_mgr = getattr(request.app.state, "workspace_manager", None)
        if ws_mgr is not None:
            workspace_skills_dir = ws_mgr._dir / "skills"
            if workspace_skills_dir.exists():
                for skill in load_skills_from_dir(str(workspace_skills_dir)):
                    registry.register(skill)

    _cached_skill_registry = registry
    return registry


# ============================================================================
# Tools
# ============================================================================


@router.get("/tools")
async def list_tools() -> List[Dict[str, Any]]:
    """List available tools with their schemas."""
    try:
        executor = _build_tool_executor()
        return [
            {
                "name": t.name,
                "description": t.description,
                "aliases": list(t.aliases),
                "input_schema": t.input_schema.to_dict(),
            }
            for t in executor.list_tools()
        ]
    except ImportError:
        return []
    except Exception as exc:
        logger.warning("Failed to list tools: %s", exc)
        return []


@router.post("/tools/{name}")
async def execute_tool(name: str, body: ToolExecuteRequest) -> Dict[str, Any]:
    """Execute a tool by name."""
    try:
        from morgan.tools.base import ToolContext

        executor = _build_tool_executor()

        tool = executor.get_tool(name)
        if tool is None:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")

        ctx = ToolContext(
            user_id=body.user_id or "api",
            conversation_id=f"api:tools:{name}",
        )
        result = await executor.execute(
            tool_name=name,
            input_data=body.input,
            context=ctx,
        )
        return {
            "output": result.output,
            "error_code": result.error_code,
            "is_error": result.is_error,
            "metadata": result.metadata,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# Skills
# ============================================================================


@router.get("/skills")
async def list_skills() -> List[Dict[str, Any]]:
    """List available skills."""
    try:
        registry = _build_skill_registry()

        return [
            {
                "name": s.name,
                "description": s.description,
                "user_invocable": s.user_invocable,
                "argument_names": list(getattr(s, "argument_names", [])),
                "allowed_tools": list(getattr(s, "allowed_tools", [])),
                "model": getattr(s, "model", None),
                "effort": getattr(s, "effort", None),
            }
            for s in registry.list_all()
        ]
    except ImportError:
        return []
    except Exception as exc:
        logger.warning("Failed to list skills: %s", exc)
        return []


@router.post("/skills/{name}")
async def execute_skill(name: str, body: SkillExecuteRequest) -> Dict[str, Any]:
    """Run a skill by name."""
    try:
        from morgan.skills import SkillExecutor

        registry = _build_skill_registry()

        skill = registry.get(name)
        if skill is None:
            raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")

        executor = SkillExecutor()
        result = await executor.execute(
            skill=skill,
            args=body.variables,
            context=body.prompt_context,
        )

        return {
            "name": skill.name,
            "rendered_prompt": skill.get_prompt(body.variables),
            "status": "ok" if result.success else "error",
            "agent_type": result.agent_type,
            "output": result.output,
            "error": result.error,
            "metadata": result.metadata,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# Tasks
# ============================================================================


@router.get("/tasks")
async def list_tasks(request: Request) -> List[Dict[str, Any]]:
    """List background tasks."""
    task_mgr = getattr(request.app.state, "task_manager", None)
    if task_mgr is None:
        return []
    return [
        {
            "task_id": t.task_id,
            "task_type": t.task_type.value if hasattr(t.task_type, "value") else str(t.task_type),
            "description": t.description,
            "status": t.status.value if hasattr(t.status, "value") else str(t.status),
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "updated_at": t.updated_at.isoformat() if t.updated_at else None,
        }
        for t in task_mgr.list_tasks()
    ]


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request) -> Dict[str, Any]:
    """Get task status by ID."""
    task_mgr = getattr(request.app.state, "task_manager", None)
    if task_mgr is None:
        raise HTTPException(status_code=503, detail="TaskManager not initialized")
    task = task_mgr.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return {
        "task_id": task.task_id,
        "task_type": task.task_type.value if hasattr(task.task_type, "value") else str(task.task_type),
        "description": task.description,
        "status": task.status.value if hasattr(task.status, "value") else str(task.status),
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at else None,
        "result": task.result,
        "error": task.error,
    }


# ============================================================================
# Channels
# ============================================================================


@router.get("/channels")
async def list_channels(request: Request) -> List[Dict[str, Any]]:
    """List active channels."""
    gateway = getattr(request.app.state, "channel_gateway", None)
    if gateway is None:
        return []
    return [
        {"name": name, "type": type(ch).__name__}
        for name, ch in gateway.channels.items()
    ]


# ============================================================================
# Workspace
# ============================================================================


@router.get("/workspace")
async def get_workspace_status(request: Request) -> Dict[str, Any]:
    """Get workspace status."""
    ws_mgr = getattr(request.app.state, "workspace_manager", None)
    if ws_mgr is None:
        return {"status": "disabled"}
    try:
        return {
            "status": "active",
            "path": str(ws_mgr._dir),
            "has_soul": (ws_mgr._dir / "SOUL.md").exists(),
            "has_user": (ws_mgr._dir / "USER.md").exists(),
            "has_memory": (ws_mgr._dir / "MEMORY.md").exists(),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@router.post("/workspace/consolidate")
async def trigger_memory_consolidation(request: Request) -> Dict[str, Any]:
    """Trigger memory consolidation."""
    ws_mgr = getattr(request.app.state, "workspace_manager", None)
    if ws_mgr is None:
        raise HTTPException(status_code=503, detail="Workspace not initialized")
    try:
        from morgan.memory_consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(workspace_dir=ws_mgr._dir)
        result = consolidator.consolidate(days_to_review=7)
        return {"status": "ok", "updated_memory": bool(result), "result": result}
    except ImportError:
        raise HTTPException(
            status_code=503, detail="Memory consolidation module not available"
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# Schedule
# ============================================================================


@router.get("/schedule")
async def list_schedule(request: Request) -> Dict[str, Any]:
    """List registered cron jobs."""
    cron_service = getattr(request.app.state, "cron_service", None)
    if cron_service is None:
        return {"enabled": False, "jobs": []}

    jobs = []
    for job in cron_service.list_jobs():
        jobs.append(
            {
                "job_id": job.job_id,
                "schedule": job.schedule,
                "prompt": job.prompt,
                "channel": job.channel,
                "model": job.model,
                "isolated": job.isolated,
                "metadata": job.metadata,
            }
        )
    return {"enabled": True, "jobs": jobs}


@router.post("/schedule")
async def add_schedule(request: Request, body: ScheduleAddRequest) -> Dict[str, Any]:
    """Register a cron job."""
    cron_service = getattr(request.app.state, "cron_service", None)
    if cron_service is None:
        raise HTTPException(status_code=503, detail="Scheduling is not enabled")

    try:
        from morgan.scheduling import CronJob

        job = CronJob(
            job_id=f"job-{uuid.uuid4().hex[:12]}",
            schedule=body.schedule,
            prompt=body.prompt,
            channel=body.channel,
            model=body.model,
            isolated=body.isolated,
            metadata=dict(body.metadata),
        )
        cron_service.add_job(job)

        if getattr(cron_service, "_started", False) and getattr(
            cron_service, "_scheduler", None
        ) is not None:
            cron_service._register_ap_job(job)

        hook_manager = getattr(request.app.state, "hook_manager", None)
        if hook_manager is not None:
            from morgan.hook_system import HookType

            await hook_manager.trigger(
                HookType.CONFIG_CHANGE,
                {
                    "component": "schedule",
                    "action": "add_job",
                    "job_id": job.job_id,
                },
            )

        return {"status": "ok", "job_id": job.job_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
