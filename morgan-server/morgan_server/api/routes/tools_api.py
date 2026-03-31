"""
API routes for the new Morgan modules (tools, skills, tasks, channels, workspace).

Task 13: Integration Wiring — exposes the 12 new modules via REST endpoints.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Modules"])


# ============================================================================
# Request / response models
# ============================================================================


class ToolExecuteRequest(BaseModel):
    """Request body for POST /api/tools/{name}."""

    input: Dict[str, Any] = {}
    user_id: Optional[str] = None


class SkillExecuteRequest(BaseModel):
    """Request body for POST /api/skills/{name}."""

    variables: Dict[str, str] = {}


# ============================================================================
# Tools
# ============================================================================


@router.get("/tools")
async def list_tools() -> List[Dict[str, Any]]:
    """List available tools with their schemas."""
    try:
        from morgan.tools import ToolExecutor
        from morgan.tools.builtin import ALL_BUILTIN_TOOLS

        executor = ToolExecutor()
        for tool in ALL_BUILTIN_TOOLS:
            executor.register(tool)

        result = []
        for tool in executor.list_tools():
            schema = None
            if tool.input_schema:
                schema = tool.input_schema.to_dict() if hasattr(tool.input_schema, "to_dict") else None
            result.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "aliases": list(tool.aliases) if tool.aliases else [],
                    "input_schema": schema,
                }
            )
        return result
    except ImportError:
        return []
    except Exception as exc:
        logger.warning("Failed to list tools: %s", exc)
        return []


@router.post("/tools/{name}")
async def execute_tool(name: str, body: ToolExecuteRequest) -> Dict[str, Any]:
    """Execute a tool by name."""
    try:
        from morgan.tools import ToolContext, ToolExecutor
        from morgan.tools.builtin import ALL_BUILTIN_TOOLS
        from morgan.tools.permissions import PermissionContext, PermissionMode

        executor = ToolExecutor()
        for tool in ALL_BUILTIN_TOOLS:
            executor.register(tool)

        tool = executor.get_tool(name)
        if tool is None:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")

        result = await executor.execute(
            tool_name=name,
            input_data=body.input,
            context=ToolContext(user_id=body.user_id or "api"),
            permission_context=PermissionContext(mode=PermissionMode.BYPASS),
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
        from morgan.skills import SkillRegistry, load_skills_from_dir
        from morgan.workspace import get_workspace_path

        registry = SkillRegistry()

        # Load bundled skills
        try:
            import importlib.resources as pkg_resources

            bundled_dir = (
                __import__("pathlib").Path(__file__).resolve().parents[4]
                / "morgan-rag"
                / "morgan"
                / "skills"
                / "bundled"
            )
            if bundled_dir.exists():
                for skill in load_skills_from_dir(str(bundled_dir)):
                    registry.register(skill)
        except Exception:
            pass

        return [
            {
                "name": s.name,
                "description": s.description,
                "user_invocable": s.user_invocable,
                "argument_names": getattr(s, "argument_names", []),
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
        from morgan.skills import SkillExecutor, SkillRegistry, load_skills_from_dir

        registry = SkillRegistry()

        # Load bundled skills
        try:
            bundled_dir = (
                __import__("pathlib").Path(__file__).resolve().parents[4]
                / "morgan-rag"
                / "morgan"
                / "skills"
                / "bundled"
            )
            if bundled_dir.exists():
                for skill in load_skills_from_dir(str(bundled_dir)):
                    registry.register(skill)
        except Exception:
            pass

        skill = registry.get(name)
        if skill is None:
            raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")

        return {
            "name": skill.name,
            "rendered_prompt": skill.get_prompt(body.variables),
            "status": "rendered",
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
        result = consolidator.consolidate()
        return {"status": "ok", "result": result}
    except ImportError:
        raise HTTPException(
            status_code=503, detail="Memory consolidation module not available"
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
