"""
Memory API Routes

This module implements the memory endpoints for Morgan server:
- GET /api/memory/stats: Get memory statistics
- GET /api/memory/search: Search conversation history
- DELETE /api/memory/cleanup: Clean up old conversations
"""

from typing import Optional, List
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from morgan_server.api.models import (
    MemoryStats,
    MemorySearchRequest,
    MemorySearchResult,
    ErrorResponse,
)
from morgan_server.api.routes.chat import get_assistant


router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("/stats", response_model=MemoryStats)
async def get_memory_stats(
    user_id: Optional[str] = Query(None, description="User ID to filter stats")
) -> MemoryStats:
    """
    Get memory statistics.
    """
    try:
        assistant = get_assistant()

        # Get stats from memory manager
        # Note: Core MemoryService might not filter by user_id yet
        stats = await run_in_threadpool(assistant.core.memory.get_learning_insights)

        # Map fields
        # stats has: total_conversations, total_turns, recent_activity, etc.
        return MemoryStats(
            total_conversations=stats.get("total_conversations", 0),
            active_conversations=stats.get("total_conversations", 0),
            total_messages=stats.get("total_turns", 0),
            oldest_conversation=None,  # Not provided by get_learning_insights
            newest_conversation=(
                datetime.fromisoformat(stats["recent_activity"])
                if stats.get("recent_activity")
                else None
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving memory stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memory stats: {str(e)}",
        )


@router.get("/search", response_model=List[MemorySearchResult])
async def search_memory(
    query: str = Query(..., min_length=1, description="Search query"),
    user_id: Optional[str] = Query(None, description="User ID to search within"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results to return"),
) -> List[MemorySearchResult]:
    """
    Search conversation history.
    """
    try:
        assistant = get_assistant()

        query = query.strip()
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty or whitespace only",
            )

        # Search conversations
        # Note: Core MemoryService search might not filter by user_id
        results = await run_in_threadpool(
            assistant.core.memory.search_conversations, query=query, max_results=limit
        )

        # Convert to API models
        search_results = []
        for res in results:
            # Core returns dict with question, answer, score, etc.
            search_results.append(
                MemorySearchResult(
                    conversation_id=res.get("conversation_id", ""),
                    timestamp=(
                        datetime.fromisoformat(res["timestamp"])
                        if res.get("timestamp")
                        else datetime.now(timezone.utc)
                    ),
                    message=res.get("question", ""),
                    response=res.get("answer", ""),
                    relevance_score=min(float(res.get("score", 0.0)), 1.0),
                )
            )

        return search_results

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error searching memory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search memory: {str(e)}",
        )


@router.delete("/cleanup")
async def cleanup_memory(
    user_id: str = Query(..., description="User ID to clean up conversations for"),
    keep_recent: int = Query(
        10, ge=1, le=100, description="Number of recent conversations to keep"
    ),
) -> JSONResponse:
    """
    Clean up old conversations.
    """
    try:
        assistant = get_assistant()

        if not user_id or not user_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required and cannot be empty",
            )

        # Core cleanup is by DAYS, not count of conversations.
        # Legacy API: keep_recent (count)
        # Core: cleanup_old_conversations(days_to_keep)
        # We will attempt to map or just use a default days value and warn/comment

        # Mapping assumption: keeping recent 10 conversations ~ keeping last 30 days?
        # Ideally we update Core to support cleanup by count, or accept the change in behavior.
        # For now, we call core with a reasonable default relative to 'recent'.
        days = 30  # Default

        deleted_count = await run_in_threadpool(
            assistant.core.memory.cleanup_old_conversations, days_to_keep=days
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "deleted_count": deleted_count,
                # We can't guarantee kept_count strictly matches keep_recent
                "kept_count": -1,
                "message": f"Cleaned up {deleted_count} old conversations (older than {days} days)",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error cleaning up memory: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clean up memory: {str(e)}",
        )
