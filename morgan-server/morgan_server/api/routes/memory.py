"""
Memory API Routes

This module implements the memory endpoints for Morgan server:
- GET /api/memory/stats: Get memory statistics
- GET /api/memory/search: Search conversation history
- DELETE /api/memory/cleanup: Clean up old conversations
"""

from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse

from morgan_server.api.models import (
    MemoryStats,
    MemorySearchRequest,
    MemorySearchResult,
    ErrorResponse,
)
from morgan_server.personalization.memory import MemoryManager


router = APIRouter(prefix="/api/memory", tags=["memory"])


# Global memory manager instance (will be injected via dependency injection)
_memory_manager: Optional[MemoryManager] = None


def set_memory_manager(memory_manager: MemoryManager) -> None:
    """
    Set the global memory manager instance.
    
    This should be called during application startup.
    
    Args:
        memory_manager: MemoryManager instance
    """
    global _memory_manager
    _memory_manager = memory_manager


def get_memory_manager() -> MemoryManager:
    """
    Get the global memory manager instance.
    
    Returns:
        MemoryManager instance
        
    Raises:
        HTTPException: If memory manager is not initialized
    """
    if _memory_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory manager not initialized",
        )
    return _memory_manager


@router.get("/stats", response_model=MemoryStats)
async def get_memory_stats(
    user_id: Optional[str] = Query(None, description="User ID to filter stats")
) -> MemoryStats:
    """
    Get memory statistics.
    
    Returns statistics about conversation memory, optionally filtered by user.
    
    Args:
        user_id: Optional user ID to filter statistics
        
    Returns:
        MemoryStats with conversation statistics
        
    Raises:
        HTTPException: If memory manager is not initialized or retrieval fails
    """
    try:
        memory_manager = get_memory_manager()
        
        # Get stats from memory manager
        stats = memory_manager.get_memory_stats(user_id=user_id)
        
        # Convert to API model
        return MemoryStats(
            total_conversations=stats["total_conversations"],
            active_conversations=stats["total_conversations"],  # All conversations are considered active
            total_messages=stats["total_messages"],
            oldest_conversation=(
                datetime.fromisoformat(stats["oldest_conversation"])
                if stats["oldest_conversation"]
                else None
            ),
            newest_conversation=(
                datetime.fromisoformat(stats["newest_conversation"])
                if stats["newest_conversation"]
                else None
            ),
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Error retrieving memory stats: {e}")
        
        # Return structured error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memory stats: {str(e)}",
        )


@router.get("/search", response_model=List[MemorySearchResult])
async def search_memory(
    query: str = Query(..., min_length=1, description="Search query"),
    user_id: Optional[str] = Query(None, description="User ID to search within"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results to return")
) -> List[MemorySearchResult]:
    """
    Search conversation history.
    
    Searches through conversation messages for the given query string.
    Returns matching messages with their context and relevance scores.
    
    Args:
        query: Search query string
        user_id: Optional user ID to limit search scope
        limit: Maximum number of results to return (1-100)
        
    Returns:
        List of MemorySearchResult with matching messages
        
    Raises:
        HTTPException: If memory manager is not initialized, user not found, or search fails
    """
    try:
        memory_manager = get_memory_manager()
        
        # Validate query
        query = query.strip()
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty or whitespace only",
            )
        
        # If no user_id provided, we can't search (need user context)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required for memory search",
            )
        
        # Search conversations
        results = memory_manager.search_conversations(
            user_id=user_id,
            query=query,
            limit=limit
        )
        
        # Convert to API models
        search_results = []
        for conversation, message, score in results:
            # Find the corresponding response (next assistant message)
            response_content = ""
            message_index = conversation.messages.index(message)
            
            # Look for the next assistant message
            for i in range(message_index + 1, len(conversation.messages)):
                if conversation.messages[i].role.value == "assistant":
                    response_content = conversation.messages[i].content
                    break
            
            search_results.append(
                MemorySearchResult(
                    conversation_id=conversation.conversation_id,
                    timestamp=message.timestamp,
                    message=message.content,
                    response=response_content,
                    relevance_score=min(score, 1.0),  # Ensure score is <= 1.0
                )
            )
        
        return search_results
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Error searching memory: {e}")
        
        # Return structured error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search memory: {str(e)}",
        )


@router.delete("/cleanup")
async def cleanup_memory(
    user_id: str = Query(..., description="User ID to clean up conversations for"),
    keep_recent: int = Query(10, ge=1, le=100, description="Number of recent conversations to keep")
) -> JSONResponse:
    """
    Clean up old conversations.
    
    Deletes old conversations for a user, keeping only the most recent ones.
    This helps manage storage and maintain performance.
    
    Args:
        user_id: User ID to clean up conversations for
        keep_recent: Number of recent conversations to keep (1-100)
        
    Returns:
        JSON response with cleanup statistics
        
    Raises:
        HTTPException: If memory manager is not initialized or cleanup fails
    """
    try:
        memory_manager = get_memory_manager()
        
        # Validate user_id
        if not user_id or not user_id.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required and cannot be empty",
            )
        
        # Perform cleanup
        deleted_count = memory_manager.cleanup_old_conversations(
            user_id=user_id,
            keep_recent=keep_recent
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "deleted_count": deleted_count,
                "kept_count": keep_recent,
                "message": f"Cleaned up {deleted_count} old conversations, kept {keep_recent} recent ones",
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Error cleaning up memory: {e}")
        
        # Return structured error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clean up memory: {str(e)}",
        )
