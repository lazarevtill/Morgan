"""
Knowledge API Routes

This module implements the knowledge endpoints for Morgan server:
- POST /api/knowledge/learn: Add documents to knowledge base
- GET /api/knowledge/search: Search knowledge base
- GET /api/knowledge/stats: Get knowledge base statistics
"""

import time
from typing import Optional, List
from fastapi import APIRouter, HTTPException, status, Query

from morgan_server.api.models import (
    LearnRequest,
    LearnResponse,
    KnowledgeStats,
    Source,
)
from morgan_server.knowledge.rag import RAGSystem


router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


# Global RAG system instance (will be injected via dependency injection)
_rag_system: Optional[RAGSystem] = None


def set_rag_system(rag_system: RAGSystem) -> None:
    """
    Set the global RAG system instance.
    
    This should be called during application startup.
    
    Args:
        rag_system: RAGSystem instance
    """
    global _rag_system
    _rag_system = rag_system


def get_rag_system() -> RAGSystem:
    """
    Get the global RAG system instance.
    
    Returns:
        RAGSystem instance
        
    Raises:
        HTTPException: If RAG system is not initialized
    """
    if _rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized",
        )
    return _rag_system


@router.post("/learn", response_model=LearnResponse)
async def learn_document(request: LearnRequest) -> LearnResponse:
    """
    Add documents to the knowledge base.
    
    Processes and indexes documents from various sources (files, URLs, or direct content)
    into the knowledge base for later retrieval.
    
    Args:
        request: LearnRequest with document source and metadata
        
    Returns:
        LearnResponse with processing statistics
        
    Raises:
        HTTPException: If RAG system is not initialized or learning fails
    """
    try:
        rag_system = get_rag_system()
        
        # Validate that at least one source is provided
        if not any([request.source, request.url, request.content]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one of source, url, or content must be provided",
            )
        
        start_time = time.time()
        
        # Determine the source to use
        if request.content:
            # For direct content, we need to create a temporary file or handle it differently
            # For now, we'll raise an error as this requires special handling
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Direct content learning not yet implemented. Please use source or url.",
            )
        
        source = request.source or request.url
        
        # Index the document
        chunks_created = await rag_system.index_document(
            source=source,
            doc_type=request.doc_type,
            metadata=request.metadata,
        )
        
        processing_time = time.time() - start_time
        
        return LearnResponse(
            status="success" if chunks_created > 0 else "no_changes",
            documents_processed=1 if chunks_created > 0 else 0,
            chunks_created=chunks_created,
            processing_time_seconds=processing_time,
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Error learning document: {e}")
        
        # Return structured error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to learn document: {str(e)}",
        )


@router.get("/search", response_model=List[Source])
async def search_knowledge(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(5, ge=1, le=50, description="Maximum results to return"),
    score_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum relevance score")
) -> List[Source]:
    """
    Search the knowledge base.
    
    Performs semantic search over the knowledge base to find relevant documents
    matching the query.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return (1-50)
        score_threshold: Minimum relevance score (0.0-1.0)
        
    Returns:
        List of Source objects with matching documents
        
    Raises:
        HTTPException: If RAG system is not initialized or search fails
    """
    try:
        rag_system = get_rag_system()
        
        # Validate query
        query = query.strip()
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty or whitespace only",
            )
        
        # Perform search
        sources = await rag_system.search_similar(
            query=query,
            limit=limit,
            filter_conditions=None,
        )
        
        # Filter by score threshold
        sources = [s for s in sources if s.score >= score_threshold]
        
        # Convert to API models
        api_sources = []
        for source in sources:
            api_sources.append(
                Source(
                    content=source.content,
                    document_id=source.document_id,
                    chunk_id=source.chunk_id,
                    score=source.score,
                    metadata=source.metadata,
                )
            )
        
        return api_sources
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Error searching knowledge: {e}")
        
        # Return structured error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search knowledge: {str(e)}",
        )


@router.get("/stats", response_model=KnowledgeStats)
async def get_knowledge_stats() -> KnowledgeStats:
    """
    Get knowledge base statistics.
    
    Returns statistics about the knowledge base including document counts,
    chunk counts, and available collections.
    
    Returns:
        KnowledgeStats with knowledge base statistics
        
    Raises:
        HTTPException: If RAG system is not initialized or retrieval fails
    """
    try:
        rag_system = get_rag_system()
        
        # Get stats from RAG system
        stats = await rag_system.get_stats()
        
        # Convert to API model
        return KnowledgeStats(
            total_documents=0,  # RAG system doesn't track documents separately
            total_chunks=stats.get("total_chunks", 0),
            total_size_bytes=0,  # Not tracked yet
            collections=[stats.get("collection_name", "knowledge_base")],
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log error (in production, use proper logging)
        print(f"Error retrieving knowledge stats: {e}")
        
        # Return structured error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve knowledge stats: {str(e)}",
        )
