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
from starlette.concurrency import run_in_threadpool

from morgan_server.api.models import (
    LearnRequest,
    LearnResponse,
    KnowledgeStats,
    Source,
)
from morgan_server.api.routes.chat import get_assistant


router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.post("/learn", response_model=LearnResponse)
async def learn_document(request: LearnRequest) -> LearnResponse:
    """
    Add documents to the knowledge base.
    """
    try:
        assistant = get_assistant()
        
        # Validate that at least one source is provided
        if not any([request.source, request.url, request.content]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one of source, url, or content must be provided",
            )
        
        start_time = time.time()
        
        if request.content:
             # Logic for direct content - creating a temp file or passing content directly?
             # KnowledgeService ingest_documents expects source_path.
             # If content is provided, we might need a workaround.
             # For now raising not implemented as before, or trying to handle it.
             raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Direct content learning not yet implemented. Please use source or url.",
            )
        
        source = request.source or request.url
        
        # Determine strict document type if specific, else 'auto'
        doc_type = request.doc_type or "auto"

        # Call knowledge service
        # ingest_documents returns Dict result
        result = await run_in_threadpool(
            assistant.core.knowledge.ingest_documents,
            source_path=source,
            document_type=doc_type,
            collection=request.metadata.get("collection") if request.metadata else None
        )
        
        processing_time = time.time() - start_time
        
        chunks_created = result.get("chunks_created", 0)
        
        return LearnResponse(
            status="success" if result.get("success") else "failed",
            documents_processed=result.get("documents_processed", 0),
            chunks_created=chunks_created,
            processing_time_seconds=processing_time,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error learning document: {e}")
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
    """
    try:
        assistant = get_assistant()
        
        query = query.strip()
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty or whitespace only",
            )
        
        # Perform search
        # search_knowledge returns List[Dict]
        results = await run_in_threadpool(
            assistant.core.knowledge.search_knowledge,
            query=query,
            max_results=limit,
            min_score=score_threshold
        )
        
        # Convert to API models
        api_sources = []
        for res in results:
            api_sources.append(
                Source(
                    content=str(res.get("content", "")),
                    document_id=str(res.get("source", "")), # fallback to source as ID
                    chunk_id=str(res.get("chunk_id", "")), # chunk_id might not be in legacy search result?
                    score=float(res.get("score", 0.0)),
                    metadata=res.get("metadata", {}),
                )
            )
        
        return api_sources
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error searching knowledge: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search knowledge: {str(e)}",
        )


@router.get("/stats", response_model=KnowledgeStats)
async def get_knowledge_stats() -> KnowledgeStats:
    """
    Get knowledge base statistics.
    """
    try:
        assistant = get_assistant()
        
        # Get stats
        stats = await run_in_threadpool(assistant.core.knowledge.get_statistics)
        
        return KnowledgeStats(
            total_documents=stats.get("document_count", 0),
            total_chunks=stats.get("chunk_count", 0),
            total_size_bytes=int(stats.get("storage_size_mb", 0) * 1024 * 1024),
            collections=[stats.get("collection_name", "")]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving knowledge stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve knowledge stats: {str(e)}",
        )
