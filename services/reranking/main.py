"""
Reranking Service - Standalone FastAPI service for document reranking.

Deployed on Host 6 (RTX 2060) in the distributed Morgan architecture.
Provides reranking capabilities for search result refinement.

Model weights are cached to avoid re-downloading on each startup.
"""

import os
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Lazy load heavy imports
_model = None
_model_name = None


def setup_model_cache():
    """
    Setup model cache directories to avoid re-downloading on each startup.
    
    Models are downloaded once to the cache directory and reused later.
    """
    cache_dir = os.getenv("MODEL_CACHE_DIR", "/app/models")
    cache_path = Path(cache_dir)
    
    # Create subdirectories
    st_cache = cache_path / "sentence-transformers"
    hf_cache = cache_path / "huggingface"
    
    for path in [cache_path, st_cache, hf_cache]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables for model caching
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(st_cache)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    
    print(f"Model cache configured at {cache_path}")
    return cache_path


def get_model():
    """Lazy load the CrossEncoder model."""
    global _model, _model_name
    
    if _model is None:
        from sentence_transformers import CrossEncoder
        
        # Self-hosted models (no API key required):
        #   - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, English)
        #   - cross-encoder/ms-marco-MiniLM-L-12-v2 (better quality)
        #   - BAAI/bge-reranker-base (multilingual)
        #   - jinaai/jina-reranker-v1-base-en (Jina, multilingual)
        model_name = os.getenv(
            "RERANKING_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        device = os.getenv("RERANKING_DEVICE", "cuda")
        
        print(f"Loading reranking model: {model_name} on {device}")
        print("(Model downloaded on first run and cached for future use)")
        _model = CrossEncoder(model_name, device=device)
        _model_name = model_name
        print("Model loaded successfully")
    
    return _model


# Setup model cache on module import
_model_cache_path = setup_model_cache()


# FastAPI app
app = FastAPI(
    title="Morgan Reranking Service",
    description="Document reranking service using CrossEncoder models",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class RerankRequest(BaseModel):
    """Reranking request."""
    query: str = Field(..., description="Search query")
    documents: List[str] = Field(..., description="Documents to rerank")
    top_k: Optional[int] = Field(None, description="Return top K results")


class RerankResultItem(BaseModel):
    """Single reranked result."""
    index: int
    text: str
    score: float


class RerankResponse(BaseModel):
    """Reranking response."""
    results: List[RerankResultItem]
    model: str
    elapsed_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    device: str


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    default_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    return HealthResponse(
        status="healthy",
        model=os.getenv("RERANKING_MODEL", default_model),
        device=os.getenv("RERANKING_DEVICE", "cuda"),
    )


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank documents by relevance to a query.
    
    Args:
        request: Reranking request with query and documents
        
    Returns:
        Reranked results sorted by score (descending)
    """
    if not request.documents:
        return RerankResponse(
            results=[],
            model=_model_name or "not_loaded",
            elapsed_ms=0.0,
        )
    
    try:
        start_time = time.time()
        
        # Get model
        model = get_model()
        
        # Create query-document pairs
        pairs = [[request.query, doc] for doc in request.documents]
        
        # Score all pairs
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
        
        # Create results with original indices
        results = []
        for i, (doc, score) in enumerate(zip(request.documents, scores)):
            results.append(RerankResultItem(
                index=i,
                text=doc,
                score=float(score),
            ))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply top_k if specified
        if request.top_k is not None:
            results = results[:request.top_k]
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return RerankResponse(
            results=results,
            model=_model_name,
            elapsed_ms=elapsed_ms,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Morgan Reranking Service",
        "version": "1.0.0",
        "status": "running",
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Pre-load model on startup."""
    print("Reranking service starting...")
    # Optionally pre-load model
    preload = os.getenv("RERANKING_PRELOAD", "true").lower() == "true"
    if preload:
        get_model()


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("RERANKING_HOST", "0.0.0.0")
    port = int(os.getenv("RERANKING_PORT", "8080"))
    
    uvicorn.run(app, host=host, port=port)

