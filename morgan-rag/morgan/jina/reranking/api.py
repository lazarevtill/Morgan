"""
FastAPI wrapper around the reranking service.

Provides a simple HTTP interface with a lightweight fallback if the full
Jina reranker cannot start (e.g., missing models or tokens).
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from morgan.jina.reranking.service import JinaRerankingService, SearchResult
except Exception:  # pragma: no cover - defensive import
    JinaRerankingService = None
    SearchResult = None

logger = logging.getLogger(__name__)


class RerankRequest(BaseModel):
    """Incoming rerank request."""

    query: str
    results: List[dict]
    top_k: Optional[int] = 5


class RerankResponse(BaseModel):
    """Outgoing rerank response."""

    results: List[dict]


def _load_service() -> Any:
    """Build reranking service or fallback."""
    if JinaRerankingService:
        try:
            return JinaRerankingService(enable_background=False)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("Falling back to simple reranker: %s", exc)
    return None


def _simple_rerank(query: str, results: List[dict], top_k: int) -> List[dict]:
    """Deterministic fallback reranker based on length/score."""
    def score(item: dict) -> float:
        base = float(item.get("score", 0.0))
        content = str(item.get("content", ""))
        # Penalize empty content, reward baseline score and length
        return base + min(len(content) / 200.0, 1.0)

    ranked = sorted(results, key=score, reverse=True)
    return ranked[:top_k]


def create_app() -> FastAPI:
    """Create FastAPI reranking app."""
    app = FastAPI(title="Morgan Reranking Service", version="1.0.0")

    service = _load_service()

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": "jina" if service else "simple"}

    @app.post("/rerank", response_model=RerankResponse)
    async def rerank(payload: RerankRequest):
        top_k = payload.top_k or 5
        if not payload.results:
            raise HTTPException(status_code=400, detail="No results provided")

        # Use real service if available
        if service and SearchResult:
            try:
                sr_items = [
                    SearchResult(
                        content=item.get("content", ""),
                        score=float(item.get("score", 0.0)),
                        metadata=item.get("metadata", {}) or {},
                        source=item.get("source", ""),
                    )
                    for item in payload.results
                ]
                reranked, _scores = service.rerank(payload.query, sr_items, top_k=top_k)
                return RerankResponse(
                    results=[
                        {
                            "content": r.content,
                            "score": r.rerank_score or r.score,
                            "metadata": r.metadata,
                            "source": r.source,
                        }
                        for r in reranked[:top_k]
                    ]
                )
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.warning("Reranking failed, falling back: %s", exc)

        # Fallback
        ranked = _simple_rerank(payload.query, payload.results, top_k)
        return RerankResponse(results=ranked)

    return app


def main():
    """Entrypoint for running via `python -m ...` or uvicorn."""
    import uvicorn

    host = os.getenv("RERANK_HOST", "0.0.0.0")
    port = int(os.getenv("RERANK_PORT", "8081"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
