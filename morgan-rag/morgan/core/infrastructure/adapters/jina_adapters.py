"""
Jina AI Adapters for Morgan Infrastructure.
Implements domain interfaces using Jina services.
"""

from typing import Any, Dict, List, Optional
from morgan.core.domain.interfaces import (
    EmbeddingProvider,
    RerankerProvider,
    ScraperProvider,
)
from morgan.jina.embeddings.service import JinaEmbeddingService
from morgan.jina.reranking.service import JinaRerankingService
from morgan.jina.scraping.service import JinaWebScrapingService


class JinaEmbeddingAdapter(EmbeddingProvider):
    """Adapter for Jina Embedding Service."""

    def __init__(
        self, jina_service: JinaEmbeddingService, model_name: str = "jina-embeddings-v4"
    ):
        self.jina = jina_service
        self.model_name = model_name

    async def embed_texts(
        self, texts: List[str], instruction: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings using Jina service."""
        return await self.jina.generate_embeddings_async(texts, self.model_name)

    def get_dimension(self) -> int:
        """Get embedding dimension from model info."""
        info = self.jina.get_model_info(self.model_name)
        return info.get("dimensions", 768)


class JinaRerankerAdapter(RerankerProvider):
    """Adapter for Jina Reranking Service."""

    def __init__(self, jina_service: JinaRerankingService):
        self.jina = jina_service

    async def rerank(
        self, query: str, documents: List[str], top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank documents using Jina service."""
        from morgan.jina.reranking.service import SearchResult

        # Convert documents to SearchResult objects
        results = [
            SearchResult(content=doc, score=0.0, metadata={}, source="unknown")
            for doc in documents
        ]

        # JinaRerankingService.rerank_results is synchronous, wrap in executor if needed
        # For simplicity in this adapter, we call it directly (assuming thread usage in service)
        reranked, metrics = self.jina.rerank_results(query, results, top_k=top_n)

        return [
            {
                "content": r.content,
                "score": r.rerank_score or r.score,
                "metadata": r.metadata,
            }
            for r in reranked
        ]


class JinaScraperAdapter(ScraperProvider):
    """Adapter for Jina Web Scraping Service."""

    def __init__(self, jina_service: JinaWebScrapingService):
        self.jina = jina_service

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape URL using Jina service."""
        # Wrap synchronous scrape in run_in_executor if necessary
        return self.jina.scrape_url(url)
