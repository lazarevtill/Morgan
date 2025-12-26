"""
Lightweight health checks for Morgan components.
"""

import requests
from qdrant_client import QdrantClient

from morgan.config.settings import get_settings
from morgan.embeddings.service import EmbeddingService, get_embedding_service


class HealthChecker:
    """Perform simple health checks across core services."""

    def __init__(self):
        self.settings = get_settings()

    def check_all_systems(self, detailed: bool = False):
        results = {}
        overall_status = "healthy"
        qdrant_ok = False
        embed_ok = False
        llm_ok = False

        # Qdrant
        try:
            q_client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key or None,
            )
            q_client.get_collections()
            qdrant_ok = True
            results["vector_db"] = {"status": "healthy"}
        except Exception as exc:
            results["vector_db"] = {"status": "error", "detail": str(exc)}
            overall_status = "error"

        # Embeddings
        try:
            emb = EmbeddingService(self.settings)
            available = emb.is_available()
            embed_ok = available
            results["embeddings"] = {
                "status": "healthy" if available else "error",
                "model": emb.model_name,
            }
            if not available:
                overall_status = "error"
        except Exception as exc:
            results["embeddings"] = {"status": "error", "detail": str(exc)}
            overall_status = "error"

        # LLM provider (basic /models check)
        try:
            base = self.settings.llm_base_url.rstrip("/")
            headers = {}
            if self.settings.llm_api_key:
                headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
            resp = requests.get(f"{base}/models", headers=headers, timeout=10)
            resp.raise_for_status()
            llm_ok = True
            results["llm"] = {"status": "healthy"}
        except Exception as exc:
            results["llm"] = {"status": "error", "detail": str(exc)}
            overall_status = "error"

        # Redis (optional)
        if getattr(self.settings, "redis_url", None):
            try:
                import redis

                client = redis.from_url(self.settings.redis_url)
                client.ping()
                results["redis"] = {"status": "healthy"}
            except Exception as exc:
                results["redis"] = {"status": "error", "detail": str(exc)}
                overall_status = "error"
        else:
            results["redis"] = {
                "status": "skipped",
                "detail": "No REDIS_URL configured",
            }

        # High-level systems expected by CLI health output
        results["knowledge"] = {
            "status": "healthy" if qdrant_ok and embed_ok else "warning"
        }
        results["memory"] = {"status": "healthy" if qdrant_ok else "error"}
        results["search"] = {
            "status": "healthy" if qdrant_ok and embed_ok else "warning"
        }

        return {
            "overall_status": overall_status,
            "components": results,
            "detailed": detailed,
        }
