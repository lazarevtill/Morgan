"""
Health checks for Morgan components.

Supports checking:
- LLM provider (single or distributed)
- Embedding provider (separate host)
- Reranking provider (optional)
- Vector database (Qdrant)
- Cache (Redis)
"""

import asyncio
from typing import Any, Dict

import requests
from qdrant_client import QdrantClient

from morgan.config.settings import get_settings
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class HealthChecker:
    """
    Perform health checks across all Morgan services.

    Checks:
    - LLM service (single or distributed endpoints)
    - Embedding service (separate host support)
    - Reranking service (optional)
    - Vector database (Qdrant)
    - Cache (Redis)
    """

    def __init__(self):
        self.settings = get_settings()

    def check_all_systems(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Check all systems and return health status.

        Args:
            detailed: Include detailed information

        Returns:
            Dict with overall status and component statuses
        """
        results = {}
        overall_status = "healthy"

        # Track individual checks for composite status
        qdrant_ok = False
        embed_ok = False
        llm_ok = False
        rerank_ok = True  # Optional, default healthy

        # Check LLM Service
        llm_result = self._check_llm()
        results["llm"] = llm_result
        llm_ok = llm_result["status"] == "healthy"
        if not llm_ok:
            overall_status = "error"

        # Check Embedding Service (separate from LLM)
        embed_result = self._check_embedding()
        results["embedding"] = embed_result
        embed_ok = embed_result["status"] == "healthy"
        if not embed_ok:
            # Embedding can fall back to local, so only warn
            if overall_status == "healthy":
                overall_status = "warning"

        # Check Reranking Service (optional)
        if self.settings.reranking_enabled:
            rerank_result = self._check_reranking()
            results["reranking"] = rerank_result
            rerank_ok = rerank_result["status"] in ("healthy", "skipped")
            if not rerank_ok and overall_status == "healthy":
                overall_status = "warning"
        else:
            results["reranking"] = {
                "status": "skipped",
                "detail": "Reranking disabled in configuration",
            }

        # Check Qdrant
        qdrant_result = self._check_qdrant()
        results["vector_db"] = qdrant_result
        qdrant_ok = qdrant_result["status"] == "healthy"
        if not qdrant_ok:
            overall_status = "error"

        # Check Redis (optional)
        redis_result = self._check_redis()
        results["redis"] = redis_result

        # High-level composite systems (for CLI output compatibility)
        results["knowledge"] = {
            "status": "healthy" if qdrant_ok and embed_ok else "warning"
        }
        results["memory"] = {"status": "healthy" if qdrant_ok else "error"}
        results["search"] = {
            "status": "healthy" if qdrant_ok and embed_ok and rerank_ok else "warning"
        }

        return {
            "overall_status": overall_status,
            "components": results,
            "detailed": detailed,
        }

    def _check_llm(self) -> Dict[str, Any]:
        """Check LLM service health."""
        try:
            # Check if distributed mode
            if self.settings.llm_distributed_enabled:
                endpoints = self.settings.get_llm_endpoints()
                healthy_count = 0
                endpoint_status = {}

                for endpoint in endpoints:
                    try:
                        base = endpoint.rstrip("/")
                        if base.endswith("/v1"):
                            base = base[:-3]

                        headers = {}
                        if self.settings.llm_api_key:
                            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"

                        # Try /v1/models first (OpenAI-style)
                        resp = requests.get(
                            f"{base}/v1/models", headers=headers, timeout=5
                        )
                        if resp.status_code == 200:
                            healthy_count += 1
                            endpoint_status[endpoint] = "healthy"
                        else:
                            # Try /api/tags (Ollama-style)
                            resp = requests.get(
                                f"{base}/api/tags", headers=headers, timeout=5
                            )
                            if resp.status_code == 200:
                                healthy_count += 1
                                endpoint_status[endpoint] = "healthy"
                            else:
                                endpoint_status[endpoint] = "error"
                    except Exception as e:
                        endpoint_status[endpoint] = f"error: {e}"

                return {
                    "status": "healthy" if healthy_count > 0 else "error",
                    "mode": "distributed",
                    "endpoints": endpoint_status,
                    "healthy_count": healthy_count,
                    "total_count": len(endpoints),
                    "strategy": self.settings.llm_load_balancing_strategy,
                }

            # Single endpoint mode
            base = self.settings.llm_base_url.rstrip("/")
            if base.endswith("/v1"):
                check_base = base[:-3]
            else:
                check_base = base

            headers = {}
            if self.settings.llm_api_key:
                headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"

            # Try different endpoint styles
            for url_suffix in ["/v1/models", "/api/tags", "/api/models"]:
                try:
                    resp = requests.get(
                        f"{check_base}{url_suffix}", headers=headers, timeout=10
                    )
                    if resp.status_code == 200:
                        return {
                            "status": "healthy",
                            "mode": "single",
                            "endpoint": self.settings.llm_base_url,
                            "model": self.settings.llm_model,
                        }
                except Exception:
                    continue

            return {
                "status": "error",
                "mode": "single",
                "endpoint": self.settings.llm_base_url,
                "detail": "Could not connect to LLM endpoint",
            }

        except Exception as exc:
            return {
                "status": "error",
                "detail": str(exc),
            }

    def _check_embedding(self) -> Dict[str, Any]:
        """Check embedding service health."""
        try:
            embedding_url = self.settings.get_embedding_base_url()

            if not embedding_url:
                # Try local fallback check
                try:
                    from sentence_transformers import SentenceTransformer

                    return {
                        "status": "healthy",
                        "mode": "local",
                        "model": self.settings.embedding_local_model,
                        "detail": "Using local embedding model",
                    }
                except ImportError:
                    return {
                        "status": "error",
                        "detail": "No embedding endpoint configured and local model unavailable",
                    }

            # Check remote embedding endpoint
            headers = {}
            if self.settings.llm_api_key:
                headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"

            # Try Ollama-style /api/tags
            try:
                resp = requests.get(
                    f"{embedding_url}/api/tags", headers=headers, timeout=5
                )
                if resp.status_code == 200:
                    return {
                        "status": "healthy",
                        "mode": "remote",
                        "endpoint": embedding_url,
                        "model": self.settings.embedding_model,
                        "dimensions": self.settings.embedding_dimensions,
                    }
            except Exception:
                pass

            # Try OpenAI-style /v1/models
            try:
                resp = requests.get(
                    f"{embedding_url}/v1/models", headers=headers, timeout=5
                )
                if resp.status_code == 200:
                    return {
                        "status": "healthy",
                        "mode": "remote",
                        "endpoint": embedding_url,
                        "model": self.settings.embedding_model,
                        "dimensions": self.settings.embedding_dimensions,
                    }
            except Exception:
                pass

            # Check local fallback
            if not self.settings.embedding_force_remote:
                try:
                    from sentence_transformers import SentenceTransformer

                    return {
                        "status": "warning",
                        "mode": "local_fallback",
                        "model": self.settings.embedding_local_model,
                        "detail": f"Remote endpoint {embedding_url} unreachable, using local fallback",
                    }
                except ImportError:
                    pass

            return {
                "status": "error",
                "endpoint": embedding_url,
                "detail": "Could not connect to embedding endpoint",
            }

        except Exception as exc:
            return {
                "status": "error",
                "detail": str(exc),
            }

    def _check_reranking(self) -> Dict[str, Any]:
        """Check reranking service health."""
        try:
            reranking_endpoint = self.settings.get_reranking_endpoint()

            if not reranking_endpoint:
                # Check local fallback
                if not self.settings.reranking_force_remote:
                    try:
                        from sentence_transformers import CrossEncoder

                        return {
                            "status": "healthy",
                            "mode": "local",
                            "model": self.settings.reranking_model,
                        }
                    except ImportError:
                        return {
                            "status": "warning",
                            "detail": "No reranking endpoint and local model unavailable",
                        }
                return {
                    "status": "skipped",
                    "detail": "No reranking endpoint configured",
                }

            # Check remote endpoint
            try:
                base_url = reranking_endpoint.replace("/rerank", "")
                resp = requests.get(f"{base_url}/health", timeout=5)
                if resp.status_code == 200:
                    return {
                        "status": "healthy",
                        "mode": "remote",
                        "endpoint": reranking_endpoint,
                    }
            except Exception:
                pass

            # Check local fallback
            if not self.settings.reranking_force_remote:
                try:
                    from sentence_transformers import CrossEncoder

                    return {
                        "status": "warning",
                        "mode": "local_fallback",
                        "model": self.settings.reranking_model,
                        "detail": f"Remote endpoint {reranking_endpoint} unreachable, using local fallback",
                    }
                except ImportError:
                    pass

            return {
                "status": "error",
                "endpoint": reranking_endpoint,
                "detail": "Could not connect to reranking endpoint",
            }

        except Exception as exc:
            return {
                "status": "error",
                "detail": str(exc),
            }

    def _check_qdrant(self) -> Dict[str, Any]:
        """Check Qdrant vector database health."""
        try:
            client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key or None,
            )
            collections = client.get_collections()

            return {
                "status": "healthy",
                "endpoint": self.settings.qdrant_url,
                "collections": len(collections.collections),
            }

        except Exception as exc:
            return {
                "status": "error",
                "endpoint": self.settings.qdrant_url,
                "detail": str(exc),
            }

    def _check_redis(self) -> Dict[str, Any]:
        """Check Redis cache health."""
        redis_url = getattr(self.settings, "redis_url", None)

        if not redis_url:
            return {
                "status": "skipped",
                "detail": "No REDIS_URL configured",
            }

        try:
            import redis

            client = redis.from_url(redis_url)
            client.ping()

            return {
                "status": "healthy",
                "endpoint": redis_url,
            }

        except ImportError:
            return {
                "status": "skipped",
                "detail": "Redis package not installed",
            }
        except Exception as exc:
            return {
                "status": "error",
                "endpoint": redis_url,
                "detail": str(exc),
            }

    def print_status(self):
        """Print health status to console."""
        status = self.check_all_systems(detailed=True)

        print("\n" + "=" * 60)
        print("Morgan System Health")
        print("=" * 60)

        # Overall status
        overall = status["overall_status"]
        icon = {"healthy": "[OK]", "warning": "[WARN]", "error": "[FAIL]"}.get(
            overall, "[?]"
        )
        print(f"\nOverall Status: {icon} {overall.upper()}")

        print("\n" + "-" * 60)
        print("Services:")
        print("-" * 60)

        # Core services
        for service in ["llm", "embedding", "reranking", "vector_db", "redis"]:
            if service in status["components"]:
                svc = status["components"][service]
                svc_status = svc.get("status", "unknown")
                icon = {
                    "healthy": "[OK]",
                    "warning": "[WARN]",
                    "error": "[FAIL]",
                    "skipped": "[SKIP]",
                }.get(svc_status, "[?]")

                print(f"\n{icon} {service.upper()}")

                if "endpoint" in svc:
                    print(f"    Endpoint: {svc['endpoint']}")
                if "mode" in svc:
                    print(f"    Mode: {svc['mode']}")
                if "model" in svc:
                    print(f"    Model: {svc['model']}")
                if "detail" in svc:
                    print(f"    Detail: {svc['detail']}")
                if "healthy_count" in svc:
                    print(f"    Healthy: {svc['healthy_count']}/{svc['total_count']}")

        print("\n" + "=" * 60)


# Async health check function
async def async_health_check() -> Dict[str, Any]:
    """
    Perform async health check using service factory.

    Returns:
        Health status dictionary
    """
    from morgan.services import get_service_factory

    factory = get_service_factory()
    return await factory.health_check_all()


# Quick health check function
def quick_health_check() -> bool:
    """
    Quick health check - returns True if all critical services are healthy.

    Returns:
        True if healthy, False otherwise
    """
    checker = HealthChecker()
    status = checker.check_all_systems()
    return status["overall_status"] in ("healthy", "warning")


if __name__ == "__main__":
    checker = HealthChecker()
    checker.print_status()
