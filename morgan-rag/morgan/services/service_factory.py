"""
Service Factory for Morgan RAG.

Provides unified access to all Morgan services with proper initialization
and configuration based on settings.

Usage:
    from morgan.services import ServiceFactory

    factory = ServiceFactory()

    # Get services
    llm = factory.get_llm()
    embedding = factory.get_embedding()
    reranking = factory.get_reranking()

    # Health check all services
    health = await factory.health_check_all()

    # Get service status
    status = factory.get_status()
"""

import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from morgan.config import get_settings
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ServiceStatus:
    """Status of a service."""

    name: str
    available: bool
    endpoint: Optional[str]
    error: Optional[str] = None


class ServiceFactory:
    """
    Factory for creating and managing Morgan services.

    Provides centralized access to:
    - LLM service (single or distributed)
    - Embedding service (with separate host support)
    - Reranking service (optional)

    All services are lazily initialized on first access.
    """

    def __init__(self):
        """Initialize service factory."""
        self.settings = get_settings()
        self._llm_service = None
        self._embedding_service = None
        self._reranking_service = None
        self._lock = threading.Lock()

        logger.info("ServiceFactory initialized")

    def get_llm(self):
        """
        Get LLM service instance.

        Returns:
            LLMService instance
        """
        if self._llm_service is None:
            with self._lock:
                if self._llm_service is None:
                    from morgan.services.llm_service import LLMService

                    self._llm_service = LLMService()
                    logger.info("LLM service created")

        return self._llm_service

    def get_embedding(self):
        """
        Get embedding service instance.

        Returns:
            EmbeddingService instance
        """
        if self._embedding_service is None:
            with self._lock:
                if self._embedding_service is None:
                    from morgan.services.embedding_service import EmbeddingService

                    self._embedding_service = EmbeddingService()
                    logger.info("Embedding service created")

        return self._embedding_service

    def get_reranking(self):
        """
        Get reranking service instance.

        Returns:
            LocalRerankingService instance or None if disabled
        """
        if not self.settings.reranking_enabled:
            return None

        if self._reranking_service is None:
            with self._lock:
                if self._reranking_service is None:
                    from morgan.infrastructure import LocalRerankingService

                    self._reranking_service = LocalRerankingService(
                        endpoint=self.settings.get_reranking_endpoint(),
                        model=self.settings.reranking_model,
                        timeout=self.settings.reranking_timeout,
                    )
                    logger.info("Reranking service created")

        return self._reranking_service

    def get_status(self) -> Dict[str, ServiceStatus]:
        """
        Get status of all services.

        Returns:
            Dict mapping service name to status
        """
        status = {}

        # LLM status
        try:
            llm = self.get_llm()
            status["llm"] = ServiceStatus(
                name="LLM",
                available=llm.is_available(),
                endpoint=self.settings.llm_base_url,
            )
        except Exception as e:
            status["llm"] = ServiceStatus(
                name="LLM",
                available=False,
                endpoint=self.settings.llm_base_url,
                error=str(e),
            )

        # Embedding status
        try:
            embedding = self.get_embedding()
            status["embedding"] = ServiceStatus(
                name="Embedding",
                available=embedding.is_available(),
                endpoint=self.settings.get_embedding_base_url(),
            )
        except Exception as e:
            status["embedding"] = ServiceStatus(
                name="Embedding",
                available=False,
                endpoint=self.settings.get_embedding_base_url(),
                error=str(e),
            )

        # Reranking status
        if self.settings.reranking_enabled:
            try:
                reranking = self.get_reranking()
                status["reranking"] = ServiceStatus(
                    name="Reranking",
                    available=reranking.is_available() if reranking else False,
                    endpoint=self.settings.get_reranking_endpoint(),
                )
            except Exception as e:
                status["reranking"] = ServiceStatus(
                    name="Reranking",
                    available=False,
                    endpoint=self.settings.get_reranking_endpoint(),
                    error=str(e),
                )
        else:
            status["reranking"] = ServiceStatus(
                name="Reranking",
                available=False,
                endpoint=None,
                error="Disabled in configuration",
            )

        return status

    async def health_check_all(self) -> Dict[str, Any]:
        """
        Perform health check on all services.

        Returns:
            Dict with health status of all services
        """
        results = {}

        # LLM health check
        try:
            llm = self.get_llm()
            results["llm"] = await llm.health_check()
        except Exception as e:
            results["llm"] = {"healthy": False, "error": str(e)}

        # Embedding health check
        try:
            embedding = self.get_embedding()
            results["embedding"] = {
                "healthy": embedding.is_available(),
                "endpoint": self.settings.get_embedding_base_url(),
                "model": self.settings.embedding_model,
                "dimensions": embedding.get_embedding_dimension(),
            }
        except Exception as e:
            results["embedding"] = {"healthy": False, "error": str(e)}

        # Reranking health check
        if self.settings.reranking_enabled:
            try:
                reranking = self.get_reranking()
                results["reranking"] = {
                    "healthy": reranking.is_available() if reranking else False,
                    "endpoint": self.settings.get_reranking_endpoint(),
                    "stats": reranking.get_stats() if reranking else {},
                }
            except Exception as e:
                results["reranking"] = {"healthy": False, "error": str(e)}
        else:
            results["reranking"] = {"healthy": True, "enabled": False}

        # Summary
        results["summary"] = {
            "all_healthy": all(
                r.get("healthy", False)
                for name, r in results.items()
                if name != "summary"
            ),
            "services_count": len(results) - 1,
        }

        return results

    def shutdown(self):
        """Shutdown all services and cleanup resources."""
        with self._lock:
            if self._llm_service:
                self._llm_service.shutdown()
                self._llm_service = None

            self._embedding_service = None
            self._reranking_service = None

        logger.info("All services shutdown")

    def print_status(self):
        """Print service status to console."""
        status = self.get_status()

        print("\n" + "=" * 60)
        print("Morgan Service Status")
        print("=" * 60)

        for name, svc_status in status.items():
            icon = "[OK]" if svc_status.available else "[FAIL]"
            print(f"\n{icon} {svc_status.name}")
            print(f"  Endpoint: {svc_status.endpoint or 'N/A'}")
            if svc_status.error:
                print(f"  Error: {svc_status.error}")

        print("\n" + "=" * 60)


# Global instance
_factory: Optional[ServiceFactory] = None
_factory_lock = threading.Lock()


def get_service_factory() -> ServiceFactory:
    """
    Get global service factory instance (singleton).

    Returns:
        ServiceFactory instance
    """
    global _factory

    if _factory is None:
        with _factory_lock:
            if _factory is None:
                _factory = ServiceFactory()

    return _factory


def reset_service_factory():
    """Reset the service factory (for testing)."""
    global _factory

    with _factory_lock:
        if _factory:
            _factory.shutdown()
        _factory = None


# Convenience functions
def get_llm_service():
    """Get LLM service from factory."""
    return get_service_factory().get_llm()


def get_embedding_service():
    """Get embedding service from factory."""
    return get_service_factory().get_embedding()


def get_reranking_service():
    """Get reranking service from factory."""
    return get_service_factory().get_reranking()


if __name__ == "__main__":
    # Test service factory
    import asyncio

    async def test():
        factory = get_service_factory()
        factory.print_status()

        print("\nRunning health checks...")
        health = await factory.health_check_all()

        print("\nHealth Check Results:")
        for name, result in health.items():
            if name == "summary":
                continue
            healthy = result.get("healthy", False)
            icon = "[OK]" if healthy else "[FAIL]"
            print(f"  {icon} {name}")
            if not healthy and "error" in result:
                print(f"      Error: {result['error']}")

        print(f"\nAll healthy: {health['summary']['all_healthy']}")

    asyncio.run(test())
