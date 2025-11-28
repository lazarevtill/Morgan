"""
Lightweight runner for Morgan's background processing service.

This keeps reindexing, reranking (when available), and cache warming
alive in a dedicated process so it can be managed separately from the
main API container.
"""

import logging
import os
import signal
import sys
import time

from prometheus_client import start_http_server

from morgan.background import BackgroundProcessingService
from morgan.infrastructure.consul_client import ConsulServiceRegistry
from morgan.utils.logger import setup_logging
from morgan.vector_db.client import VectorDBClient

try:
    # Optional dependency; service still runs if reranker is unavailable
    from morgan.jina.reranking.service import JinaRerankingService
except Exception:  # pragma: no cover - defensive import
    JinaRerankingService = None


def _build_reranker(logger: logging.Logger):
    """Create reranking service if dependencies are present."""
    if not JinaRerankingService:
        logger.warning("Reranking service not available; running without reranker")
        return None

    try:
        return JinaRerankingService(enable_background=False)
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.warning("Reranking service failed to start: %s", exc)
        return None


def main():
    """Start the background processing loop and block until interrupted."""
    setup_logging()
    logger = logging.getLogger("morgan.background.runner")

    metrics_port = int(os.getenv("BACKGROUND_METRICS_PORT", "9000"))
    start_http_server(metrics_port)
    logger.info("Background metrics server listening on :%s", metrics_port)

    vector_db = VectorDBClient()
    reranker = _build_reranker(logger)

    service = BackgroundProcessingService(
        vector_db_client=vector_db, reranking_service=reranker
    )

    registry = ConsulServiceRegistry()
    registry.register(
        name="morgan-background",
        address=None,
        port=metrics_port,
        tags=["background", "metrics"],
        check_http=f"http://morgan-background:{metrics_port}" if registry.config.enabled else None,
    )

    if not service.start():
        logger.error("Failed to start background processing service")
        sys.exit(1)

    logger.info("Background processing service is running")

    def _shutdown(signum=None, frame=None):
        logger.info("Received stop signal (%s); shutting down background service", signum)
        service.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        while service.running:
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown("keyboard", None)

    logger.info("Background processing service stopped cleanly")


if __name__ == "__main__":
    main()
