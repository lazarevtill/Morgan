"""
FastAPI application factory for Morgan.

The existing web experience lives in ``morgan.interfaces.web_interface``.
This shim exposes a lightweight ``create_app`` that the CLI ``serve``
command can import without optional extras exploding.
"""

from typing import Any

import fastapi
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

from morgan.infrastructure.consul_client import ConsulServiceRegistry
from morgan.interfaces.web_interface import MorganWebInterface


def create_app(morgan: Any = None):
    """Return a FastAPI app instance backed by the web interface."""
    interface = MorganWebInterface(morgan_assistant=morgan)
    app = interface.app

    # Metrics endpoint for Prometheus
    @app.get("/metrics")
    async def metrics():
        registry = CollectorRegistry()
        data = generate_latest(registry)
        return fastapi.Response(content=data, media_type=CONTENT_TYPE_LATEST)

    # Optional Consul registration (best-effort)
    registry = ConsulServiceRegistry()
    registry.register(
        name="morgan-core",
        address=None,
        port=8080,
        tags=["api", "web"],
        check_http="http://morgan:8080/health" if registry.config.enabled else None,
    )

    return app
