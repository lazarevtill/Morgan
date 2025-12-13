"""
Lightweight Consul client for service registration and discovery.

Uses direct HTTP calls (no extra dependency) so it is safe to import in
all environments. Registration is best-effort and will log warnings
instead of failing the application start.
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class ConsulConfig:
    """Configuration for Consul integration."""

    enabled: bool = field(
        default_factory=lambda: str(os.getenv("CONSUL_ENABLED", "false")).lower()
        in {"1", "true", "yes"}
    )
    address: str = field(
        default_factory=lambda: os.getenv("CONSUL_HTTP_ADDR", "http://consul:8500")
    )
    datacenter: Optional[str] = field(
        default_factory=lambda: os.getenv("CONSUL_DATACENTER")
    )
    token: Optional[str] = field(default_factory=lambda: os.getenv("CONSUL_HTTP_TOKEN"))


class ConsulServiceRegistry:
    """Simple Consul service registry helper."""

    def __init__(self, config: Optional[ConsulConfig] = None):
        self.config = config or ConsulConfig()
        self.session = requests.Session()
        if self.config.token:
            self.session.headers.update({"X-Consul-Token": self.config.token})

    def register(
        self,
        name: str,
        address: Optional[str],
        port: int,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, str]] = None,
        check_http: Optional[str] = None,
    ) -> bool:
        """Register a service with Consul (best-effort)."""
        if not self.config.enabled:
            logger.debug("Consul registration skipped (disabled)")
            return False

        service_address = address or os.getenv("SERVICE_ADDRESS") or self._default_ip()
        payload: Dict[str, object] = {
            "Name": name,
            "Address": service_address,
            "Port": port,
        }

        if tags:
            payload["Tags"] = tags
        if meta:
            payload["Meta"] = meta

        if check_http:
            payload["Check"] = {
                "HTTP": check_http,
                "Interval": "10s",
                "Timeout": "3s",
            }

        try:
            resp = self.session.put(
                f"{self.config.address}/v1/agent/service/register",
                json=payload,
                timeout=5,
            )
            resp.raise_for_status()
            logger.info(
                "Registered service '%s' with Consul at %s", name, self.config.address
            )
            return True
        except Exception as exc:  # pragma: no cover - networking guard
            logger.warning("Consul registration failed for %s: %s", name, exc)
            return False

    def deregister(self, name: str) -> bool:
        """Deregister a service from Consul (best-effort)."""
        if not self.config.enabled:
            return False
        try:
            resp = self.session.put(
                f"{self.config.address}/v1/agent/service/deregister/{name}", timeout=5
            )
            resp.raise_for_status()
            logger.info("Deregistered service '%s' from Consul", name)
            return True
        except Exception as exc:  # pragma: no cover - networking guard
            logger.warning("Consul deregistration failed for %s: %s", name, exc)
            return False

    def _default_ip(self) -> str:
        """Resolve a best-effort IP address for registration."""
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"
