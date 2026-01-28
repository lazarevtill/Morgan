"""
Distributed GPU Manager for Multi-Host Morgan Setup

Manages model allocation across distributed GPU hosts.
Configuration is loaded from YAML config file (config/distributed.yaml).

Unlike the single-host MultiGPUManager, this manages remote hosts
running Ollama/inference servers on the local network.
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Add shared package to path if not already there
_shared_path = Path(__file__).parent.parent.parent.parent / "shared"
if _shared_path.exists() and str(_shared_path) not in sys.path:
    sys.path.insert(0, str(_shared_path.parent))

from shared.models.enums import HostRole, HostStatus

from morgan.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HostConfig:
    """Configuration for a distributed host."""

    host_id: str
    address: str  # IP or hostname
    port: int
    role: HostRole
    gpu_model: Optional[str] = None
    gpu_vram_gb: float = 0.0
    models: List[str] = field(default_factory=list)
    api_path: str = "/v1"  # OpenAI-compatible API path


@dataclass
class HostHealth:
    """Health status of a distributed host."""

    host_id: str
    status: HostStatus
    latency_ms: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    models_loaded: List[str] = field(default_factory=list)
    last_check: float = 0.0
    error_message: Optional[str] = None


@dataclass
class DistributedConfig:
    """Complete distributed system configuration."""

    hosts: Dict[str, HostConfig] = field(default_factory=dict)
    health: Dict[str, HostHealth] = field(default_factory=dict)
    default_timeout: float = 30.0
    health_check_interval: int = 60


class DistributedGPUManager:
    """
    Manage distributed GPU hosts for Morgan.

    Configuration is loaded from YAML config file. Set the config path via:
    - Constructor parameter: config_path
    - Environment variable: MORGAN_DISTRIBUTED_CONFIG
    - Default locations: config/distributed.yaml

    Example:
        >>> # Load from default config
        >>> manager = DistributedGPUManager.from_config()
        >>>
        >>> # Or specify config path
        >>> manager = DistributedGPUManager.from_config(
        ...     config_path="config/distributed.local.yaml"
        ... )
        >>>
        >>> # Get healthy endpoints for a role
        >>> endpoints = await manager.get_endpoints(HostRole.MAIN_LLM)
        >>>
        >>> # Health check all hosts
        >>> health = await manager.check_all_health()
    """

    def __init__(
        self,
        config: Optional[DistributedConfig] = None,
    ):
        """
        Initialize distributed GPU manager.

        Args:
            config: Pre-configured DistributedConfig
        """
        self.config = config or DistributedConfig()
        self._health_check_task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None

        logger.info(
            "DistributedGPUManager initialized with %d hosts", len(self.config.hosts)
        )

    @classmethod
    def from_config(
        cls,
        config_path: Optional[str] = None,
    ) -> "DistributedGPUManager":
        """
        Create manager from YAML configuration file.

        Args:
            config_path: Path to config file (optional)

        Returns:
            Configured DistributedGPUManager
        """
        # Import here to avoid circular imports
        from morgan.config.distributed_config import get_distributed_config

        # Load configuration
        arch_config = get_distributed_config(config_path=config_path)

        # Convert to internal config
        distributed_config = DistributedConfig(
            default_timeout=arch_config.settings.default_timeout,
            health_check_interval=arch_config.settings.health_check_interval,
        )

        # Add hosts from config
        for host_def in arch_config.hosts:
            try:
                role = HostRole(host_def.role)
            except ValueError:
                logger.warning(
                    "Unknown role '%s' for host '%s', skipping",
                    host_def.role,
                    host_def.host_id,
                )
                continue

            host_config = HostConfig(
                host_id=host_def.host_id,
                address=host_def.address,
                port=host_def.port,
                role=role,
                gpu_model=host_def.gpu_model,
                gpu_vram_gb=host_def.gpu_vram_gb,
                models=host_def.models,
                api_path=host_def.api_path,
            )
            distributed_config.hosts[host_def.host_id] = host_config

        logger.info(
            "Loaded %d hosts from config: %s",
            len(distributed_config.hosts),
            arch_config.config_source or "defaults",
        )

        return cls(config=distributed_config)

    def add_host(
        self,
        host_id: str,
        address: str,
        port: int,
        role: HostRole,
        gpu_model: Optional[str] = None,
        gpu_vram_gb: float = 0.0,
        models: Optional[List[str]] = None,
        api_path: str = "/v1",
    ) -> HostConfig:
        """
        Add a host to the distributed configuration.

        Args:
            host_id: Unique identifier for the host
            address: IP address or hostname
            port: Port number for the inference server
            role: Role of the host
            gpu_model: GPU model name
            gpu_vram_gb: GPU VRAM in GB
            models: List of models available on this host
            api_path: API path for OpenAI-compatible endpoint

        Returns:
            HostConfig for the added host
        """
        host = HostConfig(
            host_id=host_id,
            address=address,
            port=port,
            role=role,
            gpu_model=gpu_model,
            gpu_vram_gb=gpu_vram_gb,
            models=models or [],
            api_path=api_path,
        )

        self.config.hosts[host_id] = host

        logger.info(
            "Added host %s: %s:%d (%s) with %d models",
            host_id,
            address,
            port,
            role.value,
            len(models or []),
        )

        return host

    def remove_host(self, host_id: str) -> bool:
        """Remove a host from the configuration."""
        if host_id in self.config.hosts:
            del self.config.hosts[host_id]
            if host_id in self.config.health:
                del self.config.health[host_id]
            logger.info("Removed host %s", host_id)
            return True
        return False

    def get_host(self, host_id: str) -> Optional[HostConfig]:
        """Get a host by ID."""
        return self.config.hosts.get(host_id)

    def get_hosts_by_role(self, role: HostRole) -> List[HostConfig]:
        """Get all hosts with a specific role."""
        return [host for host in self.config.hosts.values() if host.role == role]

    def get_endpoint_url(self, host: HostConfig) -> str:
        """Get the full endpoint URL for a host."""
        return f"http://{host.address}:{host.port}{host.api_path}"

    async def get_endpoints(
        self,
        role: HostRole,
        healthy_only: bool = True,
    ) -> List[str]:
        """
        Get endpoint URLs for hosts with a specific role.

        Args:
            role: Host role to filter by
            healthy_only: Only return endpoints for healthy hosts

        Returns:
            List of endpoint URLs
        """
        hosts = self.get_hosts_by_role(role)

        if healthy_only:
            # Check health if not recently checked
            await self.check_all_health()

            endpoints = []
            for host in hosts:
                health = self.config.health.get(host.host_id)
                if health and health.status == HostStatus.ONLINE:
                    endpoints.append(self.get_endpoint_url(host))
        else:
            endpoints = [self.get_endpoint_url(host) for host in hosts]

        return endpoints

    async def check_host_health(self, host: HostConfig) -> HostHealth:
        """
        Check health of a specific host.

        Args:
            host: Host configuration

        Returns:
            HostHealth status
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=10.0)

        health = HostHealth(
            host_id=host.host_id,
            status=HostStatus.UNKNOWN,
            last_check=time.time(),
        )

        try:
            start_time = time.time()

            # Check Ollama-style health endpoint
            base_url = f"http://{host.address}:{host.port}"

            # Try Ollama API endpoint
            response = await self._http_client.get(f"{base_url}/api/tags")

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                health.status = HostStatus.ONLINE
                health.latency_ms = latency_ms

                # Parse Ollama response for loaded models
                data = response.json()
                if "models" in data:
                    health.models_loaded = [m.get("name", "") for m in data["models"]]

                logger.debug(
                    "Host %s healthy: %.1fms, %d models",
                    host.host_id,
                    latency_ms,
                    len(health.models_loaded),
                )
            else:
                health.status = HostStatus.DEGRADED
                health.error_message = f"HTTP {response.status_code}"
                logger.warning(
                    "Host %s degraded: HTTP %d",
                    host.host_id,
                    response.status_code,
                )

        except httpx.ConnectError as e:
            health.status = HostStatus.OFFLINE
            health.error_message = f"Connection failed: {e}"
            logger.warning("Host %s offline: %s", host.host_id, e)

        except httpx.TimeoutException:
            health.status = HostStatus.OFFLINE
            health.error_message = "Connection timeout"
            logger.warning("Host %s timeout", host.host_id)

        except Exception as e:
            health.status = HostStatus.UNKNOWN
            health.error_message = str(e)
            logger.error("Host %s health check error: %s", host.host_id, e)

        # Store health status
        self.config.health[host.host_id] = health

        return health

    async def check_all_health(self) -> Dict[str, HostHealth]:
        """
        Check health of all configured hosts in parallel.

        Returns:
            Dict mapping host_id to HostHealth
        """
        if not self.config.hosts:
            return {}

        logger.debug("Checking health of %d hosts...", len(self.config.hosts))

        # Check all hosts in parallel
        tasks = [self.check_host_health(host) for host in self.config.hosts.values()]

        await asyncio.gather(*tasks, return_exceptions=True)

        # Count status
        online = sum(
            1 for h in self.config.health.values() if h.status == HostStatus.ONLINE
        )
        offline = sum(
            1 for h in self.config.health.values() if h.status == HostStatus.OFFLINE
        )

        logger.info(
            "Health check complete: %d online, %d offline, %d other",
            online,
            offline,
            len(self.config.hosts) - online - offline,
        )

        return self.config.health

    def start_health_monitoring(self):
        """Start background health monitoring."""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_monitor_loop())
            logger.info("Started background health monitoring")

    def stop_health_monitoring(self):
        """Stop background health monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None
            logger.info("Stopped background health monitoring")

    async def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self.check_all_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health monitor loop: %s", e)

    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the distributed system status.

        Returns:
            Dict with status summary
        """
        hosts_by_role = {}
        for role in HostRole:
            hosts = self.get_hosts_by_role(role)
            online = sum(
                1
                for h in hosts
                if self.config.health.get(
                    h.host_id, HostHealth(host_id=h.host_id, status=HostStatus.UNKNOWN)
                ).status
                == HostStatus.ONLINE
            )
            hosts_by_role[role.value] = {
                "total": len(hosts),
                "online": online,
            }

        return {
            "total_hosts": len(self.config.hosts),
            "hosts_by_role": hosts_by_role,
            "health_summary": {
                host_id: {
                    "status": h.status.value,
                    "latency_ms": h.latency_ms,
                    "models": h.models_loaded,
                }
                for host_id, h in self.config.health.items()
            },
        }

    async def close(self):
        """Clean up resources."""
        self.stop_health_monitoring()
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# Global instance
_manager: Optional[DistributedGPUManager] = None


def get_distributed_gpu_manager(
    config_path: Optional[str] = None,
    reload: bool = False,
) -> DistributedGPUManager:
    """
    Get global distributed GPU manager instance.

    Loads configuration from YAML file on first call.

    Args:
        config_path: Path to config file (optional)
        reload: Force reload configuration

    Returns:
        Singleton DistributedGPUManager instance
    """
    global _manager
    if _manager is None or reload:
        _manager = DistributedGPUManager.from_config(config_path=config_path)
    return _manager
