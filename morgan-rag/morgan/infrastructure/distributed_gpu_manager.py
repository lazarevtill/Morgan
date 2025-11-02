"""
Distributed GPU Manager for Morgan 6-Host Setup

Manages GPU resources across distributed hosts:
- Host 3: RTX 3090 (24GB) - Main LLM #1
- Host 4: RTX 3090 (24GB) - Main LLM #2
- Host 5: RTX 4070 (8GB) - Embeddings + Fast LLM
- Host 6: RTX 2060 (6GB) - Reranking

Provides centralized GPU monitoring and resource management.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from morgan.utils.logger import get_logger


logger = get_logger(__name__)


class GPURole(str, Enum):
    """GPU role assignment in distributed architecture."""
    MAIN_LLM_1 = "main_llm_1"       # Host 3 - RTX 3090 #1
    MAIN_LLM_2 = "main_llm_2"       # Host 4 - RTX 3090 #2
    EMBEDDINGS = "embeddings"        # Host 5 - RTX 4070 (embeddings)
    FAST_LLM = "fast_llm"           # Host 5 - RTX 4070 (fast LLM)
    RERANKING = "reranking"          # Host 6 - RTX 2060


@dataclass
class GPUStatus:
    """GPU status information."""
    hostname: str                    # Host hostname/IP
    gpu_id: int = 0                  # GPU ID on host (usually 0 for 1 GPU per host)
    gpu_model: str = "Unknown"       # GPU model (RTX 3090, etc.)
    vram_total_mb: int = 0           # Total VRAM in MB
    vram_used_mb: int = 0            # Used VRAM in MB
    vram_free_mb: int = 0            # Free VRAM in MB
    utilization_percent: int = 0     # GPU utilization %
    temperature_c: int = 0           # Temperature in Celsius
    power_draw_w: int = 0            # Power draw in Watts
    role: Optional[GPURole] = None   # Assigned role
    model_loaded: Optional[str] = None  # Currently loaded model
    healthy: bool = True             # Health status
    last_check: float = 0.0          # Last health check timestamp

    @property
    def vram_usage_percent(self) -> float:
        """Calculate VRAM usage percentage."""
        if self.vram_total_mb == 0:
            return 0.0
        return (self.vram_used_mb / self.vram_total_mb) * 100

    @property
    def vram_available_gb(self) -> float:
        """Calculate available VRAM in GB."""
        return self.vram_free_mb / 1024


@dataclass
class HostGPUConfig:
    """GPU configuration for a host."""
    hostname: str
    role: GPURole
    gpu_model: str
    expected_vram_gb: int
    expected_model: Optional[str] = None


class DistributedGPUManager:
    """
    Manage GPU resources across distributed Morgan hosts.

    Provides:
    - GPU status monitoring across all hosts
    - Resource usage tracking
    - Health checks
    - Performance metrics
    - Load recommendations

    Example:
        >>> manager = DistributedGPUManager()
        >>> manager.register_host(
        ...     hostname="192.168.1.20",
        ...     role=GPURole.MAIN_LLM_1,
        ...     gpu_model="RTX 3090",
        ...     expected_vram_gb=24
        ... )
        >>>
        >>> # Get GPU status
        >>> status = await manager.get_gpu_status("192.168.1.20")
        >>>
        >>> # Check all GPUs
        >>> all_status = await manager.check_all_gpus()
        >>>
        >>> # Get recommendations
        >>> recommendations = manager.get_load_recommendations()
    """

    def __init__(self, distributed_manager=None):
        """
        Initialize distributed GPU manager.

        Args:
            distributed_manager: DistributedHostManager instance (optional)
        """
        self.distributed_manager = distributed_manager
        self.hosts: Dict[str, HostGPUConfig] = {}
        self.status_cache: Dict[str, GPUStatus] = {}

        logger.info("DistributedGPUManager initialized")

    def register_host(
        self,
        hostname: str,
        role: GPURole,
        gpu_model: str,
        expected_vram_gb: int,
        expected_model: Optional[str] = None
    ):
        """
        Register a GPU host.

        Args:
            hostname: Hostname or IP
            role: GPU role
            gpu_model: GPU model name
            expected_vram_gb: Expected VRAM in GB
            expected_model: Expected model name (optional)
        """
        config = HostGPUConfig(
            hostname=hostname,
            role=role,
            gpu_model=gpu_model,
            expected_vram_gb=expected_vram_gb,
            expected_model=expected_model
        )

        self.hosts[hostname] = config
        logger.info(f"Registered GPU host: {hostname} ({gpu_model}, {role.value})")

    def load_default_config(self):
        """
        Load default 6-host GPU configuration.

        Network: 192.168.1.x
        - Host 3 (20): RTX 3090 - Main LLM #1
        - Host 4 (21): RTX 3090 - Main LLM #2
        - Host 5 (22): RTX 4070 - Embeddings + Fast LLM
        - Host 6 (23): RTX 2060 - Reranking
        """
        # Host 3 - Main LLM #1 (RTX 3090)
        self.register_host(
            hostname="192.168.1.20",
            role=GPURole.MAIN_LLM_1,
            gpu_model="RTX 3090",
            expected_vram_gb=24,
            expected_model="qwen2.5:32b-instruct-q4_K_M"
        )

        # Host 4 - Main LLM #2 (RTX 3090)
        self.register_host(
            hostname="192.168.1.21",
            role=GPURole.MAIN_LLM_2,
            gpu_model="RTX 3090",
            expected_vram_gb=24,
            expected_model="qwen2.5:32b-instruct-q4_K_M"
        )

        # Host 5 - Embeddings (RTX 4070)
        self.register_host(
            hostname="192.168.1.22",
            role=GPURole.EMBEDDINGS,
            gpu_model="RTX 4070",
            expected_vram_gb=8,
            expected_model="nomic-embed-text"
        )

        # Host 6 - Reranking (RTX 2060)
        self.register_host(
            hostname="192.168.1.23",
            role=GPURole.RERANKING,
            gpu_model="RTX 2060",
            expected_vram_gb=6,
            expected_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        logger.info(f"Loaded default GPU config with {len(self.hosts)} hosts")

    async def get_gpu_status(self, hostname: str) -> Optional[GPUStatus]:
        """
        Get GPU status for a specific host.

        Args:
            hostname: Target hostname

        Returns:
            GPUStatus or None if unavailable
        """
        config = self.hosts.get(hostname)
        if not config:
            logger.error(f"Host {hostname} not registered")
            return None

        if not self.distributed_manager:
            logger.warning("DistributedHostManager not configured, using mock data")
            return self._get_mock_gpu_status(hostname)

        # Query GPU via SSH
        command = (
            "nvidia-smi --query-gpu=memory.total,memory.used,memory.free,"
            "utilization.gpu,temperature.gpu,power.draw "
            "--format=csv,noheader,nounits"
        )

        try:
            success, stdout, stderr = await self.distributed_manager._run_ssh_command(
                hostname,
                command,
                timeout=10.0
            )

            if not success:
                logger.error(f"Failed to query GPU on {hostname}: {stderr}")
                return GPUStatus(
                    hostname=hostname,
                    role=config.role,
                    gpu_model=config.gpu_model,
                    healthy=False,
                    last_check=time.time()
                )

            # Parse output: vram_total, vram_used, vram_free, util, temp, power
            parts = stdout.strip().split(",")
            if len(parts) < 6:
                logger.error(f"Invalid nvidia-smi output from {hostname}")
                return None

            status = GPUStatus(
                hostname=hostname,
                gpu_id=0,
                gpu_model=config.gpu_model,
                vram_total_mb=int(float(parts[0].strip())),
                vram_used_mb=int(float(parts[1].strip())),
                vram_free_mb=int(float(parts[2].strip())),
                utilization_percent=int(float(parts[3].strip())),
                temperature_c=int(float(parts[4].strip())),
                power_draw_w=int(float(parts[5].strip())),
                role=config.role,
                model_loaded=config.expected_model,
                healthy=True,
                last_check=time.time()
            )

            # Cache status
            self.status_cache[hostname] = status

            logger.debug(
                f"GPU status for {hostname}: "
                f"{status.utilization_percent}% util, "
                f"{status.vram_usage_percent:.1f}% VRAM, "
                f"{status.temperature_c}°C"
            )

            return status

        except Exception as e:
            logger.error(f"Error querying GPU on {hostname}: {e}")
            return GPUStatus(
                hostname=hostname,
                role=config.role,
                gpu_model=config.gpu_model,
                healthy=False,
                last_check=time.time()
            )

    def _get_mock_gpu_status(self, hostname: str) -> GPUStatus:
        """Get mock GPU status for testing."""
        config = self.hosts.get(hostname)
        if not config:
            return None

        # Return mock data
        return GPUStatus(
            hostname=hostname,
            gpu_id=0,
            gpu_model=config.gpu_model,
            vram_total_mb=config.expected_vram_gb * 1024,
            vram_used_mb=config.expected_vram_gb * 1024 // 2,  # 50% used
            vram_free_mb=config.expected_vram_gb * 1024 // 2,
            utilization_percent=50,
            temperature_c=65,
            power_draw_w=200,
            role=config.role,
            model_loaded=config.expected_model,
            healthy=True,
            last_check=time.time()
        )

    async def check_all_gpus(self) -> Dict[str, GPUStatus]:
        """
        Check GPU status on all hosts.

        Returns:
            Dict mapping hostname to GPUStatus
        """
        logger.info(f"Checking GPU status on {len(self.hosts)} hosts...")

        # Query all hosts in parallel
        tasks = [
            self.get_gpu_status(hostname)
            for hostname in self.hosts.keys()
        ]
        results = await asyncio.gather(*tasks)

        # Build result dict
        status_dict = {}
        for hostname, status in zip(self.hosts.keys(), results):
            if status:
                status_dict[hostname] = status

        return status_dict

    def get_load_recommendations(self) -> Dict[str, Any]:
        """
        Get load balancing recommendations based on current GPU status.

        Returns:
            Recommendations dict
        """
        recommendations = {
            "timestamp": time.time(),
            "hosts": []
        }

        for hostname, config in self.hosts.items():
            status = self.status_cache.get(hostname)

            if not status:
                recommendations["hosts"].append({
                    "hostname": hostname,
                    "role": config.role.value,
                    "recommendation": "Unable to assess - no status data",
                    "priority": "unknown"
                })
                continue

            # Analyze load
            rec = {
                "hostname": hostname,
                "role": config.role.value,
                "gpu_model": status.gpu_model,
                "utilization": f"{status.utilization_percent}%",
                "vram_usage": f"{status.vram_usage_percent:.1f}%",
                "temperature": f"{status.temperature_c}°C"
            }

            # Recommendations based on metrics
            if not status.healthy:
                rec["recommendation"] = "CRITICAL: GPU unhealthy or unreachable"
                rec["priority"] = "critical"
            elif status.temperature_c > 85:
                rec["recommendation"] = "WARNING: High temperature, reduce load"
                rec["priority"] = "high"
            elif status.vram_usage_percent > 95:
                rec["recommendation"] = "WARNING: VRAM near capacity"
                rec["priority"] = "high"
            elif status.utilization_percent > 90:
                rec["recommendation"] = "High utilization, consider load balancing"
                rec["priority"] = "medium"
            elif status.utilization_percent < 20:
                rec["recommendation"] = "Low utilization, can handle more load"
                rec["priority"] = "low"
            else:
                rec["recommendation"] = "Normal operation"
                rec["priority"] = "normal"

            recommendations["hosts"].append(rec)

        return recommendations

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all GPU resources.

        Returns:
            Summary dict
        """
        summary = {
            "total_hosts": len(self.hosts),
            "healthy_hosts": 0,
            "unhealthy_hosts": 0,
            "total_vram_gb": 0,
            "used_vram_gb": 0,
            "average_utilization": 0.0,
            "average_temperature": 0.0,
            "hosts_by_role": {}
        }

        for hostname, config in self.hosts.items():
            status = self.status_cache.get(hostname)

            if status:
                if status.healthy:
                    summary["healthy_hosts"] += 1
                else:
                    summary["unhealthy_hosts"] += 1

                summary["total_vram_gb"] += status.vram_total_mb / 1024
                summary["used_vram_gb"] += status.vram_used_mb / 1024
                summary["average_utilization"] += status.utilization_percent
                summary["average_temperature"] += status.temperature_c

                # Group by role
                role = config.role.value
                if role not in summary["hosts_by_role"]:
                    summary["hosts_by_role"][role] = []

                summary["hosts_by_role"][role].append({
                    "hostname": hostname,
                    "gpu_model": status.gpu_model,
                    "utilization": status.utilization_percent,
                    "vram_usage_percent": status.vram_usage_percent
                })

        # Calculate averages
        if summary["healthy_hosts"] > 0:
            summary["average_utilization"] /= summary["healthy_hosts"]
            summary["average_temperature"] /= summary["healthy_hosts"]

        summary["vram_usage_percent"] = (
            (summary["used_vram_gb"] / summary["total_vram_gb"] * 100)
            if summary["total_vram_gb"] > 0 else 0.0
        )

        return summary


# Global instance for singleton pattern
_gpu_manager: Optional[DistributedGPUManager] = None


def get_distributed_gpu_manager(
    distributed_manager=None,
    auto_load_config: bool = True
) -> DistributedGPUManager:
    """
    Get global distributed GPU manager instance (singleton).

    Args:
        distributed_manager: DistributedHostManager instance
        auto_load_config: Automatically load default config

    Returns:
        DistributedGPUManager instance
    """
    global _gpu_manager

    if _gpu_manager is None:
        _gpu_manager = DistributedGPUManager(
            distributed_manager=distributed_manager
        )

        if auto_load_config:
            _gpu_manager.load_default_config()

    return _gpu_manager
