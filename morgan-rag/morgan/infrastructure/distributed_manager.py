"""
Distributed Host Manager for Morgan 6-Host Setup

Automated deployment and update system for managing all hosts from central location.

Features:
- Automated deployment to all hosts via SSH
- Git-based updates (pull, restart services)
- Health monitoring across all hosts
- Service management (start/stop/restart)
- Configuration synchronization
- Zero-downtime rolling updates
- Docker-based deployment for isolation

Architecture:
- Host 1 (CPU): Central manager + Morgan Core
- Host 2-6: Remote workers (managed by Host 1)
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import asyncssh

    ASYNCSSH_AVAILABLE = True
except ImportError:
    ASYNCSSH_AVAILABLE = False

from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class HostRole(str, Enum):
    """Host roles in distributed architecture."""

    MANAGER = "manager"  # Host 1 - Central orchestrator
    BACKGROUND = "background"  # Host 2 - Background services
    MAIN_LLM_1 = "main_llm_1"  # Host 3 - RTX 3090 #1
    MAIN_LLM_2 = "main_llm_2"  # Host 4 - RTX 3090 #2
    EMBEDDINGS = "embeddings"  # Host 5 - RTX 4070
    RERANKING = "reranking"  # Host 6 - RTX 2060


class ServiceType(str, Enum):
    """Service types per host."""

    MORGAN_CORE = "morgan_core"
    OLLAMA = "ollama"
    QDRANT = "qdrant"
    REDIS = "redis"
    RERANKING_API = "reranking_api"
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"


@dataclass
class HostConfig:
    """Configuration for a single host."""

    hostname: str  # Hostname or IP
    role: HostRole  # Host role
    ssh_user: str = "morgan"  # SSH username
    ssh_port: int = 22  # SSH port
    ssh_key: Optional[str] = None  # Path to SSH private key
    project_path: str = "/opt/Morgan"  # Project path on host
    python_path: str = "/opt/Morgan/morgan-venv/bin/python"  # Python path
    services: List[ServiceType] = field(default_factory=list)  # Services on host
    gpu_available: bool = False  # GPU available
    gpu_model: Optional[str] = None  # GPU model (e.g., "RTX 3090")

    # Health check settings
    health_check_url: Optional[str] = None
    health_check_interval: int = 60


@dataclass
class DeploymentResult:
    """Result from deployment operation."""

    host: str
    success: bool
    message: str
    duration: float
    output: Optional[str] = None
    error: Optional[str] = None


class DistributedHostManager:
    """
    Manage distributed Morgan deployment across 6 hosts.

    Features:
    - Automated SSH-based deployment
    - Git pull and service restart
    - Health monitoring
    - Rolling updates with zero downtime
    - Configuration synchronization

    Example:
        >>> manager = DistributedHostManager()
        >>> manager.add_host(
        ...     hostname="192.168.1.20",
        ...     role=HostRole.MAIN_LLM_1,
        ...     services=[ServiceType.OLLAMA],
        ...     gpu_model="RTX 3090"
        ... )
        >>>
        >>> # Deploy to all hosts
        >>> await manager.deploy_all()
        >>>
        >>> # Update code on all hosts
        >>> await manager.update_all()
        >>>
        >>> # Restart services
        >>> await manager.restart_service("ollama", hosts=["192.168.1.20"])
        >>>
        >>> # Health check
        >>> status = await manager.health_check_all()
    """

    def __init__(self, ssh_key_path: Optional[str] = None):
        """
        Initialize distributed host manager.

        Args:
            ssh_key_path: Path to SSH private key (default: ~/.ssh/id_rsa)
        """
        if not ASYNCSSH_AVAILABLE:
            raise ImportError("asyncssh required. Install with: pip install asyncssh")

        self.ssh_key_path = ssh_key_path or str(Path.home() / ".ssh" / "id_rsa")
        self.hosts: Dict[str, HostConfig] = {}

        logger.info("DistributedHostManager initialized")

    def add_host(
        self,
        hostname: str,
        role: HostRole,
        services: List[ServiceType],
        ssh_user: str = "morgan",
        gpu_model: Optional[str] = None,
        **kwargs,
    ):
        """
        Add host to managed cluster.

        Args:
            hostname: Hostname or IP address
            role: Host role
            services: List of services running on host
            ssh_user: SSH username
            gpu_model: GPU model (if GPU host)
            **kwargs: Additional HostConfig parameters
        """
        config = HostConfig(
            hostname=hostname,
            role=role,
            ssh_user=ssh_user,
            services=services,
            gpu_available=gpu_model is not None,
            gpu_model=gpu_model,
            **kwargs,
        )

        self.hosts[hostname] = config
        logger.info(
            f"Added host: {hostname} ({role.value}, GPU: {gpu_model or 'None'})"
        )

    def load_default_config(self):
        """
        Load default 6-host configuration for Morgan.

        Network: 192.168.1.x
        - Host 1 (10): Manager + Morgan Core
        - Host 2 (11): Background services
        - Host 3 (20): Main LLM #1 (RTX 3090)
        - Host 4 (21): Main LLM #2 (RTX 3090)
        - Host 5 (22): Embeddings (RTX 4070)
        - Host 6 (23): Reranking (RTX 2060)
        """
        # Host 1 - Manager (CPU)
        self.add_host(
            hostname="192.168.1.10",
            role=HostRole.MANAGER,
            services=[ServiceType.MORGAN_CORE, ServiceType.QDRANT, ServiceType.REDIS],
        )

        # Host 2 - Background (CPU)
        self.add_host(
            hostname="192.168.1.11",
            role=HostRole.BACKGROUND,
            services=[ServiceType.PROMETHEUS, ServiceType.GRAFANA],
        )

        # Host 3 - Main LLM #1 (RTX 3090)
        self.add_host(
            hostname="192.168.1.20",
            role=HostRole.MAIN_LLM_1,
            services=[ServiceType.OLLAMA],
            gpu_model="RTX 3090",
        )

        # Host 4 - Main LLM #2 (RTX 3090)
        self.add_host(
            hostname="192.168.1.21",
            role=HostRole.MAIN_LLM_2,
            services=[ServiceType.OLLAMA],
            gpu_model="RTX 3090",
        )

        # Host 5 - Embeddings (RTX 4070)
        self.add_host(
            hostname="192.168.1.22",
            role=HostRole.EMBEDDINGS,
            services=[ServiceType.OLLAMA],
            gpu_model="RTX 4070",
        )

        # Host 6 - Reranking (RTX 2060)
        self.add_host(
            hostname="192.168.1.23",
            role=HostRole.RERANKING,
            services=[ServiceType.RERANKING_API],
            gpu_model="RTX 2060",
        )

        logger.info(f"Loaded default config with {len(self.hosts)} hosts")

    async def _run_ssh_command(
        self, hostname: str, command: str, timeout: float = 300.0
    ) -> tuple[bool, str, str]:
        """
        Run command on remote host via SSH.

        Args:
            hostname: Target hostname
            command: Command to execute
            timeout: Command timeout in seconds

        Returns:
            Tuple of (success, stdout, stderr)
        """
        config = self.hosts.get(hostname)
        if not config:
            return False, "", f"Host {hostname} not configured"

        try:
            async with asyncssh.connect(
                host=hostname,
                port=config.ssh_port,
                username=config.ssh_user,
                client_keys=[self.ssh_key_path],
                known_hosts=None,  # Skip host key verification (adjust for production)
            ) as conn:
                result = await asyncio.wait_for(conn.run(command), timeout=timeout)

                stdout = result.stdout if result.stdout else ""
                stderr = result.stderr if result.stderr else ""
                success = result.exit_status == 0

                return success, stdout, stderr

        except asyncio.TimeoutError:
            return False, "", f"Command timeout after {timeout}s"
        except Exception as e:
            return False, "", f"SSH error: {e}"

    async def deploy_to_host(
        self, hostname: str, git_branch: str = "v2-0.0.1", force: bool = False
    ) -> DeploymentResult:
        """
        Deploy/update Morgan on a single host.

        Args:
            hostname: Target hostname
            git_branch: Git branch to deploy
            force: Force update (discard local changes)

        Returns:
            DeploymentResult
        """
        start_time = time.time()
        config = self.hosts.get(hostname)

        if not config:
            return DeploymentResult(
                host=hostname,
                success=False,
                message="Host not configured",
                duration=0.0,
            )

        logger.info(f"Deploying to {hostname} ({config.role.value})...")

        # Build deployment commands
        commands = []

        # 1. Navigate to project directory
        commands.append(f"cd {config.project_path}")

        # 2. Git operations
        if force:
            commands.append("git fetch origin")
            commands.append("git reset --hard origin/{git_branch}")
        else:
            commands.append("git pull origin {git_branch}")

        # 3. Update Python dependencies
        commands.append(
            f"{config.python_path} -m pip install -r morgan-rag/requirements.txt"
        )

        # 4. Restart services based on host role
        for service in config.services:
            if service == ServiceType.OLLAMA:
                commands.append("sudo systemctl restart ollama")
            elif service == ServiceType.MORGAN_CORE:
                commands.append("sudo systemctl restart morgan")
            elif service == ServiceType.QDRANT:
                commands.append("docker restart qdrant")
            elif service == ServiceType.REDIS:
                commands.append("docker restart redis")
            elif service == ServiceType.RERANKING_API:
                commands.append("sudo systemctl restart morgan-reranking")

        # Combine commands
        full_command = " && ".join(commands)

        # Execute
        success, stdout, stderr = await self._run_ssh_command(
            hostname, full_command, timeout=600.0  # 10 minutes for deployment
        )

        duration = time.time() - start_time

        if success:
            logger.info(f"✓ Deployed to {hostname} in {duration:.2f}s")
            return DeploymentResult(
                host=hostname,
                success=True,
                message=f"Deployed successfully ({config.role.value})",
                duration=duration,
                output=stdout,
            )
        else:
            logger.error(f"✗ Deployment failed on {hostname}: {stderr}")
            return DeploymentResult(
                host=hostname,
                success=False,
                message="Deployment failed",
                duration=duration,
                error=stderr,
            )

    async def deploy_all(
        self, git_branch: str = "v2-0.0.1", force: bool = False, parallel: bool = True
    ) -> List[DeploymentResult]:
        """
        Deploy to all hosts.

        Args:
            git_branch: Git branch to deploy
            force: Force update
            parallel: Deploy in parallel (faster but less safe)

        Returns:
            List of DeploymentResult
        """
        logger.info(f"Deploying to {len(self.hosts)} hosts...")

        if parallel:
            # Deploy to all hosts in parallel
            tasks = [
                self.deploy_to_host(hostname, git_branch, force)
                for hostname in self.hosts.keys()
            ]
            results = await asyncio.gather(*tasks)
        else:
            # Deploy sequentially (safer for rolling updates)
            results = []
            for hostname in self.hosts.keys():
                result = await self.deploy_to_host(hostname, git_branch, force)
                results.append(result)

        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(
            f"Deployment complete: {successful}/{len(results)} hosts successful"
        )

        return results

    async def update_all(
        self, git_branch: str = "v2-0.0.1", rolling: bool = True
    ) -> List[DeploymentResult]:
        """
        Update all hosts with zero-downtime rolling update.

        Args:
            git_branch: Git branch to update to
            rolling: Use rolling update (one host at a time)

        Returns:
            List of DeploymentResult
        """
        if rolling:
            logger.info("Starting rolling update (zero-downtime)...")

            # Update order for zero-downtime:
            # 1. Update non-critical hosts first (background, reranking)
            # 2. Update LLM hosts one at a time (keep at least one running)
            # 3. Update manager last

            update_order = [
                HostRole.BACKGROUND,  # Host 2 - no user-facing impact
                HostRole.RERANKING,  # Host 6 - fallback available
                HostRole.EMBEDDINGS,  # Host 5 - fallback available
                HostRole.MAIN_LLM_2,  # Host 4 - LLM #1 still serving
                HostRole.MAIN_LLM_1,  # Host 3 - LLM #2 now serving
                HostRole.MANAGER,  # Host 1 - last (critical)
            ]

            results = []
            for role in update_order:
                # Find host with this role
                hostname = None
                for h, config in self.hosts.items():
                    if config.role == role:
                        hostname = h
                        break

                if hostname:
                    logger.info(f"Updating {role.value} ({hostname})...")
                    result = await self.deploy_to_host(
                        hostname, git_branch, force=False
                    )
                    results.append(result)

                    # Wait a bit for service to stabilize
                    if result.success:
                        await asyncio.sleep(5)

            return results
        else:
            # Parallel update (faster but no zero-downtime)
            return await self.deploy_all(git_branch, force=False, parallel=True)

    async def restart_service(
        self, service: ServiceType, hosts: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Restart a service on specified hosts.

        Args:
            service: Service to restart
            hosts: List of hostnames (None = all hosts with this service)

        Returns:
            Dict mapping hostname to success status
        """
        # Determine which hosts to restart
        if hosts is None:
            hosts = [
                h for h, config in self.hosts.items() if service in config.services
            ]

        logger.info(f"Restarting {service.value} on {len(hosts)} hosts...")

        # Build restart command
        if service == ServiceType.OLLAMA:
            command = "sudo systemctl restart ollama"
        elif service == ServiceType.MORGAN_CORE:
            command = "sudo systemctl restart morgan"
        elif service == ServiceType.QDRANT:
            command = "docker restart qdrant"
        elif service == ServiceType.REDIS:
            command = "docker restart redis"
        elif service == ServiceType.RERANKING_API:
            command = "sudo systemctl restart morgan-reranking"
        else:
            logger.error(f"Unknown service: {service}")
            return dict.fromkeys(hosts, False)

        # Restart on all hosts in parallel
        tasks = [
            self._run_ssh_command(hostname, command, timeout=60.0) for hostname in hosts
        ]
        results = await asyncio.gather(*tasks)

        # Build result dict
        status = {}
        for hostname, (success, stdout, stderr) in zip(hosts, results):
            status[hostname] = success
            if success:
                logger.info(f"✓ Restarted {service.value} on {hostname}")
            else:
                logger.error(
                    f"✗ Failed to restart {service.value} on {hostname}: {stderr}"
                )

        return status

    async def health_check_host(self, hostname: str) -> Dict[str, Any]:
        """
        Check health of a single host.

        Args:
            hostname: Target hostname

        Returns:
            Health status dict
        """
        config = self.hosts.get(hostname)
        if not config:
            return {"hostname": hostname, "healthy": False, "error": "Not configured"}

        # Check SSH connectivity
        success, stdout, stderr = await self._run_ssh_command(
            hostname, "echo 'alive'", timeout=5.0
        )

        if not success:
            return {
                "hostname": hostname,
                "role": config.role.value,
                "healthy": False,
                "error": "SSH connection failed",
            }

        # Check services
        service_status = {}
        for service in config.services:
            if service == ServiceType.OLLAMA:
                success, stdout, stderr = await self._run_ssh_command(
                    hostname, "systemctl is-active ollama", timeout=5.0
                )
                service_status["ollama"] = stdout.strip() == "active"

            elif service == ServiceType.MORGAN_CORE:
                success, stdout, stderr = await self._run_ssh_command(
                    hostname, "systemctl is-active morgan", timeout=5.0
                )
                service_status["morgan"] = stdout.strip() == "active"

        # Check GPU if available
        gpu_status = None
        if config.gpu_available:
            success, stdout, stderr = await self._run_ssh_command(
                hostname,
                "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits",
                timeout=5.0,
            )
            if success:
                try:
                    util, mem = stdout.strip().split(",")
                    gpu_status = {
                        "utilization": f"{util.strip()}%",
                        "memory_used": f"{mem.strip()}MB",
                        "model": config.gpu_model,
                    }
                except Exception:
                    pass

        return {
            "hostname": hostname,
            "role": config.role.value,
            "healthy": True,
            "services": service_status,
            "gpu": gpu_status,
        }

    async def health_check_all(self) -> Dict[str, Any]:
        """
        Check health of all hosts.

        Returns:
            Comprehensive health status
        """
        logger.info(f"Health checking {len(self.hosts)} hosts...")

        # Check all hosts in parallel
        tasks = [self.health_check_host(hostname) for hostname in self.hosts.keys()]
        results = await asyncio.gather(*tasks)

        # Build summary
        healthy_count = sum(1 for r in results if r.get("healthy", False))

        return {
            "total_hosts": len(self.hosts),
            "healthy_hosts": healthy_count,
            "unhealthy_hosts": len(self.hosts) - healthy_count,
            "hosts": results,
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.

        Returns:
            Configuration dict
        """
        return {
            "ssh_key": self.ssh_key_path,
            "total_hosts": len(self.hosts),
            "hosts": [
                {
                    "hostname": config.hostname,
                    "role": config.role.value,
                    "services": [s.value for s in config.services],
                    "gpu": config.gpu_model,
                }
                for config in self.hosts.values()
            ],
        }

    async def sync_config(
        self, config_file: str = ".env", source_host: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Synchronize configuration file across all hosts.

        Args:
            config_file: Configuration file to sync (relative to project root)
            source_host: Source hostname (None = use local file)

        Returns:
            Dict mapping hostname to success status
        """
        logger.info(f"Syncing {config_file} to all hosts...")

        # Read source config
        if source_host:
            # Copy from source host
            config = self.hosts.get(source_host)
            if not config:
                logger.error(f"Source host {source_host} not found")
                return dict.fromkeys(self.hosts.keys(), False)

            success, content, stderr = await self._run_ssh_command(
                source_host, f"cat {config.project_path}/{config_file}", timeout=10.0
            )

            if not success:
                logger.error(f"Failed to read config from {source_host}")
                return dict.fromkeys(self.hosts.keys(), False)
        else:
            # Use local file
            local_path = Path(config_file)
            if not local_path.exists():
                logger.error(f"Local config file not found: {config_file}")
                return dict.fromkeys(self.hosts.keys(), False)

            content = local_path.read_text()

        # Write to all hosts
        results = {}
        for hostname, config in self.hosts.items():
            # Escape content for shell
            escaped_content = content.replace("'", "'\"'\"'")

            command = f"echo '{escaped_content}' > {config.project_path}/{config_file}"

            success, stdout, stderr = await self._run_ssh_command(
                hostname, command, timeout=10.0
            )

            results[hostname] = success

            if success:
                logger.info(f"✓ Synced {config_file} to {hostname}")
            else:
                logger.error(f"✗ Failed to sync {config_file} to {hostname}")

        return results


# Global instance for singleton pattern
_manager: Optional[DistributedHostManager] = None


def get_distributed_manager(
    ssh_key_path: Optional[str] = None, auto_load_config: bool = True
) -> DistributedHostManager:
    """
    Get global distributed host manager instance (singleton).

    Args:
        ssh_key_path: Path to SSH private key
        auto_load_config: Automatically load default 6-host config

    Returns:
        DistributedHostManager instance
    """
    global _manager

    if _manager is None:
        _manager = DistributedHostManager(ssh_key_path=ssh_key_path)

        if auto_load_config:
            _manager.load_default_config()

    return _manager
