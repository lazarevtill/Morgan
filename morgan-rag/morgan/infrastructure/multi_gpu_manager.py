"""
Multi-GPU Manager for Self-Hosted JARVIS

Manages model allocation across multiple GPUs:
- RTX 3090s (GPU 0+1): Main reasoning LLM with tensor parallelism
- RTX 4070 (GPU 2): Embeddings + fast LLM
- RTX 2060 (GPU 3): Reranking + utilities

This enables optimal resource utilization for JARVIS-like performance.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import psutil

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class GPURole(str, Enum):
    """GPU role assignment"""

    MAIN_LLM = "main_llm"  # Heavy reasoning (RTX 3090s)
    FAST_LLM = "fast_llm"  # Quick responses (RTX 4070)
    EMBEDDINGS = "embeddings"  # Embedding generation (RTX 4070)
    RERANKING = "reranking"  # Result reranking (RTX 2060)
    UTILITY = "utility"  # Misc tasks (RTX 2060/CPU)


@dataclass
class GPUAllocation:
    """GPU allocation configuration"""

    gpu_ids: List[int]  # GPU device IDs
    role: GPURole  # GPU role
    model: Optional[str] = None  # Model name
    vram_gb: float = 0.0  # Total VRAM (GB)
    utilization: float = 0.0  # Current utilization %
    temperature: float = 0.0  # Temperature (C)


class MultiGPUManager:
    """
    Manage multi-GPU model allocation for JARVIS.

    Optimizes model placement across GPUs based on:
    - Hardware capabilities (VRAM, compute)
    - Task requirements (reasoning, embedding, reranking)
    - Performance targets (latency, throughput)

    Example:
        >>> manager = MultiGPUManager()
        >>> manager.allocate_model("qwen2.5:32b", GPURole.MAIN_LLM, gpu_ids=[0, 1])
        >>> manager.get_gpu_for_task(GPURole.EMBEDDINGS)
        GPUAllocation(gpu_ids=[2], role=GPURole.EMBEDDINGS, ...)
    """

    def __init__(self):
        """Initialize multi-GPU manager"""
        self.allocations: Dict[GPURole, GPUAllocation] = {}
        self.torch_available = TORCH_AVAILABLE

        # Detect available GPUs
        self._detect_gpus()

        logger.info("MultiGPUManager initialized")
        logger.info(f"Torch available: {self.torch_available}")
        logger.info(f"GPU count: {self.gpu_count}")

    def _detect_gpus(self):
        """Detect available GPUs and their capabilities"""
        self.gpu_count = 0
        self.gpu_info = {}

        if self.torch_available and torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()

            for i in range(self.gpu_count):
                props = torch.cuda.get_device_properties(i)
                self.gpu_info[i] = {
                    "name": props.name,
                    "vram_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "cuda_cores": props.multi_processor_count,
                }

                logger.info(
                    f"GPU {i}: {props.name}, "
                    f"VRAM: {props.total_memory / (1024**3):.1f}GB, "
                    f"Compute: {props.major}.{props.minor}"
                )
        else:
            logger.warning("CUDA not available, running CPU-only mode")

    def allocate_model(
        self,
        model_name: str,
        role: GPURole,
        gpu_ids: List[int],
        vram_estimate_gb: Optional[float] = None,
    ) -> GPUAllocation:
        """
        Allocate a model to specific GPU(s).

        Args:
            model_name: Name of the model (e.g., "qwen2.5:32b-instruct")
            role: GPU role assignment
            gpu_ids: List of GPU IDs to use (e.g., [0, 1] for tensor parallelism)
            vram_estimate_gb: Estimated VRAM usage in GB

        Returns:
            GPUAllocation with assignment details

        Example:
            >>> manager.allocate_model(
            ...     "qwen2.5:32b-instruct-q4_K_M",
            ...     GPURole.MAIN_LLM,
            ...     gpu_ids=[0, 1]  # Tensor parallelism across 2x RTX 3090
            ... )
        """
        # Validate GPU IDs
        for gpu_id in gpu_ids:
            if gpu_id >= self.gpu_count:
                raise ValueError(
                    f"GPU {gpu_id} not available (only {self.gpu_count} GPUs)"
                )

        # Calculate total VRAM
        total_vram = sum(
            self.gpu_info[gpu_id]["vram_gb"]
            for gpu_id in gpu_ids
            if gpu_id in self.gpu_info
        )

        # Check VRAM if estimate provided
        if vram_estimate_gb and vram_estimate_gb > total_vram:
            logger.warning(
                f"Model {model_name} requires ~{vram_estimate_gb}GB but only "
                f"{total_vram:.1f}GB available on GPUs {gpu_ids}"
            )

        allocation = GPUAllocation(
            gpu_ids=gpu_ids, role=role, model=model_name, vram_gb=total_vram
        )

        self.allocations[role] = allocation

        logger.info(
            f"Allocated {model_name} to GPUs {gpu_ids} for {role.value} "
            f"(VRAM: {total_vram:.1f}GB)"
        )

        return allocation

    def get_gpu_for_task(self, role: GPURole) -> Optional[GPUAllocation]:
        """
        Get GPU allocation for a specific task.

        Args:
            role: Task role (MAIN_LLM, EMBEDDINGS, etc.)

        Returns:
            GPUAllocation if assigned, None otherwise
        """
        return self.allocations.get(role)

    def get_cuda_visible_devices(self, role: GPURole) -> str:
        """
        Get CUDA_VISIBLE_DEVICES environment variable value for a role.

        Args:
            role: GPU role

        Returns:
            Comma-separated GPU IDs (e.g., "0,1")

        Example:
            >>> os.environ["CUDA_VISIBLE_DEVICES"] = manager.get_cuda_visible_devices(GPURole.MAIN_LLM)
        """
        allocation = self.allocations.get(role)
        if not allocation:
            raise ValueError(f"No GPU allocation for role {role}")

        return ",".join(str(gpu_id) for gpu_id in allocation.gpu_ids)

    def get_gpu_stats(self, gpu_id: int) -> Dict[str, float]:
        """
        Get current GPU statistics.

        Args:
            gpu_id: GPU device ID

        Returns:
            Dict with utilization, memory, temperature
        """
        stats = {
            "utilization": 0.0,
            "memory_used_gb": 0.0,
            "memory_total_gb": 0.0,
            "memory_percent": 0.0,
            "temperature": 0.0,
        }

        if not self.torch_available:
            return stats

        if gpu_id >= self.gpu_count:
            return stats

        try:
            # Memory stats
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                torch.cuda.memory_reserved(gpu_id) / (1024**3)
                mem_total = self.gpu_info[gpu_id]["vram_gb"]

                stats["memory_used_gb"] = mem_allocated
                stats["memory_total_gb"] = mem_total
                stats["memory_percent"] = (mem_allocated / mem_total) * 100

            # For detailed stats, would need nvidia-ml-py
            # For now, return basic stats

        except Exception as e:
            logger.error(f"Error getting GPU stats for GPU {gpu_id}: {e}")

        return stats

    def get_all_gpu_stats(self) -> Dict[int, Dict[str, float]]:
        """Get statistics for all GPUs"""
        return {gpu_id: self.get_gpu_stats(gpu_id) for gpu_id in range(self.gpu_count)}

    def monitor_health(self) -> Dict[str, any]:
        """
        Monitor overall system health.

        Returns:
            Dict with health metrics
        """
        health = {
            "gpu_count": self.gpu_count,
            "allocations": len(self.allocations),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "gpus": {},
        }

        # GPU stats
        for gpu_id in range(self.gpu_count):
            health["gpus"][gpu_id] = self.get_gpu_stats(gpu_id)

        return health

    def setup_default_allocation(self):
        """
        Setup default GPU allocation for JARVIS.

        Assumes hardware:
        - GPU 0+1: RTX 3090 (12GB each)
        - GPU 2: RTX 4070 (8GB)
        - GPU 3: RTX 2060 (6GB)
        """
        if self.gpu_count < 2:
            logger.warning("Less than 2 GPUs available, using CPU fallback")
            return

        try:
            # Main LLM on RTX 3090s (tensor parallelism)
            if self.gpu_count >= 2:
                self.allocate_model(
                    "qwen2.5:32b-instruct-q4_K_M",
                    GPURole.MAIN_LLM,
                    gpu_ids=[0, 1],
                    vram_estimate_gb=20.0,  # Q4 quantization ~19GB
                )
                logger.info("✓ Main LLM allocated to GPU 0+1 (RTX 3090s)")

            # Embeddings + Fast LLM on RTX 4070
            if self.gpu_count >= 3:
                self.allocate_model(
                    "nomic-embed-text",
                    GPURole.EMBEDDINGS,
                    gpu_ids=[2],
                    vram_estimate_gb=1.5,
                )

                self.allocate_model(
                    "qwen2.5:7b-instruct-q5_K_M",
                    GPURole.FAST_LLM,
                    gpu_ids=[2],
                    vram_estimate_gb=5.0,
                )
                logger.info("✓ Embeddings + Fast LLM allocated to GPU 2 (RTX 4070)")

            # Reranking on RTX 2060
            if self.gpu_count >= 4:
                self.allocate_model(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    GPURole.RERANKING,
                    gpu_ids=[3],
                    vram_estimate_gb=0.5,
                )
                logger.info("✓ Reranking allocated to GPU 3 (RTX 2060)")

        except Exception as e:
            logger.error(f"Error setting up default allocation: {e}")


# Global instance
_manager: Optional[MultiGPUManager] = None


def get_gpu_manager() -> MultiGPUManager:
    """
    Get global GPU manager instance.

    Returns:
        Singleton MultiGPUManager instance
    """
    global _manager
    if _manager is None:
        _manager = MultiGPUManager()
        _manager.setup_default_allocation()
    return _manager
