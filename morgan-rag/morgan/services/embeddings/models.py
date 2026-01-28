"""
Embedding Service Models.

Standardized dataclasses for embedding operations.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class EmbeddingStats:
    """
    Statistics for embedding operations.

    Tracks performance metrics for monitoring and optimization.
    """

    total_requests: int = 0
    total_embeddings: int = 0
    total_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    remote_calls: int = 0
    local_calls: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate (0.0 to 1.0)."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def average_time(self) -> float:
        """Calculate average embedding time in seconds."""
        return self.total_time / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def throughput(self) -> float:
        """Calculate throughput (embeddings/sec)."""
        return self.total_embeddings / self.total_time if self.total_time > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate (0.0 to 1.0)."""
        return self.errors / self.total_requests if self.total_requests > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "total_embeddings": self.total_embeddings,
            "total_time": f"{self.total_time:.2f}s",
            "average_time": f"{self.average_time:.3f}s",
            "throughput": f"{self.throughput:.1f} embeddings/sec",
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{self.cache_hit_rate * 100:.1f}%",
            "errors": self.errors,
            "error_rate": f"{self.error_rate * 100:.1f}%",
            "remote_calls": self.remote_calls,
            "local_calls": self.local_calls,
        }
