"""
Reranking Service Models.

Standardized dataclasses for reranking operations.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RerankResult:
    """
    Result from reranking operation.

    Attributes:
        text: Document text
        score: Relevance score (0.0 to 1.0)
        original_index: Original position in the input list
    """

    text: str
    score: float
    original_index: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "score": self.score,
            "original_index": self.original_index,
        }


@dataclass
class RerankStats:
    """
    Statistics for reranking operations.

    Tracks performance metrics for monitoring and optimization.
    """

    total_requests: int = 0
    total_pairs: int = 0
    total_time: float = 0.0
    errors: int = 0
    remote_calls: int = 0
    local_calls: int = 0
    embedding_calls: int = 0
    bm25_calls: int = 0

    @property
    def average_time(self) -> float:
        """Calculate average reranking time in seconds."""
        return self.total_time / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def throughput(self) -> float:
        """Calculate throughput (pairs/sec)."""
        return self.total_pairs / self.total_time if self.total_time > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate (0.0 to 1.0)."""
        return self.errors / self.total_requests if self.total_requests > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "total_pairs": self.total_pairs,
            "total_time": f"{self.total_time:.2f}s",
            "average_time": f"{self.average_time:.3f}s",
            "throughput": f"{self.throughput:.1f} pairs/sec",
            "errors": self.errors,
            "error_rate": f"{self.error_rate * 100:.1f}%",
            "remote_calls": self.remote_calls,
            "local_calls": self.local_calls,
            "embedding_calls": self.embedding_calls,
            "bm25_calls": self.bm25_calls,
        }
