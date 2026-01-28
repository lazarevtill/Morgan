"""
LLM Service Models.

Standardized dataclasses for LLM operations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class LLMMode(str, Enum):
    """LLM operation mode."""

    SINGLE = "single"  # Single endpoint (OpenAI-compatible)
    DISTRIBUTED = "distributed"  # Multi-host with load balancing


@dataclass
class LLMResponse:
    """
    Standardized LLM response wrapper.

    Attributes:
        content: Generated text content
        model: Model name used for generation
        finish_reason: Why generation stopped (stop, length, etc.)
        usage: Token usage statistics (if available)
        latency_ms: Response time in milliseconds
        endpoint_used: Which endpoint was used (for distributed mode)
        metadata: Additional response metadata
    """

    content: str
    model: str
    finish_reason: str = "stop"
    usage: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None
    endpoint_used: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        """Get total token count from usage."""
        return self.usage.get("total_tokens", 0)

    @property
    def prompt_tokens(self) -> int:
        """Get prompt token count."""
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        """Get completion token count."""
        return self.usage.get("completion_tokens", 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "endpoint_used": self.endpoint_used,
            "metadata": self.metadata,
        }
