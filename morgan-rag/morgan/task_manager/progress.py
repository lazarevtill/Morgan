"""Progress tracking for tool usage and token consumption."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolActivity:
    """Record of a single tool invocation."""

    tool_name: str
    input_data: Any = None
    description: str = ""


@dataclass
class ProgressTracker:
    """Tracks tool usage and token consumption for a running task."""

    tool_use_count: int = 0
    cumulative_tokens: int = 0
    recent_activities: list[ToolActivity] = field(default_factory=list)

    def record_activity(self, activity: ToolActivity) -> None:
        """Record a tool activity, keeping the list capped at 20."""
        self.recent_activities.append(activity)
        if len(self.recent_activities) > 20:
            self.recent_activities = self.recent_activities[-20:]
        self.tool_use_count += 1

    def add_tokens(self, count: int) -> None:
        """Add to the cumulative token count."""
        self.cumulative_tokens += count
