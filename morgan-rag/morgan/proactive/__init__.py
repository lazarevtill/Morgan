"""
Morgan Proactive Module - Anticipatory Assistance

Provides proactive assistance capabilities:
- Context monitoring
- Suggestion generation
- Scheduled check-ins
- Task anticipation
"""

from morgan.proactive.monitor import (
    ContextMonitor,
    ContextEvent,
    get_context_monitor,
)
from morgan.proactive.suggestions import (
    SuggestionEngine,
    Suggestion,
    SuggestionType,
    get_suggestion_engine,
)
from morgan.proactive.anticipator import (
    TaskAnticipator,
    AnticipatedTask,
    get_task_anticipator,
)

__all__ = [
    "ContextMonitor",
    "ContextEvent",
    "get_context_monitor",
    "SuggestionEngine",
    "Suggestion",
    "SuggestionType",
    "get_suggestion_engine",
    "TaskAnticipator",
    "AnticipatedTask",
    "get_task_anticipator",
]

