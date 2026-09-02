"""The contracts. Every module is reachable only through one of these Protocols, and the
orchestrator depends on these — never on concrete implementations. This is what makes a module
swappable (text→audio perception) and promotable (in-proc→its own service) with no code change.
"""

from morgan_brain.interfaces.events import Event, EventBus, EventType
from morgan_brain.interfaces.learning import Learner
from morgan_brain.interfaces.memory import MemoryStore
from morgan_brain.interfaces.perception import Perception
from morgan_brain.interfaces.personalization import PersonalizedContext, Personalizer
from morgan_brain.interfaces.reasoning import Reasoner, ReasoningRequest, ReasoningResult
from morgan_brain.interfaces.skills import Skill, SkillEngine
from morgan_brain.interfaces.tools import BaseTool, ToolExecutor, ToolResult

__all__ = [
    "BaseTool",
    "Event",
    "EventBus",
    "EventType",
    "Learner",
    "MemoryStore",
    "Perception",
    "PersonalizedContext",
    "Personalizer",
    "Reasoner",
    "ReasoningRequest",
    "ReasoningResult",
    "Skill",
    "SkillEngine",
    "ToolExecutor",
    "ToolResult",
]
