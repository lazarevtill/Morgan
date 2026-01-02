"""
Morgan Reasoning Module - Multi-Step Reasoning and Task Planning

Provides advanced reasoning capabilities:
- Chain-of-thought reasoning
- Problem decomposition
- Logical reasoning chains
- Assumption tracking and validation
- Reasoning explanation generation
- Task decomposition and planning
- Progress tracking
"""

from morgan.reasoning.engine import (
    ReasoningEngine,
    ReasoningStep,
    ReasoningResult,
    ReasoningType,
    ReasoningStatus,
    get_reasoning_engine,
)
from morgan.reasoning.decomposer import (
    ProblemDecomposer,
    ProblemDecomposition,
    SubProblem,
    DecompositionStrategy,
    ProblemType,
    get_problem_decomposer,
)
from morgan.reasoning.chains import (
    ReasoningChainBuilder,
    ReasoningChain,
    ChainStep,
    ReasoningChainType,
    ChainStepType,
    get_reasoning_chain_builder,
)
from morgan.reasoning.assumptions import (
    AssumptionTracker,
    AssumptionSet,
    Assumption,
    AssumptionType,
    AssumptionStatus,
    AssumptionRisk,
    get_assumption_tracker,
)
from morgan.reasoning.explainer import (
    ReasoningExplainer,
    ReasoningExplanation,
    ExplanationSection,
    ExplanationStyle,
    ExplanationAudience,
    get_reasoning_explainer,
)
from morgan.reasoning.planner import (
    TaskPlanner,
    Task,
    TaskPlan,
    TaskStatus,
    get_task_planner,
)

__all__ = [
    # Core reasoning engine
    "ReasoningEngine",
    "ReasoningStep",
    "ReasoningResult",
    "ReasoningType",
    "ReasoningStatus",
    "get_reasoning_engine",
    
    # Problem decomposition
    "ProblemDecomposer",
    "ProblemDecomposition",
    "SubProblem",
    "DecompositionStrategy",
    "ProblemType",
    "get_problem_decomposer",
    
    # Reasoning chains
    "ReasoningChainBuilder",
    "ReasoningChain",
    "ChainStep",
    "ReasoningChainType",
    "ChainStepType",
    "get_reasoning_chain_builder",
    
    # Assumption tracking
    "AssumptionTracker",
    "AssumptionSet",
    "Assumption",
    "AssumptionType",
    "AssumptionStatus",
    "AssumptionRisk",
    "get_assumption_tracker",
    
    # Explanation generation
    "ReasoningExplainer",
    "ReasoningExplanation",
    "ExplanationSection",
    "ExplanationStyle",
    "ExplanationAudience",
    "get_reasoning_explainer",
    
    # Task planning
    "TaskPlanner",
    "Task",
    "TaskPlan",
    "TaskStatus",
    "get_task_planner",
]
