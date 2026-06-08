"""Learning-lifecycle seam.

Exposes the ``PromptRegistry`` and ``Optimizer`` Protocols, the ``PromptVersion`` /
``EvalScore`` value types, and the dependency-light ``LocalPromptRegistry`` /
``NoopOptimizer`` implementations.

The real GEPA-via-MLflow optimizer and the MLflow-backed registry land in Wave 1/5.
"""
from morgan_brain.learning_lifecycle.interfaces import (
    EvalScore,
    Optimizer,
    PromptRegistry,
    PromptVersion,
)
from morgan_brain.learning_lifecycle.local import LocalPromptRegistry, NoopOptimizer

__all__ = [
    "EvalScore",
    "Optimizer",
    "PromptRegistry",
    "PromptVersion",
    "LocalPromptRegistry",
    "NoopOptimizer",
]
