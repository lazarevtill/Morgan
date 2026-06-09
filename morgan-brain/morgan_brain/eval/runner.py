"""Eval runner — wires the real assistant into the eval harness.

make_predict_fn
---------------
Returns an async ``predict_fn(item, *, system_override="") -> str`` that:
1. Seeds ``item.setup`` strings into a *per-item isolated* MemoryGate (new
   MemoryModule + MemoryGate instantiated fresh for each item call), so eval
   items never pollute each other or the real assistant memory.
2. Honours ``item.should_inject``: for OVER_PERSONALIZATION_NEGATIVE items
   with should_inject=False, the stale preference is NOT seeded.
3. Calls ``orchestrator.handle_turn(user_id=EVAL_USER_ID, text=item.query,
   system_override=system_override)`` and returns ``result.text``.

FIREWALL: eval content is written only to a scratch MemoryGate that is
discarded after each item.  The real orchestrator's MemoryGate is NEVER
written to by eval code.

make_eval_scorer
----------------
Returns an async ``scorer(body: str) -> float`` that runs the harness over
all golden items with predict_fn using body as system_override, and returns
the ``overall_preference_following_accuracy`` float from the Scorecard.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING

from morgan_brain.eval.golden import GoldenItem
from morgan_brain.eval.harness import EvalHarness
from morgan_brain.models.memory import Memory, MemoryKind, MemorySource
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex
from morgan_brain.security.memory_gate import MemoryGate

if TYPE_CHECKING:
    from morgan_brain.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

# Dedicated user_id for all eval turns — isolated from any real user data.
EVAL_USER_ID: str = "__eval__"

# Scalar key used for the gating metric inside Scorecard.layer2.
_OVERALL_ACCURACY_KEY = "overall_preference_following_accuracy"


def _make_scratch_gate(clock: Callable[[], datetime]) -> MemoryGate:
    """Build a fresh, fully isolated in-memory MemoryGate for one eval item.

    Uses FakeEmbedder (no network) and InMemoryVectorIndex + in-memory SQLite,
    so it is entirely self-contained and discarded after the item run.
    """
    module = MemoryModule(
        embedder=FakeEmbedder(dim=16),
        vectors=InMemoryVectorIndex(),
        temporal=SqliteTemporalStore(":memory:"),
        clock=clock,
    )
    return MemoryGate(module)


def make_predict_fn(
    *,
    orchestrator: "Orchestrator",
    clock: Callable[[], datetime] = datetime.utcnow,
) -> Callable[[GoldenItem, str], Awaitable[str]]:
    """Return an async predict_fn over the REAL orchestrator.

    The returned callable signature is:

        async def predict_fn(item: GoldenItem, system_override: str = "") -> str

    The function creates a fresh scratch MemoryGate per item, seeds
    ``item.setup`` into it (unless ``item.should_inject is False``), then
    drives the orchestrator.  The scratch gate is NEVER the orchestrator's
    own gate — eval content is completely firewalled.

    Args:
        orchestrator: The real Orchestrator to evaluate.
        clock:        Deterministic clock (injected for tests).

    Returns:
        Async callable ``(item, system_override="") → str``.
    """

    async def predict_fn(item: GoldenItem, system_override: str = "") -> str:
        # Build a per-item scratch gate so eval items are isolated.
        scratch_gate = _make_scratch_gate(clock)

        # Seed setup facts into the scratch gate (honours should_inject).
        if item.should_inject:
            for fact_text in item.setup:
                mem = Memory(
                    user_id=EVAL_USER_ID,
                    kind=MemoryKind.SEMANTIC,
                    content=fact_text,
                    source=MemorySource.USER_STATED,
                )
                await scratch_gate.store(mem)
        else:
            # OVER_PERSONALIZATION_NEGATIVE: do NOT seed the stale preference.
            logger.debug(
                "eval predict_fn: skipping setup seeding for item %r (should_inject=False)",
                item.id,
            )

        # Temporarily swap in the scratch gate's store for this item's memory recall.
        # The orchestrator's _memory is the real gate; we monkey-patch only within
        # this call and restore it immediately after — guaranteeing FIREWALL isolation.
        original_memory = orchestrator._memory
        orchestrator._memory = scratch_gate
        try:
            result = await orchestrator.handle_turn(
                user_id=EVAL_USER_ID,
                text=item.query,
                system_override=system_override,
            )
        finally:
            # FIREWALL: always restore the real gate, even on exception.
            orchestrator._memory = original_memory

        return result.text

    return predict_fn


def make_eval_scorer(
    *,
    harness: EvalHarness,
    golden_items: list[GoldenItem],
    predict_fn: Callable[[GoldenItem, str], Awaitable[str]],
) -> Callable[[str], Awaitable[float]]:
    """Return an async scorer ``(body: str) -> float`` backed by the eval harness.

    The scorer runs ``harness.run_l2(golden_items, ...)`` using ``predict_fn``
    with ``body`` as the system_override on each item, and returns the
    ``overall_preference_following_accuracy`` figure from the Scorecard.

    Args:
        harness:       The EvalHarness instance (holds the judge).
        golden_items:  List of GoldenItems to evaluate over.
        predict_fn:    The async callable from ``make_predict_fn``.

    Returns:
        Async callable ``(body: str) → float`` in [0, 1].
    """

    async def scorer(body: str) -> float:
        # Wrap predict_fn with the body as system_override to match PredictFn signature.
        async def _item_predict(item: GoldenItem) -> str:
            return await predict_fn(item, body)

        scorecard = await harness.run_l2(golden_items, _item_predict)
        return scorecard.layer2.get(_OVERALL_ACCURACY_KEY, 0.0)

    return scorer
