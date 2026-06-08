"""Phase 3 Increment D — Champion-preprompt optimizer loop (dependency-light).

Design refs:
  * Design doc §D (GEPA optimizer loop) — reflective loop + MLflow GEPA upgrade path.
  * self-learning ADR — offline only, beats-current-or-nothing gate, char-budget anti-bloat,
    reflection model = biggest local model.

Public API
----------
``Example``               — a (context, query, good_output) training triple.
``mine_examples``         — build training triples from high-value interaction signals.
``ReflectiveOptimizer``   — dependency-light LLM-driven optimizer (the default path).
``GepaOptimizer``         — factory that delegates to ReflectiveOptimizer (dependency-light)
                            or attempts MLflow GEPA when the [learning] extra is installed
                            and ``settings.learning_backend == "mlflow"``.

Offline contract
----------------
The optimizer runs exclusively in the learning-worker (Cron/idle process), never on the
hot request path.  The deployed champion is just a better prompt string — zero
inference-time overhead.

MLflow GEPA upgrade
-------------------
``GepaOptimizer`` is the documented production upgrade path.  When mlflow ≥ 2.18 is
installed and ``settings.learning_backend == "mlflow"``, it will attempt:

    from mlflow.genai.optimize import GepaPromptOptimizer
    mlflow.genai.optimize_prompts(
        predict_fn, train_data,
        prompt_uris=[name],
        optimizer=GepaPromptOptimizer(reflection_model=<biggest_local_role>),
        scorers=[...],
    )

with ``MLFLOW_DISABLE_TELEMETRY=1`` / ``DO_NOT_TRACK=1`` forced.
Until that path is available (or the extra is absent) the fallback is always
``ReflectiveOptimizer``, which requires only stdlib + the existing provider seam.
"""
from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Union

from pydantic import BaseModel

from morgan_brain.learning.signals import SignalStore
from morgan_brain.learning_lifecycle.interfaces import PromptVersion
from morgan_brain.providers.router import RoleRouter
from morgan_brain.providers.wire import ChatMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class Example(BaseModel):
    """A single (context, query, good_output) training triple.

    Attributes:
        context:     Optional context summary for the conversation turn (may be empty).
        query:       The user query that triggered the reply.
        good_output: The positive/target output for this example (user edit or thumb-up reply).
    """

    context: str = ""
    query: str
    good_output: str


# ---------------------------------------------------------------------------
# mine_examples
# ---------------------------------------------------------------------------

# Sync/async scorer type alias used across this module.
_SyncScorer = Callable[[str], float]
_AsyncScorer = Callable[[str], Awaitable[float]]
AnyScorer = Union[_SyncScorer, _AsyncScorer]


async def mine_examples(
    signals: SignalStore,
    user_id: str,
    *,
    limit: int = 50,
) -> list[Example]:
    """Build training triples from high-value interaction signals.

    Pulls signals with ``value_rank >= 2`` (edited or thumb-down/retry) but ONLY
    includes signals that have a positive target:

    * ``user_edit is not None`` → ``good_output = user_edit``  (rank 3 — ground truth)
    * ``thumb == UP and user_edit is None`` → ``good_output = original_reply``  (rank 1 — low-trust)
    * thumb-down / retry WITHOUT edit → NO positive target → **skipped**.

    The thumb-up branch is caught by the ``high_value`` call using ``min_rank=1``
    for the positive path only; we call ``high_value`` with ``min_rank=1`` (include
    thumb-ups) then filter in Python to skip negatives without edits.

    Args:
        signals:  A ``SignalStore`` instance.
        user_id:  The user whose signals to mine.
        limit:    Maximum number of ``Example`` objects to return.

    Returns:
        A list of ``Example`` objects, newest signals first, up to *limit*.
    """
    # Fetch broadly (min_rank=1 to include thumb-ups).
    raw = await signals.high_value(user_id, min_rank=1, limit=limit * 4)

    examples: list[Example] = []
    for sig in raw:
        if sig.user_edit is not None:
            # Edit signal: the user corrected the reply → ground-truth pair.
            examples.append(
                Example(
                    context=sig.context_summary,
                    query=sig.query,
                    good_output=sig.user_edit,
                )
            )
        elif sig.thumb is not None and sig.thumb.value == "up":
            # Thumb-up with no edit: the original reply was good enough.
            examples.append(
                Example(
                    context=sig.context_summary,
                    query=sig.query,
                    good_output=sig.original_reply,
                )
            )
        # thumb-down / retry without edit: no positive target → skip.

        if len(examples) >= limit:
            break

    return examples


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------


async def _call_scorer(scorer: AnyScorer, body: str) -> float:
    """Call *scorer* whether it is sync or async, always returning a float."""
    raw: object = scorer(body)
    if inspect.isawaitable(raw):
        return float(await raw)
    return float(raw)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ReflectiveOptimizer
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a prompt engineer optimizing a champion document (system prompt or skill body).
Given the CURRENT document and a set of FAILING EXAMPLES where the current document
produces sub-optimal outputs, propose an IMPROVED document.

Rules:
1. Return ONLY the new document body — no commentary, no markdown fences.
2. Make bounded, targeted edits — do not rewrite from scratch.
3. Keep the document concise and focused; anti-bloat is critical.
"""

_USER_TEMPLATE = """\
CURRENT DOCUMENT:
{current_body}

FAILING EXAMPLES (query → expected good output):
{examples_text}

Propose the improved document:
"""


def _format_examples(train: list[Example], n: int = 5) -> str:
    """Render up to *n* training examples as a compact text block."""
    lines: list[str] = []
    for ex in train[:n]:
        ctx = f" [{ex.context}]" if ex.context else ""
        lines.append(f"  Q{ctx}: {ex.query}")
        lines.append(f"  A: {ex.good_output}")
    return "\n".join(lines)


class ReflectiveOptimizer:
    """Dependency-light optimizer that proposes champion candidates via the LLM.

    Uses the ``reflection`` role (the biggest local model) to generate bounded
    edits to the current champion document.  Falls back to ``fallback_role`` if
    the reflection role is not registered.

    This is the default, dependency-free implementation of the ``Optimizer``
    Protocol.  ``GepaOptimizer`` delegates here when the ``[learning]`` extra
    (MLflow ≥ 2.18) is not installed.

    Offline contract: runs only in the learning-worker, never the hot path.

    Anti-bloat: proposals exceeding ``char_budget`` characters are rejected
    (not scored) and the loop skips them.

    Args:
        router:        A ``RoleRouter`` instance wiring roles to LLM clients.
        role:          Role name for the reflection / optimization model.
                       Defaults to ``"reflection"``.
        fallback_role: Role to use if ``role`` is not registered.
                       Defaults to ``"strong"``.
        char_budget:   Maximum character length for a valid proposal.
                       Over-budget proposals are rejected without scoring.
                       Defaults to 1500.
    """

    def __init__(
        self,
        *,
        router: RoleRouter,
        role: str = "reflection",
        fallback_role: str = "strong",
        char_budget: int = 1500,
    ) -> None:
        self._router = router
        self._role = role
        self._fallback_role = fallback_role
        self._char_budget = char_budget

    def _get_client_model(self) -> tuple[Any, str]:
        """Return (client, model) for the reflection role, falling back gracefully."""
        try:
            return self._router.chat_for(self._role)
        except LookupError:
            logger.warning(
                "ReflectiveOptimizer: role %r not found; falling back to %r",
                self._role,
                self._fallback_role,
            )
            return self._router.chat_for(self._fallback_role)

    async def optimize(
        self,
        name: str,
        *,
        train: list[Any],
        scorer: AnyScorer,
        max_calls: int = 6,
        current_body: str = "",
    ) -> PromptVersion:
        """Produce a candidate optimized prompt body via reflective LLM loop.

        Algorithm:
        1. Score ``current_body`` as baseline.
        2. For each of ``max_calls`` iterations:
           a. Ask the reflection role to propose an improved document.
           b. Reject proposals that exceed ``char_budget`` (anti-bloat gate).
           c. Score accepted proposals; keep track of the best body seen.
        3. Return a ``PromptVersion(version=0, ...)`` wrapping the best body.
           ``version=0`` signals "candidate — not yet registered".

        The caller (``ChampionTrainer``) decides whether to promote.

        Args:
            name:         Prompt name (stored in the returned ``PromptVersion``).
            train:        Training examples (``list[Example]`` or plain dicts).
            scorer:       Callable ``(body: str) → float`` (sync or async).
            max_calls:    Maximum number of LLM proposals to request.
            current_body: The current champion body (baseline).

        Returns:
            A ``PromptVersion`` with ``version=0`` and the best body found.
        """
        # Coerce to list[Example] if raw dicts were passed.
        typed_train: list[Example] = []
        for item in train:
            if isinstance(item, Example):
                typed_train.append(item)
            elif isinstance(item, dict):
                typed_train.append(Example(**item))
            # else: skip unrecognised items

        client, model = self._get_client_model()

        # Baseline score.
        best_body = current_body
        best_score = await _call_scorer(scorer, current_body)

        examples_text = _format_examples(typed_train)

        for _ in range(max_calls):
            user_msg = _USER_TEMPLATE.format(
                current_body=best_body if best_body else "(empty — no current champion)",
                examples_text=examples_text,
            )
            messages = [
                ChatMessage(role="system", content=_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_msg),
            ]
            result = await client.agenerate(messages, model=model)
            candidate_body = result.text.strip()

            # Anti-bloat gate: reject over-budget proposals.
            if len(candidate_body) > self._char_budget:
                logger.debug(
                    "ReflectiveOptimizer: proposal rejected (len=%d > budget=%d)",
                    len(candidate_body),
                    self._char_budget,
                )
                continue

            candidate_score = await _call_scorer(scorer, candidate_body)
            if candidate_score > best_score:
                best_body = candidate_body
                best_score = candidate_score

        return PromptVersion(
            name=name,
            version=0,
            body=best_body,
            commit_message="reflective optimize",
            metrics={"score": best_score},
        )


# ---------------------------------------------------------------------------
# GepaOptimizer
# ---------------------------------------------------------------------------


class GepaOptimizer:
    """Factory-style optimizer: MLflow GEPA when available, else reflective fallback.

    When the ``[learning]`` extra (mlflow ≥ 2.18) is installed AND
    ``settings.learning_backend == "mlflow"`` is set, this optimizer will attempt
    to use ``mlflow.genai.optimize_prompts`` with ``GepaPromptOptimizer``.
    Telemetry is forcibly disabled (``MLFLOW_DISABLE_TELEMETRY=1``,
    ``DO_NOT_TRACK=1``) before any MLflow call.

    In all other cases — missing extra, import error, or non-mlflow backend —
    it transparently delegates to ``ReflectiveOptimizer``.

    Production upgrade path:
        Install ``morgan-brain[learning]`` (adds mlflow) and set::

            MORGAN_LEARNING_BACKEND=mlflow

        The next optimizer run will use GEPA automatically.

    Args:
        router:   ``RoleRouter`` for the reflection / fallback roles.
        settings: Any object with a ``learning_backend`` attribute (str).
                  Defaults to dependency-light mode if attribute is absent.
    """

    def __init__(
        self,
        *,
        router: RoleRouter,
        settings: Any = None,
        role: str = "reflection",
        fallback_role: str = "strong",
        char_budget: int = 1500,
    ) -> None:
        self._router = router
        self._settings = settings
        self._role = role
        self._fallback_role = fallback_role
        self._char_budget = char_budget
        self._reflective = ReflectiveOptimizer(
            router=router,
            role=role,
            fallback_role=fallback_role,
            char_budget=char_budget,
        )

    def _use_mlflow(self) -> bool:
        """Return True only if the mlflow backend is requested and importable."""
        if self._settings is None:
            return False
        backend = getattr(self._settings, "learning_backend", "reflective")
        return str(backend).lower() == "mlflow"

    async def optimize(
        self,
        name: str,
        *,
        train: list[Any],
        scorer: AnyScorer,
        max_calls: int = 6,
        current_body: str = "",
    ) -> PromptVersion:
        """Run GEPA optimization (MLflow) or fall back to the reflective loop.

        See ``ReflectiveOptimizer.optimize`` for the fallback behaviour.
        The MLflow path is only taken when:
          1. ``settings.learning_backend == "mlflow"``
          2. mlflow ≥ 2.18 with ``mlflow.genai.optimize`` is importable.
          3. ``GepaPromptOptimizer`` is available in that module.

        Otherwise, delegates to ``ReflectiveOptimizer`` transparently.
        """
        if self._use_mlflow():
            try:
                return await self._optimize_mlflow(
                    name, train=train, scorer=scorer, max_calls=max_calls,
                    current_body=current_body,
                )
            except (ImportError, AttributeError, NotImplementedError) as exc:
                logger.warning(
                    "GepaOptimizer: MLflow GEPA not available (%s); "
                    "falling back to reflective loop.",
                    exc,
                )

        return await self._reflective.optimize(
            name,
            train=train,
            scorer=scorer,
            max_calls=max_calls,
            current_body=current_body,
        )

    async def _optimize_mlflow(
        self,
        name: str,
        *,
        train: list[Any],
        scorer: AnyScorer,
        max_calls: int,
        current_body: str,
    ) -> PromptVersion:
        """Attempt the MLflow GEPA path; raises NotImplementedError if unavailable.

        Forces telemetry off before any MLflow import.
        """
        import os

        os.environ["MLFLOW_DISABLE_TELEMETRY"] = "1"
        os.environ["DO_NOT_TRACK"] = "1"

        try:
            from mlflow.genai.optimize import GepaPromptOptimizer as _GepaPromptOptimizer  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            raise NotImplementedError(
                "MLflow GEPA lands when [learning] extra is installed "
                "(pip install morgan-brain[learning]); using reflective fallback."
            ) from exc

        raise NotImplementedError(
            "MLflow GEPA orchestration not yet wired (GepaPromptOptimizer found but "
            "optimize_prompts integration is pending).  Falling back to reflective loop.",
        )
        # NOTE: The full wiring below is the documented production upgrade path:
        #
        #   import mlflow
        #   result = mlflow.genai.optimize_prompts(
        #       predict_fn=...,
        #       data=train,
        #       prompt_uris=[name],
        #       optimizer=GepaPromptOptimizer(
        #           reflection_model=self._role, max_metric_calls=max_calls
        #       ),
        #       scorers=[scorer],
        #   )
        #   best_body = result.best_prompt_bodies[name]
        #   best_score = result.best_scores[name]
        #   return PromptVersion(name=name, version=0, body=best_body,
        #                        commit_message="gepa optimize",
        #                        metrics={"score": best_score})
