"""Phase 2B — Bi-temporal consolidation worker.

``MemoryConsolidator`` reads recent episodics + current facts, asks an LLM
(via the role router) to propose ``FactOp`` operations, and applies them to
the bi-temporal store through the MemoryGate.

Design invariants:
- Contradiction → close old interval (valid_to = now), never hard-delete.
- Deterministic: clock injected, no datetime.now() calls.
- Provider-agnostic: uses roles, never model names directly.
- Dedup pre-filter: ADD whose (subject, predicate, object) matches a currently-
  valid fact is silently skipped (treated as NOOP).
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field

from morgan_brain.models.memory import Memory, MemoryKind, MemoryQuery, MemorySource, TemporalFact
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import RoleRouter
from morgan_brain.providers.structured import generate_structured
from morgan_brain.providers.wire import ChatMessage
from morgan_brain.security.memory_gate import MemoryGate


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class FactOpKind(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


class FactOp(BaseModel):
    """A single fact operation proposed by the LLM."""

    op: FactOpKind
    subject: str
    predicate: str
    object: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    reason: str = ""


class FactOpBatch(BaseModel):
    """Batch of fact operations — the schema passed to ``generate_structured``."""

    ops: list[FactOp]


# ---------------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------------


class MemoryConsolidator:
    """Bi-temporal consolidation worker.

    Parameters
    ----------
    gate:
        The MemoryGate (all reads/writes pass through here).
    temporal:
        Direct reference to the SqliteTemporalStore for low-level operations
        (``close_fact``, ``set_confidence``) not exposed on the gate.
    router:
        RoleRouter for LLM dispatch.
    capability_registry:
        CapabilityRegistry for building CapabilityDescriptor passed to
        ``generate_structured``.
    clock:
        Injected callable returning the current datetime. Never calls
        ``datetime.now()`` internally.
    role:
        LLM role to request (default "strong").
    """

    def __init__(
        self,
        *,
        gate: MemoryGate,
        temporal: SqliteTemporalStore,
        router: RoleRouter,
        capability_registry: CapabilityRegistry,
        clock: Callable[[], datetime],
        role: str = "strong",
    ) -> None:
        self._gate = gate
        self._temporal = temporal
        self._router = router
        self._reg = capability_registry
        self._clock = clock
        self._role = role

    # ------------------------------------------------------------------
    # propose
    # ------------------------------------------------------------------

    async def propose(
        self,
        user_id: str,
        episodics: list[Memory],
        existing_facts: list[TemporalFact],
    ) -> FactOpBatch:
        """Ask the LLM to propose fact operations from episodics + existing facts.

        Falls back gracefully when no JSON-schema-capable binding exists.
        """
        # Try to get a json_schema-capable binding first; fall back to any binding.
        try:
            client, model = self._router.chat_for(
                self._role, needs_json_schema=True
            )
            provider = self._provider_for_model(model)
        except LookupError:
            client, model = self._router.chat_for(self._role)
            provider = self._provider_for_model(model)

        descriptor = self._reg.get(provider, model)

        episodic_text = "\n".join(
            f"- [{m.source.value}] {m.content}" for m in episodics
        ) or "(none)"
        facts_text = "\n".join(
            f"- {f.subject} {f.predicate} {f.object} (conf={f.confidence:.2f})"
            for f in existing_facts
        ) or "(none)"

        system_msg = ChatMessage(
            role="system",
            content=(
                "You are a memory consolidation engine. "
                "Given recent episodic memories and the user's existing known facts, "
                "produce a batch of fact operations (ADD, UPDATE, DELETE, NOOP) in JSON. "
                "Use subject/predicate/object triples. "
                "Prefer UPDATE over ADD when a fact for the same subject+predicate already exists "
                "with a different object. Use NOOP when no change is needed. "
                "Dates are provided by the system — do NOT hallucinate timestamps."
            ),
        )
        user_msg = ChatMessage(
            role="user",
            content=(
                f"Recent episodics:\n{episodic_text}\n\n"
                f"Existing facts:\n{facts_text}\n\n"
                "Propose fact operations."
            ),
        )
        messages: list[ChatMessage] = [system_msg, user_msg]

        return await generate_structured(
            client,
            messages,
            model=model,
            schema=FactOpBatch,
            descriptor=descriptor,
        )

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    async def apply(self, user_id: str, batch: FactOpBatch) -> list[FactOp]:
        """Apply a batch of fact operations.

        Dedup pre-filter: an ADD whose (subject, predicate, object) exactly
        matches a currently-valid fact is silently dropped (treated as NOOP).

        Returns the list of ops that were actually applied (excludes NOOPs and
        deduped ADDs).
        """
        now = self._clock()
        current = await self._temporal.current_facts(user_id=user_id)
        current_set = {
            (f.subject, f.predicate, f.object) for f in current
        }

        applied: list[FactOp] = []

        for op in batch.ops:
            if op.op is FactOpKind.NOOP:
                continue

            if op.op is FactOpKind.ADD:
                key = (op.subject, op.predicate, op.object)
                if key in current_set:
                    # Dedup — already a current fact with the exact same triple.
                    continue
                await self._gate.upsert_fact(
                    TemporalFact(
                        user_id=user_id,
                        subject=op.subject,
                        predicate=op.predicate,
                        object=op.object,
                        confidence=op.confidence,
                        source=MemorySource.AGENT_INFERRED,
                    )
                )
                applied.append(op)

            elif op.op is FactOpKind.UPDATE:
                # upsert_fact closes any existing (subject, predicate) interval
                # and opens a new one — this is the "supersede not delete" pattern.
                await self._gate.upsert_fact(
                    TemporalFact(
                        user_id=user_id,
                        subject=op.subject,
                        predicate=op.predicate,
                        object=op.object,
                        confidence=op.confidence,
                        source=MemorySource.AGENT_INFERRED,
                    )
                )
                applied.append(op)

            elif op.op is FactOpKind.DELETE:
                # Close the currently-valid fact's interval without hard-deleting it.
                matching = [
                    f
                    for f in current
                    if f.subject == op.subject and f.predicate == op.predicate
                ]
                for fact in matching:
                    await self._temporal.close_fact(fact.id, now=now)
                if matching:
                    applied.append(op)

        return applied

    # ------------------------------------------------------------------
    # consolidate
    # ------------------------------------------------------------------

    async def consolidate(self, user_id: str) -> list[FactOp]:
        """Orchestrate propose → apply for *user_id*.

        Pulls recent episodics via the gate and current facts from the temporal
        store, then runs propose + apply.
        """
        # Recall recent episodics (up to 50).
        episodics = await self._gate.recall(
            MemoryQuery(user_id=user_id, text="", top_k=50)
        )
        # Filter to episodic kind only (fact_memories are also returned by recall).
        episodics = [m for m in episodics if m.kind is MemoryKind.EPISODIC]

        existing_facts = await self._temporal.current_facts(user_id=user_id)

        batch = await self.propose(user_id, episodics, existing_facts)
        return await self.apply(user_id, batch)

    # ------------------------------------------------------------------
    # decay_confidence
    # ------------------------------------------------------------------

    async def decay_confidence(
        self,
        user_id: str,
        *,
        half_life_days: float = 30.0,
        now: datetime,
        stale_threshold: float = 0.2,
    ) -> list[TemporalFact]:
        """Apply exponential confidence decay based on age since ``last_confirmed``.

        For each currently-valid fact, the confidence is updated to::

            new_conf = original_conf * 0.5 ** (age_days / half_life_days)

        Facts whose decayed confidence falls below *stale_threshold* are returned
        as the "stale" list for re-confirmation.  All updates are persisted via
        ``SqliteTemporalStore.set_confidence``.

        Parameters
        ----------
        user_id:
            User whose facts to decay.
        half_life_days:
            Half-life for exponential decay (default 30 days).
        now:
            Injected current time — deterministic.
        stale_threshold:
            Confidence below which a fact is considered stale (default 0.2).

        Returns
        -------
        list[TemporalFact]
            Facts whose confidence is below *stale_threshold* after decay
            (the fact objects reflect the *pre-decay* state; callers should
            re-query for the updated confidence).
        """
        facts = await self._temporal.current_facts(user_id=user_id)
        stale: list[TemporalFact] = []

        for fact in facts:
            reference = fact.last_confirmed or fact.valid_from
            if reference is None:
                # No timestamp → skip (cannot compute age).
                continue

            # Ensure both datetimes are comparable (both tz-aware or both naive).
            ref_ts = _ensure_comparable(reference, now)
            now_ts = now

            age_seconds = (now_ts - ref_ts).total_seconds()
            age_days = age_seconds / 86400.0

            decayed = fact.confidence * (0.5 ** (age_days / half_life_days))
            # Clamp to [0, 1].
            decayed = max(0.0, min(1.0, decayed))

            await self._temporal.set_confidence(fact.id, decayed)

            if decayed < stale_threshold:
                stale.append(fact)

        return stale

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provider_for_model(self, model: str) -> str:
        """Best-effort: find the provider string for a model from registered bindings."""
        for binding_list in self._router._bindings.values():  # noqa: SLF001
            for binding in binding_list:
                if binding.model == model:
                    return binding.provider
        return "unknown"


# ---------------------------------------------------------------------------
# Timezone normalisation helper
# ---------------------------------------------------------------------------


def _ensure_comparable(ref: datetime, now: datetime) -> datetime:
    """Return *ref* in the same tz-awareness as *now* to allow subtraction."""
    if now.tzinfo is not None and ref.tzinfo is None:
        # now is tz-aware, ref is naive — treat ref as UTC.
        from datetime import timezone

        return ref.replace(tzinfo=timezone.utc)
    if now.tzinfo is None and ref.tzinfo is not None:
        # now is naive, ref is tz-aware — strip tz from ref.
        return ref.replace(tzinfo=None)
    return ref
