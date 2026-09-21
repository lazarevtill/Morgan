"""Episodics into valid-time facts.

``MemoryConsolidator`` reads the recent episodics a project's current facts do not already
predict, asks the model to propose ``FactOp`` operations as schema-validated JSON, and
applies them through the gate.

What it must keep doing:
- A contradiction closes the old interval (``valid_to = now``); nothing is hard-deleted.
- The clock is injected. No ``datetime.now()`` call appears here, so a run is reproducible.
- An ADD whose (subject, predicate, object) already matches a currently-valid fact is a
  NOOP, not a duplicate row.
- It runs when asked. Nothing here is on the path of a recall.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.knowledge.fact_ops import FactOp, FactOpBatch, FactOpKind
from morgan_brain.memory.knowledge.surprise import keep_surprising
from morgan_brain.models import (
    Memory,
    MemoryKind,
    MemoryQuery,
    MemorySource,
    Scope,
    TemporalFact,
)
from morgan_brain.providers.structured import JsonMode, generate_structured
from morgan_brain.providers.wire import ChatClient, ChatMessage

# ---------------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------------


class MemoryConsolidator:
    """Bi-temporal consolidation worker.

    Parameters
    ----------
    gate:
        The MemoryGate (all reads and writes, including ``close_fact`` and
        ``set_confidence``, pass through here — the consolidator holds no raw store).
    client, model, json_mode:
        The chat model that proposes the fact operations, and how it is asked for JSON.
    clock:
        Injected callable returning the current datetime. Never calls
        ``datetime.now()`` internally.
    """

    def __init__(
        self,
        *,
        gate: MemoryGate,
        client: ChatClient,
        model: str,
        clock: Callable[[], datetime],
        json_mode: JsonMode = "json_schema",
    ) -> None:
        self._gate = gate
        self._client = client
        self._model = model
        self._json_mode = json_mode
        self._clock = clock

    # ------------------------------------------------------------------
    # propose
    # ------------------------------------------------------------------

    async def propose(
        self,
        user_id: str,
        episodics: list[Memory],
        existing_facts: list[TemporalFact],
    ) -> FactOpBatch:
        """Ask the model to propose fact operations from episodics + existing facts."""
        episodic_text = (
            "\n".join(f"- [{m.source.value}] {m.content}" for m in episodics) or "(none)"
        )
        facts_text = (
            "\n".join(
                f"- {f.subject} {f.predicate} {f.object} (conf={f.confidence:.2f})"
                for f in existing_facts
            )
            or "(none)"
        )

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
            self._client,
            messages,
            model=self._model,
            schema=FactOpBatch,
            json_mode=self._json_mode,
        )

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    async def apply(self, user_id: str, batch: FactOpBatch, *, project: str) -> list[FactOp]:
        """Apply a batch of fact operations, scoped to *project*.

        Dedup pre-filter: an ADD whose (subject, predicate, object) exactly
        matches a currently-valid fact is silently dropped (treated as NOOP).

        Returns the list of ops that were actually applied (excludes NOOPs and
        deduped ADDs).
        """
        now = self._clock()
        # The current facts are read under the same lock the ops are applied with. Two runs
        # over one database -- a cron job and a manual `morgan consolidate` -- that each read
        # first both saw a new fact as absent and both added it, and a DELETE could close a
        # fact that the other run had already replaced. Holding the lock makes the second run
        # see the first run's result.
        with self._gate.write_transaction():
            return await self._apply(user_id, batch, project=project, now=now)

    async def _apply(
        self, user_id: str, batch: FactOpBatch, *, project: str, now: datetime
    ) -> list[FactOp]:
        """``apply``'s body. The caller holds the write transaction."""
        current = await self._gate.current_facts(user_id=user_id, project=project)
        current_set = {(f.subject, f.predicate, f.object) for f in current}

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
                        project=project,
                        subject=op.subject,
                        predicate=op.predicate,
                        object=op.object,
                        confidence=op.confidence,
                        source=MemorySource.AGENT_INFERRED,
                        author_id=user_id,
                        scope=Scope.PRIVATE,
                    )
                )
                applied.append(op)

            elif op.op is FactOpKind.UPDATE:
                # upsert_fact closes any existing (subject, predicate) interval
                # and opens a new one — this is the "supersede not delete" pattern.
                await self._gate.upsert_fact(
                    TemporalFact(
                        user_id=user_id,
                        project=project,
                        subject=op.subject,
                        predicate=op.predicate,
                        object=op.object,
                        confidence=op.confidence,
                        source=MemorySource.AGENT_INFERRED,
                        author_id=user_id,
                        scope=Scope.PRIVATE,
                    )
                )
                applied.append(op)

            elif op.op is FactOpKind.DELETE:
                # Close the currently-valid fact's interval without hard-deleting it.
                # Anti-amnesia guard: the consolidator (agent-inferred) must NEVER erase a
                # fact the user explicitly stated — that is exactly the "low-frequency,
                # high-importance fact silently vanishes" failure the 2026 memory literature
                # flags (e.g. "never deploy on Friday"). User-stated facts evolve only via a
                # new user-stated supersession, never an inferred DELETE.
                matching = [
                    f
                    for f in current
                    if f.subject == op.subject
                    and f.predicate == op.predicate
                    and f.source is not MemorySource.USER_STATED
                ]
                for fact in matching:
                    await self._gate.close_fact(fact.id, user_id=user_id, project=project, now=now)
                if matching:
                    applied.append(op)

        return applied

    # ------------------------------------------------------------------
    # consolidate
    # ------------------------------------------------------------------

    async def consolidate(self, user_id: str, *, project: str) -> list[FactOp]:
        """Orchestrate propose → apply for *user_id*, scoped to *project*.

        Pulls recent episodics via the gate and current facts from the temporal
        store, then runs propose + apply.
        """
        # Recall recent episodics (up to 50).
        episodics = await self._gate.recall(
            MemoryQuery(user_id=user_id, project=project, text="", top_k=50)
        )
        # Filter to episodic kind only (fact_memories are also returned by recall).
        episodics = [m for m in episodics if m.kind is MemoryKind.EPISODIC]

        existing_facts = await self._gate.current_facts(user_id=user_id, project=project)

        # Surprise-gate: consolidate what the current model did NOT already predict.
        # Neuro-grounded (the hippocampus preferentially encodes prediction errors): episodics
        # whose content is already covered by current facts carry little new signal, so we skip
        # them and focus the LLM call on the surprising remainder — cheaper and better-targeted.
        episodics = keep_surprising(episodics, existing_facts)

        batch = await self.propose(user_id, episodics, existing_facts)
        return await self.apply(user_id, batch, project=project)

    # ------------------------------------------------------------------
    # decay_confidence
    # ------------------------------------------------------------------

    async def decay_confidence(
        self,
        user_id: str,
        *,
        project: str,
        half_life_days: float = 30.0,
        now: datetime,
        stale_threshold: float = 0.2,
        protected_floor: float = 0.5,
    ) -> list[TemporalFact]:
        """Apply exponential confidence decay based on age since ``last_confirmed``.

        For each currently-valid fact, the confidence is updated to::

            new_conf = original_conf * 0.5 ** (age_days / half_life_days)

        Facts whose decayed confidence falls below *stale_threshold* are returned
        as the "stale" list for re-confirmation.  All updates are persisted via
        ``MemoryGate.set_confidence``.

        Parameters
        ----------
        user_id:
            User whose facts to decay.
        project:
            Project to scope the decay to.
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
        facts = await self._gate.current_facts(user_id=user_id, project=project)
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

            # Importance-weighted retention: a fact the user explicitly stated never decays
            # below ``protected_floor``, so high-importance / low-frequency user statements are
            # not silently lost to staleness. Agent-inferred and tool-observed facts decay
            # freely. This is the retention half of the hoarding-vs-amnesia tradeoff.
            if fact.source is MemorySource.USER_STATED:
                decayed = max(decayed, protected_floor)

            await self._gate.set_confidence(
                fact.id, user_id=user_id, project=project, value=decayed
            )

            if decayed < stale_threshold:
                stale.append(fact)

        return stale


def _ensure_comparable(ref: datetime, now: datetime) -> datetime:
    """Return *ref* in the same tz-awareness as *now* to allow subtraction."""
    if now.tzinfo is not None and ref.tzinfo is None:
        # now is tz-aware, ref is naive — treat ref as UTC.

        return ref.replace(tzinfo=UTC)
    if now.tzinfo is None and ref.tzinfo is not None:
        # now is naive, ref is tz-aware — strip tz from ref.
        return ref.replace(tzinfo=None)
    return ref
