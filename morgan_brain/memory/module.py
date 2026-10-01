"""MemoryModule — the interfaces.MemoryStore implementation.

Recall is two signals, vector (semantic) and FTS5 (keyword), combined with reciprocal rank
fusion (the single rerank layer). The entity index is the relevance floor's evidence of an exact
name match, not a third ranking. Facts are delegated to the bi-temporal store.
All access is user-scoped; callers reach it only through the MemoryGate.

Every signal is durable: the vector index, the keyword index, the entity index, and the
episodic records themselves each live in SQLite, so recall survives a process restart. Episodic
rehydration reads the full record from ``EpisodicStore`` -- never a subset carried in a vector
payload -- so a memory recovered after a restart is exactly the one that was stored.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime

import structlog

from morgan_brain.memory.checked_embedder import CheckedEmbedder
from morgan_brain.memory.checkpoints import (
    CHECKPOINT_PREDICATE,
    AmbiguousCheckpoint,
    StaleCheckpoint,
)
from morgan_brain.memory.embedder import Embedder
from morgan_brain.memory.errors import EvidenceChanged
from morgan_brain.memory.evidence import ScopedEvidenceReader, fact_memory
from morgan_brain.memory.gate import EvidenceResult, ForgetReport, RecallOutcome, RecallReason
from morgan_brain.memory.knowledge.basis import (
    MAX_FACT_INPUTS,
    MAX_SOURCE_INPUTS,
    ConsolidationBasis,
    ConsolidationInput,
    ConsolidationInputLimit,
    SourceBasis,
    StaleConsolidationProposal,
    event_fingerprint,
    fact_fingerprint,
)
from morgan_brain.memory.knowledge.extract import extract_entity_names, words
from morgan_brain.memory.recall import language
from morgan_brain.memory.recall.floor import answer_margin, should_answer
from morgan_brain.memory.recall.fusion import reciprocal_rank_fusion
from morgan_brain.memory.revisions import EventCandidates, RevisionError, RevisionResolver, instant
from morgan_brain.memory.store import erasure as erasure_store
from morgan_brain.memory.store import projects
from morgan_brain.memory.store import tables as registry
from morgan_brain.memory.store.db import read_transaction, write_transaction
from morgan_brain.memory.store.entities import EntityIndex, delete_entities
from morgan_brain.memory.store.episodic import EpisodicStore, delete_memories
from morgan_brain.memory.store.fts import FtsIndex, delete_keywords
from morgan_brain.memory.store.history import SessionHistoryStore, delete_history
from morgan_brain.memory.store.projects import delete_project
from morgan_brain.memory.store.tables import Deleter, Erasure
from morgan_brain.memory.store.temporal import SqliteTemporalStore, delete_facts
from morgan_brain.memory.store.vectors import (
    SqliteVectorIndex,
    VectorHit,
    VectorRecord,
    delete_meta,
    delete_vec_items,
    space_deleter,
    vector_rowids,
)
from morgan_brain.memory.working_context import (
    MAX_CONTEXT_BYTES,
    WORKING_CONTEXT_PREDICATE,
    AttributedSpan,
    WorkingContext,
    WorkingContextEntry,
    WorkingContextList,
    WorkingContextResult,
    validate_working_context,
    working_subject,
)
from morgan_brain.models import (
    PERSONAL_PROJECT,
    Entity,
    Memory,
    MemoryKind,
    MemoryQuery,
    MemorySource,
    MemoryStatus,
    Message,
    Role,
    TemporalFact,
)
from morgan_brain.providers.wire import EmbedOutcome, ProviderRefused, ProviderUnreachable

log = structlog.get_logger("recall")
_forget_log = structlog.get_logger("forget")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


#: Each table ``store/tables.py`` registers, by name, mapped to the deleter of the store that
#: owns it. Another embedding space's vec0 table, named in ``embedding_spaces``, is erased by
#: ``vectors.space_deleter`` for that name instead. A store that registers a table maps its
#: deleter here; ``forget()`` refuses a registered table it finds no deleter for.
_DELETERS: dict[str, Deleter] = {
    "memories": delete_memories,
    "facts": delete_facts,
    "memory_entities": delete_entities,
    "vec_meta": delete_meta,
    "vec_items": delete_vec_items,
    "fts_memories": delete_keywords,
    "session_history": delete_history,
    "projects": delete_project,
}


def _erasure_plan(conn: sqlite3.Connection) -> tuple[list[tuple[str, Deleter]], list[str]]:
    """Each registered table *conn* has, with its store's deleter; and the tables
    ``project_tables`` registers that *conn* does not have.

    Every table is resolved before ``forget()`` deletes anything, so a present table with no
    deleter -- or an embedding space's table without the ``user_id`` and ``project`` columns
    its deleter erases by -- raises here, by name, with nothing erased. An absent table of
    ``NAME_KEYED_PROJECT_TABLES`` is passed over and not reported: ``tables_skipped`` names
    the tables with a ``project`` column. The plan erases the name-keyed tables last: the
    ``projects`` row goes only once no project-keyed table holds a row of the project.
    """
    registered = registry.project_tables(conn)
    skipped = [t for t in registered if not _table_exists(conn, t)]
    present = [t for t in registered if t not in skipped]
    present += [t for t in registry.NAME_KEYED_PROJECT_TABLES if _table_exists(conn, t)]
    spaces = set(registry.space_tables(conn))
    plan: list[tuple[str, Deleter]] = []
    unresolved: list[str] = []
    for table in present:
        deleter = _DELETERS.get(table)
        if deleter is not None:
            plan.append((table, deleter))
        elif table not in spaces:
            unresolved.append(
                f"{table} has no deleter (its store's belongs in memory/module.py's _DELETERS)"
            )
        elif (deleter := space_deleter(conn, table)) is not None:
            plan.append((table, deleter))
        else:
            unresolved.append(
                f"{table}, an embedding space's table, has no user_id and project columns to "
                "erase by"
            )
    if unresolved:
        raise RuntimeError(
            "forget() erased nothing: it cannot erase every registered table: "
            + "; ".join(unresolved)
        )
    return plan, skipped


def _truncate_wal(conn: sqlite3.Connection) -> None:
    """Empty the write-ahead log, whose frames hold what the erasure deleted until they are
    overwritten. A connection in the middle of a read keeps SQLite from truncating it. The
    erasure has committed by then, so that is a warning naming the log, never a failure."""
    busy = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
    if busy:
        main = next(r["file"] for r in conn.execute("PRAGMA database_list") if r["name"] == "main")
        _forget_log.warning(
            "forget.wal-not-truncated",
            wal=f"{main}-wal",
            reason="another connection is reading the database",
            hint="the erased rows stay in the database file and its log until a later checkpoint "
            "completes, once no connection is reading",
        )


class MemoryModule:
    def __init__(
        self,
        *,
        embedder: Embedder,
        vectors: SqliteVectorIndex,
        temporal: SqliteTemporalStore,
        clock: Callable[[], datetime],
        fts: FtsIndex,
        entities: EntityIndex,
        episodics: EpisodicStore,
        floor_margin: float | None = None,
    ) -> None:
        # store() and forget() are each one transaction on the episodic store's connection. An
        # index on any other connection would commit on its own, outside that transaction, and
        # forget() would never reach it -- so a module assembled that way is refused.
        index_connections = {
            "vectors": vectors._conn,
            "temporal": temporal._conn,
            "fts": fts._conn,
            "entities": entities._conn,
        }
        strays = sorted(name for name, c in index_connections.items() if c is not episodics._conn)
        if strays:
            raise ValueError(
                "every index must share the episodic store's connection; "
                f"not shared: {', '.join(strays)}"
            )
        self._embedder = embedder
        self._vectors = vectors
        self._temporal = temporal
        self._clock = clock
        self._fts = fts
        self._entities = entities
        self._episodics = episodics
        self._floor_margin = floor_margin

    @property
    def _conn(self) -> sqlite3.Connection:
        """The one connection every index shares, so a write across indexes is one transaction."""
        return self._episodics._conn

    def write_transaction(self) -> AbstractContextManager[None]:
        """One atomic write across every call made inside the block. See ``store.db``."""
        return write_transaction(self._conn)

    def capture_erasure_generation(self) -> int:
        if self._conn.in_transaction:
            raise ValueError("Preparation requires no active transaction")
        return erasure_store.read_generation(self._conn)

    async def _prepare_event(
        self, memory: Memory, *, allow_replay: bool, expected_generation: int | None = None
    ) -> tuple[Memory, Memory | None, int]:
        if self._conn.in_transaction:
            raise ValueError("Event preparation requires no active transaction")
        with read_transaction(self._conn):
            generation = erasure_store.read_generation(self._conn)
            if expected_generation is not None:
                erasure_store.require_generation(self._conn, expected_generation)
            original = memory.model_copy(deep=True)
            original.revises_event_ids = sorted(original.revises_event_ids)
            original.revision_root_id = RevisionResolver(
                self._episodics, conn=self._conn, at=self._clock()
            ).validate_parents(original)
            if self._episodics.check_replay(original) is not None:
                if not allow_replay:
                    raise ValueError("Atomic turn events must be new; turn replay is not supported")
                return original, None, generation
            self._check_fact_identity(original)
        prepared = original.model_copy(deep=True)
        if prepared.created_at is None:
            prepared.created_at = self._clock()
        if not prepared.entities:
            prepared.entities = [Entity(name=n) for n in extract_entity_names(prepared.content)]
        prepared.embedding = await self._embedder.embed(prepared.content)
        return original, prepared, generation

    def _check_fact_identity(self, memory: Memory) -> None:
        if self._conn.execute(
            "SELECT 1 FROM facts WHERE id=? AND user_id=? AND project=?",
            (memory.id, memory.user_id, memory.project),
        ).fetchone():
            raise ValueError("event ID is already used by a fact")

    async def _persist_prepared_event(
        self, original: Memory, memory: Memory, *, allow_replay: bool
    ) -> None:
        """SQLite-only calls; the caller owns the write transaction, with no external await."""
        memory.revision_root_id = RevisionResolver(
            self._episodics, conn=self._conn, at=self._clock()
        ).validate_parents(original)
        if self._episodics.check_replay(original) is not None:
            if not allow_replay:
                raise ValueError("Atomic turn events must be new; turn replay is not supported")
            return
        self._check_fact_identity(original)
        vector = memory.embedding
        if vector is None:
            raise ValueError("Prepared memory requires an embedding")
        registered_at = self._clock()
        memory.recorded_at = self._clock()
        projects.register(self._conn, memory.project, now=registered_at)
        self._episodics.put(memory)
        await self._vectors.upsert(
            VectorRecord(
                id=memory.id,
                user_id=memory.user_id,
                project=memory.project,
                vector=vector,
                payload={"content": memory.content, "user_id": memory.user_id},
                status=memory.status,
                scope=memory.scope,
                author_id=memory.author_id,
            )
        )
        self._fts.add(
            memory.id,
            memory.content,
            user_id=memory.user_id,
            project=memory.project,
            status=memory.status,
            scope=memory.scope,
            author_id=memory.author_id,
        )
        self._entities.add(
            memory.id,
            [e.name for e in memory.entities],
            user_id=memory.user_id,
            project=memory.project,
        )

    async def store(self, memory: Memory, *, expected_generation: int | None = None) -> str:
        """Prepare outside the lock, then atomically persist every index."""
        if expected_generation is not None and (
            not isinstance(expected_generation, int)
            or isinstance(expected_generation, bool)
            or expected_generation < 0
        ):
            raise ValueError("Store requires a captured erasure generation")
        original, prepared, generation = await self._prepare_event(
            memory, allow_replay=True, expected_generation=expected_generation
        )
        if prepared is None:
            return original.id
        with write_transaction(self._conn):
            erasure_store.require_generation(self._conn, generation)
            await self._persist_prepared_event(original, prepared, allow_replay=True)
        return original.id

    async def store_turn(
        self,
        memories: list[Memory],
        *,
        history: SessionHistoryStore,
        history_entries: list[tuple[str, str, Message]],
        expected_generation: int,
        evidence_basis: list[Memory] | None = None,
        fresh_session: bool = False,
    ) -> None:
        """Exactly two new events and history rows, prepared before one atomic write."""
        if not isinstance(fresh_session, bool):
            raise TypeError("fresh_session must be an explicit boolean")
        if (
            not isinstance(expected_generation, int)
            or isinstance(expected_generation, bool)
            or expected_generation < 0
        ):
            raise ValueError("Atomic turn requires a captured erasure generation")
        if len(memories) != 2 or len(history_entries) != 2:
            raise ValueError("Atomic turn requires exactly two events and history rows")
        if self._conn.in_transaction:
            raise ValueError("Atomic turn preparation requires no active transaction")
        memories = [memory.model_copy(deep=True) for memory in memories]
        history_entries = [
            (key, context, message.model_copy(deep=True))
            for key, context, message in history_entries
        ]
        if evidence_basis is not None:
            if len(evidence_basis) > 32:
                raise ValueError("Atomic turn evidence basis is limited to 32 records")
            evidence_basis = [record.model_copy(deep=True) for record in evidence_basis]
        if not history.shares_connection(self._conn):
            raise ValueError("Atomic turn history must share the memory connection")
        if memories[0].id == memories[1].id:
            raise ValueError("Atomic turn event IDs must be distinct")
        scope = (memories[0].user_id, memories[0].project)
        for memory, (key, context, message), role in zip(
            memories, history_entries, (Role.USER, Role.ASSISTANT), strict=True
        ):
            if (memory.user_id, memory.project) != scope or (
                message.user_id,
                message.project,
                context,
            ) != (*scope, scope[1]):
                raise ValueError("Atomic turn ownership and context must match")
            if message.role is not role or message.content != memory.content:
                raise ValueError("Atomic turn history must match event content and roles")
            if not isinstance(key, str) or not key.startswith(f"{scope[0]}:"):
                raise ValueError("Atomic turn session must match its owner")
            if key != history_entries[0][0]:
                raise ValueError("Atomic turn history must share one session")
        if evidence_basis is not None and any(
            (record.user_id, record.project) != scope for record in evidence_basis
        ):
            raise ValueError("Atomic turn evidence basis must match its owner and context")
        erasure_store.require_generation(self._conn, expected_generation)
        prepared = [
            await self._prepare_event(
                memory, allow_replay=False, expected_generation=expected_generation
            )
            for memory in memories
        ]
        with write_transaction(self._conn):
            erasure_store.require_generation(self._conn, expected_generation)
            if fresh_session and history.recent(
                history_entries[0][0], project=scope[1], user_id=scope[0], limit=1
            ):
                raise ValueError("Fresh session is already occupied")
            if evidence_basis:
                await self._check_turn_evidence(evidence_basis, owner=scope[0], project=scope[1])
            for original, event, _generation in prepared:
                if event is None:
                    raise ValueError("Atomic turn preparation requires new events")
                await self._persist_prepared_event(original, event, allow_replay=False)
            for key, context, message in history_entries:
                history.append(key, message, project=context)

    async def _check_turn_evidence(
        self, evidence_basis: list[Memory], *, owner: str, project: str
    ) -> None:
        """SQL-only validation inside the turn transaction; no provider I/O."""
        if not self._conn.in_transaction:
            raise ValueError("Turn evidence validation requires an active transaction")
        basis_at = self._clock()
        resolved = await self.evidence(
            user_id=owner,
            project=project,
            evidence_ids=[record.id for record in evidence_basis],
            effective_at=basis_at,
        )
        if any(
            record.kind is MemoryKind.SEMANTIC
            and (
                (record.valid_from is not None and instant(record.valid_from) > instant(basis_at))
                or (record.valid_to is not None and instant(record.valid_to) <= instant(basis_at))
            )
            for record in resolved.records
        ):
            raise EvidenceChanged()
        excluded = {"embedding", "entities", "importance"}
        expected = {
            record.id: record.model_dump(mode="json", exclude=excluded) for record in evidence_basis
        }
        actual = {
            record.id: record.model_dump(mode="json", exclude=excluded)
            for record in resolved.records
        }
        if resolved.missing_ids or expected != actual:
            raise EvidenceChanged()

    async def get(self, memory_id: str, *, user_id: str) -> Memory | None:
        """One memory by id, scoped to its owner.

        The id alone would be enough to find the row; the owner check is what stops an id
        guessed or carried over from another scope from reading across it.
        """
        memory = self._episodics.get(memory_id)
        return memory if memory is not None and memory.user_id == user_id else None

    async def check_embedding_space(self) -> None:
        """Re-check the active embedding space right now, for an import's canary.

        Delegates to the embedder's own ``check()`` when it is a ``CheckedEmbedder`` --
        ``FakeEmbedder``, the hash stub, answers no model and checks nothing, the same as
        every other call this module makes to it, so this is a no-op then.
        """
        if isinstance(self._embedder, CheckedEmbedder):
            await self._embedder.check()

    async def recall(self, query: MemoryQuery) -> RecallOutcome:
        # None means "no project filter" at the store layer -- the cross-project escape hatch.
        project = None if query.all_projects else query.project
        embed_started = time.monotonic()
        try:
            q_vector = await self._embedder.embed(query.text)
        except ProviderRefused:
            self._log_recall_done(query, time.monotonic() - embed_started, "refused", reason=None)
            raise
        except ProviderUnreachable as exc:
            self._log_recall_done(query, time.monotonic() - embed_started, exc.outcome, reason=None)
            raise
        except Exception:
            self._log_recall_done(query, time.monotonic() - embed_started, "error", reason=None)
            raise
        embed_elapsed = time.monotonic() - embed_started
        with read_transaction(self._conn):
            resolver = RevisionResolver(
                self._episodics, conn=self._conn, at=query.effective_at or self._clock()
            )
            candidates = resolver.ranked_candidates(user_id=query.user_id, project=project)
            vec_hits = await self._vectors.search(
                user_id=query.user_id,
                vector=q_vector,
                top_k=query.top_k * 2,
                project=project,
                candidates=candidates,
            )
            # The floor judges on vector evidence alone, so it rules before anything else is
            # gathered: a decline returns nothing, facts included. Judged after the fact merge, a
            # decline dropped the facts along with the memories and said nothing about why.
            verdict = self._floor_verdict(query, project, vec_hits, candidates=candidates)
            if verdict == "declined":
                return self._recall_done(
                    query, embed_elapsed, RecallOutcome([], abstained=True, reason="declined")
                )
            vector_ranking = [h.id for h in vec_hits]
            fts_ranking = self._fts.search(
                query.text,
                user_id=query.user_id,
                top_k=query.top_k * 2,
                project=project,
                candidates=candidates,
            )
            # The entity ranking is not fused. A stored name is in the memory's text, so the keyword
            # search already counts it; a third vote for the same evidence pushed paraphrased
            # answers down on a real archive (recall@8 0.83 fused, 0.90 not).
            fused_ids = reciprocal_rank_fusion([vector_ranking, fts_ranking])
            episodic = [m for m in (self._episodics.get(mid) for mid in fused_ids) if m is not None]
            # Defense in depth: every signal above is already project-scoped, but fusion resolves
            # ids through episodic rehydration, which isn't -- drop anything that slipped through.
            if not query.all_projects:
                episodic = [m for m in episodic if m.project == query.project]
            episodic = [resolver.annotate(event) for event in episodic if resolver.is_leaf(event)]

            # Currently-valid facts are authoritative, so they are surfaced alongside episodic
            # recall -- but alongside, never instead of. This used to prepend every fact and then
            # truncate, so once a project held top_k facts no episodic memory could be returned
            # at all, however exactly it matched. current_facts has no limit, so that threshold
            # is crossed silently as consolidation runs, and the probe harness stores no facts
            # and could never see it. Verbatim memories also measure better than extracted
            # artifacts on the published comparisons, so crowding them out loses twice.
            facts = await self._temporal.current_facts(
                user_id=query.user_id, project=project, at=resolver.at
            )
            fact_memories = []
            for fact in facts:
                if fact.predicate == WORKING_CONTEXT_PREDICATE:
                    continue
                state = resolver.support_state(fact)
                if state not in ("inactive_support", "conflicted_support"):
                    fact_memories.append(
                        fact_memory(fact).model_copy(update={"support_state": state})
                    )
            merged = _merge_facts_and_episodics(fact_memories, episodic, query.text, query.top_k)
            # "empty" is decided on what comes back, facts included: a project holding only facts
            # answers with them, and "abstained" beside them would contradict the result.
            if not merged:
                outcome = RecallOutcome([], abstained=True, reason="empty")
            else:
                outcome = RecallOutcome(merged, abstained=False, reason=verdict)
            return self._recall_done(query, embed_elapsed, outcome)

    def _recall_done(
        self, query: MemoryQuery, embed_elapsed_seconds: float, outcome: RecallOutcome
    ) -> RecallOutcome:
        """Log ``recall.done`` for a recall whose embedding came back, and return *outcome*.

        Every such path ends here, so the line carries the reason the caller is given."""
        self._log_recall_done(query, embed_elapsed_seconds, "ok", reason=outcome.reason)
        return outcome

    def _log_recall_done(
        self,
        query: MemoryQuery,
        embed_elapsed_seconds: float,
        embed_outcome: EmbedOutcome,
        *,
        reason: RecallReason | None,
    ) -> None:
        """One line per recall -- answered, declined, or one whose embedding never came back:
        1a's availability trigger reads it, not memory, and a recall that failed to embed is
        exactly the case it counts. *reason* is the outcome's, and ``None`` when the embedding
        never came back, since there is no outcome then. `degraded` is None until 1a's
        keyword-only fallback fills it in."""
        log.info(
            "recall.done",
            embed_latency_ms=round(embed_elapsed_seconds * 1000, 1),
            embed_outcome=embed_outcome,
            degraded=None,
            reason=reason,
            query_language=language.of(query.text),
        )

    def _floor_verdict(
        self,
        query: MemoryQuery,
        project: str | None,
        vec_hits: list[VectorHit],
        *,
        candidates: EventCandidates,
    ) -> RecallReason | None:
        """The relevance floor's verdict: ``"declined"`` when this query found only the nearest
        of many unrelated things, ``None`` when it was judged and answered, or why it was not
        judged at all -- ``"no_floor"`` or ``"too_few_to_judge"``.

        Off unless a threshold is configured: the right value depends on the corpus and the
        embedding model, and shipping someone else's constant would reject real answers
        quietly. See ``recall.floor`` for why the test is a margin rather than a similarity.

        An exact entity match overrules the margin only when the memory it found is also one
        the vector search ranked within ``top_k``. The entity index is searched with every word
        of the question, so on a real corpus some word is nearly always stored on some memory:
        counting any match let 31 of 38 unanswerable questions through the floor on the
        owner's archive. A genuine identifier hit is ranked by the vector search too.
        """
        if self._floor_margin is None:
            return "no_floor"
        margin = answer_margin([h.score for h in vec_hits])
        if margin is None:
            # Fewer than floor.MIN_RESULTS_TO_JUDGE hits: no background to judge against, so
            # the results go back unjudged.
            return "too_few_to_judge"
        # words(), not split(): the index matches a name exactly, and a raw split leaves the
        # punctuation attached, so "harbor?" at the end of a question never matched "harbor".
        entity_ranking = self._entities.search(
            set(words(query.text)),
            user_id=query.user_id,
            top_k=query.top_k * 2,
            project=project,
            candidates=candidates,
        )
        ranked = {h.id for h in vec_hits[: query.top_k]}
        answered = should_answer(
            margin=margin,
            threshold=self._floor_margin,
            has_exact_match=any(memory_id in ranked for memory_id in entity_ranking),
        )
        return None if answered else "declined"

    def _source_basis(
        self, event_id: str, *, user_id: str, project: str, resolver: RevisionResolver
    ) -> tuple[Memory, SourceBasis]:
        event = self._episodics.get(event_id, user_id=user_id, project=project)
        if (
            event is None
            or event.kind is not MemoryKind.EPISODIC
            or event.status is not MemoryStatus.STORED
            or event.source not in (MemorySource.USER_STATED, MemorySource.TOOL_OBSERVED)
            or not resolver.is_leaf(event)
        ):
            raise StaleConsolidationProposal("Consolidation source is missing or inactive")
        family = resolver.family(event)
        if family.conflicted:
            raise StaleConsolidationProposal("Consolidation source has an unresolved conflict")
        return event, SourceBasis(
            event.id, event.revision_root_id or event.id, family.leaf_ids, event_fingerprint(event)
        )

    async def capture_consolidation_basis(
        self, *, user_id: str, project: str, event_ids: list[str], generation: int
    ) -> ConsolidationInput:
        """Capture exact prompt inputs after recall, before external generation."""
        if len(event_ids) > MAX_SOURCE_INPUTS or len(set(event_ids)) != len(event_ids):
            raise ConsolidationInputLimit("Consolidation needs at most 50 distinct source IDs")
        with read_transaction(self._conn):
            if erasure_store.read_generation(self._conn) != generation:
                raise StaleConsolidationProposal(
                    "Consolidation preparation crossed a committed forget"
                )
            prepared_at = self._clock()
            resolver = RevisionResolver(self._episodics, conn=self._conn, at=prepared_at)
            sources = [
                self._source_basis(identity, user_id=user_id, project=project, resolver=resolver)
                for identity in event_ids
            ]
            facts = await self._consolidation_facts(user_id, project, resolver)
            if len(facts) > MAX_FACT_INPUTS:
                raise ConsolidationInputLimit(
                    "Consolidation fact inputs exceed the explicit 256 limit"
                )
            basis = ConsolidationBasis(
                user_id,
                project,
                generation,
                prepared_at,
                tuple(source for _, source in sources),
                fact_fingerprint(facts),
                tuple(sorted(fact.id for fact in facts)),
            )
            return ConsolidationInput(basis, tuple(event for event, _ in sources), tuple(facts))

    async def check_consolidation_basis(
        self, basis: ConsolidationBasis, *, effective_at: datetime | None = None
    ) -> datetime:
        """Caller holds the apply write transaction; no I/O may suspend here."""
        if not self._conn.in_transaction:
            raise RuntimeError("Consolidation basis must be checked inside its write transaction")
        effective_at = effective_at if effective_at is not None else self._clock()
        if erasure_store.read_generation(self._conn) != basis.generation:
            raise StaleConsolidationProposal("Consolidation proposal crossed a committed forget")
        resolver = RevisionResolver(self._episodics, conn=self._conn, at=effective_at)
        for expected in basis.sources:
            _, actual = self._source_basis(
                expected.event_id, user_id=basis.user_id, project=basis.project, resolver=resolver
            )
            if actual != expected:
                raise StaleConsolidationProposal("Consolidation source basis changed")
        # The complete current prompt inventory is compared, not only target IDs:
        # a new key or same-ID confidence/confirmation change can alter a proposal.
        facts = await self._consolidation_facts(basis.user_id, basis.project, resolver)
        if fact_fingerprint(facts) != basis.fact_fingerprint:
            raise StaleConsolidationProposal("Consolidation fact basis changed")
        return effective_at

    async def _consolidation_facts(
        self, user_id: str, project: str, resolver: RevisionResolver
    ) -> list[TemporalFact]:
        facts = await self._temporal.current_facts(user_id=user_id, project=project, at=resolver.at)
        return [
            fact
            for fact in facts
            if resolver.support_state(fact) not in ("inactive_support", "conflicted_support")
        ]

    async def upsert_fact(self, fact: TemporalFact, *, now: datetime | None = None) -> str:
        """Assert *fact*, registering its project in the same transaction.

        The registration is here rather than in the temporal store because ``projects`` is not
        that store's table: ``SqliteTemporalStore`` is built over connections that have no
        such table at all. The store's own ``write_transaction`` joins this one as a savepoint,
        so the fact and the row still commit or roll back together. The upsert is awaited but
        never suspends: it is SQL on this connection, nothing else, so nothing awaits real I/O
        while the write lock is held.
        """
        now = now if now is not None else self._clock()
        with write_transaction(self._conn):
            if self._episodics.get(fact.id, user_id=fact.user_id, project=fact.project) is not None:
                raise ValueError("fact ID is already used by an event")
            for event_id in fact.support_event_ids:
                event = self._episodics.get(event_id, user_id=fact.user_id, project=fact.project)
                if (
                    event is None
                    or (event.user_id, event.project) != (fact.user_id, fact.project)
                    or event.kind is not MemoryKind.EPISODIC
                    or event.status is not MemoryStatus.STORED
                    or event.source not in (MemorySource.USER_STATED, MemorySource.TOOL_OBSERVED)
                ):
                    raise ValueError("fact support requires scoped source events")
            if (
                fact.support_event_ids
                and RevisionResolver(self._episodics, conn=self._conn, at=now).support_state(fact)
                != "current"
            ):
                raise RevisionError("stale_revision_basis")
            projects.register(self._conn, fact.project, now=now)
            return await self._temporal.upsert_fact(fact, now=now)

    async def put_checkpoint_fact(
        self,
        fact: TemporalFact,
        *,
        expected_fact_id: str | None,
        predicate: str = CHECKPOINT_PREDICATE,
    ) -> str:
        """Create-only or compare-and-swap the structural checkpoint head under lock."""
        with write_transaction(self._conn):
            now = self._clock()
            heads = await self._temporal.current_facts(
                user_id=fact.user_id, project=fact.project, subject=fact.subject
            )
            heads = [head for head in heads if head.predicate == predicate]
            if not heads:
                effective = await self._temporal.current_facts(
                    user_id=fact.user_id, project=fact.project, subject=fact.subject, at=now
                )
                heads = [head for head in effective if head.predicate == predicate]
            if [head.id for head in heads] != (
                [] if expected_fact_id is None else [expected_fact_id]
            ):
                raise StaleCheckpoint(
                    "checkpoint head changed; prepare against the current fact ID"
                )
            return await self.upsert_fact(fact, now=now)

    async def checkpoint_fact(
        self,
        *,
        user_id: str,
        project: str,
        subject: str,
        predicate: str = CHECKPOINT_PREDICATE,
    ) -> TemporalFact | None:
        """Return the effective checkpoint even when its basis requires rebuilding."""
        with read_transaction(self._conn):
            now = self._clock()
            facts = await self._temporal.current_facts(
                user_id=user_id, project=project, subject=subject, at=now
            )
            facts = [fact for fact in facts if fact.predicate == predicate]
            if not facts:
                return None
            if len(facts) != 1:
                raise AmbiguousCheckpoint([fact.id for fact in facts])
            fact = facts[0]
            state = RevisionResolver(self._episodics, conn=self._conn, at=now).support_state(fact)
            return fact.model_copy(update={"support_state": state})

    async def put_working_context_fact(
        self,
        fact: TemporalFact,
        *,
        expected_fact_id: str | None,
        expected_generation: int,
        evidence_basis: list[Memory],
    ) -> str:
        with write_transaction(self._conn):
            erasure_store.require_generation(self._conn, expected_generation)
            await self._check_turn_evidence(
                evidence_basis, owner=fact.user_id, project=fact.project
            )
            return await self.put_checkpoint_fact(
                fact, expected_fact_id=expected_fact_id, predicate=WORKING_CONTEXT_PREDICATE
            )

    async def working_context_list(
        self, *, user_id: str, project: str, limit: int
    ) -> WorkingContextList:
        # Structural discovery must precede support eligibility: stale views need rebuilding.
        # Output is bounded; the existing temporal store scans effective facts within this scope.
        with read_transaction(self._conn):
            facts = await self._temporal.current_facts(
                user_id=user_id, project=project, at=self._clock()
            )
            identities = set()
            prefix = "working_checkpoint:"
            for fact in facts:
                if fact.predicate != WORKING_CONTEXT_PREDICATE:
                    continue
                if not fact.subject.startswith(prefix):
                    raise ValueError("invalid working context subject")
                identity = fact.subject[len(prefix) :]
                if working_subject(identity) != fact.subject:
                    raise ValueError("invalid working context subject")
                identities.add(identity)
            ordered = sorted(identities)
            items = []
            for identity in ordered[:limit]:
                view = await self.working_context_view(
                    user_id=user_id, project=project, subject=working_subject(identity)
                )
                if view is not None:
                    items.append(WorkingContextEntry(context_id=identity, view=view))
            return WorkingContextList(items=items, truncated=len(ordered) > limit)

    async def working_context_view(
        self, *, user_id: str, project: str, subject: str
    ) -> WorkingContextResult | None:
        with read_transaction(self._conn):
            fact = await self.checkpoint_fact(
                user_id=user_id,
                project=project,
                subject=subject,
                predicate=WORKING_CONTEXT_PREDICATE,
            )
            if fact is None:
                return None
            invalid = WorkingContextResult(fact_id=fact.id, eligibility="invalid_state")
            try:
                if len(fact.object.encode("utf-8")) > MAX_CONTEXT_BYTES:
                    return invalid
                raw = json.loads(fact.object)
                if isinstance(raw, dict) and raw.get("version") != "morgan.working_context.v1":
                    return WorkingContextResult(fact_id=fact.id, eligibility="unsupported_version")
                state = WorkingContext.model_validate(raw)
            except (ValueError, RecursionError):
                return invalid
            resolved = await self.evidence(
                user_id=user_id, project=project, evidence_ids=state.event_ids()
            )
            try:
                validate_working_context(state, resolved.records)
            except ValueError:
                return WorkingContextResult(fact_id=fact.id, eligibility="needs_rebuild")
            sources = {record.id: record for record in resolved.records}
            return WorkingContextResult(
                fact_id=fact.id,
                eligibility="current",
                state=state,
                sources=[
                    AttributedSpan(
                        **span.model_dump(),
                        source=sources[span.event_id].source,
                        author_id=sources[span.event_id].author_id,
                    )
                    for span in state.spans()
                ],
            )

    async def evidence(
        self,
        *,
        user_id: str,
        project: str,
        evidence_ids: list[str],
        effective_at: datetime | None = None,
    ) -> EvidenceResult:
        return await ScopedEvidenceReader(
            self._episodics, self._temporal, conn=self._conn
        ).evidence(
            user_id=user_id,
            project=project,
            evidence_ids=evidence_ids,
            effective_at=effective_at or self._clock(),
        )

    async def record_project(
        self, project: str, *, classification: str, remote: str | None, root: str | None
    ) -> bool:
        """Record what repository *project* is; ``False`` when it has no row to record it in.

        One statement under the write lock (``store/projects.py::record``). A CLI write from
        inside a repository is the only caller: an MCP server never sees the client's checkout.
        """
        with write_transaction(self._conn):
            return projects.record(
                self._conn,
                project,
                classification=classification,
                remote=remote,
                root=root,
            )

    async def current_facts(
        self,
        *,
        user_id: str,
        subject: str | None = None,
        project: str | None = PERSONAL_PROJECT,
        all_projects: bool = False,
        effective_at: datetime | None = None,
    ) -> list[TemporalFact]:
        resolved_project = None if all_projects else project
        with read_transaction(self._conn):
            resolver = RevisionResolver(
                self._episodics, conn=self._conn, at=effective_at or self._clock()
            )
            facts = await self._temporal.current_facts(
                user_id=user_id, subject=subject, project=resolved_project, at=resolver.at
            )
            result = []
            for fact in facts:
                state = resolver.support_state(fact)
                if state not in ("inactive_support", "conflicted_support"):
                    result.append(fact.model_copy(update={"support_state": state}))
            return result

    async def close_fact(
        self, fact_id: str, *, user_id: str, project: str, now: datetime | None = None
    ) -> None:
        resolved_now = now if now is not None else self._clock()
        await self._temporal.close_fact(fact_id, user_id=user_id, project=project, now=resolved_now)

    async def set_confidence(
        self, fact_id: str, *, user_id: str, project: str, value: float
    ) -> None:
        await self._temporal.set_confidence(fact_id, user_id=user_id, project=project, value=value)

    async def distinct_projects(self, user_id: str) -> list[str]:
        """Return the distinct project names *user_id* has stored memories under."""
        return self._episodics.distinct_projects(user_id)

    async def consolidate_enabled(self, project: str) -> bool:
        """``False`` only when *project*'s ``projects`` row says ``consolidate_enabled = 0``;
        a project with no row (``store/projects.py::get``) is consolidated."""
        row = projects.get(self._conn, project)
        return row is None or row.consolidate_enabled

    async def forget(self, *, user_id: str, project: str) -> ForgetReport:
        """Erase everything *user_id* stored under *project*, in one transaction.

        Walks the registry in ``store/tables.py``: each table ``project_tables`` or
        ``NAME_KEYED_PROJECT_TABLES`` names is erased by the deleter its store owns
        (``_erasure_plan``). Every table is resolved before any row is deleted, so a
        registered table with no deleter stops the erasure by name with nothing erased. A
        registered table absent here -- ``session_history``, opened only by
        ``build_memory_context`` -- is named in ``report.tables_skipped`` rather than counted
        as zero. Every index lives in the same SQLite database, so the whole erasure is one
        write transaction; once it has committed, the database is vacuumed and its write-ahead
        log truncated (`_truncate_wal`).
        """
        conn = self._conn
        if conn.in_transaction:
            # The erasure is followed by a VACUUM, which SQLite refuses inside a transaction.
            # Nested in a caller's write, forget() would erase, fail at the vacuum, and have the
            # erasure rolled back with the caller's block -- so it refuses before touching anything.
            raise RuntimeError(
                "forget() cannot run inside a write transaction: it vacuums the database once "
                "the erasure has committed"
            )

        with write_transaction(conn):
            plan, skipped = _erasure_plan(conn)
            erasure_store.advance_generation(conn)
            # The ids are selected inside the write transaction, which holds the lock from its
            # first statement. Selecting before the lock left a window in which another process
            # -- morgan-mcp storing a memory while `morgan forget` runs -- could insert a memory
            # for this project between the SELECT and the DELETE: the new row is absent from
            # `ids`, survives the erasure, and forget() still reports success. Holding the lock
            # for the whole read-then-delete sequence is what makes the id list authoritative.
            ids = [
                str(r["id"])
                for r in conn.execute(
                    "SELECT id FROM memories WHERE user_id = ? AND project = ?",
                    (user_id, project),
                )
            ]
            memory_ids = json.dumps(ids)
            erasure = Erasure(
                user_id=user_id,
                project=project,
                memory_ids=memory_ids,
                vector_rowids=vector_rowids(conn, memory_ids, user_id, project),
            )
            erased = {table: delete(conn, erasure) for table, delete in plan}

        conn.execute("VACUUM")  # cannot run inside a transaction
        _truncate_wal(conn)
        # `ForgetReport` counts what the owner asked to erase: memories, facts and history.
        # The index rows follow from the memories, and a `projects` row is none of the three.
        return ForgetReport(
            memories=len(ids),
            facts=erased.get("facts", 0),
            history=erased.get("session_history", 0),
            tables_skipped=skipped,
        )


#: The share of the window episodic memories are guaranteed when they exist. Facts may use
#: the whole window when little else comes back, so the reservation costs nothing on a
#: project with no episodic hits -- it only stops facts from evicting memories that matched.
_EPISODIC_RESERVE = 0.5


def _merge_facts_and_episodics(
    facts: list[Memory], episodic: list[Memory], query_text: str, top_k: int
) -> list[Memory]:
    """Facts first, but never so many that a matching memory is pushed out of the window.

    Facts are ranked by how much of the query they mention, so the ones that survive a narrow
    budget are the ones asked about. Ordering them arbitrarily would drop the relevant fact as
    readily as an irrelevant one, trading one silent failure for another.
    """
    reserved = min(len(episodic), int(top_k * _EPISODIC_RESERVE))
    budget = max(top_k - reserved, 0)
    if len(facts) > budget:
        terms = {t for t in words(query_text.lower()) if t}
        facts = sorted(
            facts,
            key=lambda m: -len(terms & set(words(m.content.lower()))),
        )[:budget]
    return (facts + episodic)[:top_k]
