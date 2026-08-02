"""MemoryModule — the interfaces.MemoryStore implementation.

Recall is multi-signal: vector (semantic) + FTS5 (keyword) + entity overlap, combined with
reciprocal rank fusion (the single rerank layer). Facts are delegated to the bi-temporal store.
All access is user-scoped; callers reach it only through the MemoryGate.

Every signal is durable: the vector index, the keyword index, the entity index, and the
episodic records themselves each live in SQLite, so recall survives a process restart. Episodic
rehydration reads the full record from ``EpisodicStore`` -- never a subset carried in a vector
payload -- so a memory recovered after a restart is exactly the one that was stored.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Callable

from morgan_brain.interfaces.memory import ForgetReport
from morgan_brain.models.memory import (
    DEFAULT_PROJECT,
    Memory,
    MemoryKind,
    MemoryQuery,
    TemporalFact,
)
from morgan_brain.modules.memory.indexing.embedder import Embedder
from morgan_brain.modules.memory.retrieval.entities import EntityIndex
from morgan_brain.modules.memory.retrieval.fts import FtsIndex
from morgan_brain.modules.memory.retrieval.fusion import reciprocal_rank_fusion
from morgan_brain.modules.memory.stores.episodic import EpisodicStore
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import VectorIndex, VectorRecord


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


class MemoryModule:
    def __init__(
        self,
        *,
        embedder: Embedder,
        vectors: VectorIndex,
        temporal: SqliteTemporalStore,
        clock: Callable[[], datetime],
        fts: FtsIndex,
        entities: EntityIndex,
        episodics: EpisodicStore,
    ) -> None:
        self._embedder = embedder
        self._vectors = vectors
        self._temporal = temporal
        self._clock = clock
        self._fts = fts
        self._entities = entities
        self._episodics = episodics

    async def store(self, memory: Memory) -> str:
        if memory.created_at is None:
            memory.created_at = self._clock()
        vector = await self._embedder.embed(memory.content)
        memory.embedding = vector
        self._episodics.put(memory)
        await self._vectors.upsert(
            VectorRecord(
                id=memory.id,
                user_id=memory.user_id,
                project=memory.project,
                vector=vector,
                payload={"content": memory.content, "user_id": memory.user_id},
            )
        )
        self._fts.add(memory.id, memory.content, user_id=memory.user_id, project=memory.project)
        self._entities.add(
            memory.id,
            [e.name for e in memory.entities],
            user_id=memory.user_id,
            project=memory.project,
        )
        return memory.id

    async def recall(self, query: MemoryQuery) -> list[Memory]:
        # None means "no project filter" at the store layer -- the cross-project escape hatch.
        project = None if query.all_projects else query.project
        q_vector = await self._embedder.embed(query.text)
        vec_hits = await self._vectors.search(
            user_id=query.user_id, vector=q_vector, top_k=query.top_k * 2, project=project
        )
        vector_ranking = [h.id for h in vec_hits]
        fts_ranking = self._fts.search(
            query.text, user_id=query.user_id, top_k=query.top_k * 2, project=project
        )
        entity_ranking = self._entities.search(
            {t for t in query.text.split()},
            user_id=query.user_id,
            top_k=query.top_k * 2,
            project=project,
        )

        fused_ids = reciprocal_rank_fusion([vector_ranking, fts_ranking, entity_ranking])
        episodic = [m for m in (self._episodics.get(mid) for mid in fused_ids) if m is not None]
        # Defense in depth: every signal above is already project-scoped, but fusion resolves
        # ids through episodic rehydration, which isn't -- drop anything that slipped through.
        if not query.all_projects:
            episodic = [m for m in episodic if m.project == query.project]

        # Currently-valid facts are authoritative; surface them alongside episodic recall.
        # Phase 1 includes all current facts (volume is small until Phase 2 extraction);
        # relevance-ranking of facts is a Phase 2 concern.
        facts = await self._temporal.current_facts(user_id=query.user_id, project=project)
        fact_memories = [
            Memory(
                user_id=query.user_id,
                project=f.project,
                kind=MemoryKind.SEMANTIC,
                content=f"{f.subject} {f.predicate} {f.object}".replace("_", " "),
                source=f.source,
            )
            for f in facts
        ]
        return (fact_memories + episodic)[: query.top_k]

    async def upsert_fact(self, fact: TemporalFact) -> str:
        return await self._temporal.upsert_fact(fact, now=self._clock())

    async def current_facts(
        self,
        *,
        user_id: str,
        subject: str | None = None,
        project: str | None = DEFAULT_PROJECT,
        all_projects: bool = False,
    ) -> list[TemporalFact]:
        resolved_project = None if all_projects else project
        return await self._temporal.current_facts(
            user_id=user_id, subject=subject, project=resolved_project
        )

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

    async def forget(self, *, user_id: str, project: str) -> ForgetReport:
        """Erase everything *user_id* stored under *project*, in one transaction.

        Every durable index lives in the same SQLite database (the point of Task 7/13A), so
        the affected memory ids are collected first and every dependent row is deleted with
        plain SQL inside a single ``BEGIN IMMEDIATE`` -- including the vector rows, which are
        NOT erased via ``self._vectors.delete()`` because that call commits on its own and
        would break the atomicity a single connection exists to provide.

        Champion preprompts are NOT erased: a promoted champion may embed text mined from a
        forgotten conversation and cannot be un-learned, only rolled back. Flagging affected
        versions requires the ``PromptRegistry``, which this module does not hold and no
        caller wires in, so ``champions_flagged`` stays empty -- for the owner to review by
        hand until that wiring exists.

        ``vec_items``/``vec_meta`` (only present with the sqlite vector backend) and
        ``interaction_signals``/``session_history`` (only present once a ``SignalStore`` or
        ``SessionHistoryStore`` has opened on this same connection) are each optional parts
        of the shared database -- deleted when present, skipped (not an error) when not,
        since there is nothing under *project* to forget from a table that was never created.
        Every table skipped this way is named in ``report.tables_skipped`` so a caller can
        tell "erased zero" apart from "nothing to erase from" (see ``ForgetReport``).
        """
        conn = self._episodics._conn  # forget() owns the whole database
        ids = [
            str(r["id"])
            for r in conn.execute(
                "SELECT id FROM memories WHERE user_id = ? AND project = ?", (user_id, project)
            )
        ]
        report = ForgetReport(memories=len(ids))
        placeholders = ",".join("?" * len(ids))
        has_vectors = _table_exists(conn, "vec_items") and _table_exists(conn, "vec_meta")
        has_signals = _table_exists(conn, "interaction_signals")
        has_history = _table_exists(conn, "session_history")
        if not has_vectors:
            report.tables_skipped.append("vec_items")
        if not has_signals:
            report.tables_skipped.append("interaction_signals")
        if not has_history:
            report.tables_skipped.append("session_history")

        conn.execute("BEGIN IMMEDIATE")
        try:
            if ids:
                conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
                conn.execute(f"DELETE FROM fts_memories WHERE memory_id IN ({placeholders})", ids)
                conn.execute(
                    f"DELETE FROM memory_entities WHERE memory_id IN ({placeholders})", ids
                )
                if has_vectors:
                    # Vectors live in this same database, so they go inside the transaction.
                    conn.execute(
                        f"DELETE FROM vec_items WHERE rowid IN "
                        f"(SELECT rowid FROM vec_meta WHERE id IN ({placeholders}))",
                        ids,
                    )
                    conn.execute(f"DELETE FROM vec_meta WHERE id IN ({placeholders})", ids)
            report.facts = conn.execute(
                "DELETE FROM facts WHERE user_id = ? AND project = ?", (user_id, project)
            ).rowcount
            if has_signals:
                report.signals = conn.execute(
                    "DELETE FROM interaction_signals WHERE user_id = ? AND project = ?",
                    (user_id, project),
                ).rowcount
            if has_history:
                report.history = conn.execute(
                    "DELETE FROM session_history WHERE user_id = ? AND project = ?",
                    (user_id, project),
                ).rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        conn.execute("VACUUM")  # cannot run inside a transaction
        return report
