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

import json
import sqlite3
from collections.abc import Callable
from datetime import datetime

from morgan_brain.memory.embedder import Embedder
from morgan_brain.memory.gate import ForgetReport
from morgan_brain.memory.knowledge.extract import extract_entity_names, words
from morgan_brain.memory.knowledge.schema_classifier import SemanticIndexBuilder
from morgan_brain.memory.recall.floor import answer_margin, should_answer
from morgan_brain.memory.recall.fusion import reciprocal_rank_fusion
from morgan_brain.memory.recall.semantic_index import SemanticIndex
from morgan_brain.memory.store.db import write_transaction
from morgan_brain.memory.store.entities import EntityIndex
from morgan_brain.memory.store.episodic import EpisodicStore
from morgan_brain.memory.store.fts import FtsIndex
from morgan_brain.memory.store.temporal import SqliteTemporalStore
from morgan_brain.memory.store.vectors import SqliteVectorIndex, VectorHit, VectorRecord
from morgan_brain.models import (
    DEFAULT_PROJECT,
    Entity,
    Memory,
    MemoryKind,
    MemoryQuery,
    TemporalFact,
)

#: Everything `forget()` erases that is *derived* from the memories it is erasing: the
#: semantic upper index. Each entry is a literal statement rather than a table name to
#: interpolate -- see the note at the call site.
_DERIVED_TABLE_DELETES: dict[str, str] = {
    "mem_entity_edges": "DELETE FROM mem_entity_edges WHERE user_id = ? AND project = ?",
    "mem_schema_edges": "DELETE FROM mem_schema_edges WHERE user_id = ? AND project = ?",
    "mem_entity_nodes": "DELETE FROM mem_entity_nodes WHERE user_id = ? AND project = ?",
    "mem_schemas": "DELETE FROM mem_schemas WHERE user_id = ? AND project = ?",
}


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
        vectors: SqliteVectorIndex,
        temporal: SqliteTemporalStore,
        clock: Callable[[], datetime],
        fts: FtsIndex,
        entities: EntityIndex,
        episodics: EpisodicStore,
        semantic: SemanticIndex,
        index_builder: SemanticIndexBuilder,
        floor_margin: float | None = None,
    ) -> None:
        self._embedder = embedder
        self._vectors = vectors
        self._temporal = temporal
        self._clock = clock
        self._fts = fts
        self._entities = entities
        self._episodics = episodics
        self._semantic = semantic
        self._index_builder = index_builder
        self._floor_margin = floor_margin

    @property
    def _conn(self) -> sqlite3.Connection:
        """The one connection every index shares, so a write across indexes is one transaction."""
        return self._episodics._conn

    async def store(self, memory: Memory) -> str:
        """Write *memory* to every index at once, as one transaction.

        Entities are extracted here when the caller supplied none, and the memory is filed
        into the semantic upper index in the same call. There is exactly one write path on
        purpose: a memory indexed by one signal and invisible to another is the failure
        that routing turns into lost recall.
        """
        if memory.created_at is None:
            memory.created_at = self._clock()
        if not memory.entities:
            memory.entities = [Entity(name=n) for n in extract_entity_names(memory.content)]
        # Everything that awaits real work -- the embedding, the schema classification --
        # happens before the write lock is taken, so the lock is never held across a model call.
        vector = await self._embedder.embed(memory.content)
        memory.embedding = vector
        plan = await self._index_builder.plan(
            user_id=memory.user_id, project=memory.project, memories=[memory]
        )
        # One transaction for all five indexes. Written one at a time, an erasure of the
        # project from another process could land between two of them and leave the rest --
        # with the memory's text -- behind for a memory that no longer exists; and a failure
        # part-way left a memory stored in some indexes and missing from others. The vector
        # upsert is awaited but never suspends: it is SQL on this connection, nothing else.
        with write_transaction(self._conn):
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
            self._index_builder.apply(plan)
        return memory.id

    async def get(self, memory_id: str, *, user_id: str) -> Memory | None:
        """One memory by id, scoped to its owner.

        The id alone would be enough to find the row; the owner check is what stops an id
        guessed or carried over from another scope from reading across it.
        """
        memory = self._episodics.get(memory_id)
        return memory if memory is not None and memory.user_id == user_id else None

    async def recall(self, query: MemoryQuery) -> list[Memory]:
        # None means "no project filter" at the store layer -- the cross-project escape hatch.
        project = None if query.all_projects else query.project
        restrict_ids = self._route(query)
        q_vector = await self._embedder.embed(query.text)
        vec_hits = await self._vectors.search(
            user_id=query.user_id,
            vector=q_vector,
            top_k=query.top_k * 2,
            project=project,
            restrict_ids=restrict_ids,
        )
        vector_ranking = [h.id for h in vec_hits]
        fts_ranking = self._fts.search(
            query.text,
            user_id=query.user_id,
            top_k=query.top_k * 2,
            project=project,
            restrict_ids=restrict_ids,
        )
        # words(), not split(): the index matches a name exactly, and a raw split leaves the
        # punctuation attached, so "harbor?" at the end of a question never matched "harbor".
        entity_ranking = self._entities.search(
            set(words(query.text)),
            user_id=query.user_id,
            top_k=query.top_k * 2,
            project=project,
            restrict_ids=restrict_ids,
        )

        fused_ids = reciprocal_rank_fusion([vector_ranking, fts_ranking, entity_ranking])
        episodic = [m for m in (self._episodics.get(mid) for mid in fused_ids) if m is not None]
        # Defense in depth: every signal above is already project-scoped, but fusion resolves
        # ids through episodic rehydration, which isn't -- drop anything that slipped through.
        if not query.all_projects:
            episodic = [m for m in episodic if m.project == query.project]

        # Currently-valid facts are authoritative, so they are surfaced alongside episodic
        # recall -- but alongside, never instead of. This used to prepend every fact and then
        # truncate, so once a project held top_k facts no episodic memory could be returned
        # at all, however exactly it matched. current_facts has no limit, so that threshold
        # is crossed silently as consolidation runs, and the probe harness stores no facts
        # and could never see it. Verbatim memories also measure better than extracted
        # artifacts on the published comparisons, so crowding them out loses twice.
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
        merged = _merge_facts_and_episodics(fact_memories, episodic, query.text, query.top_k)
        if not self._answer_is_worth_returning(vec_hits, entity_ranking, query.top_k):
            return []
        return merged

    def _answer_is_worth_returning(
        self, vec_hits: list[VectorHit], entity_ranking: list[str], top_k: int
    ) -> bool:
        """Whether this query found anything, or only the nearest of many unrelated things.

        Off unless a threshold is configured: the right value depends on the corpus and the
        embedding model, and shipping someone else's constant would reject real answers
        quietly. See ``recall.floor`` for why the test is a margin rather than a similarity.

        An exact entity match overrules the margin only when the memory it found is also one
        the vector search ranked within ``top_k``. The entity signal matches every word of
        the question, so on a real corpus some word is nearly always stored on some memory:
        counting any match let 31 of 38 unanswerable questions through the floor on the
        owner's archive. A genuine identifier hit is ranked by the vector search too.
        """
        if self._floor_margin is None:
            return True
        ranked = {h.id for h in vec_hits[:top_k]}
        return should_answer(
            margin=answer_margin([h.score for h in vec_hits]),
            threshold=self._floor_margin,
            has_exact_match=any(memory_id in ranked for memory_id in entity_ranking),
        )

    def _route(self, query: MemoryQuery) -> list[str] | None:
        """Ask the semantic upper index for a candidate pool, or ``None`` to search all.

        Cross-project recall is deliberately never routed: the index is built per
        ``(user_id, project)``, so a pool derived from one project would narrow a search
        that was explicitly asked to cross them -- turning the escape hatch into a
        stricter filter than the default. ``None`` here is the honest answer.

        The pool is advisory in one direction only. Every signal treats ``None`` as
        "search everything", so a routing miss costs precision, never recall.
        """
        if query.all_projects:
            return None
        return self._semantic.route(
            query.text.split(), user_id=query.user_id, project=query.project
        )

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

        Every index lives in the same SQLite database, so the affected memory ids are
        collected first and every dependent row -- including the vectors -- is deleted inside
        a single write transaction. ``session_history`` is optional (present once a
        ``SessionHistoryStore`` has opened on this connection); when absent it is named in
        ``report.tables_skipped`` rather than counted as zero.
        """
        conn = self._conn

        # The ids are selected inside the write transaction, which holds the lock from its
        # first statement. Selecting before the lock left a window in which another process --
        # morgan-mcp storing a memory while `morgan forget` runs -- could insert a memory for
        # this project between the SELECT and the DELETE: the new row is absent from `ids`,
        # survives the erasure, and forget() still reports success. Holding the lock for the
        # whole read-then-delete sequence is what makes the id list authoritative.
        with write_transaction(conn):
            ids = [
                str(r["id"])
                for r in conn.execute(
                    "SELECT id FROM memories WHERE user_id = ? AND project = ?",
                    (user_id, project),
                )
            ]
            report = ForgetReport(memories=len(ids))
            # The id list is bound once as a JSON array and expanded by json_each, so every
            # statement below stays a literal and a project with more memories than
            # SQLITE_MAX_VARIABLE_NUMBER still erases in one statement each.
            id_json = json.dumps(ids)
            has_history = _table_exists(conn, "session_history")
            if not has_history:
                report.tables_skipped.append("session_history")

            if ids:
                conn.execute(
                    "DELETE FROM memories WHERE id IN (SELECT value FROM json_each(?))",
                    (id_json,),
                )
                conn.execute(
                    "DELETE FROM fts_memories WHERE memory_id IN (SELECT value FROM json_each(?))",
                    (id_json,),
                )
                conn.execute(
                    "DELETE FROM memory_entities "
                    "WHERE memory_id IN (SELECT value FROM json_each(?))",
                    (id_json,),
                )
                # Vectors live in this same database, so they go inside the transaction.
                conn.execute(
                    "DELETE FROM vec_items WHERE rowid IN "
                    "(SELECT rowid FROM vec_meta WHERE id IN (SELECT value FROM json_each(?)))",
                    (id_json,),
                )
                conn.execute(
                    "DELETE FROM vec_meta WHERE id IN (SELECT value FROM json_each(?))",
                    (id_json,),
                )
            report.facts = conn.execute(
                "DELETE FROM facts WHERE user_id = ? AND project = ?", (user_id, project)
            ).rowcount
            # The semantic upper index is *derived* from the memories above, so it belongs
            # inside the same transaction rather than being cleaned up afterwards. Each
            # statement is a literal, keyed by table name rather than assembled by
            # interpolating one: a table name cannot be a bound parameter.
            for table, statement in _DERIVED_TABLE_DELETES.items():
                if _table_exists(conn, table):
                    report.index_entries += conn.execute(statement, (user_id, project)).rowcount
                elif table not in report.tables_skipped:
                    report.tables_skipped.append(table)
            if has_history:
                report.history = conn.execute(
                    "DELETE FROM session_history WHERE user_id = ? AND project = ?",
                    (user_id, project),
                ).rowcount

        conn.execute("VACUUM")  # cannot run inside a transaction
        return report


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
