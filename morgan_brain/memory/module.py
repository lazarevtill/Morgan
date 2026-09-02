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
from morgan_brain.memory.entities import EntityIndex
from morgan_brain.memory.episodic import EpisodicStore
from morgan_brain.memory.extract import extract_entity_names
from morgan_brain.memory.fts import FtsIndex
from morgan_brain.memory.fusion import reciprocal_rank_fusion
from morgan_brain.memory.gate import ForgetReport
from morgan_brain.memory.schema_classifier import SemanticIndexBuilder
from morgan_brain.memory.semantic_index import SemanticIndex
from morgan_brain.memory.temporal import SqliteTemporalStore
from morgan_brain.memory.vectors import SqliteVectorIndex, VectorRecord
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

    async def store(self, memory: Memory) -> str:
        """Write *memory* to every index at once.

        Entities are extracted here when the caller supplied none, and the memory is filed
        into the semantic upper index in the same call. There is exactly one write path on
        purpose: a memory indexed by one signal and invisible to another is the failure
        that routing turns into lost recall.
        """
        if memory.created_at is None:
            memory.created_at = self._clock()
        if not memory.entities:
            memory.entities = [Entity(name=n) for n in extract_entity_names(memory.content)]
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
        await self._index_builder.index(
            user_id=memory.user_id, project=memory.project, memories=[memory]
        )
        return memory.id

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
        entity_ranking = self._entities.search(
            set(query.text.split()),
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
        a single ``BEGIN IMMEDIATE``. ``session_history`` is optional (present once a
        ``SessionHistoryStore`` has opened on this connection); when absent it is named in
        ``report.tables_skipped`` rather than counted as zero.
        """
        conn = self._episodics._conn  # forget() owns the whole database

        # The write lock is taken BEFORE the ids are selected, not after. Selecting first left
        # a window in which another process could insert a memory for this project between
        # the SELECT and the DELETE: the new row is absent from `ids`, survives the erasure,
        # and forget() still reports success. On the documented 2-process topology the worker
        # writes memories off the bus continuously, so "erase this project" racing an in-flight
        # turn is the normal case, not an exotic one. BEGIN IMMEDIATE blocks other writers for
        # the whole read-then-delete sequence, which is what makes the id list authoritative.
        conn.execute("BEGIN IMMEDIATE")
        try:
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
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        conn.execute("VACUUM")  # cannot run inside a transaction
        return report
