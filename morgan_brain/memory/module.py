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
from morgan_brain.memory.embedder import Embedder
from morgan_brain.memory.gate import ForgetReport, RecallOutcome, RecallReason
from morgan_brain.memory.knowledge.extract import extract_entity_names, words
from morgan_brain.memory.recall import language
from morgan_brain.memory.recall.floor import answer_margin, should_answer
from morgan_brain.memory.recall.fusion import reciprocal_rank_fusion
from morgan_brain.memory.store import projects
from morgan_brain.memory.store import tables as registry
from morgan_brain.memory.store.db import write_transaction
from morgan_brain.memory.store.entities import EntityIndex, delete_entities
from morgan_brain.memory.store.episodic import EpisodicStore, delete_memories
from morgan_brain.memory.store.fts import FtsIndex, delete_keywords
from morgan_brain.memory.store.history import delete_history
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
from morgan_brain.models import (
    PERSONAL_PROJECT,
    Entity,
    Memory,
    MemoryKind,
    MemoryQuery,
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

    async def store(self, memory: Memory) -> str:
        """Write *memory* to every index at once, as one transaction.

        Entities are extracted here when the caller supplied none. There is exactly one write
        path on purpose: a memory indexed by one signal and invisible to another is found by a
        search that should not find it, or missed by one that should.

        The project's ``projects`` row is registered in the same transaction
        (``store/projects.py::register``), so a project first written to after migration step 7
        seeded the table has a row as well -- and a store that fails leaves none.
        """
        if memory.created_at is None:
            memory.created_at = self._clock()
        if not memory.entities:
            memory.entities = [Entity(name=n) for n in extract_entity_names(memory.content)]
        # The embedding awaits a model server, so it happens before the write lock is taken: the
        # lock is never held across a model call.
        vector = await self._embedder.embed(memory.content)
        memory.embedding = vector
        # Now, not `memory.created_at`: the row records when Morgan first wrote to the project,
        # and an import carries the timestamps of conversations years old.
        registered_at = self._clock()
        # One transaction for all four indexes. Written one at a time, an erasure of the
        # project from another process could land between two of them and leave the rest --
        # with the memory's text -- behind for a memory that no longer exists; and a failure
        # part-way left a memory stored in some indexes and missing from others. The vector
        # upsert is awaited but never suspends: it is SQL on this connection, nothing else.
        with write_transaction(self._conn):
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
        return memory.id

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
        vec_hits = await self._vectors.search(
            user_id=query.user_id,
            vector=q_vector,
            top_k=query.top_k * 2,
            project=project,
        )
        # The floor judges on vector evidence alone, so it rules before anything else is
        # gathered: a decline returns nothing, facts included. Judged after the fact merge, a
        # decline dropped the facts along with the memories and said nothing about why.
        verdict = self._floor_verdict(query, project, vec_hits)
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
        self, query: MemoryQuery, project: str | None, vec_hits: list[VectorHit]
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
        )
        ranked = {h.id for h in vec_hits[: query.top_k]}
        answered = should_answer(
            margin=margin,
            threshold=self._floor_margin,
            has_exact_match=any(memory_id in ranked for memory_id in entity_ranking),
        )
        return None if answered else "declined"

    async def upsert_fact(self, fact: TemporalFact) -> str:
        """Assert *fact*, registering its project in the same transaction.

        The registration is here rather than in the temporal store because ``projects`` is not
        that store's table: ``SqliteTemporalStore`` is built over connections that have no
        such table at all. The store's own ``write_transaction`` joins this one as a savepoint,
        so the fact and the row still commit or roll back together. The upsert is awaited but
        never suspends: it is SQL on this connection, nothing else, so nothing awaits real I/O
        while the write lock is held.
        """
        now = self._clock()
        with write_transaction(self._conn):
            projects.register(self._conn, fact.project, now=now)
            return await self._temporal.upsert_fact(fact, now=now)

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
