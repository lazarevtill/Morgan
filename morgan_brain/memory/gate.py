"""MemoryGate — the single choke point for all memory reads and writes.

Every store/recall/forget passes through here. It enforces user- and project-scope (the
basis of multi-tenant readiness) and is the one place to add redaction, consent, and audit
later. No caller holds the ``MemoryModule`` directly.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from morgan_brain.memory.migrations import DatabaseNeedsMigration
from morgan_brain.models import PERSONAL_PROJECT, Memory, MemoryQuery, TemporalFact

if TYPE_CHECKING:
    from morgan_brain.memory.module import MemoryModule


@dataclass
class ForgetReport:
    """What a single ``forget()`` call erased.

    ``history`` is ``0`` for two different reasons that used to be indistinguishable: the
    table exists and genuinely had nothing under this project, or it was never created on
    this connection. ``tables_skipped`` names every table that was absent (and therefore not
    touched) so a caller can print "not tracked here" instead of a false "0 erased".
    ``sessions``, ``turns`` and ``digests`` count what the session archive's tables lost.
    """

    memories: int = 0
    facts: int = 0
    history: int = 0
    sessions: int = 0
    turns: int = 0
    digests: int = 0
    tables_skipped: list[str] = field(default_factory=list)


@dataclass
class SessionForgetReport:
    """What a session-grain erasure erased: sessions, turns, links (from either end), digests,
    and how many exclusions it wrote. ``memories_reached`` is always ``False``: memories and
    facts are not keyed by a session, and the report says so rather than printing a 0."""

    sessions: int = 0
    turns: int = 0
    links: int = 0
    digests: int = 0
    excluded: int = 0
    memories_reached: bool = False


#: Why a recall returned what it did. ``empty``: nothing in scope came back, not even a fact
#: (abstained). ``declined``: the relevance floor judged that nothing stood out above the
#: background (abstained). ``too_few_to_judge``: fewer than ``floor.MIN_RESULTS_TO_JUDGE``
#: vector hits, returned unjudged. ``no_floor``: returned, and no floor is configured.
#: ``keyword_only``: reserved for a fallback when embeddings are down, which is not built;
#: nothing emits it. ``None`` on the outcome: the floor judged the query and it answered.
RecallReason = Literal["empty", "declined", "too_few_to_judge", "no_floor", "keyword_only"]


@dataclass(frozen=True)
class RecallOutcome:
    """What one ``recall()`` returned, and why.

    An empty list used to stand for four situations the caller could not tell apart --
    nothing stored, the floor declining, too few results to judge, no floor configured -- and
    the owner cannot act on one without knowing which it is. ``reason`` is the same string the
    ``recall.done`` log line carries.
    """

    memories: list[Memory]
    abstained: bool
    reason: RecallReason | None


class MemoryGate:
    """*read_only_reason*, when given, refuses every write -- ``store``, ``upsert_fact``,
    ``close_fact``, ``set_confidence``, ``forget`` -- with ``DatabaseNeedsMigration`` carrying
    it: the database behind the gate waits for a heavy migration step (``memory.migrations``).
    Reads are unaffected, because they work on the schema the database already has.
    """

    def __init__(self, store: MemoryModule, read_only_reason: str | None = None) -> None:
        self._store = store
        self._read_only_reason = read_only_reason

    @property
    def read_only_reason(self) -> str | None:
        """Why writes are refused, or ``None`` when they are not."""
        return self._read_only_reason

    async def store(self, memory: Memory) -> str:
        self.require_writable()
        self._require_scope(memory.user_id)
        return await self._store.store(memory)

    async def get(self, memory_id: str, *, user_id: str) -> Memory | None:
        """One memory by id, or ``None`` if this owner has no such memory.

        A read like any other, so it comes through the gate: a caller that reached into the
        episodic store directly would be a caller whose scope nobody checked.
        """
        self._require_scope(user_id)
        return await self._store.get(memory_id, user_id=user_id)

    async def recall(self, query: MemoryQuery) -> RecallOutcome:
        self._require_scope(query.user_id)
        return await self._store.recall(query)

    async def check_embedding_space(self) -> None:
        """Re-check the active embedding space now, rather than trusting the check a process
        made on its first embedding call.

        The import canary calls this every ``MORGAN_IMPORT_CANARY_EVERY`` memories
        and once more at the end, so a model that starts answering wrong mid-import is caught
        within one stretch instead of at the end. Not user- or project-scoped -- the embedding
        space is a property of the database, not of any one owner's data -- and not a write:
        it reads the recorded fingerprint and sends its own small embedding request, but
        stores nothing. Raises ``EmbeddingSpaceMismatch`` like any other embedding-space
        failure; a no-op on an embedder that does none of this checking (the hash backend).
        """
        await self._store.check_embedding_space()

    async def upsert_fact(self, fact: TemporalFact) -> str:
        self.require_writable()
        self._require_scope(fact.user_id)
        return await self._store.upsert_fact(fact)

    async def record_project(
        self,
        *,
        user_id: str,
        project: str,
        classification: str,
        remote: str | None,
        root: str | None,
    ) -> bool:
        """Record what repository *project* is: its classification, remote and root.

        A write like any other -- refused with the rest while a heavy migration step waits --
        and the one gate method that carries the owner's remote URL and checkout path, which
        is why it comes through here rather than reaching into the store. It returns whether a
        row was updated: a project with no row (nothing was written under it, or another
        process forgot it) is left without one.

        *user_id* scopes the caller, not the row: ``projects`` has no owner column, because a
        repository's classification is the same for everyone with data in it.
        """
        self.require_writable()
        self._require_scope(user_id, project)
        return await self._store.record_project(
            project, classification=classification, remote=remote, root=root
        )

    async def current_facts(
        self,
        *,
        user_id: str,
        subject: str | None = None,
        project: str | None = PERSONAL_PROJECT,
        all_projects: bool = False,
    ) -> list[TemporalFact]:
        self._require_scope(user_id, None if all_projects else project)
        return await self._store.current_facts(
            user_id=user_id, subject=subject, project=project, all_projects=all_projects
        )

    async def close_fact(
        self, fact_id: str, *, user_id: str, project: str, now: datetime | None = None
    ) -> None:
        self.require_writable()
        self._require_scope(user_id, project)
        await self._store.close_fact(fact_id, user_id=user_id, project=project, now=now)

    async def set_confidence(
        self, fact_id: str, *, user_id: str, project: str, value: float
    ) -> None:
        self.require_writable()
        self._require_scope(user_id, project)
        await self._store.set_confidence(fact_id, user_id=user_id, project=project, value=value)

    async def distinct_projects(self, user_id: str) -> list[str]:
        self._require_scope(user_id)
        return await self._store.distinct_projects(user_id)

    async def consolidate_enabled(self, *, user_id: str, project: str) -> bool:
        """Whether ``consolidate --all-projects`` reaches *project*: ``False`` only when its
        ``projects`` row turns consolidation off.

        A read like any other, so it comes through the gate rather than into the store. A
        project with no row -- the seed never reached it, or the database predates the switch
        -- is consolidated. Like ``record_project``, *user_id* scopes the caller, not the row.
        """
        self._require_scope(user_id, project)
        return await self._store.consolidate_enabled(project)

    def write_transaction(self) -> AbstractContextManager[None]:
        """Make every gate call inside the block one atomic write, under the write lock.

        For a caller whose writes depend on what it reads through the gate: the reads see
        what other processes committed before the block, and nothing else can write until
        the block ends. Nothing inside may await real I/O. ``forget`` is the exception: it
        vacuums once its erasure commits, so it refuses to run inside the block.
        """
        return self._store.write_transaction()

    async def forget(self, *, user_id: str, project: str) -> ForgetReport:
        self.require_writable()
        self._require_scope(user_id, project)
        return await self._store.forget(user_id=user_id, project=project)

    def require_writable(self) -> None:
        """Raise ``DatabaseNeedsMigration`` if writes are refused; every write method does.

        Public for a command whose write comes after costly work -- a model call, an embedding,
        a history row written outside the gate: it refuses first, and nothing else happens.
        """
        if self._read_only_reason is not None:
            raise DatabaseNeedsMigration(self._read_only_reason)

    @staticmethod
    def _require_scope(user_id: str, project: str | None = None) -> None:
        if not user_id:
            raise PermissionError("memory access requires a user_id")
        if project is not None and not project:
            raise PermissionError("memory access requires a project")
