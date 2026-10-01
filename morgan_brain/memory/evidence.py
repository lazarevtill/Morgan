"""Exact scoped source reads, independent of indexes, migrations and model backends."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from morgan_brain.memory.gate import EvidenceResult
from morgan_brain.memory.revisions import RevisionResolver
from morgan_brain.memory.store.db import read_transaction
from morgan_brain.memory.store.episodic import EpisodicStore
from morgan_brain.memory.store.temporal import SqliteTemporalStore
from morgan_brain.models import Memory, MemoryKind, TemporalFact


def fact_memory(fact: TemporalFact) -> Memory:
    """Preserve persisted fact identity and provenance in the existing recall interface."""
    return Memory(
        id=fact.id,
        user_id=fact.user_id,
        project=fact.project,
        kind=MemoryKind.SEMANTIC,
        content=f"{fact.subject} {fact.predicate.replace('_', ' ')} {fact.object}",
        source=fact.source,
        author_id=fact.author_id,
        scope=fact.scope,
        created_at=fact.created_at,
        recorded_at=fact.recorded_at,
        confidence=fact.confidence,
        valid_from=fact.valid_from,
        valid_to=fact.valid_to,
        superseded_by=fact.superseded_by,
        last_confirmed=fact.last_confirmed,
        support_event_ids=fact.support_event_ids,
    )


def validate_source_schema(conn: sqlite3.Connection) -> None:
    required = {
        "memories": {
            "id",
            "user_id",
            "project",
            "kind",
            "source",
            "content",
            "importance",
            "entities",
            "created_at",
        },
        "facts": {
            "id",
            "user_id",
            "project",
            "subject",
            "predicate",
            "object",
            "source",
            "confidence",
            "valid_from",
            "valid_to",
            "superseded_by",
            "last_confirmed",
        },
    }
    source_columns = {
        "memories": {row["name"] for row in conn.execute("PRAGMA table_info(memories)")},
        "facts": {row["name"] for row in conn.execute("PRAGMA table_info(facts)")},
    }
    for table, fields in required.items():
        columns = source_columns[table]
        if missing := fields - columns:
            raise ValueError(
                f"Morgan evidence source schema is missing {table} columns: "
                + ", ".join(sorted(missing))
            )


class ScopedEvidenceReader:
    """Read source records; MemoryGate owns request validation and caller boundaries."""

    def __init__(self, episodics: EpisodicStore, temporal: SqliteTemporalStore) -> None:
        self._episodics = episodics
        self._temporal = temporal

    async def evidence(
        self,
        *,
        user_id: str,
        project: str,
        evidence_ids: list[str],
        effective_at: datetime | None = None,
    ) -> EvidenceResult:
        records: list[Memory] = []
        missing: list[str] = []
        with read_transaction(self._episodics._conn):
            resolver = RevisionResolver(self._episodics, at=effective_at or datetime.now(UTC))
            for identity in evidence_ids:
                event = self._episodics.get(identity, user_id=user_id, project=project)
                fact = await self._temporal.get_fact(identity, user_id=user_id, project=project)
                if event is not None and fact is not None:
                    raise ValueError(
                        "Ambiguous evidence identity: event and fact exist in named scope"
                    )
                if event is not None:
                    records.append(resolver.annotate(event))
                    continue
                if fact is not None:
                    records.append(
                        fact_memory(fact).model_copy(
                            update={"support_state": resolver.support_state(fact)}
                        )
                    )
                else:
                    missing.append(identity)
        return EvidenceResult(
            user_id=user_id,
            project=project,
            requested_ids=evidence_ids,
            records=records,
            missing_ids=missing,
        )
