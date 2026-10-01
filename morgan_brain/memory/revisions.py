"""Scoped immutable correction families; no recursive graph or derived state writes."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from morgan_brain.memory.store.episodic import EpisodicStore
from morgan_brain.models import Memory, MemoryKind, MemorySource, MemoryStatus, TemporalFact


class RevisionError(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def instant(value: datetime) -> datetime:
    return (value.replace(tzinfo=UTC) if value.tzinfo is None else value).astimezone(UTC)


def _effective_us(value: str | None) -> int | None:
    if value is None:
        return None
    return (instant(datetime.fromisoformat(value)) - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(
        microseconds=1
    )


@dataclass(frozen=True)
class FamilyState:
    leaf_ids: tuple[str, ...]
    leaf_count: int

    @property
    def conflicted(self) -> bool:
        return self.leaf_count > 1


@dataclass(frozen=True)
class EventCandidates:
    """Trusted metadata-only SQL selection shared by both ranked indexes."""

    sql: str
    params: tuple[object, ...]


class RevisionResolver:
    def __init__(self, episodics: EpisodicStore, *, conn: sqlite3.Connection, at: datetime) -> None:
        self._episodics = episodics
        self._conn = conn
        self.at = instant(at)
        self._cache: dict[tuple[str, str, str], FamilyState] = {}
        # SQLite's built-in date functions round submillisecond times. This pure
        # connection-local function keeps exact UTC comparisons without source rewrites.
        conn.create_function("morgan_effective_us", 1, _effective_us, deterministic=True)
        self._columns = {row["name"] for row in conn.execute("PRAGMA table_info(memories)")}

    def ranked_candidates(self, *, user_id: str, project: str | None) -> EventCandidates:
        """Select eligible IDs inside SQLite before KNN/FTS limits, never source text."""
        sql = """
            SELECT candidate.id FROM memories AS candidate
            WHERE candidate.user_id=? AND (? IS NULL OR candidate.project=?)
            AND (candidate.created_at IS NULL OR
                morgan_effective_us(candidate.created_at)<=?)
        """
        cutoff_us = (self.at - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1)
        params: tuple[object, ...] = (user_id, project, project, cutoff_us)
        if "status" in self._columns:
            sql += " AND candidate.status='stored'"
        if "revision_root_id" in self._columns:
            sql = """
                WITH activated AS MATERIALIZED (
                    SELECT id,user_id,project,revision_root_id,revises_event_ids FROM memories
                    WHERE user_id=? AND (? IS NULL OR project=?) AND status='stored'
                    AND (created_at IS NULL OR
                        morgan_effective_us(created_at)<=?)
                ), suppressed AS MATERIALIZED (
                    SELECT child.user_id,child.project,child.revision_root_id,
                        link.value AS parent_id
                    FROM activated AS child, json_each(child.revises_event_ids) AS link
                )
                SELECT candidate.id FROM activated AS candidate
                WHERE NOT EXISTS (
                    SELECT 1 FROM suppressed AS child WHERE child.user_id=candidate.user_id
                    AND child.project=candidate.project AND child.parent_id=candidate.id
                    AND child.revision_root_id=COALESCE(candidate.revision_root_id,candidate.id)
                )
            """
        if "origin_kind" in self._columns:
            # Procedural clarification turns remain durable exact evidence/history,
            # but must not crowd their unresolved source family out of ranked recall.
            sql += (
                " AND NOT EXISTS (SELECT 1 FROM memories AS procedural "
                "WHERE procedural.id=candidate.id "
                "AND procedural.origin_kind='ask_conflict_guard')"
            )
        return EventCandidates(sql, params)

    def validate_parents(self, memory: Memory) -> str:
        if memory.created_at is not None and memory.created_at.utcoffset() is None:
            raise RevisionError("revision_effective_time")
        parents = memory.revises_event_ids
        if (
            not isinstance(parents, list)
            or len(parents) > 8
            or any(
                not isinstance(identity, str) or not identity.strip() or len(identity) > 256
                for identity in parents
            )
            or len(set(parents)) != len(parents)
        ):
            raise RevisionError("revision_parent_limit")
        if not parents:
            if memory.revision_root_id not in (None, memory.id):
                raise RevisionError("revision_family_boundary")
            return memory.id
        if memory.source is MemorySource.UNKNOWN or not memory.author_id.strip():
            raise RevisionError("revision_actor_boundary")
        if memory.kind is not MemoryKind.EPISODIC:
            raise RevisionError("revision_parent_unavailable")
        if memory.created_at is None or memory.created_at.utcoffset() is None:
            raise RevisionError("revision_effective_time")
        roots = set()
        for identity in parents:
            parent = self._episodics.get(identity, user_id=memory.user_id, project=memory.project)
            if (
                parent is None
                or parent.status is not MemoryStatus.STORED
                or parent.kind is not MemoryKind.EPISODIC
            ):
                raise RevisionError("revision_parent_unavailable")
            if (parent.source, parent.author_id, parent.scope) != (
                memory.source,
                memory.author_id,
                memory.scope,
            ):
                raise RevisionError("revision_actor_boundary")
            if parent.created_at is None or instant(memory.created_at) < instant(parent.created_at):
                raise RevisionError("revision_effective_time")
            roots.add(parent.revision_root_id or parent.id)
        if len(roots) != 1:
            raise RevisionError("revision_family_boundary")
        root = roots.pop()
        if memory.revision_root_id not in (None, root):
            raise RevisionError("revision_family_boundary")
        return root

    def family(self, event: Memory) -> FamilyState:
        root = event.revision_root_id or event.id
        key = (event.user_id, event.project, root)
        if key in self._cache:
            return self._cache[key]
        conn = self._conn
        if "revision_root_id" not in self._columns:
            eligible = event.status is MemoryStatus.STORED and (
                event.created_at is None or instant(event.created_at) <= self.at
            )
            state = FamilyState((event.id,) if eligible else (), int(eligible))
        else:
            # Both branches use an index: scoped family root or the legacy root's ID.
            # Activated stored children suppress only explicitly named parents.
            sql = """
                WITH family AS (
                    SELECT id, created_at, status, revises_event_ids FROM memories
                    WHERE user_id=? AND project=? AND revision_root_id=?
                    UNION ALL
                    SELECT id, created_at, status, revises_event_ids FROM memories
                    WHERE user_id=? AND project=? AND id=? AND revision_root_id IS NULL
                ), activated AS (
                    SELECT * FROM family WHERE status='stored'
                    AND (created_at IS NULL OR
                        morgan_effective_us(created_at)<=morgan_effective_us(?))
                ), leaves AS (
                    SELECT id FROM activated WHERE id NOT IN (
                        SELECT link.value FROM activated AS child,
                        json_each(child.revises_event_ids) AS link
                    )
                ) SELECT id, COUNT(*) OVER () AS total FROM leaves ORDER BY id LIMIT 32
            """
            rows = conn.execute(
                sql, (*key, event.user_id, event.project, root, self.at.isoformat())
            ).fetchall()
            state = FamilyState(tuple(row["id"] for row in rows), rows[0]["total"] if rows else 0)
        self._cache[key] = state
        return state

    def is_leaf(self, event: Memory) -> bool:
        family = self.family(event)
        if family.leaf_count <= 32:
            return event.id in family.leaf_ids
        # A truncated reference set must not misclassify a later-ID active leaf.
        if event.status is not MemoryStatus.STORED or (
            event.created_at is not None and instant(event.created_at) > self.at
        ):
            return False
        root = event.revision_root_id or event.id
        child = self._conn.execute(
            "SELECT 1 FROM memories AS child, json_each(child.revises_event_ids) AS link "
            "WHERE child.user_id=? AND child.project=? AND child.revision_root_id=? "
            "AND child.status='stored' AND link.value=? "
            "AND (child.created_at IS NULL OR "
            "morgan_effective_us(child.created_at)<=morgan_effective_us(?)) LIMIT 1",
            (event.user_id, event.project, root, event.id, self.at.isoformat()),
        ).fetchone()
        return child is None

    def annotate(self, event: Memory) -> Memory:
        state = self.family(event)
        eligible = self.is_leaf(event)
        return event.model_copy(
            update={
                "revision_root_id": event.revision_root_id or event.id,
                "revision_state": "inactive"
                if not eligible
                else "conflicted"
                if state.conflicted
                else "active",
                "eligible_leaf_ids": list(state.leaf_ids),
                "eligible_leaf_count": state.leaf_count,
                "revision_truncated": state.leaf_count > len(state.leaf_ids),
            }
        )

    def support_state(self, fact: TemporalFact) -> str:
        if not fact.support_event_ids:
            return (
                "intrinsic"
                if fact.source in (MemorySource.USER_STATED, MemorySource.TOOL_OBSERVED)
                else "unsupported"
            )
        roots = [
            self._episodics.get(identity, user_id=fact.user_id, project=fact.project)
            for identity in fact.support_event_ids
        ]
        if any(
            event is None
            or event.kind is not MemoryKind.EPISODIC
            or event.source not in (MemorySource.USER_STATED, MemorySource.TOOL_OBSERVED)
            or not self.is_leaf(event)
            for event in roots
        ):
            return "inactive_support"
        if any(event is not None and self.family(event).conflicted for event in roots):
            return "conflicted_support"
        return "current"
