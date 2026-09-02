"""The persona graph — VoiceMem's right brain (arXiv:2608.26005 §3.2).

Where the left brain records *what happened*, this records *who the user is*. Morgan
already had the second idea as a flat ``UserModel.traits`` list scored by token overlap.
What that list cannot express is the distinction this module is built around::

    G_R = (V_I, V_C)
      v_I = (description, evidence)               intrinsic: an enduring disposition
      v_C = (description, evidence, anchor)       cross-entity: an attitude toward something

    "This distinction is fundamental: v_I explains persistent user characteristics,
     whereas v_C_e preserves whom or what an emotion concerns. Collapsing the two would
     either mistake situational reactions for stable traits or remove the real-world
     causes that give affect its meaning."  -- §3.2

"He is impatient" and "he is impatient *with the weekly Harbor sync*" are different
claims. A flat trait list records the first when only the second is true, which is the
over-personalization failure Morgan's golden eval already probes as
``OVER_PERSONALIZATION_NEGATIVE``.

It is also the same discipline ``MemorySource`` already enforces for facts: an inference
is never silently upgraded to a user's statement. Here, a situational reading stays
situational until the evidence earns the generalisation.

**Promotion is deliberately hard.** An observation becomes a stable trait only when the
same disposition shows up toward *several different anchors* across *several distinct
sessions*. Recurrence toward one anchor -- however often -- is a fact about that anchor,
not about the person: someone impatient with one recurring meeting is not an impatient
person, and saying so in the system prompt is worse than saying nothing. Requiring
multiple anchors is what makes "situational" and "dispositional" separable without a
model having to judge it.

Two horizons, mapped onto Morgan's existing path split:

* **Short horizon** -- ``observe()``, once per turn, on the cold path beside the signal
  recorder. Never during a request.
* **Long horizon** -- ``consolidate()``, nightly, promoting recurrent evidence.

``activate()`` is the only method the hot path calls, and it is read-only.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

#: A disposition must appear toward at least this many distinct anchors before it is a
#: statement about the person rather than about one thing in their life.
MIN_ANCHORS_TO_PROMOTE = 3

#: ...and across at least this many distinct sessions, so one long afternoon of
#: complaining about three things does not become a personality.
MIN_SESSIONS_TO_PROMOTE = 3

#: Confidence saturates below 1.0. Nothing inferred about a person from conversation is
#: ever certain, and a trait injected at certainty is one the assistant will not revise.
_CONFIDENCE_CEILING = 0.95


class PersonaKind(str, Enum):
    INTRINSIC = "intrinsic"
    CROSS_ENTITY = "cross_entity"


@dataclass(frozen=True)
class PersonaNode:
    id: str
    kind: PersonaKind
    description: str
    entity: str | None
    valence: float
    confidence: float
    observations: int
    sessions: int
    anchors: list[str] = field(default_factory=list)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS persona_nodes (
    user_id      TEXT    NOT NULL,
    project      TEXT    NOT NULL,
    id           TEXT    NOT NULL,
    kind         TEXT    NOT NULL,
    description  TEXT    NOT NULL,
    entity       TEXT,
    valence      REAL    NOT NULL DEFAULT 0.0,
    observations INTEGER NOT NULL DEFAULT 0,
    sessions     INTEGER NOT NULL DEFAULT 0,
    last_session TEXT,
    anchors      TEXT    NOT NULL DEFAULT '[]',
    first_seen   TEXT,
    last_seen    TEXT,
    PRIMARY KEY (user_id, project, id)
);

CREATE INDEX IF NOT EXISTS idx_persona_anchor
    ON persona_nodes (user_id, project, entity);
"""


def _node_id(kind: PersonaKind, description: str, entity: str | None) -> str:
    """Stable identity, so the same observation always lands on the same node.

    Derived from the content rather than assigned, because the cold path has no handle
    on a node it has not read -- and reading first to decide whether to insert is the
    race that produces duplicate personas for the same disposition.
    """
    raw = f"{kind.value}|{description.strip().casefold()}|{(entity or '').strip().casefold()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _confidence(observations: int, sessions: int) -> float:
    """Evidence → confidence, saturating below certainty.

    Sessions count for more than repetitions: coming back to something on a different day
    is stronger evidence than saying it twice in one sitting.
    """
    weight = observations + 2 * sessions
    return round(_CONFIDENCE_CEILING * (1.0 - 1.0 / (1.0 + 0.25 * weight)), 4)


class PersonaGraph:
    """Persona nodes over the one shared SQLite connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.executescript(_SCHEMA)
        conn.commit()

    # ------------------------------------------------------------------
    # Short horizon (cold path, per turn)
    # ------------------------------------------------------------------

    def observe(
        self,
        *,
        user_id: str,
        project: str,
        description: str,
        entity: str | None,
        valence: float,
        session_id: str,
        now: datetime,
    ) -> PersonaNode:
        """Record one observation. An *entity* makes it cross-entity; no entity makes it
        intrinsic (a stated preference, not an inferred disposition)."""
        kind = PersonaKind.CROSS_ENTITY if entity else PersonaKind.INTRINSIC
        anchor = entity.strip() if entity else None
        node_id = _node_id(kind, description, anchor)
        stamp = now.isoformat()

        row = self._conn.execute(
            "SELECT observations, sessions, last_session FROM persona_nodes "
            "WHERE user_id = ? AND project = ? AND id = ?",
            (user_id, project, node_id),
        ).fetchone()

        if row is None:
            observations, sessions = 1, 1
            self._conn.execute(
                "INSERT INTO persona_nodes (user_id, project, id, kind, description, entity, "
                "valence, observations, sessions, last_session, first_seen, last_seen) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    project,
                    node_id,
                    kind.value,
                    description.strip(),
                    anchor,
                    valence,
                    observations,
                    sessions,
                    session_id,
                    stamp,
                    stamp,
                ),
            )
        else:
            observations = int(row["observations"]) + 1
            # A second remark in the same sitting is repetition, not recurrence. Counting
            # it would let one bad afternoon look like a pattern.
            sessions = int(row["sessions"]) + (1 if row["last_session"] != session_id else 0)
            self._conn.execute(
                "UPDATE persona_nodes SET observations = ?, sessions = ?, last_session = ?, "
                "valence = (valence * (? - 1) + ?) / ?, last_seen = ? "
                "WHERE user_id = ? AND project = ? AND id = ?",
                (
                    observations,
                    sessions,
                    session_id,
                    observations,
                    valence,
                    observations,
                    stamp,
                    user_id,
                    project,
                    node_id,
                ),
            )
        self._conn.commit()
        return self._get(user_id=user_id, project=project, node_id=node_id)

    # ------------------------------------------------------------------
    # Activation (hot path, read-only)
    # ------------------------------------------------------------------

    def activate(
        self,
        *,
        user_id: str,
        project: str,
        terms: list[str],
        entities: set[str],
        limit: int = 5,
    ) -> list[PersonaNode]:
        """The nodes this turn should see, strongest first.

        Joint retrieval, as in eq. (5): intrinsic nodes matched from the turn's own words,
        plus cross-entity nodes anchored to entities the turn made relevant -- which is
        why the caller passes the entity set the left brain activated rather than only the
        literal ones in the text.

        A cross-entity node without its anchor is never returned. Surfacing "impatient"
        with no idea what it is about is the collapse this whole structure exists to
        prevent.
        """
        anchors = sorted({e.strip().casefold() for e in entities if e and e.strip()})
        words = sorted({t.strip().casefold() for t in terms if t and t.strip()})
        rows = self._conn.execute(
            """
            SELECT * FROM persona_nodes
            WHERE user_id = ? AND project = ?
              AND (
                    (kind = 'cross_entity'
                     AND lower(trim(entity)) IN (SELECT value FROM json_each(?)))
                 OR (kind = 'intrinsic' AND ? > 0)
              )
            ORDER BY observations DESC, sessions DESC, id ASC
            """,
            (user_id, project, json.dumps(anchors), len(words)),
        ).fetchall()

        out: list[PersonaNode] = []
        for r in rows:
            if r["kind"] == PersonaKind.INTRINSIC.value and not _matches(
                str(r["description"]), words
            ):
                continue
            out.append(_to_node(r))
        return out[:limit]

    # ------------------------------------------------------------------
    # Long horizon (nightly)
    # ------------------------------------------------------------------

    def consolidate(self, *, user_id: str, project: str, now: datetime) -> list[PersonaNode]:
        """Promote recurrent cross-entity evidence into intrinsic traits.

        Returns only the traits promoted by *this* call, so a nightly run that promotes
        nothing is distinguishable from one that re-promoted what already existed.

        The cross-entity nodes are left in place. They are the evidence, and they are the
        half a flat trait list throws away -- "impatient" is not a substitute for knowing
        it was about the sync, the checklist and the docs.
        """
        rows = self._conn.execute(
            "SELECT description, entity, sessions, last_session, valence, observations "
            "FROM persona_nodes "
            "WHERE user_id = ? AND project = ? AND kind = 'cross_entity'",
            (user_id, project),
        ).fetchall()

        grouped: dict[str, list[sqlite3.Row]] = {}
        for r in rows:
            grouped.setdefault(str(r["description"]).strip().casefold(), []).append(r)

        promoted: list[PersonaNode] = []
        for members in grouped.values():
            anchors = sorted({str(m["entity"]).strip() for m in members if m["entity"]})
            sessions = {str(m["last_session"]) for m in members if m["last_session"]}
            total_sessions = sum(int(m["sessions"]) for m in members)
            if len(anchors) < MIN_ANCHORS_TO_PROMOTE:
                # A disposition seen toward exactly one thing is a fact about that thing.
                continue
            if max(len(sessions), total_sessions) < MIN_SESSIONS_TO_PROMOTE:
                continue

            description = str(members[0]["description"]).strip()
            node_id = _node_id(PersonaKind.INTRINSIC, description, None)
            if self._exists(user_id=user_id, project=project, node_id=node_id):
                continue
            observations = sum(int(m["observations"]) for m in members)
            stamp = now.isoformat()
            self._conn.execute(
                "INSERT INTO persona_nodes (user_id, project, id, kind, description, entity, "
                "valence, observations, sessions, last_session, anchors, first_seen, last_seen) "
                "VALUES (?, ?, ?, 'intrinsic', ?, NULL, ?, ?, ?, NULL, ?, ?, ?)",
                (
                    user_id,
                    project,
                    node_id,
                    description,
                    sum(float(m["valence"]) for m in members) / len(members),
                    observations,
                    max(len(sessions), total_sessions),
                    json.dumps(anchors),
                    stamp,
                    stamp,
                ),
            )
            self._conn.commit()
            promoted.append(self._get(user_id=user_id, project=project, node_id=node_id))
        return promoted

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def all_nodes(self, *, user_id: str, project: str) -> list[PersonaNode]:
        rows = self._conn.execute(
            "SELECT * FROM persona_nodes WHERE user_id = ? AND project = ? "
            "ORDER BY observations DESC, id ASC",
            (user_id, project),
        ).fetchall()
        return [_to_node(r) for r in rows]

    def _exists(self, *, user_id: str, project: str, node_id: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM persona_nodes WHERE user_id = ? AND project = ? AND id = ?",
                (user_id, project, node_id),
            ).fetchone()
            is not None
        )

    def _get(self, *, user_id: str, project: str, node_id: str) -> PersonaNode:
        row = self._conn.execute(
            "SELECT * FROM persona_nodes WHERE user_id = ? AND project = ? AND id = ?",
            (user_id, project, node_id),
        ).fetchone()
        return _to_node(row)


def _matches(description: str, words: list[str]) -> bool:
    tokens = {w.casefold() for w in description.split()}
    return any(w in tokens for w in words)


def _to_node(row: sqlite3.Row) -> PersonaNode:
    return PersonaNode(
        id=str(row["id"]),
        kind=PersonaKind(str(row["kind"])),
        description=str(row["description"]),
        entity=str(row["entity"]) if row["entity"] is not None else None,
        valence=float(row["valence"]),
        confidence=_confidence(int(row["observations"]), int(row["sessions"])),
        observations=int(row["observations"]),
        sessions=int(row["sessions"]),
        anchors=list(json.loads(str(row["anchors"] or "[]"))),
    )
