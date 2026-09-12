"""The semantic upper index — schema → entity → memory routing above the store.

From VoiceMem (arXiv:2608.26005) §3.1. The idea is that at a small retrieval budget,
accuracy depends on how *dense* the candidate space is, not on how sophisticated the
ranking over it is: several memories mentioning the same entity in irrelevant contexts
will fill the top-k while saying nothing useful. So a lightweight index is built above
the backend memories and narrows retrieval to a compact, semantically coherent pool
*before* any signal searches. In the paper's ablation this is the single largest
contributor -- removing it costs 9.9 points on LoCoMo at K=5 -- and it transfers across
backends unchanged (+15.8 to +29.5 dropped onto three different stores).

Two levels, no more::

    G_L = (S, V, E)
      s = (description, N_macro, V_s)   a schema: a coarse slot holding entities
      v = (description, N_micro, I_v)   an entity: in exactly ONE schema; I_v its memories
      E = E_micro ∪ E_macro             entity↔entity and schema↔schema co-occurrence

Schema membership lives on the entity row rather than as schema→entity edges. That is the
paper's own simplification and it is what keeps routing a couple of flat queries instead
of a graph traversal.

``I_v`` is not a new table. ``memory_entities`` already maps entity name → memory id under
``(user_id, project)``, so the leaf level of this index is the table ``EntityIndex``
already owns, and the two cannot drift apart.

**Routing never costs recall.** ``route()`` returns ``None`` -- meaning "no narrowing,
search everything" -- whenever it has nothing useful to say: no term matched, the matched
entities index no memories, or the pool came out so wide that filtering by it would
remove nothing. It returns a pool only when that pool is genuinely narrower than the
store. An index that can return a *wrong* pool silently deletes memories from recall,
which is worse than having no index, so those cases are the first thing the tests pin.

Written by the cold path (``learning/semantic_index_builder.py``), read by the hot path.
Nothing here writes during a request.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence

#: The six coarse slots VoiceMem starts from (§5, "six preset slots"). They are a
#: starting partition, not the final one: emergent clustering re-partitions across these
#: boundaries later, which is why `mem_schemas.emerged` exists from the start.
PRESET_SCHEMAS: tuple[str, ...] = (
    "work",
    "health",
    "daily_life",
    "relationships",
    "knowledge",
    "goals",
)

#: Above this many candidates, routing has not narrowed anything worth the filter, so the
#: index steps aside and lets every signal search unrestricted. Deliberately generous:
#: the failure this guards is "the pool is the whole store", not "the pool is large".
DEFAULT_MAX_CANDIDATES = 512

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mem_schemas (
    user_id     TEXT    NOT NULL,
    project     TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    description TEXT    NOT NULL DEFAULT '',
    emerged     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, project, name)
);

CREATE TABLE IF NOT EXISTS mem_entity_nodes (
    user_id     TEXT NOT NULL,
    project     TEXT NOT NULL,
    name        TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (user_id, project, name)
);

CREATE INDEX IF NOT EXISTS idx_entity_nodes_schema
    ON mem_entity_nodes (user_id, project, schema_name);

-- E_micro. Stored canonically (src < dst) so the pair is undirected by construction
-- rather than by two rows that can fall out of step.
CREATE TABLE IF NOT EXISTS mem_entity_edges (
    user_id TEXT    NOT NULL,
    project TEXT    NOT NULL,
    src     TEXT    NOT NULL,
    dst     TEXT    NOT NULL,
    weight  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, project, src, dst)
);

-- E_macro. Written alongside E_micro and read by cluster emergence, which needs to know
-- which slots keep being activated together before it proposes re-partitioning them.
CREATE TABLE IF NOT EXISTS mem_schema_edges (
    user_id TEXT    NOT NULL,
    project TEXT    NOT NULL,
    src     TEXT    NOT NULL,
    dst     TEXT    NOT NULL,
    weight  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, project, src, dst)
);
"""


def _pair(a: str, b: str) -> tuple[str, str]:
    """Canonical undirected pair, so (x, y) and (y, x) are one row."""
    return (a, b) if a <= b else (b, a)


class SemanticIndex:
    """The upper index. Shares the one SQLite connection with every other store."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.executescript(_SCHEMA)
        conn.commit()

    # ------------------------------------------------------------------
    # Building (cold path)
    # ------------------------------------------------------------------

    def ensure_schemas(
        self, *, user_id: str, project: str, names: Sequence[str] = PRESET_SCHEMAS
    ) -> None:
        """Create the preset slots for this scope. Idempotent."""
        self._conn.executemany(
            "INSERT OR IGNORE INTO mem_schemas (user_id, project, name) VALUES (?, ?, ?)",
            [(user_id, project, n) for n in names],
        )
        self._conn.commit()

    def add_emergent_schema(
        self, *, user_id: str, project: str, name: str, description: str = ""
    ) -> None:
        """Register a slot that emerged from retrieval patterns rather than the presets."""
        self._conn.execute(
            "INSERT OR REPLACE INTO mem_schemas (user_id, project, name, description, emerged) "
            "VALUES (?, ?, ?, ?, 1)",
            (user_id, project, name, description),
        )
        self._conn.commit()

    def schemas(self, *, user_id: str, project: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT name FROM mem_schemas WHERE user_id = ? AND project = ? ORDER BY name",
            (user_id, project),
        ).fetchall()
        return [str(r["name"]) for r in rows]

    def schema_of(self, *, user_id: str, project: str, entity: str) -> str | None:
        """The schema *entity* is filed under, or ``None`` when it has never been filed."""
        return self._schema_of(user_id=user_id, project=project, entity=entity.lower())

    def assign(
        self,
        *,
        user_id: str,
        project: str,
        entity: str,
        schema_name: str,
        description: str = "",
    ) -> None:
        """Put *entity* in *schema_name*, replacing any previous membership.

        Exactly one schema per entity is the paper's constraint, and the reason routing
        never has to traverse: reassignment moves the entity, it does not add a second
        membership that a query would then have to reconcile.
        """
        known = self._conn.execute(
            "SELECT 1 FROM mem_schemas WHERE user_id = ? AND project = ? AND name = ?",
            (user_id, project, schema_name),
        ).fetchone()
        if known is None:
            raise ValueError(f"unknown schema {schema_name!r} for project {project!r}")
        self._conn.execute(
            "INSERT INTO mem_entity_nodes (user_id, project, name, schema_name, description) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (user_id, project, name) DO UPDATE SET "
            "schema_name = excluded.schema_name, "
            "description = CASE WHEN excluded.description != '' "
            "THEN excluded.description ELSE mem_entity_nodes.description END",
            (user_id, project, entity.lower(), schema_name, description),
        )
        self._conn.commit()

    def observe_cooccurrence(self, *, user_id: str, project: str, names: Iterable[str]) -> None:
        """Record that these entities appeared in the same memory.

        Only entities that already have a node participate. An edge to an unassigned name
        would route into an entity with no schema, which is a pool member no query can
        explain -- so unknown names are dropped rather than half-registered.
        """
        known = self._known_entities(user_id=user_id, project=project, names=names)
        if len(known) < 2:
            return
        ordered = sorted(known)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                src, dst = _pair(a, b)
                self._conn.execute(
                    "INSERT INTO mem_entity_edges (user_id, project, src, dst, weight) "
                    "VALUES (?, ?, ?, ?, 1) "
                    "ON CONFLICT (user_id, project, src, dst) DO UPDATE SET "
                    "weight = weight + 1",
                    (user_id, project, src, dst),
                )
        schemas = {
            s
            for s in (self._schema_of(user_id=user_id, project=project, entity=e) for e in ordered)
            if s is not None
        }
        ordered_schemas = sorted(schemas)
        for i, a in enumerate(ordered_schemas):
            for b in ordered_schemas[i + 1 :]:
                src, dst = _pair(a, b)
                self._conn.execute(
                    "INSERT INTO mem_schema_edges (user_id, project, src, dst, weight) "
                    "VALUES (?, ?, ?, ?, 1) "
                    "ON CONFLICT (user_id, project, src, dst) DO UPDATE SET "
                    "weight = weight + 1",
                    (user_id, project, src, dst),
                )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Routing (hot path, read-only)
    # ------------------------------------------------------------------

    def route(
        self,
        terms: Iterable[str],
        *,
        user_id: str,
        project: str,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ) -> list[str] | None:
        """Return the candidate memory ids for *terms*, or ``None`` for "do not narrow".

        Implements eq. (1): ``Z_t = V_t ∪ V_{S_t} ∪ N_1(V_t ∪ V_{S_t})`` over one hop,
        then ``C_L = ⋃_{z∈Z_t} I_z``.

        ``None`` is returned -- never an empty list -- whenever narrowing would be wrong
        or pointless: nothing matched, the matched entities index no memories, or the
        pool exceeded *max_candidates*. The caller treats ``None`` as "search everything",
        so the index can only ever improve precision, never remove a memory recall would
        otherwise have found.
        """
        wanted = sorted({t.lower() for t in terms if t})
        if not wanted:
            return None

        matched_entities = set(self._known_entities(user_id=user_id, project=project, names=wanted))
        matched_schemas = self._match_schemas(user_id=user_id, project=project, names=wanted)
        seeds = matched_entities | self._entities_in_schemas(
            user_id=user_id, project=project, schemas=matched_schemas
        )
        if not seeds:
            return None

        expanded = seeds | self._one_hop(user_id=user_id, project=project, seeds=seeds)
        ids = self._memories_for(user_id=user_id, project=project, entities=expanded)
        if not ids:
            # The index knows these entities but nothing has been filed under them yet.
            # Narrowing to nothing would silence the turn.
            return None
        if len(ids) > max_candidates:
            return None
        return ids

    def neighbours(self, names: Sequence[str], *, user_id: str, project: str) -> set[str]:
        """The entities one hop from *names*, plus *names* themselves.

        This is ``Z_t`` from eq. (1) without the memory lookup, and it is what the right
        brain expands over for joint retrieval (eq. 5): an attitude anchored to something
        the turn implies but does not name still has to be reachable.
        """
        seeds = {n.lower() for n in names if n}
        if not seeds:
            return set()
        known = set(self._known_entities(user_id=user_id, project=project, names=seeds))
        if not known:
            return seeds
        return seeds | known | self._one_hop(user_id=user_id, project=project, seeds=known)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _known_entities(self, *, user_id: str, project: str, names: Iterable[str]) -> list[str]:
        wanted = sorted({n.lower() for n in names if n})
        if not wanted:
            return []
        rows = self._conn.execute(
            "SELECT name FROM mem_entity_nodes WHERE user_id = ? AND project = ? "
            "AND name IN (SELECT value FROM json_each(?)) ORDER BY name",
            (user_id, project, json.dumps(wanted)),
        ).fetchall()
        return [str(r["name"]) for r in rows]

    def _schema_of(self, *, user_id: str, project: str, entity: str) -> str | None:
        row = self._conn.execute(
            "SELECT schema_name FROM mem_entity_nodes "
            "WHERE user_id = ? AND project = ? AND name = ?",
            (user_id, project, entity),
        ).fetchone()
        return str(row["schema_name"]) if row is not None else None

    def _match_schemas(self, *, user_id: str, project: str, names: Sequence[str]) -> set[str]:
        rows = self._conn.execute(
            "SELECT name FROM mem_schemas WHERE user_id = ? AND project = ? "
            "AND name IN (SELECT value FROM json_each(?))",
            (user_id, project, json.dumps(list(names))),
        ).fetchall()
        return {str(r["name"]) for r in rows}

    def _entities_in_schemas(self, *, user_id: str, project: str, schemas: set[str]) -> set[str]:
        if not schemas:
            return set()
        rows = self._conn.execute(
            "SELECT name FROM mem_entity_nodes WHERE user_id = ? AND project = ? "
            "AND schema_name IN (SELECT value FROM json_each(?))",
            (user_id, project, json.dumps(sorted(schemas))),
        ).fetchall()
        return {str(r["name"]) for r in rows}

    def _one_hop(self, *, user_id: str, project: str, seeds: set[str]) -> set[str]:
        """Neighbours one edge away. Strong and weak links both count, and neither is
        followed further -- two hops is how a routed pool quietly becomes the whole
        store again."""
        if not seeds:
            return set()
        seed_json = json.dumps(sorted(seeds))
        rows = self._conn.execute(
            """
            SELECT dst AS other FROM mem_entity_edges
            WHERE user_id = ? AND project = ? AND src IN (SELECT value FROM json_each(?))
            UNION
            SELECT src AS other FROM mem_entity_edges
            WHERE user_id = ? AND project = ? AND dst IN (SELECT value FROM json_each(?))
            """,
            (user_id, project, seed_json, user_id, project, seed_json),
        ).fetchall()
        return {str(r["other"]) for r in rows} - seeds

    def _memories_for(self, *, user_id: str, project: str, entities: set[str]) -> list[str]:
        if not entities:
            return []
        rows = self._conn.execute(
            "SELECT DISTINCT memory_id FROM memory_entities "
            "WHERE user_id = ? AND project = ? "
            "AND name IN (SELECT value FROM json_each(?)) "
            "ORDER BY memory_id",
            (user_id, project, json.dumps(sorted(entities))),
        ).fetchall()
        return [str(r["memory_id"]) for r in rows]
