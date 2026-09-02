"""Cluster emergence — letting the schema partition follow what is actually retrieved.

VoiceMem §3.1 (*Cluster Emergence Mechanism*) and Algorithm 1. Preset slots go stale:
in the paper's own store, 49.8% of items ended up in two slots that were never preset,
and neither sat inside a single preset -- ``Pets & Outdoor`` drew 48% from daily_life,
27% from health and 17% from relationships. Emergence *re-partitions across* preset
boundaries rather than refining within them, which is why disabling it costs most on the
longest sessions, where the presets are furthest from the real topic structure.

Rule-based splitting is the thing this replaces. Splitting on size fragments related
memories and costs retrieval coverage; instead, coherent subclusters are allowed to
emerge from what queries actually activate together. For a connected entity subset ``H``
and the queries ``Q`` observed in a window::

    ρ(H) = (1/|Q|) · Σ_{q∈Q} |A_q ∩ H| / |A_q ∪ H|

``A_q`` is the entity set query ``q`` activated. A high ρ means these entities are
repeatedly retrieved *together* -- not merely that they are related, which the edges
already said, but that the owner's questions treat them as one thing.

Two guards make this safe to run unattended:

* **An LLM judge scores relevance, importance and completeness before promotion.** A
  high ρ can come from a handful of near-identical queries; the judge is what stops a
  week of asking about one incident from permanently re-partitioning the store.
* **A rejected candidate is remembered as rejected.** Without that, the same subgraph is
  re-proposed every night, costs a judge call every night, and eventually gets promoted
  by a judge having an off day. The paper calls this disabling the candidate; here it is
  a row, so the refusal survives a restart.

Cold path only, nightly. Nothing here runs during a request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel

from morgan_brain.interfaces.llm import ProviderUnreachable
from morgan_brain.modules.memory.retrieval.semantic_index import SemanticIndex
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import RoleRouter
from morgan_brain.providers.structured import StructuredError, generate_structured
from morgan_brain.providers.wire import ChatMessage

logger = logging.getLogger(__name__)

#: Coherence threshold α. A subset must be activated together in a good fraction of the
#: queries that touch it before it counts as one thing the owner thinks about.
DEFAULT_ALPHA = 0.35

#: A candidate below this size is not a cluster, it is a pair of related entities -- the
#: micro edges already express that, and promoting it would only add a slot.
MIN_CANDIDATE_SIZE = 3

#: Fewer observed queries than this and ρ is measuring noise: one afternoon of asking
#: about the same incident would read as a permanent structure.
MIN_QUERIES = 5

_SCHEMA = """
-- The co-retrieval log. One row per (query, activated entity), written by the cold path
-- after a turn has been answered -- never during it.
CREATE TABLE IF NOT EXISTS mem_query_activations (
    user_id  TEXT NOT NULL,
    project  TEXT NOT NULL,
    query_id TEXT NOT NULL,
    entity   TEXT NOT NULL,
    seen_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, project, query_id, entity)
);

CREATE INDEX IF NOT EXISTS idx_activations_scope
    ON mem_query_activations (user_id, project, seen_at DESC);

-- Candidates a judge refused. Without this the same subgraph is re-proposed every night
-- until a judge has an off day.
CREATE TABLE IF NOT EXISTS mem_emergence_rejected (
    user_id     TEXT NOT NULL,
    project     TEXT NOT NULL,
    signature   TEXT NOT NULL,
    reason      TEXT NOT NULL DEFAULT '',
    rejected_at TEXT NOT NULL,
    PRIMARY KEY (user_id, project, signature)
);
"""


def _signature(entities: Iterable[str]) -> str:
    """Identity of a candidate subgraph, stable across runs."""
    joined = "\n".join(sorted(e.lower() for e in entities))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class Candidate:
    entities: tuple[str, ...]
    coherence: float
    schema_name: str

    @property
    def signature(self) -> str:
        return _signature(self.entities)


class Verdict(BaseModel):
    """The judge's answer. All three must hold, per Algorithm 1."""

    relevant: bool = False
    important: bool = False
    complete: bool = False
    #: A short slot name for the promoted cluster, e.g. "pets_and_outdoor".
    name: str = ""
    reason: str = ""


class EmergenceJudge(Protocol):
    async def judge(self, entities: list[str], *, schema_name: str) -> Verdict:
        """Score a candidate. Must not raise."""
        ...


class RefusingJudge:
    """Promotes nothing. The fallback when no model is reachable.

    Re-partitioning the memory index on a heuristic is not a degraded version of doing it
    with judgement -- it is a different, worse operation whose mistakes are invisible and
    permanent. Doing nothing is the honest degradation.
    """

    async def judge(self, entities: list[str], *, schema_name: str) -> Verdict:
        return Verdict(reason="no judge available")


class LLMEmergenceJudge:
    """Scores a candidate on the ``reflection`` role."""

    def __init__(
        self,
        *,
        router: RoleRouter,
        capability_registry: CapabilityRegistry,
        role: str = "reflection",
    ) -> None:
        self._router = router
        self._reg = capability_registry
        self._role = role

    async def judge(self, entities: list[str], *, schema_name: str) -> Verdict:
        try:
            client, model = self._router.chat_for(self._role, needs_json_schema=True)
        except LookupError:
            try:
                client, model = self._router.chat_for(self._role)
            except LookupError:
                return Verdict(reason="no binding for the reflection role")

        provider = model.split("/", 1)[0] if "/" in model else "fake"
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "These concepts are repeatedly retrieved together and currently sit in "
                    f"the slot {schema_name!r}. Decide whether they form a coherent topic "
                    "worth its own slot. `relevant`: do they genuinely belong together? "
                    "`important`: is this a recurring part of the person's life rather than "
                    "one episode? `complete`: is the set whole, not an arbitrary fragment? "
                    "All three must be true to split. Give the new slot a short snake_case "
                    "name."
                ),
            ),
            ChatMessage(role="user", content="Concepts: " + ", ".join(entities)),
        ]
        try:
            return await generate_structured(
                client,
                messages,
                model=model,
                schema=Verdict,
                descriptor=self._reg.get(provider, model),
            )
        except (StructuredError, ProviderUnreachable) as exc:
            logger.warning("cluster-emergence: judge failed, promoting nothing: %s", exc)
            return Verdict(reason="judge call failed")
        except Exception:
            logger.exception("cluster-emergence: judge failed; promoting nothing this run")
            return Verdict(reason="judge call failed")


class ClusterEmergence:
    """Proposes and promotes emergent schemas from the co-retrieval log."""

    def __init__(
        self,
        *,
        semantic: SemanticIndex,
        conn: sqlite3.Connection,
        judge: EmergenceJudge,
        alpha: float = DEFAULT_ALPHA,
    ) -> None:
        self._semantic = semantic
        self._conn = conn
        self._judge = judge
        self._alpha = alpha
        conn.executescript(_SCHEMA)
        conn.commit()

    @property
    def semantic(self) -> SemanticIndex:
        """The index this runs over. Exposed so a caller holding the emergence job need
        not also be handed the index to compute an activation set."""
        return self._semantic

    # ------------------------------------------------------------------
    # The co-retrieval log (cold path)
    # ------------------------------------------------------------------

    def log_activation(
        self, *, user_id: str, project: str, entities: Iterable[str], now: datetime
    ) -> str | None:
        """Record which entities one query activated together. Returns the query id.

        A query that activated fewer than two entities says nothing about co-retrieval
        and is not logged -- keeping it would dilute every ρ by a row that can never
        contribute to one.
        """
        names = sorted({e.lower() for e in entities if e})
        if len(names) < 2:
            return None
        query_id = uuid.uuid4().hex
        stamp = now.isoformat()
        self._conn.executemany(
            "INSERT OR IGNORE INTO mem_query_activations "
            "(user_id, project, query_id, entity, seen_at) VALUES (?, ?, ?, ?, ?)",
            [(user_id, project, query_id, n, stamp) for n in names],
        )
        self._conn.commit()
        return query_id

    # ------------------------------------------------------------------
    # Candidates
    # ------------------------------------------------------------------

    def candidates(self, *, user_id: str, project: str) -> list[Candidate]:
        """Connected subsets that clear α, largest first.

        Candidates are the connected components of each schema's induced co-occurrence
        subgraph. A component *is* the natural split: entities the store has never seen
        together cannot be one topic, so enumerating every subset would be both
        exponential and pointless.

        A component covering its whole schema is skipped -- promoting it would rename a
        slot, not partition one.
        """
        queries = self._query_sets(user_id=user_id, project=project)
        if len(queries) < MIN_QUERIES:
            return []
        rejected = self._rejected(user_id=user_id, project=project)

        out: list[Candidate] = []
        for schema in self._semantic.schemas(user_id=user_id, project=project):
            members = self._entities_in(user_id=user_id, project=project, schema=schema)
            if len(members) <= MIN_CANDIDATE_SIZE:
                continue
            for component in self._components(user_id=user_id, project=project, members=members):
                if len(component) < MIN_CANDIDATE_SIZE or component == members:
                    continue
                if _signature(component) in rejected:
                    continue
                rho = coherence(component, queries)
                if rho > self._alpha:
                    out.append(
                        Candidate(
                            entities=tuple(sorted(component)),
                            coherence=round(rho, 4),
                            schema_name=schema,
                        )
                    )
        out.sort(key=lambda c: (-len(c.entities), -c.coherence, c.entities))
        return out

    # ------------------------------------------------------------------
    # Promotion (nightly)
    # ------------------------------------------------------------------

    async def run(self, *, user_id: str, project: str, now: datetime) -> list[str]:
        """Judge the strongest candidate and promote it if all three checks hold.

        One candidate per run, deliberately. Re-partitioning the index changes what every
        future query can route to, and doing several at once makes the effect of any one
        of them unattributable if recall gets worse.
        """
        found = self.candidates(user_id=user_id, project=project)
        if not found:
            return []
        best = found[0]
        verdict = await self._judge.judge(list(best.entities), schema_name=best.schema_name)

        if not (verdict.relevant and verdict.important and verdict.complete):
            self._reject(
                user_id=user_id,
                project=project,
                signature=best.signature,
                reason=verdict.reason or "judge declined",
                now=now,
            )
            return []

        name = _slug(verdict.name) or f"emergent_{best.signature[:8]}"
        self._semantic.add_emergent_schema(
            user_id=user_id,
            project=project,
            name=name,
            description=verdict.reason,
        )
        for entity in best.entities:
            self._semantic.assign(user_id=user_id, project=project, entity=entity, schema_name=name)
        logger.info(
            "cluster-emergence: promoted %d entities out of %r into %r (rho=%.3f)",
            len(best.entities),
            best.schema_name,
            name,
            best.coherence,
        )
        return [name]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _query_sets(self, *, user_id: str, project: str) -> list[set[str]]:
        rows = self._conn.execute(
            "SELECT query_id, entity FROM mem_query_activations WHERE user_id = ? AND project = ?",
            (user_id, project),
        ).fetchall()
        grouped: dict[str, set[str]] = {}
        for r in rows:
            grouped.setdefault(str(r["query_id"]), set()).add(str(r["entity"]))
        return list(grouped.values())

    def _entities_in(self, *, user_id: str, project: str, schema: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT name FROM mem_entity_nodes "
            "WHERE user_id = ? AND project = ? AND schema_name = ?",
            (user_id, project, schema),
        ).fetchall()
        return {str(r["name"]) for r in rows}

    def _components(self, *, user_id: str, project: str, members: set[str]) -> list[set[str]]:
        """Connected components of the co-occurrence subgraph induced by *members*."""
        member_json = json.dumps(sorted(members))
        rows = self._conn.execute(
            "SELECT src, dst FROM mem_entity_edges WHERE user_id = ? AND project = ? "
            "AND src IN (SELECT value FROM json_each(?)) "
            "AND dst IN (SELECT value FROM json_each(?))",
            (user_id, project, member_json, member_json),
        ).fetchall()
        adjacency: dict[str, set[str]] = {m: set() for m in members}
        for r in rows:
            adjacency[str(r["src"])].add(str(r["dst"]))
            adjacency[str(r["dst"])].add(str(r["src"]))

        seen: set[str] = set()
        components: list[set[str]] = []
        for start in sorted(members):
            if start in seen:
                continue
            stack, component = [start], set()
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                stack.extend(adjacency[node] - component)
            seen |= component
            components.append(component)
        return components

    def _rejected(self, *, user_id: str, project: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT signature FROM mem_emergence_rejected WHERE user_id = ? AND project = ?",
            (user_id, project),
        ).fetchall()
        return {str(r["signature"]) for r in rows}

    def _reject(
        self, *, user_id: str, project: str, signature: str, reason: str, now: datetime
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO mem_emergence_rejected "
            "(user_id, project, signature, reason, rejected_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, project, signature, reason, now.isoformat()),
        )
        self._conn.commit()


def coherence(subset: set[str], queries: list[set[str]]) -> float:
    """ρ(H) from §3.1: mean Jaccard overlap between *subset* and each query's activations.

    Queries that activated nothing in the subset still count in the denominator -- that
    is what stops a subset from looking coherent merely because it is rarely touched.
    """
    if not queries or not subset:
        return 0.0
    total = 0.0
    for activated in queries:
        union = activated | subset
        if union:
            total += len(activated & subset) / len(union)
    return total / len(queries)


def _slug(name: str) -> str:
    cleaned = "".join(c if c.isalnum() else "_" for c in name.strip().lower())
    return "_".join(part for part in cleaned.split("_") if part)[:40]
