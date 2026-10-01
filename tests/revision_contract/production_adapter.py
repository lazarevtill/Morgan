"""Execute operation-only fixtures; obtain revision state exclusively from production APIs.

Expected answers are rejected at this boundary. Historical facts are recorded snapshots,
not proposals: seed those through the temporal store; candidate proposals go through Gate.
No revision leaf-selection algorithm or scoring labels belong here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from morgan_brain.app.chat import build_messages
from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.revisions import RevisionError
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.episodic import EpisodicStore
from morgan_brain.memory.store.temporal import SqliteTemporalStore
from morgan_brain.models import Memory, MemoryQuery, TemporalFact

REASONS = frozenset(
    {
        "revision_actor_boundary",
        "revision_parent_unavailable",
        "revision_effective_time",
        "revision_family_boundary",
        "revision_parent_limit",
        "stale_revision_basis",
    }
)


def time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def operation_only(scenario: dict[str, Any]) -> dict[str, Any]:
    """The caller/scorer strips all gold before the observer sees an operation."""
    return {
        "id": scenario["id"],
        "events": scenario["events"],
        "facts": scenario["facts"],
        "observations": [
            {key: value for key, value in row.items() if key != "expected"}
            for row in scenario["observations"]
        ],
        "write_observations": [
            {key: value for key, value in row.items() if key != "expected"}
            for row in scenario["write_observations"]
        ],
    }


def require_no_gold(value: Any) -> None:
    if isinstance(value, dict):
        if "expected" in value:
            raise ValueError("Gold cannot enter the production observer")
        for child in value.values():
            require_no_gold(child)
    elif isinstance(value, list):
        for child in value:
            require_no_gold(child)


def error_reason(error: Exception) -> str:
    if isinstance(error, RevisionError):
        if error.reason not in REASONS:
            raise ValueError(f"Unrecognized production reason: {error.reason}") from error
        return error.reason
    if isinstance(error, ValidationError):
        reasons = {
            str(item.get("ctx", {}).get("error", "")).split(":", 1)[0] for item in error.errors()
        }
        if len(reasons) == 1 and reasons <= REASONS:
            return reasons.pop()
    raise ValueError("Unrecognized production error; no gold-derived fallback") from error


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__(dim=4)
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return await super().embed(text)


def fact_input(row: dict[str, Any], scope: tuple[str, str]) -> TemporalFact:
    return TemporalFact.model_validate(
        {
            **row,
            "user_id": row.get("user_id", scope[0]),
            "project": row.get("project", scope[1]),
            "subject": row["id"],
            "predicate": "synthetic_contract",
            "object": "recorded assertion",
        }
    )


async def observe(scenario: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    require_no_gold(scenario)
    conn = open_db(":memory:")
    clock = {"now": datetime(2026, 1, 1, tzinfo=UTC)}
    embedder = CountingEmbedder()
    module = build_memory_module(conn, embedder=embedder, dim=4, clock=lambda: clock["now"])
    gate = MemoryGate(module)
    temporal = SqliteTemporalStore(conn=conn, initialize=False)
    try:
        for row in scenario["events"]:
            if row["recorded_at"] is None:
                # A preserved legacy snapshot is not a new ingestion. Keep missing
                # recorded time and reported attribution exactly as supplied.
                EpisodicStore(conn, initialize=False).put(Memory.model_validate(row))
            else:
                clock["now"] = time(row["recorded_at"])
                await gate.store(Memory.model_validate(row))
        first = scenario["events"][0]
        scope = (first["user_id"], first.get("project", "personal"))
        for row in scenario["facts"]:
            await temporal.upsert_fact(fact_input(row, scope), now=clock["now"])
        observations = [
            await observe_at(gate, scenario, scope, row) for row in scenario["observations"]
        ]
        writes = [
            await observe_write(gate, conn, clock, embedder, scenario["id"], row, scope)
            for row in scenario["write_observations"]
        ]
        return {"observations": observations, "writes": writes}
    finally:
        conn.close()


def inventory(scenario, scope, owner, context):
    """Use input inventory only, never the gold's selected IDs."""
    return [
        row["id"]
        for row in scenario["events"] + scenario["facts"]
        if (row.get("user_id", scope[0]), row.get("project", scope[1])) == (owner, context)
    ]


async def instruction_exposure(gate, owner, context, at):
    recalled = await gate.recall(
        MemoryQuery(
            user_id=owner, project=context, text="synthetic contract", top_k=32, effective_at=at
        )
    )
    messages = build_messages(memories=recalled.memories, history=[], text="synthetic contract")
    # Native-prompt exposure proxy only, not actual command execution/model obedience.
    return {
        item.id
        for item in recalled.memories
        if item.instruction_like and item.content in messages[0].content
    }


async def observe_at(gate, scenario, scope, row):
    records = []
    promoted = set()
    scopes = {(item["user_id"], item.get("project", "personal")) for item in scenario["events"]}
    for owner, context in sorted(scopes):
        ids = inventory(scenario, scope, owner, context)
        for start in range(0, len(ids), 32):
            evidence = await gate.evidence(
                user_id=owner,
                project=context,
                evidence_ids=ids[start : start + 32],
                effective_at=time(row["effective_at"]),
            )
            records.extend(evidence.records)
        promoted.update(await instruction_exposure(gate, owner, context, time(row["effective_at"])))
    events = [record for record in records if record.revision_state is not None]
    return {
        "scenario_id": scenario["id"],
        "observation_id": row["id"],
        "eligible_leaf_ids": sorted(
            record.id for record in events if record.revision_state in ("active", "conflicted")
        ),
        "conflicted_root_ids": sorted(
            {record.revision_root_id for record in events if record.eligible_leaf_count > 1}
        ),
        "raw_history_ids": sorted(record.id for record in events),
        "fact_states": {
            record.id: record.support_state
            for record in records
            if record.support_state is not None
        },
        "executable_instruction_ids": sorted(promoted),
    }


async def observe_write(gate, conn, clock, embedder, scenario_id, row, scope):
    before = {item[0] for item in conn.execute("SELECT id FROM memories")}
    before_facts = {item[0] for item in conn.execute("SELECT id FROM facts")}
    snapshot = conn.serialize()
    recorded = dict(conn.execute("SELECT id,recorded_at FROM memories"))
    calls = embedder.calls
    clock["now"] = time(
        row["applied_at"] if "candidate_fact" in row else row["candidate"]["recorded_at"]
    )
    output = {"scenario_id": scenario_id, "write_id": row["id"]}
    try:
        if "candidate_fact" in row:
            await gate.upsert_fact(fact_input(row["candidate_fact"], scope))
        else:
            await gate.store(Memory.model_validate(row["candidate"]))
        output["outcome"] = "replayed" if conn.serialize() == snapshot else "accepted"
    except (RevisionError, ValidationError) as error:
        output.update(outcome="rejected", reason=error_reason(error))
        if conn.serialize() != snapshot:
            raise AssertionError("Rejected operation mutated persisted data") from error
    if "candidate_fact" in row:
        after = {item[0] for item in conn.execute("SELECT id FROM facts")}
        output["inserted_fact_ids"] = sorted(after - before_facts)
    else:
        after = {item[0] for item in conn.execute("SELECT id FROM memories")}
        output["inserted_event_ids"] = sorted(after - before)
        output["embedding_calls"] = embedder.calls - calls
        output["recorded_at_unchanged"] = recorded == dict(
            conn.execute("SELECT id,recorded_at FROM memories")
        )
    return output
