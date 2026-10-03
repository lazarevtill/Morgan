"""Seal input provenance across CLI restarts while leaving output selections editable."""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from dataclasses import dataclass

from morgan_brain.memory.checkpoints import CheckpointContext
from morgan_brain.memory.store.proposal_key import read_key
from morgan_brain.models import Memory


@dataclass(frozen=True)
class ProposalInputs:
    context_id: str
    context: CheckpointContext
    expected_fact_id: str | None
    generation: int
    records: list[Memory]


def seal_inputs(conn: sqlite3.Connection, inputs: ProposalInputs) -> str:
    context_id, context = inputs.context_id, inputs.context
    expected_fact_id, generation = inputs.expected_fact_id, inputs.generation
    records = inputs.records
    if not 1 <= len(records) <= 36 or len({r.id for r in records}) != len(records):
        raise ValueError("proposal input basis requires 1 to 36 distinct records")
    if any((r.user_id, r.project) != (context.user_id, context.project) for r in records):
        raise ValueError("proposal input basis scope mismatch")
    payload = {
        "contract": "morgan.proposal-input-seal.v1",
        "context_id": context_id,
        "context": context.model_dump(mode="json"),
        "expected_fact_id": expected_fact_id,
        "generation": generation,
        "records": [
            r.model_dump(mode="json", exclude={"embedding", "entities", "importance"})
            for r in sorted(records, key=lambda r: r.id)
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hmac.new(read_key(conn), encoded, hashlib.sha256).hexdigest()


def verify_inputs(conn: sqlite3.Connection, seal: str, inputs: ProposalInputs) -> None:
    expected = seal_inputs(conn, inputs)
    if not hmac.compare_digest(seal, expected):
        raise ValueError("proposal input basis changed; prepare a new proposal")
