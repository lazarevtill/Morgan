"""Bounded, ephemeral preparation state; never model-authored or persisted."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from morgan_brain.models import Memory, TemporalFact

MAX_SOURCE_INPUTS = 50
MAX_FACT_INPUTS = 256


class StaleConsolidationProposal(ValueError):
    """Prepare again explicitly; the old proposal must not be retried automatically."""


class UnsupportedConsolidationOperation(ValueError):
    """Automatic writes need declared trusted supports from their preparation inputs."""


class ConsolidationInputLimit(ValueError):
    """Refuse an oversized input instead of silently claiming a complete basis."""


@dataclass(frozen=True)
class SourceBasis:
    event_id: str
    root_id: str
    leaf_ids: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class ConsolidationBasis:
    user_id: str
    project: str
    generation: int
    prepared_at: datetime
    sources: tuple[SourceBasis, ...]
    fact_fingerprint: str
    fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class ConsolidationInput:
    basis: ConsolidationBasis
    episodics: tuple[Memory, ...]
    facts: tuple[TemporalFact, ...]


def event_fingerprint(event: Memory) -> str:
    # Raw durable source fields include reported provenance; derived recall metadata
    # must not affect CAS. No source text is retained in the basis itself.
    raw = event.model_dump(
        mode="json",
        exclude={
            "revision_state",
            "eligible_leaf_ids",
            "eligible_leaf_count",
            "revision_truncated",
            "support_state",
        },
    )
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()


def fact_fingerprint(facts: list[TemporalFact]) -> str:
    digest = hashlib.sha256()
    for fact in sorted(facts, key=lambda value: value.id):
        # Includes confidence, last-confirmed and intervals as well as identity/value.
        digest.update(
            json.dumps(
                fact.model_dump(mode="json", exclude={"support_state"}), sort_keys=True
            ).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()
