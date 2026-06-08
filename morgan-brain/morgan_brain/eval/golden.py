"""Golden eval set — hand-authored probe items for L2 preference-following accuracy.

ProbeType taxonomy (per the ADR / LongMemEval):
- EXPLICIT_RECALL        — a fact the assistant was explicitly told.
- IMPLICIT_TRAIT         — a trait inferred from patterns, not stated directly.
- PREFERENCE_UPDATE      — the value AFTER the owner changed their mind (tests anti-staleness).
- LONG_GAP_DECAY         — fact mentioned long ago; may have decayed (tests uncertainty signalling).
- OVER_PERSONALIZATION_NEGATIVE — a stale preference that must NOT be applied (anti-sycophancy gate).
- ABSTENTION             — no known fact; assistant should say "I don't know".

GoldenItem.should_inject:
- True (default) for all positive probes — the preference SHOULD be applied.
- False for OVER_PERSONALIZATION_NEGATIVE — the stale pref must NOT be injected.

Eval items are FIREWALLED from what the assistant may consolidate:
the harness reads predict_fn output only; it never writes to memory.
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class ProbeType(str, Enum):
    """Taxonomy of golden-set probe types."""

    EXPLICIT_RECALL = "EXPLICIT_RECALL"
    IMPLICIT_TRAIT = "IMPLICIT_TRAIT"
    PREFERENCE_UPDATE = "PREFERENCE_UPDATE"
    LONG_GAP_DECAY = "LONG_GAP_DECAY"
    OVER_PERSONALIZATION_NEGATIVE = "OVER_PERSONALIZATION_NEGATIVE"
    ABSTENTION = "ABSTENTION"


class GoldenItem(BaseModel):
    """One hand-authored eval probe.

    Attributes:
        id:            Unique identifier for this item.
        probe:         Probe type classification.
        setup:         Facts / memories to seed before the query (list of strings).
        query:         The question or request to the assistant.
        expected:      Reference answer (ground truth).
        should_inject: True → the preference/fact SHOULD be applied.
                       False → a stale pref must NOT be applied (NEGATIVE items).
    """

    id: str
    probe: ProbeType
    setup: list[str] = Field(default_factory=list)
    query: str
    expected: str
    should_inject: bool = True


def load_golden_set(path: Path | str) -> list[GoldenItem]:
    """Load and validate a golden set from a JSON file.

    Args:
        path: Path to the JSON file containing a list of GoldenItem dicts.

    Returns:
        List of validated ``GoldenItem`` instances, preserving file order.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError:        If the JSON is malformed or fails Pydantic validation.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        raw: list[dict[str, object]] = json.load(fh)
    return [GoldenItem.model_validate(item) for item in raw]
