"""Recovery must not admit unrelated changes hidden in a pending renewal journal."""

import json

import pytest
from pydantic import ValidationError

from morgan_brain.guided_renewal import RenewalUpdate


def review():
    """An explicitly reviewed synthetic date-only change with a stable generated source ID."""
    before = {
        "version": "morgan.commitment.v1",
        "title": "Return the book.",
        "due": "2027-02-01",
        "minutes": 10,
        "priority": "normal",
        "status": "open",
    }
    return {
        "event_id": "guided-renew-example",
        "parent_id": "original",
        "source_text": "I did not return it. Remind me again.",
        "effective_at": "2026-11-06T12:00:00+00:00",
        "before": before,
        "after": {**before, "due": "2027-02-02"},
    }


@pytest.mark.parametrize(
    "field,value", [("title", "An unrelated task"), ("status", "done"), ("minutes", 30)]
)
def test_pending_renewal_cannot_change_unreviewed_fields(field, value):
    """A resumed file cannot smuggle a new target, completion, or estimate into date renewal."""
    payload = review()
    payload["after"][field] = value
    with pytest.raises(ValidationError):
        RenewalUpdate.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_text", "bad\x1b[31m"),
        ("parent_id", "guided-renew-example-source"),
        ("effective_at", "2026-11-06T12:00:00"),
    ],
)
def test_pending_renewal_rejects_bad_provenance(field, value):
    """Sources are printable, identity links cannot self-reference, and time is explicit."""
    payload = review()
    payload[field] = value
    with pytest.raises(ValidationError):
        RenewalUpdate.model_validate_json(json.dumps(payload))
