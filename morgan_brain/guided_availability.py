"""Typed pending dialogue writes for confirmed unavailability; no inference or services."""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from morgan_brain.availability import Unavailability, planning_zone


class AvailabilityUpdate(BaseModel):
    """Persist source and assertion identities together before either immutable capture."""

    model_config = ConfigDict(extra="forbid")
    source_id: str
    source_text: str = Field(min_length=1, max_length=1000)
    event_id: str
    effective_at: str
    interval: Unavailability
    parent_id: str | None = None

    @field_validator("source_text")
    @classmethod
    def check_source(cls, value: str) -> str:
        """Only one visible user-entered statement; never terminal controls or blank text."""
        return source_statement(value)

    @model_validator(mode="after")
    def check_binding(self) -> Self:
        """The assertion must refer to exactly its saved source; validate recovery time too."""
        if self.interval.source_ids != [self.source_id] or self.source_id == self.event_id:
            raise ValueError("pending interval must bind its exact source")
        if datetime.fromisoformat(self.effective_at).utcoffset() is None:
            raise ValueError("pending write requires an offset-bearing event time")
        return self


def source_statement(value: str) -> str:
    """Retain the user's own words, not a generated reason or a location inference."""
    if not value.strip() or len(value) > 1000 or not value.isprintable():
        raise ValueError("enter one printable statement of 1..1000 characters")
    return value


def explicit_zone(value: str) -> str:
    """Validate the user's chosen planning zone without guessing from a device or location."""
    value = value.strip()
    planning_zone(value)
    return value


def interval_entry(raw: str, timezone: str, source_id: str) -> Unavailability:
    """Require two reviewed ISO timestamps; offsets explicitly disambiguate repeated hours."""
    values = raw.split()
    if len(values) != 2:
        raise ValueError("enter start and exclusive end, both with UTC offsets")
    return Unavailability(
        version="morgan.unavailability.v1",
        starts_at=datetime.fromisoformat(values[0]),
        ends_at=datetime.fromisoformat(values[1]),
        timezone=timezone,
        status="active",
        source_ids=[source_id],
    )
