"""Reviewed unavailability intervals: temporal constraints, never a location assertion."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, model_validator


def planning_zone(name: str) -> ZoneInfo:
    """Require an explicit IANA zone rather than inheriting the machine's local zone."""
    if not isinstance(name, str) or not name or len(name) > 128:
        raise ValueError("an explicit IANA timezone is required")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError("unknown IANA timezone") from error


class Unavailability(BaseModel):
    """Account-confirmed half-open interval, anchored to explicit source evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["morgan.unavailability.v1"]
    starts_at: datetime
    ends_at: datetime
    timezone: str
    status: Literal["active", "cancelled"]
    source_ids: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def check_interval(self) -> Self:
        """Explicit offsets disambiguate folds; reject nonexistent times and offset mismatch."""
        zone = planning_zone(self.timezone)
        for value in (self.starts_at, self.ends_at):
            if value.utcoffset() is None:
                raise ValueError("interval endpoints require explicit UTC offsets")
            local = value.astimezone(UTC).astimezone(zone)
            if local.replace(tzinfo=None) != value.replace(tzinfo=None) or (
                local.utcoffset() != value.utcoffset()
            ):
                raise ValueError("endpoint offset or wall time does not match timezone")
        duration = self.ends_at.astimezone(UTC) - self.starts_at.astimezone(UTC)
        if not timedelta(0) < duration <= timedelta(days=366):
            raise ValueError("interval must be positive and at most 366 days")
        if len(set(self.source_ids)) != len(self.source_ids) or any(
            not identity.strip() or not identity.isprintable() or len(identity) > 256
            for identity in self.source_ids
        ):
            raise ValueError("select distinct printable source IDs")
        return self

    def overlaps_day(self, day: date, timezone: str) -> bool:
        """Use real UTC day boundaries, including 23/25-hour DST days; end is exclusive."""
        zone = planning_zone(timezone)
        try:
            start = datetime.combine(day, time.min, zone).astimezone(UTC)
            end = datetime.combine(day + timedelta(days=1), time.min, zone).astimezone(UTC)
        except (OverflowError, ValueError) as error:
            raise ValueError("planning day has no representable boundaries") from error
        if end <= start:
            raise ValueError("planning day does not exist in this timezone")
        return self.status == "active" and (
            self.starts_at.astimezone(UTC) < end and self.ends_at.astimezone(UTC) > start
        )
