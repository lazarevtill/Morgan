"""Bounded shared CLI/MCP input contracts validated before opening a database."""

from datetime import datetime


def effective_time(value: str | None) -> datetime | None:
    """Accept an explicit timezone-aware ISO timestamp; never infer a caller timezone."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 64:
        raise ValueError("effective_at must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("effective_at must be a timezone-aware ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("effective_at must include a timezone")
    return parsed


def event_identity(value: str | None) -> str | None:
    if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 256):
        raise ValueError("event_id must be a nonempty string of at most 256 characters")
    return value


def revision_parents(values: list[str] | None) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list) or len(values) > 8:
        raise ValueError("revises_event_ids must contain at most 8 distinct IDs")
    for value in values:
        if value is None:
            raise ValueError("revision parent IDs must be nonempty strings")
        event_identity(value)
    if len(set(values)) != len(values):
        raise ValueError("revises_event_ids must contain distinct IDs")
    return sorted(values)


def evidence_ids(values: list[str]) -> list[str]:
    if not isinstance(values, list) or not 1 <= len(values) <= 32:
        raise ValueError("evidence requires between 1 and 32 IDs")
    if any(not isinstance(value, str) or not value.strip() or len(value) > 256 for value in values):
        raise ValueError("evidence IDs must be nonempty strings of at most 256 characters")
    return list(dict.fromkeys(values))
