"""Bounded shared CLI/MCP input contracts validated before opening a database."""


def evidence_ids(values: list[str]) -> list[str]:
    if not isinstance(values, list) or not 1 <= len(values) <= 32:
        raise ValueError("evidence requires between 1 and 32 IDs")
    if any(not isinstance(value, str) or not value.strip() or len(value) > 256 for value in values):
        raise ValueError("evidence IDs must be nonempty strings of at most 256 characters")
    return list(dict.fromkeys(values))
