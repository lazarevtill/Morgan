"""Typed memory refusals shared by persistence and application code."""


class SourceProtectionError(ValueError):
    """An unattributed or inferred operation would replace a user statement."""
