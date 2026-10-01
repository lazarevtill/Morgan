"""Typed memory refusals shared by persistence and application code."""


class SourceProtectionError(ValueError):
    """An unattributed or inferred operation would replace a user statement."""


class DatabaseSchemaTooNew(ValueError):
    """The recorded schema belongs to a newer build; initialization must not write."""

    def __init__(self, database_version: int, supported_version: int) -> None:
        self.database_version = database_version
        self.supported_version = supported_version
        super().__init__(
            f"Morgan database user_version {database_version} is newer than this build's "
            f"supported schema {supported_version}. Use a matching newer build; "
            "do not downgrade this database in place."
        )


class EvidenceChanged(ValueError):
    """Prepared answer evidence no longer describes the commit-time scoped state."""

    reason = "evidence_changed"

    def __init__(self) -> None:
        super().__init__(self.reason)
