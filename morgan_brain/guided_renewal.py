"""Explicit reviewed renewal of one current task; message text cannot select a target."""

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from morgan_brain.daily_brief import Commitment
from morgan_brain.guided_availability import source_statement


class RenewalUpdate(BaseModel):
    """Bind the chosen parent, exact source and new date before either immutable write."""

    model_config = ConfigDict(extra="forbid")
    event_id: str
    parent_id: str
    source_text: str
    effective_at: str
    before: Commitment
    after: Commitment

    @field_validator("source_text")
    @classmethod
    def visible_source(cls, value: str) -> str:
        """Preserve one printable owner-entered statement, without terminal controls."""
        return source_statement(value)

    @model_validator(mode="after")
    def valid_review(self) -> Self:
        """Renew only the date of an open task; never reopen or silently alter another field."""
        if self.before.status != "open" or self.after.status != "open":
            raise ValueError("renewal requires an open task")
        if self.before.model_copy(update={"due": self.after.due}) != self.after:
            raise ValueError("renewal may change only the reviewed date")
        for identity, limit in ((self.event_id, 200), (self.parent_id, 256)):
            if not identity.strip() or len(identity) > limit or not identity.isprintable():
                raise ValueError("invalid renewal identities")
        if not self.event_id.startswith("guided-renew-") or self.parent_id in (
            self.event_id,
            self.source_id,
        ):
            raise ValueError("invalid renewal identities")
        if datetime.fromisoformat(self.effective_at).utcoffset() is None:
            raise ValueError("renewal requires an offset-bearing time")
        return self

    @property
    def source_id(self) -> str:
        """Generated companion identity, not inferred message ancestry or a guessed parent."""
        return f"{self.event_id}-source"
