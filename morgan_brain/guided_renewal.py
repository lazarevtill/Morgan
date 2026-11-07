"""Reviewed task transitions; message text cannot select a target."""

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
        """Preserve source line feeds for owner inbox reviews, without other controls."""
        return source_statement(value, multiline=True)

    @model_validator(mode="after")
    def valid_review(self) -> Self:
        """Validate a date renewal, closure or reopening, never arbitrary edits."""
        completion = self.event_id.startswith("guided-complete-")
        cancellation = self.event_id.startswith("guided-cancel-")
        reopening = self.event_id.startswith("guided-reopen-")
        if reopening:
            if self.before.status != "done" or self.after != self.before.model_copy(
                update={"status": "open", "due": self.after.due}
            ):
                raise ValueError("reopening may only restore a completed task with a reviewed date")
        elif completion or cancellation:
            if self.before.status != "open" or self.after != self.before.model_copy(
                update={"status": "done" if completion else "cancelled"}
            ):
                raise ValueError("completion/cancellation may only close an open task")
        elif self.before.status != "open" or self.after.status != "open":
            raise ValueError("renewal requires an open task")
        if (
            not (completion or cancellation or reopening)
            and self.before.model_copy(update={"due": self.after.due}) != self.after
        ):
            raise ValueError("renewal may change only the reviewed date")
        for identity, limit in ((self.event_id, 200), (self.parent_id, 256)):
            if not identity.strip() or len(identity) > limit or not identity.isprintable():
                raise ValueError("invalid renewal identities")
        if not self.event_id.startswith(
            ("guided-renew-", "guided-complete-", "guided-cancel-", "guided-reopen-")
        ) or self.parent_id in (
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
