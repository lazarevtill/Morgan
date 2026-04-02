"""Session permission modes."""

from enum import Enum


class SessionPermissionMode(Enum):
    """Permission levels for a session."""

    INTERACTIVE = "interactive"
    AUTONOMOUS = "autonomous"
    RESTRICTED = "restricted"
