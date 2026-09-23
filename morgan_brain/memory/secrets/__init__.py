"""The secret gate: what a secret looks like (``rules``), the scanner that applies the rules
(``scan``) and the report-only pass over the transcripts (``report``). CPU only: no model, no
embedding, no network. Nothing this package produces carries a matched value."""

from morgan_brain.memory.secrets.rules import GATE_VERSION, GateLimits
from morgan_brain.memory.secrets.scan import (
    REDACTION,
    Hit,
    Scanner,
    ScanResult,
    SecretRefused,
    TextVerdict,
    Verdict,
    build_scanner,
    strip_userinfo,
)

__all__ = [
    "GATE_VERSION",
    "REDACTION",
    "GateLimits",
    "Hit",
    "ScanResult",
    "Scanner",
    "SecretRefused",
    "TextVerdict",
    "Verdict",
    "build_scanner",
    "strip_userinfo",
]
