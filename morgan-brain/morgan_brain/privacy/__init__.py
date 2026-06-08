"""Privacy foundation — data classification, egress redaction, field encryption.

All controls are opt-in via config flags (``redact_egress``, ``encryption``).
The seams (EgressGate, EgressRedactor, crypto functions) exist unconditionally
so enabling privacy is a flag flip, not a refactor.

Public surface::

    from morgan_brain.privacy import DataClass, classify
    from morgan_brain.privacy import EgressRedactor, RedactionMap, rehydrate_stream
    from morgan_brain.privacy import EgressGate
    from morgan_brain.privacy import new_dek, seal, open_sealed, derive_kek
"""

from morgan_brain.privacy.classification import DataClass, classify
from morgan_brain.privacy.redaction import EgressRedactor, RedactionMap, rehydrate_stream
from morgan_brain.privacy.egress import EgressGate
from morgan_brain.privacy.crypto import new_dek, seal, open_sealed, derive_kek

__all__ = [
    "DataClass",
    "classify",
    "EgressRedactor",
    "RedactionMap",
    "rehydrate_stream",
    "EgressGate",
    "new_dek",
    "seal",
    "open_sealed",
    "derive_kek",
]
