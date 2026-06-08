"""Data classification — assign a sensitivity tier to text or metadata.

Tiers (ordered lowest → highest):
    PUBLIC    — no identifiable information detected.
    PERSONAL  — PII signals present (email, phone number, person-like names).
    SENSITIVE — health, financial, or credential keywords / formatted identifiers.
    SECRET    — secret-material patterns: API keys, tokens, passwords, private keys.

Usage::

    from morgan_brain.privacy.classification import DataClass, classify

    tier = classify("my password is hunter2")        # → DataClass.SECRET
    tier = classify("call me at 555-123-4567")       # → DataClass.PERSONAL (at least)
    tier = classify("the weather is nice")           # → DataClass.PUBLIC
    tier = classify("anything", explicit=DataClass.SENSITIVE)  # → DataClass.SENSITIVE

Heuristics are intentionally **conservative** — a match promotes to that tier or
higher, never demotes.  The ordering is PUBLIC < PERSONAL < SENSITIVE < SECRET,
so the most sensitive match wins.

Non-English names and informal PII are a known gap; Presidio NER (optional) closes
most of that gap at the redaction layer (see ``redaction.py``).
"""
from __future__ import annotations

import re
from enum import Enum

# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------

_TIER_ORDER = {"PUBLIC": 0, "PERSONAL": 1, "SENSITIVE": 2, "SECRET": 3}


class DataClass(str, Enum):
    """Sensitivity classification for a piece of text or metadata field."""

    PUBLIC = "PUBLIC"
    PERSONAL = "PERSONAL"
    SENSITIVE = "SENSITIVE"
    SECRET = "SECRET"

    # Convenience: comparison by tier ordering (PUBLIC < PERSONAL < SENSITIVE < SECRET).
    def __lt__(self, other: object) -> bool:
        if not isinstance(other, DataClass):
            return NotImplemented
        return _TIER_ORDER[self.value] < _TIER_ORDER[other.value]

    def __le__(self, other: object) -> bool:
        if not isinstance(other, DataClass):
            return NotImplemented
        return _TIER_ORDER[self.value] <= _TIER_ORDER[other.value]

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, DataClass):
            return NotImplemented
        return _TIER_ORDER[self.value] > _TIER_ORDER[other.value]

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, DataClass):
            return NotImplemented
        return _TIER_ORDER[self.value] >= _TIER_ORDER[other.value]


# ---------------------------------------------------------------------------
# SECRET patterns — credential-shaped strings that should never leave the device
# ---------------------------------------------------------------------------

# Generic API key / token patterns:
#   - starts with a typical prefix (sk-, ghp_, xoxb-, etc.) or the word "key"/"token"
#   - ≥20 chars of base64/hex-ish characters
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    # Prefixed secrets: sk-, ghp_, xoxb-, xapp-, xoxa-, ghs_, glpat-, bearer, etc.
    re.compile(r"\b(?:sk|ghp|ghs|gho|ghu|ghx|xoxb|xapp|xoxa|glpat)[-_][A-Za-z0-9_\-]{16,}", re.I),
    # AWS-style keys: AKIA / ASIA followed by 16+ uppercase alnum chars
    re.compile(r"\b(?:AKIA|ASIA|AROA|AIPA|ANPA|ANVA|APKA)[A-Z0-9]{16,}\b"),
    # 32–64 hex chars next to a "secret" / "api_key" / "token" / "password" label
    re.compile(
        r'(?:api[_\s-]?key|api[_\s-]?secret|token|secret|password|passwd|private[_\s-]?key'
        r'|auth[_\s-]?key|access[_\s-]?key)\s*[=:"\s]+[A-Za-z0-9+/=_\-]{20,}',
        re.I,
    ),
    # Private key PEM header
    re.compile(r"-----BEGIN\s+(?:RSA |EC |OPENSSH |PGP |DSA )?PRIVATE KEY-----", re.I),
    # Generic long random-looking hex string next to a key-word (conservative — only if labeled)
    re.compile(
        r'(?:key|token|secret|password)\s*[=:]\s*[0-9a-fA-F]{32,}',
        re.I,
    ),
    # Bearer token in Authorization header
    re.compile(r"\bBearer\s+[A-Za-z0-9\-_.~+/]+=*\b", re.I),
]

# ---------------------------------------------------------------------------
# SENSITIVE patterns — health/financial/credential-ish
# ---------------------------------------------------------------------------

# US Social Security Number: NNN-NN-NNNN or NNNNNNNNN
_SSN_RE = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")

# Credit/debit card: 13–19 consecutive digits with optional spaces/dashes (Luhn not checked)
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")

# Keywords indicating health or financial context
_SENSITIVE_KW = re.compile(
    r"\b(?:diagnosis|prescription|medication|therapy|blood[_ ]?type|hiv|cancer|diabetes"
    r"|salary|income|tax|ssn|credit[_ ]?card|bank[_ ]?account|iban|routing[_ ]?number"
    r"|password|passwd|credential|secret|token|api[_ ]?key|private[_ ]?key"
    r"|social[_ ]?security)\b",
    re.I,
)

# ---------------------------------------------------------------------------
# PERSONAL patterns — PII signals
# ---------------------------------------------------------------------------

# RFC 5322 simplified email (local@domain.tld)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# Phone: international +N... or North-American NXX-NXX-XXXX / (NXX) NXX-XXXX
_PHONE_RE = re.compile(
    r"(?:\+\d{1,3}[-.\s]?)?"
    r"(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    r"|\d{3}[-.\s]\d{3}[-.\s]\d{4})",
)


def _max_class(*classes: DataClass) -> DataClass:
    """Return the highest-tier DataClass from the given set."""
    return max(classes, key=lambda c: _TIER_ORDER[c.value])


def classify(text: str, *, explicit: DataClass | None = None) -> DataClass:
    """Classify *text* into a ``DataClass`` sensitivity tier.

    If *explicit* is provided it is returned immediately (caller override).

    Otherwise heuristic regex scanning is applied in priority order:
    SECRET → SENSITIVE → PERSONAL → PUBLIC.

    Parameters
    ----------
    text:
        The text to classify.
    explicit:
        Override the heuristic; if given, the text is not scanned.

    Returns
    -------
    DataClass
        The most sensitive tier detected, or PUBLIC if no patterns match.
    """
    if explicit is not None:
        return explicit

    current = DataClass.PUBLIC

    # --- SECRET tier checks (short-circuit: once SECRET, nothing can raise it further) ---
    for pat in _SECRET_PATTERNS:
        if pat.search(text):
            return DataClass.SECRET

    # --- SENSITIVE tier checks ---
    sensitive_hit = (
        bool(_SSN_RE.search(text))
        or bool(_CARD_RE.search(text))
        or bool(_SENSITIVE_KW.search(text))
    )
    if sensitive_hit:
        current = _max_class(current, DataClass.SENSITIVE)

    # --- PERSONAL tier checks ---
    personal_hit = bool(_EMAIL_RE.search(text)) or bool(_PHONE_RE.search(text))
    if personal_hit:
        current = _max_class(current, DataClass.PERSONAL)

    return current
