"""Reversible egress PII redaction.

Two tiers:
  Tier 1 (always active) — regex patterns for email, phone, credit card, IP,
      API-key-shaped strings, and SSN.  Deterministic within a redactor instance:
      the same input span always maps to the same placeholder in a session.
  Tier 2 (optional) — Presidio NER via lazy import. If ``presidio-analyzer`` is
      not installed, silently skip (regex-only mode).

Placeholders use the format ``«TYPE_N»`` (e.g. ``«EMAIL_1»``).  Guillemet delimiters
are chosen because they are extremely unlikely to appear in normal LLM input/output.

Round-trip guarantee::

    redactor = EgressRedactor()
    redacted, rmap = redactor.redact(text)
    assert redactor.rehydrate(redacted, rmap) == text  # exact for all redacted spans

Known gap — non-English names:
    Regex-only mode cannot detect names written in Latin script without strong
    contextual cues (e.g. "Alice Smith called").  Presidio's NER model closes
    this gap.  The test suite asserts this limitation explicitly so it is
    documented and visible, not silent.

Streaming helper::

    fn = rehydrate_stream(rmap)
    for chunk in chunks:
        out = fn(chunk)     # may be empty if a placeholder straddles the boundary
    out = fn(None)          # flush — returns any buffered remainder
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable

# ``RedactionMap`` is a plain dict: placeholder → original text.
RedactionMap = dict[str, str]


# ---------------------------------------------------------------------------
# Regex patterns (Tier 1)
# ---------------------------------------------------------------------------

# Email: simplified RFC 5322
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# Phone: international or North-American style
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+\d{1,3}[-.\s]?)?"
    r"(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    r"|\d{3}[-.\s]\d{3}[-.\s]\d{4})"
    r"(?!\d)"
)

# Credit / debit card: 13–19 digits with optional space/dash separators
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")

# IPv4 address
_IP_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# US SSN: NNN-NN-NNNN (not starting with 000, 666, 9xx)
_SSN_RE = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")

# API-key-shaped strings: prefixed secrets and bare hex-ish long values adjacent to a label
_API_KEY_RE = re.compile(
    r"\b(?:sk|ghp|ghs|gho|ghu|ghx|xoxb|xapp|xoxa|glpat)[-_][A-Za-z0-9_\-]{16,}"
    r"|\b(?:AKIA|ASIA|AROA|AIPA|ANPA|ANVA|APKA)[A-Z0-9]{16,}\b"
    r"|(?:api[_\s-]?key|api[_\s-]?secret|token|secret|password)\s*[=:\"'\s]+[A-Za-z0-9+/=_\-]{20,}"
    r"|\bBearer\s+[A-Za-z0-9\-_.~+/]+=*\b",
    re.I,
)

# Ordered patterns: (type_tag, compiled_pattern)
# Order matters: more specific patterns first (API_KEY before CARD, SSN before PHONE)
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("API_KEY", _API_KEY_RE),
    ("SSN", _SSN_RE),
    ("CARD", _CARD_RE),
    ("EMAIL", _EMAIL_RE),
    ("PHONE", _PHONE_RE),
    ("IP", _IP_RE),
]

# Placeholder delimiters: guillemets (extremely unlikely in normal text)
_PLACEHOLDER_L = "«"  # «
_PLACEHOLDER_R = "»"  # »


def _make_placeholder(tag: str, n: int) -> str:
    return f"{_PLACEHOLDER_L}{tag}_{n}{_PLACEHOLDER_R}"


# ---------------------------------------------------------------------------
# EgressRedactor
# ---------------------------------------------------------------------------


class EgressRedactor:
    """Session-stable PII redactor.

    Within a single ``EgressRedactor`` instance, the same input span always maps
    to the same placeholder (deterministic and stable for the session lifetime).

    Tier 2 (Presidio) is activated automatically if ``presidio-analyzer`` is
    installed; otherwise regex-only mode is used silently.
    """

    def __init__(self) -> None:
        # span_text → placeholder  (original text → assigned token)
        self._span_to_placeholder: dict[str, str] = {}
        # placeholder → original  (for rehydration)
        self._placeholder_to_span: dict[str, str] = {}
        # counters per tag
        self._counters: dict[str, int] = defaultdict(int)

        # Presidio: attempt lazy import once; cache availability
        self._presidio_available: bool | None = None
        self._presidio_analyzer: object | None = None  # AnalyzerEngine instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def redact(self, text: str) -> tuple[str, RedactionMap]:
        """Redact PII from *text*, returning the sanitised string and a map
        that can be passed to :meth:`rehydrate` to restore the originals.

        The returned ``RedactionMap`` is a *snapshot* of the placeholders used in
        this call.  It is safe to pass directly to :meth:`rehydrate`.
        """
        redacted = self._apply_regex(text)
        redacted = self._apply_presidio(redacted)
        # Build snapshot map of only the placeholders that appear in this result
        snapshot: RedactionMap = {
            ph: orig
            for ph, orig in self._placeholder_to_span.items()
            if ph in redacted
        }
        return redacted, snapshot

    def rehydrate(self, text: str, rmap: RedactionMap) -> str:
        """Restore all placeholders in *text* to their original spans using *rmap*."""
        result = text
        # Sort by placeholder length descending to avoid partial replacements
        for placeholder, original in sorted(rmap.items(), key=lambda x: -len(x[0])):
            result = result.replace(placeholder, original)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_placeholder(self, tag: str, span: str) -> str:
        """Return the existing placeholder for *span*, or create a new one."""
        if span in self._span_to_placeholder:
            return self._span_to_placeholder[span]
        self._counters[tag] += 1
        placeholder = _make_placeholder(tag, self._counters[tag])
        self._span_to_placeholder[span] = placeholder
        self._placeholder_to_span[placeholder] = span
        return placeholder

    def _apply_regex(self, text: str) -> str:
        """Apply all Tier-1 regex patterns, replacing each match with a placeholder."""
        # Collect all matches with their positions; process from right to left to
        # keep offsets valid as we replace.
        matches: list[tuple[int, int, str, str]] = []  # (start, end, tag, span)
        covered: set[range] = set()

        for tag, pat in _PATTERNS:
            for m in pat.finditer(text):
                start, end = m.start(), m.end()
                span = m.group(0)
                # Skip if already covered by a higher-priority match
                if any(start < r.stop and end > r.start for r in covered):
                    continue
                matches.append((start, end, tag, span))
                covered.add(range(start, end))

        # Sort by start position descending (right-to-left replacement)
        matches.sort(key=lambda x: -x[0])

        result = text
        for start, end, tag, span in matches:
            placeholder = self._get_or_create_placeholder(tag, span)
            result = result[:start] + placeholder + result[end:]

        return result

    def _apply_presidio(self, text: str) -> str:
        """Apply Tier-2 Presidio NER if available; otherwise return *text* unchanged."""
        analyzer = self._get_presidio_analyzer()
        if analyzer is None:
            return text

        try:
            # presidio-analyzer API: analyzer.analyze(text, language="en") → list[RecognizerResult]
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]  # optional dep

            assert isinstance(analyzer, AnalyzerEngine)
            results = analyzer.analyze(text=text, language="en")
            # Sort by start descending for right-to-left replacement
            results_sorted = sorted(results, key=lambda r: -r.start)
            result = text
            for r in results_sorted:
                span = text[r.start : r.end]
                tag = r.entity_type
                placeholder = self._get_or_create_placeholder(tag, span)
                result = result[: r.start] + placeholder + result[r.end :]
            return result
        except Exception:  # noqa: BLE001 — Presidio errors must never break the pipeline
            return text

    def _get_presidio_analyzer(self) -> object | None:
        """Lazy-initialise the Presidio AnalyzerEngine (once per instance)."""
        if self._presidio_available is not None:
            return self._presidio_analyzer
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]  # optional dep

            self._presidio_analyzer = AnalyzerEngine()
            self._presidio_available = True
        except ImportError:
            self._presidio_available = False
            self._presidio_analyzer = None
        return self._presidio_analyzer


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------


def rehydrate_stream(rmap: RedactionMap) -> Callable[[str | None], str]:
    """Return a stateful callable that rehydrates placeholders across streaming chunks.

    The callable should be called with each chunk as it arrives.  When called with
    ``None`` it flushes any buffered remainder and returns it (call once at the end).

    Strategy: accumulate chunks in a buffer.  After each new chunk, scan the buffer
    for *complete* placeholders (i.e. sequences containing both ``«`` and ``»``).
    Emit all text up to and including the last complete placeholder as rehydrated text;
    retain any suffix that might be an incomplete placeholder.

    Limitations: if a placeholder is split across more than two chunks the helper
    still handles it correctly as long as the full placeholder eventually arrives.
    """
    buffer: list[str] = []
    pl = _PLACEHOLDER_L  # «
    pr = _PLACEHOLDER_R  # »

    def _flush_complete(force: bool = False) -> str:
        accumulated = "".join(buffer)
        buffer.clear()

        if force:
            # Replace all complete placeholders; return everything
            result = accumulated
            for ph, orig in sorted(rmap.items(), key=lambda x: -len(x[0])):
                result = result.replace(ph, orig)
            return result

        # Find the rightmost complete placeholder boundary
        last_close = accumulated.rfind(pr)
        if last_close == -1:
            # No complete placeholder; check if we might be in the middle of one
            if pl in accumulated:
                # Hold back everything from the last «
                cut = accumulated.rfind(pl)
                emit = accumulated[:cut]
                buffer.append(accumulated[cut:])
                return emit
            return accumulated

        # Emit everything up to and including the last »; buffer the rest
        emit_part = accumulated[: last_close + 1]
        remainder = accumulated[last_close + 1 :]
        if remainder:
            buffer.append(remainder)

        # Rehydrate complete placeholders in the emit part
        result = emit_part
        for ph, orig in sorted(rmap.items(), key=lambda x: -len(x[0])):
            result = result.replace(ph, orig)
        return result

    def step(chunk: str | None) -> str:
        if chunk is None:
            return _flush_complete(force=True)
        buffer.append(chunk)
        return _flush_complete(force=False)

    return step
