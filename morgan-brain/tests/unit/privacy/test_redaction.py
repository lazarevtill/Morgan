"""Unit tests for morgan_brain.privacy.redaction.

Coverage:
- Email and phone addresses are replaced with stable «TYPE_N» placeholders.
- The same input span always maps to the same placeholder within a session.
- rehydrate() restores the originals exactly (round-trip guarantee).
- IP addresses, SSNs, credit-card-shaped numbers, API keys are redacted.
- rehydrate_stream: correct reassembly when a placeholder is split across chunks.
- KNOWN GAP: regex-only mode does NOT catch non-English names (Presidio needed).
- The gap test is marked xfail so it is visible and documented.
"""

from __future__ import annotations

import pytest

from morgan_brain.privacy.redaction import EgressRedactor, RedactionMap, rehydrate_stream

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_redactor() -> EgressRedactor:
    return EgressRedactor()


# ---------------------------------------------------------------------------
# Basic email + phone redaction
# ---------------------------------------------------------------------------


class TestBasicRedaction:
    def test_email_is_redacted(self) -> None:
        r = make_redactor()
        redacted, rmap = r.redact("contact me at user@example.com please")
        assert "user@example.com" not in redacted
        assert "«" in redacted and "»" in redacted
        assert len(rmap) == 1

    def test_phone_is_redacted(self) -> None:
        r = make_redactor()
        redacted, rmap = r.redact("call 415-555-0100 for info")
        assert "415-555-0100" not in redacted
        assert len(rmap) >= 1

    def test_email_and_phone_both_redacted(self) -> None:
        r = make_redactor()
        text = "email user@example.com or call 555-123-4567"
        redacted, rmap = r.redact(text)
        assert "user@example.com" not in redacted
        assert "555-123-4567" not in redacted
        assert len(rmap) >= 2

    def test_ip_address_is_redacted(self) -> None:
        r = make_redactor()
        redacted, rmap = r.redact("server at 192.168.1.100 is down")
        assert "192.168.1.100" not in redacted
        assert len(rmap) >= 1

    def test_ssn_is_redacted(self) -> None:
        r = make_redactor()
        redacted, rmap = r.redact("SSN is 123-45-6789 for the form")
        assert "123-45-6789" not in redacted
        assert len(rmap) >= 1

    def test_api_key_is_redacted(self) -> None:
        r = make_redactor()
        redacted, rmap = r.redact("token: ghp_abcdefghijklmnopqrstuvwxyz1234")
        assert "ghp_abcdefghijklmnopqrstuvwxyz1234" not in redacted
        assert len(rmap) >= 1

    def test_plain_text_unchanged(self) -> None:
        r = make_redactor()
        text = "the quick brown fox jumps over the lazy dog"
        redacted, rmap = r.redact(text)
        assert redacted == text
        assert rmap == {}


# ---------------------------------------------------------------------------
# Session-stable placeholders (same span → same placeholder)
# ---------------------------------------------------------------------------


class TestSessionStability:
    def test_same_email_same_placeholder(self) -> None:
        r = make_redactor()
        redacted1, rmap1 = r.redact("send to alice@example.org")
        redacted2, rmap2 = r.redact("also ping alice@example.org")
        # Extract placeholder from each
        ph1 = next(ph for ph in rmap1 if rmap1[ph] == "alice@example.org")
        ph2 = next(ph for ph in rmap2 if rmap2[ph] == "alice@example.org")
        assert ph1 == ph2, "Same email must map to the same placeholder in a session"

    def test_different_emails_different_placeholders(self) -> None:
        r = make_redactor()
        _, rmap1 = r.redact("alice@example.org")
        _, rmap2 = r.redact("bob@example.org")
        phs1 = set(rmap1.keys())
        phs2 = set(rmap2.keys())
        assert phs1.isdisjoint(phs2), "Different values must use different placeholders"

    def test_same_phone_same_placeholder(self) -> None:
        r = make_redactor()
        _, rmap1 = r.redact("call 415-555-0199")
        _, rmap2 = r.redact("missed call from 415-555-0199")
        ph1 = next((ph for ph in rmap1 if rmap1[ph] == "415-555-0199"), None)
        ph2 = next((ph for ph in rmap2 if rmap2[ph] == "415-555-0199"), None)
        assert ph1 is not None and ph2 is not None
        assert ph1 == ph2


# ---------------------------------------------------------------------------
# Round-trip: rehydrate(redact(text), map) == text
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_email_round_trip(self) -> None:
        r = make_redactor()
        text = "contact me at user@example.com please"
        redacted, rmap = r.redact(text)
        assert r.rehydrate(redacted, rmap) == text

    def test_phone_round_trip(self) -> None:
        r = make_redactor()
        text = "call 415-555-0100 for info"
        redacted, rmap = r.redact(text)
        assert r.rehydrate(redacted, rmap) == text

    def test_multiple_pii_round_trip(self) -> None:
        r = make_redactor()
        text = "reach alice@test.com or bob@test.com at 555-123-4567"
        redacted, rmap = r.redact(text)
        assert r.rehydrate(redacted, rmap) == text

    def test_no_pii_round_trip(self) -> None:
        r = make_redactor()
        text = "nothing sensitive here"
        redacted, rmap = r.redact(text)
        assert r.rehydrate(redacted, rmap) == text

    def test_api_key_round_trip(self) -> None:
        r = make_redactor()
        text = "use token: ghp_abcdefghijklmnopqrstuvwxyz1234 for auth"
        redacted, rmap = r.redact(text)
        assert r.rehydrate(redacted, rmap) == text

    def test_ssn_round_trip(self) -> None:
        r = make_redactor()
        text = "SSN: 234-56-7890 on the form"
        redacted, rmap = r.redact(text)
        assert r.rehydrate(redacted, rmap) == text

    def test_ip_round_trip(self) -> None:
        r = make_redactor()
        text = "connect to 10.0.0.1 on port 8080"
        redacted, rmap = r.redact(text)
        assert r.rehydrate(redacted, rmap) == text


# ---------------------------------------------------------------------------
# Streaming rehydration helper
# ---------------------------------------------------------------------------


class TestRehydrateStream:
    def test_whole_placeholder_in_one_chunk(self) -> None:
        r = make_redactor()
        text = "email me at user@example.com thanks"
        redacted, rmap = r.redact(text)
        # Single chunk: should rehydrate fully
        step = rehydrate_stream(rmap)
        out = step(redacted) + step(None)
        assert "user@example.com" in out

    def test_placeholder_split_across_two_chunks(self) -> None:
        """Placeholder split across a chunk boundary must be reassembled correctly."""
        r = make_redactor()
        text = "email me at user@example.com thanks"
        redacted, rmap = r.redact(text)
        assert rmap, "Expected at least one placeholder"

        # Find the placeholder and split in its middle
        placeholder = next(iter(rmap))
        idx = redacted.index(placeholder)
        mid = idx + len(placeholder) // 2

        chunk1 = redacted[:mid]
        chunk2 = redacted[mid:]

        step = rehydrate_stream(rmap)
        part1 = step(chunk1)  # may be empty or partial (before «)
        part2 = step(chunk2)  # receives rest of placeholder
        flush = step(None)  # flush remainder

        full = part1 + part2 + flush
        assert "user@example.com" in full
        assert placeholder not in full

    def test_no_placeholder_passes_through(self) -> None:
        step = rehydrate_stream({})
        out = step("hello world") + step(None)
        assert out == "hello world"

    def test_flush_with_none_returns_buffer(self) -> None:
        rmap: RedactionMap = {"«EMAIL_1»": "x@y.com"}
        step = rehydrate_stream(rmap)
        # Feed only part of the placeholder (no «, just plain text)
        out = step("hello ")
        rest = step(None)
        assert (out + rest) == "hello "

    def test_placeholder_split_three_ways(self) -> None:
        """Placeholder split into three chunks is still reassembled."""
        r = make_redactor()
        text = "server at 10.0.0.1 is ready"
        redacted, rmap = r.redact(text)
        assert rmap, "Expected an IP placeholder"

        placeholder = next(iter(rmap))
        # Split placeholder into three roughly equal parts
        n = len(placeholder)
        chunk_a = redacted[: redacted.index(placeholder) + n // 3]
        chunk_b = redacted[
            redacted.index(placeholder) + n // 3 : redacted.index(placeholder) + 2 * n // 3
        ]
        chunk_c = redacted[redacted.index(placeholder) + 2 * n // 3 :]

        step = rehydrate_stream(rmap)
        out = step(chunk_a) + step(chunk_b) + step(chunk_c) + step(None)
        assert "10.0.0.1" in out


# ---------------------------------------------------------------------------
# KNOWN GAP: regex-only mode does NOT catch non-English names
#
# Purpose: make the limitation explicit and visible, not silent.
# Presidio NER is required for accurate name detection.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "KNOWN GAP: regex-only mode cannot detect Latin-script person names "
        "without surrounding context cues. Presidio NER (optional dep) is "
        "required to close this gap.  This test documents the limitation and "
        "must NOT be 'fixed' by hacking name detection into the regex tier — "
        "the false-positive rate would be unacceptable."
    ),
    strict=True,
)
def test_non_english_name_not_caught_by_regex() -> None:
    """Assert that a common non-English person name IS redacted by regex alone.

    This test is expected to FAIL in regex-only mode, documenting the gap.
    With Presidio installed the test would pass (the xfail would then become an
    unexpected success — xpass — which should be changed to a regular pass once
    Presidio is added as a hard dependency or the test guarded by importorskip).
    """
    r = make_redactor()
    # "Björn Andersen" — a Scandinavian name, no email/phone/key signal around it.
    text = "meeting with Björn Andersen tomorrow"
    redacted, rmap = r.redact(text)
    # We ASSERT it IS redacted — which will FAIL in regex-only mode (xfail).
    assert "Björn Andersen" not in redacted, (
        "regex-only mode did NOT redact the name — this is the known gap"
    )
