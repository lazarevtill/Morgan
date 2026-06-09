"""MCP tool-description security hardening.

Heuristics for detecting prompt-injection in MCP-provided tool descriptions,
stable fingerprinting for rug-pull detection, and a server allowlist.

Why this matters
----------------
5.5% of public MCP servers have been found to carry tool-poisoning payloads —
instructions embedded in tool *descriptions* that attempt to redirect the model
(e.g., "ignore previous instructions", hidden Unicode overrides, fake "system:"
preambles). Since tool descriptions are rendered verbatim into the model's
context, a malicious server can mount a prompt-injection attack just by serving
a rogue description.

Mitigations implemented here:
  1. ``sanitize_tool_description`` — strips/rewrites known injection patterns.
  2. ``tool_fingerprint`` — SHA-256 of canonical (name + description + schema)
     so the hub can detect a server quietly mutating a description after approval.
  3. ``verify_fingerprint`` — compares a freshly computed hash to a pinned one.
  4. ``ServerAllowlist`` — gate that rejects unlisted MCP servers entirely.
"""

from __future__ import annotations

import hashlib
import json
import re

# ---------------------------------------------------------------------------
# Injection-pattern catalogue (heuristics; keep documented)
# ---------------------------------------------------------------------------

# Each tuple: (compiled pattern, replacement).  Applied in order.
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Direct meta-instruction verbs targeting the model's instruction context.
    (
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.I),
        "[REDACTED]",
    ),
    (
        re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.I),
        "[REDACTED]",
    ),
    (
        re.compile(r"forget\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.I),
        "[REDACTED]",
    ),
    # Fake system / assistant turn injections.
    (re.compile(r"\bsystem\s*:", re.I), "[REDACTED]:"),
    (re.compile(r"\bassistant\s*:", re.I), "[REDACTED]:"),
    # Hidden-instruction Unicode tricks: zero-width characters, soft-hyphens, RTL overrides.
    (re.compile(r"[​‌‍⁠­‪-‮⁦-⁩]+"), ""),
    # "Now do X" / "Your new task is" imperative override patterns.
    (re.compile(r"now\s+(?:you\s+must|do|perform|execute)\s+", re.I), "[REDACTED] "),
    (
        re.compile(r"your\s+(?:new\s+)?(?:task|goal|objective|role|purpose)\s+is\b", re.I),
        "[REDACTED] is",
    ),
    # Jailbreak phrase "DAN" / "developer mode".
    (re.compile(r"\bDAN\b"), "[REDACTED]"),
    (re.compile(r"developer\s+mode", re.I), "[REDACTED]"),
    # Exfiltration marker: directing output to an external URL.
    (re.compile(r"send\s+(?:this\s+)?(?:to|at)\s+https?://", re.I), "[REDACTED] "),
]

# Hard cap on description length (characters).  Descriptions longer than this
# are almost certainly attempts to smuggle instructions; truncate.
_MAX_DESCRIPTION_LENGTH = 2_000


def sanitize_tool_description(desc: str) -> str:
    """Strip or rewrite known prompt-injection patterns in *desc*.

    The returned string is safe to include in model context.  Sanitization is
    applied in two phases:

    1. Pattern substitution — each regex in ``_INJECTION_PATTERNS`` is applied.
    2. Length truncation — the result is capped at ``_MAX_DESCRIPTION_LENGTH``
       characters (excess silently dropped; callers can inspect the difference).

    The function is *conservative*: it only removes or marks content that matches
    high-confidence patterns.  Borderline cases are kept so legitimate tools are
    not broken.  The caller (``McpHub``) should surface the cleaned description
    to the owner for review.
    """
    result = desc
    for pattern, replacement in _INJECTION_PATTERNS:
        result = pattern.sub(replacement, result)
    if len(result) > _MAX_DESCRIPTION_LENGTH:
        result = result[:_MAX_DESCRIPTION_LENGTH]
    return result


# ---------------------------------------------------------------------------
# Fingerprinting (rug-pull detection)
# ---------------------------------------------------------------------------


def _canonical_bytes(name: str, description: str, schema: dict[str, object]) -> bytes:
    """Return a deterministic canonical serialisation of the three fields."""
    # json.dumps with sort_keys gives a stable string regardless of dict insertion order.
    payload = json.dumps(
        {"name": name, "description": description, "schema": schema},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return payload.encode("utf-8")


def tool_fingerprint(name: str, description: str, schema: dict[str, object]) -> str:
    """Return the stable SHA-256 hex fingerprint of a tool definition.

    The hash covers ``name``, ``description``, and ``schema`` (all three)
    serialised in canonical form so any single-character mutation changes it.
    Use ``verify_fingerprint`` to check that a previously pinned value still
    matches.
    """
    return hashlib.sha256(_canonical_bytes(name, description, schema)).hexdigest()


def verify_fingerprint(name: str, description: str, schema: dict[str, object], pinned: str) -> bool:
    """Return True if the recomputed fingerprint equals *pinned*.

    A False return means the server silently changed the tool definition after
    the owner pinned it — a classic rug-pull.  The hub should SKIP (not execute)
    the tool and alert the owner.
    """
    return tool_fingerprint(name, description, schema) == pinned


# ---------------------------------------------------------------------------
# Server allowlist
# ---------------------------------------------------------------------------


class ServerAllowlist:
    """Gate that rejects MCP server connections to servers not on the allowlist.

    Parameters
    ----------
    allowed:
        Set of server names (strings) that are permitted.  An empty set means
        *all* servers are blocked.  ``None`` is not a valid value — pass
        ``set()`` to block everything or use an explicit opt-in set.
    """

    def __init__(self, allowed: set[str]) -> None:
        self._allowed = frozenset(allowed)

    def is_allowed(self, server_name: str) -> bool:
        """Return True if *server_name* is on the allowlist."""
        return server_name in self._allowed
