"""Conservative credential-shaped text refusal, not a complete secret detector.

Adapted from phase1a's provider rules, without redaction or archive dependencies.
Only explicit GitHub, OpenAI project/service-account, Anthropic API and armored
private-key shapes are covered. No entropy, generic password or personal-data rules.
Scan the whole supplied text: splitting or windowing first can hide a credential.
"""

from __future__ import annotations

import re

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "github_token",
        re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,}"),
    ),
    (
        "openai_key",
        re.compile(r"sk-(?:proj|svcacct)-[A-Za-z0-9_-]{20,}"),
    ),
    ("anthropic_key", re.compile(r"sk-ant-api03-[A-Za-z0-9_-]{20,}")),
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:(?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY"
            r"|PGP PRIVATE KEY BLOCK)-----\r?\n"
            r"(?:[ \t]*\r?\n|(?:Version|Comment|Proc-Type|DEK-Info):[^\r\n]*\r?\n)*"
            r"(?:[A-Za-z0-9+/=][ \t\r\n]*){32}"
        ),
    ),
)


class CredentialRefused(ValueError):
    """Only a fixed rule name is retained, never the matching text or its position."""

    def __init__(self, rule: str) -> None:
        self.rule = rule
        super().__init__(f"Credential-shaped text refused ({rule}); remove it before retrying")


def refuse_credentials(text: str) -> None:
    """Refuse supported shapes without changing the caller's evidence or checking validity.

    Matching examples and revoked tokens also refuse. Encoded, split or unknown formats
    may pass; this is admission screening, not permission to store other secrets.
    """
    for rule, pattern in _RULES:
        if pattern.search(text) is not None:
            raise CredentialRefused(rule)
