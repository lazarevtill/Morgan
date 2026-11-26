"""Bound and redact quoted provider errors before they cross either surface."""

from __future__ import annotations

import json
import re
from urllib.parse import quote, quote_plus

#: How much of a server's words an error quotes -- its answer to a refused request, or the
#: text of the failure that ended a call -- once every key in them is redacted.
_QUOTED_CHARS = 200

#: A bearer token anywhere in a server's words: everything after "Bearer " up to whitespace or
#: a quote, backslashes and percent signs included, however the token was encoded.
_BEARER = re.compile(r"(?i)\bbearer\s+[^\s\"']+")

#: What stands in quoted text for a key.
_REDACTED = "[redacted]"


class ResponseRedactor:
    """Makes a server's words fit to print: every key in them redacted, then cut.

    A gateway that echoes a credential back, or a malformed answer httpx quotes in its error,
    would otherwise put the key into a message, the MCP result that carries it, a log and a
    transcript. The key this adapter sends is matched however the server wrote it (``_forms``),
    and any bearer token at all is matched by ``_BEARER``. Whitespace is collapsed and the text
    cut to ``_QUOTED_CHARS`` only after both, so no cut leaves a piece of a key behind.
    """

    def __init__(self, key: str) -> None:
        self._key = re.compile(_forms(key), re.IGNORECASE) if key else None

    def __call__(self, text: str) -> str:
        if self._key is not None:
            text = self._key.sub(_REDACTED, text)
        text = _BEARER.sub(f"Bearer {_REDACTED}", text)
        return " ".join(text.split())[:_QUOTED_CHARS]


def _forms(key: str) -> str:
    """A pattern for *key* in every encoding a server might write it back in, built once per
    adapter: each character raw, JSON-escaped (with ``ensure_ascii`` on or off, and ``/`` as
    ``\\/``), ``\\u``-escaped, or percent-encoded (a space also as ``+``), in any mix, and with
    whitespace allowed between characters, where a server broke the key across a line. Hex digits
    match in either case. An accepted residual: a key broken inside one character's escape, or
    by an escaped line break (``\\n``), is not matched, and the part after the break prints.
    """
    return r"\s*".join(_forms_of(char) for char in key)


def _forms_of(char: str) -> str:
    """One character of a key, as any of its encodings: an alternation, longest first."""
    utf16 = char.encode("utf-16-be")
    forms = {
        char,
        json.dumps(char)[1:-1],
        json.dumps(char, ensure_ascii=False)[1:-1],
        "".join(f"\\u{int.from_bytes(utf16[i : i + 2]):04x}" for i in range(0, len(utf16), 2)),
        quote(char, safe=""),
        quote_plus(char),
        "".join(f"%{byte:02x}" for byte in char.encode()),
    }
    if char == "/":
        forms.add("\\/")
    return "(?:" + "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + ")"
