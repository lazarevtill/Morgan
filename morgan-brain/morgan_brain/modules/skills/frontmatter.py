"""Dependency-light YAML-subset frontmatter parser for markdown skill files.

Supports the small subset needed for skill frontmatter:
- ``key: value`` — string scalar
- ``key: [a, b, c]`` — inline list (comma-separated, bracket-wrapped)
- ``key:`` followed by ``- item`` lines — dash-list (block sequence)
- Numeric values are coerced to ``int``

No PyYAML dependency; no standard library ``tomllib`` import needed.

Returns ``({}, full_text)`` when no frontmatter block is found, so any plain
markdown file is accepted without error.
"""

from __future__ import annotations

import re

# Regex for an inline list: [a, b, c] — trims surrounding whitespace on each item.
_INLINE_LIST_RE = re.compile(r"^\s*\[(.+)\]\s*$")


def _coerce(value: str) -> str | int:
    """Return *value* as int if it looks like a bare integer, else str."""
    stripped = value.strip()
    if re.fullmatch(r"-?\d+", stripped):
        return int(stripped)
    return stripped


def _parse_inline_list(raw: str) -> list[str | int]:
    """Parse ``[a, b, c]`` into a list; each element coerced."""
    return [_coerce(item) for item in raw.split(",")]


def parse_frontmatter(text: str) -> tuple[dict[str, str | int | list[str | int]], str]:
    """Split a leading ``---\\n...\\n---\\n`` block from *text*.

    Parameters
    ----------
    text:
        Raw file content (potentially starting with a frontmatter block).

    Returns
    -------
    tuple[dict, str]
        ``(meta, body)`` where *meta* contains the parsed key/value pairs and
        *body* is the remaining markdown text after the closing ``---`` line.
        When no frontmatter block is present, ``meta`` is an empty dict and
        ``body`` is the full *text*.
    """
    # Normalise line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if not text.startswith("---\n"):
        return {}, text

    # Find closing "---" line.
    rest = text[4:]  # skip the opening "---\n"
    close_idx = rest.find("\n---\n")
    if close_idx == -1:
        # Try trailing "---" at the very end of file (no newline after).
        if rest.rstrip("\n").endswith("\n---"):
            alt = rest.rstrip("\n")
            close_idx = alt.rfind("\n---")
            fm_block = rest[:close_idx]
            body = ""
        else:
            return {}, text
    else:
        fm_block = rest[:close_idx]
        body = rest[close_idx + 5 :]  # skip "\n---\n"

    meta: dict[str, str | int | list[str | int]] = {}
    lines = fm_block.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.strip().startswith("#"):
            i += 1
            continue

        # Key-only line (block sequence follows)?
        key_only = re.match(r"^(\w[\w\-]*)\s*:\s*$", line)
        if key_only:
            key = key_only.group(1)
            # Consume subsequent "- item" lines as a list.
            items: list[str | int] = []
            i += 1
            while i < len(lines) and re.match(r"^\s*-\s+", lines[i]):
                item_text = re.sub(r"^\s*-\s+", "", lines[i])
                items.append(_coerce(item_text))
                i += 1
            meta[key] = items
            continue

        # Key: value (scalar or inline list)
        kv = re.match(r"^(\w[\w\-]*)\s*:\s*(.*)", line)
        if kv:
            key = kv.group(1)
            raw_val = kv.group(2).strip()
            m_list = _INLINE_LIST_RE.match(raw_val)
            if m_list:
                meta[key] = _parse_inline_list(m_list.group(1))
            else:
                meta[key] = _coerce(raw_val)
        i += 1

    return meta, body
