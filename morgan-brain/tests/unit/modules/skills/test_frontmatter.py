"""Unit tests for the dependency-light frontmatter parser.

All tests are deterministic and pure-Python — no I/O, no network.
"""
from __future__ import annotations

from morgan_brain.modules.skills.frontmatter import parse_frontmatter


# ---------------------------------------------------------------------------
# Happy-path: scalar + integer version
# ---------------------------------------------------------------------------


def test_parses_name_and_version_int() -> None:
    text = "---\nname: coding\nversion: 3\n---\nBody text here.\n"
    meta, body = parse_frontmatter(text)
    assert meta["name"] == "coding"
    assert meta["version"] == 3
    assert isinstance(meta["version"], int)
    assert body.strip() == "Body text here."


# ---------------------------------------------------------------------------
# Inline list: key: [a, b, c]
# ---------------------------------------------------------------------------


def test_parses_inline_list() -> None:
    text = "---\ntriggers: [chat, greeting, hello]\n---\n# heading\n"
    meta, body = parse_frontmatter(text)
    assert meta["triggers"] == ["chat", "greeting", "hello"]
    assert "# heading" in body


def test_inline_list_strips_whitespace() -> None:
    text = "---\ntools: [ calculator , memory_search ]\n---\n"
    meta, _ = parse_frontmatter(text)
    assert meta["tools"] == ["calculator", "memory_search"]


# ---------------------------------------------------------------------------
# Dash / block-sequence list
# ---------------------------------------------------------------------------


def test_parses_dash_list() -> None:
    text = "---\ntriggers:\n- sad\n- stressed\n- support\n---\nEmpathy body.\n"
    meta, body = parse_frontmatter(text)
    assert meta["triggers"] == ["sad", "stressed", "support"]
    assert "Empathy body." in body


def test_dash_list_mixed_with_scalars() -> None:
    text = "---\nname: empathy\ntriggers:\n- sad\n- feeling\nversion: 2\n---\nbody\n"
    meta, body = parse_frontmatter(text)
    assert meta["name"] == "empathy"
    assert meta["triggers"] == ["sad", "feeling"]
    assert meta["version"] == 2


# ---------------------------------------------------------------------------
# No-frontmatter passthrough
# ---------------------------------------------------------------------------


def test_no_frontmatter_returns_empty_meta_and_full_text() -> None:
    text = "# Just a heading\n\nSome content.\n"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_partial_frontmatter_no_close_passthrough() -> None:
    """A file starting with --- but no closing --- is not treated as frontmatter."""
    text = "---\nname: broken\n"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


# ---------------------------------------------------------------------------
# Full realistic skill file
# ---------------------------------------------------------------------------


def test_full_skill_file() -> None:
    skill_md = (
        "---\n"
        "name: research\n"
        "version: 1\n"
        "triggers: [research, find, look up, sources]\n"
        "tools: [web_search, fetch_url]\n"
        "---\n"
        "## Research Skill\n\n"
        "Break down the question.\nFetch primary sources.\nCite everything.\n"
    )
    meta, body = parse_frontmatter(skill_md)
    assert meta["name"] == "research"
    assert meta["version"] == 1
    assert "find" in meta["triggers"]  # type: ignore[operator]
    assert "web_search" in meta["tools"]  # type: ignore[operator]
    assert "Break down" in body


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_string() -> None:
    meta, body = parse_frontmatter("")
    assert meta == {}
    assert body == ""


def test_string_version_stays_string() -> None:
    text = "---\nversion: 1a\n---\nbody\n"
    meta, _ = parse_frontmatter(text)
    assert meta["version"] == "1a"
    assert isinstance(meta["version"], str)


def test_numeric_list_items_coerced() -> None:
    text = "---\nversions: [1, 2, 3]\n---\n"
    meta, _ = parse_frontmatter(text)
    assert meta["versions"] == [1, 2, 3]
