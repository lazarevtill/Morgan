"""The query's language is decided by script, not by a model.

The owner writes most queries in Russian, and a keyword-only recall while embeddings are down
-- not built -- would run over unicode61 with no stemming, so it is weakest exactly there. The
log records which language each call was in.
"""

from __future__ import annotations

import pytest

from morgan_brain.memory.recall import language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Проверь конфиг Harbor", "ru"),
        ("check the Harbor config", "en"),
        ("Harbor конфиг upgrade план сегодня", "mixed"),
        ("ハーバー設定", "other"),
        ("", "other"),
        ("12:45 2026-09-21", "other"),
    ],
)
def test_the_script_decides(text, expected):
    assert language.of(text) == expected


@pytest.mark.parametrize(
    ("cyrillic_letters", "latin_letters", "expected"),
    [
        # Exactly a factor of two either way is still "within" it: inclusive on the boundary.
        ("аб", "A", "mixed"),
        ("а", "AB", "mixed"),
        # One letter past the boundary tips it to whichever script has more.
        ("абв", "A", "ru"),
        ("а", "ABC", "en"),
    ],
)
def test_the_factor_of_two_boundary_is_inclusive(cyrillic_letters, latin_letters, expected):
    assert language.of(f"{cyrillic_letters} {latin_letters}") == expected
