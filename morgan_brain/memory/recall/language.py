"""The query's language, decided by script alone -- no model call.

The owner writes most queries in Russian, and phase 1a's keyword-only degrade runs over FTS5's
``unicode61`` tokenizer with no stemming, which is weakest exactly there. ``recall.done``
records which language each query was in, so a degraded recall can be told apart by it.
"""

from __future__ import annotations

import unicodedata
from typing import Literal

Language = Literal["ru", "en", "mixed", "other"]


def of(text: str) -> Language:
    """More Cyrillic letters than Latin gives ``"ru"``, the reverse ``"en"``, both present
    within a factor of two ``"mixed"``, neither ``"other"``.

    Only ``str.isalpha`` letters are counted, so digits and punctuation never decide; a letter
    counts as Cyrillic or Latin by its Unicode character name, which also covers accented Latin
    letters. A script with neither -- Chinese, Japanese, Arabic, Hebrew -- always answers
    ``"other"``, whatever else the text contains.
    """
    cyrillic = 0
    latin = 0
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "CYRILLIC" in name:
            cyrillic += 1
        elif "LATIN" in name:
            latin += 1
    if cyrillic == 0 and latin == 0:
        return "other"
    if cyrillic > 0 and latin > 0 and max(cyrillic, latin) <= 2 * min(cyrillic, latin):
        return "mixed"
    return "ru" if cyrillic > latin else "en"
