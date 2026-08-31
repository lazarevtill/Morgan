"""Entity extraction — the single definition of "entity" in the codebase.

Both paths need it and they must agree. The hot path calls it through
``TextPerception.analyze`` to populate ``FusedPerception.entities`` (which drives trait
selection). The cold path calls it directly to populate ``Memory.entities``, which is
what ``EntityIndex`` indexes and therefore what the semantic upper index routes over. Two
extractors would build two indexes that disagree about what is in them, which is the
duplication the "one of each" invariant exists to prevent.

**Script-aware, not ASCII-aware.** The predecessor matched ``[A-Z][a-z]{2,}`` and could
not see a single Cyrillic name. That is the same defect the reshape fixed one layer down
when it replaced the ``[a-z0-9]+`` BM25 tokeniser with FTS5 ``unicode61``: a large part of
the owner's corpus is Russian, and an index that cannot see it returns nothing for it.
Case detection here goes through Python's own Unicode casing (``str.istitle()`` /
``str.isupper()``), so Cyrillic, Greek and Latin all work without a per-script rule.

**Known limit, stated rather than hidden.** Scripts without letter case -- Chinese,
Japanese, Arabic, Hebrew -- carry no capitalisation signal, so a deterministic extractor
finds nothing in them. This function returns an empty list there rather than guessing.
Closing that gap needs the model-backed extractor the semantic-index job layers on top;
this is the floor it falls back to, and the floor is honest about where it ends.
"""

from __future__ import annotations

import re

#: A word is a run of letters, digits, and the joiners that appear inside real names
#: (``kube-proxy``, ``O'Neill``, ``asyncio_bus``). Punctuation ends a word.
_WORD = re.compile(r"[^\W_]+(?:[-'’_][^\W_]+)*", re.UNICODE)

#: Sentences start with a capital in cased scripts, so the openers that begin an
#: instruction or a question would otherwise be extracted from every other turn.
_STOPWORDS = frozenset(
    {
        # English sentence/question openers
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "what",
        "when",
        "where",
        "why",
        "how",
        "who",
        "whom",
        "which",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "my",
        "your",
        "our",
        "remind",
        "create",
        "add",
        "delete",
        "set",
        "schedule",
        "run",
        "please",
        "thanks",
        "thank",
        "hello",
        "hi",
        "hey",
        "yes",
        "no",
        "ok",
        "okay",
        "and",
        "or",
        "but",
        "if",
        "then",
        "so",
        "also",
        "not",
        # Russian sentence/question openers -- the same problem in the other script
        "что",
        "кто",
        "где",
        "когда",
        "почему",
        "как",
        "какой",
        "какая",
        "какие",
        "это",
        "этот",
        "эта",
        "эти",
        "тот",
        "та",
        "те",
        "он",
        "она",
        "они",
        "оно",
        "я",
        "ты",
        "мы",
        "вы",
        "мой",
        "моя",
        "наш",
        "ваш",
        "и",
        "или",
        "но",
        "если",
        "то",
        "так",
        "тоже",
        "не",
        "да",
        "нет",
        "спасибо",
        "привет",
        "пожалуйста",
        "напомни",
        "создай",
        "удали",
        "поставь",
    }
)

#: Calendar words are capitalised in English and are never the subject of a memory.
_CALENDAR = frozenset(
    {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "today",
        "tomorrow",
        "yesterday",
        "tonight",
        "понедельник",
        "вторник",
        "среда",
        "четверг",
        "пятница",
        "суббота",
        "воскресенье",
        "январь",
        "февраль",
        "март",
        "апрель",
        "июнь",
        "июль",
        "август",
        "сентябрь",
        "октябрь",
        "ноябрь",
        "декабрь",
        "сегодня",
        "завтра",
        "вчера",
    }
)

#: Below this length a capitalised token is an initial ("A."), not a name. Acronyms are
#: admitted separately by the all-caps branch, which is why GDPR survives and "A" does not.
_MIN_NAME_LENGTH = 2
_MIN_ACRONYM_LENGTH = 2


def _is_candidate(word: str) -> bool:
    """True when *word* looks like a name or an acronym in a cased script.

    ``istitle()`` and ``isupper()`` are Unicode-aware, so this is one rule for every
    cased script rather than one regex per alphabet. A word that is neither -- lower
    case, or from a script without case -- is not a candidate.
    """
    if word.isupper():
        # ALL CAPS: an acronym (GDPR, SQL) or shouting. Digits are allowed inside
        # (S3, K8S) but a bare number is not a name.
        return len(word) >= _MIN_ACRONYM_LENGTH and any(c.isalpha() for c in word)
    if word.istitle():
        return len(word) >= _MIN_NAME_LENGTH
    return False


def extract_entity_names(text: str) -> list[str]:
    """Return the entity names in *text*, in order of first appearance, deduplicated.

    Order is part of the contract: the caller writes these into ``memory_entities`` and
    the index reads them back, so a set would make the stored order depend on hash seed
    and two processes would disagree about the same memory.
    """
    seen: set[str] = set()
    names: list[str] = []
    for match in _WORD.finditer(text):
        word = match.group(0)
        folded = word.casefold()
        if folded in _STOPWORDS or folded in _CALENDAR:
            continue
        if not _is_candidate(word):
            continue
        if folded in seen:
            continue
        seen.add(folded)
        names.append(word)
    return names
