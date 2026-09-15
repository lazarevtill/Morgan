"""Entity extraction — the single definition of "entity" in the codebase.

``MemoryModule.store`` calls it to fill ``Memory.entities``, which is what ``EntityIndex``
indexes, and ``migrations`` calls it to re-extract a database stored under an older rule. Two
extractors would build two indexes that disagree about what is in them, which is the
duplication the "one of each" invariant exists to prevent.

**Script-aware, not ASCII-aware.** The predecessor matched ``[A-Z][a-z]{2,}`` and could
not see a single non-Latin name. That is the same defect the reshape fixed one layer down
when it replaced the ``[a-z0-9]+`` BM25 tokeniser with FTS5 ``unicode61``: an index that
cannot see part of the corpus returns nothing for it, silently. Case detection here goes
through Python's own Unicode casing (``str.istitle()`` / ``str.isupper()``), so Cyrillic,
Greek and Latin all work without a per-script rule.

**Known limit, stated rather than hidden.** Scripts without letter case -- Chinese,
Japanese, Arabic, Hebrew -- carry no capitalisation signal, so a deterministic extractor
finds nothing in them. This function returns an empty list there rather than guessing.
Closing that gap needs a model-backed extractor; this is the floor it would fall back to,
and the floor is honest about where it ends.
"""

from __future__ import annotations

import re

#: A word is a run of letters, digits, and the joiners that appear inside real names
#: (``kube-proxy``, ``O'Neill``, ``asyncio_bus``). Punctuation ends a word.
_WORD = re.compile(r"[^\W_]+(?:[-'’_][^\W_]+)*", re.UNICODE)

#: Function words that arrive capitalised where no boundary explains it: after an opening
#: quote or bracket, or in a title-cased heading.
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

#: What ends a sentence, a clause introduced by a colon, or a line. The word after one is
#: capitalised by where it stands, not by what it names.
_BOUNDARY = re.compile(r"[.!?:;\n]")


def _is_candidate(word: str) -> bool:
    """True when *word* looks like a name, an acronym, or a CamelCase brand.

    ``istitle()`` and ``isupper()`` are Unicode-aware, so this is one rule for every
    cased script rather than one regex per alphabet. A word that matches none of the
    three -- lower case throughout, or from a script without case -- is not a candidate.
    """
    if len(word) < _MIN_NAME_LENGTH or not any(c.isalpha() for c in word):
        return False
    if word.isupper():
        # ALL CAPS: an acronym (GDPR, SQL) or shouting. Digits are allowed inside
        # (S3, K8S) but a bare number is not a name.
        return True
    if word.istitle():
        return True
    # CamelCase. `istitle()` is False for GitLab, PyPI, JavaScript, iPhone and McDonald
    # -- an internal capital breaks it -- and those are exactly the proper nouns a
    # technical corpus is full of. An interior capital in an otherwise-lowercase word is
    # not something ordinary prose does, so this is a narrow rule, not a loose one.
    return any(c.isupper() for c in word[1:])


def words(text: str) -> list[str]:
    """Every word in *text*, in any script, in order.

    The one tokenizer. There were three -- here, in surprise gating, and inline in recall --
    and each had to be taught about non-Latin scripts separately. Two were; the third was not,
    and every Cyrillic memory was silently dropped before consolidation for months. A second
    definition of "word" is how that happens again.
    """
    return [m.group(0) for m in _WORD.finditer(text)]


def _words_with_position(text: str) -> list[tuple[str, bool]]:
    """Every word in *text*, each with whether a sentence, clause or line opens right before it."""
    positioned: list[tuple[str, bool]] = []
    previous_end = None
    for match in _WORD.finditer(text):
        opens = previous_end is None or bool(_BOUNDARY.search(text, previous_end, match.start()))
        positioned.append((match.group(0), opens))
        previous_end = match.end()
    return positioned


def _named_where_it_stands(word: str, opens: bool) -> bool:
    """True when the capital in *word* is not explained by its position.

    A title-case word that opens a sentence, clause or line is capitalised by grammar. An
    acronym (GDPR) or an interior capital (GitLab) is not a shape any position produces, so
    those count wherever they stand.
    """
    if not _is_candidate(word):
        return False
    return not opens or word.isupper() or any(c.isupper() for c in word[1:])


def extract_entity_names(text: str) -> list[str]:
    """Return the entity names in *text*, in order of first appearance, deduplicated.

    Order is part of the contract: the caller writes these into ``memory_entities`` and
    the index reads them back, so a set would make the stored order depend on hash seed
    and two processes would disagree about the same memory.

    A word is a name when the text capitalises it somewhere its position does not explain.
    Every sentence, line and list item opens with a capital; counting those put "Проверь",
    "Теперь", "Install" and "Check" on every other memory of a real archive. A name that
    opens one sentence is still found when the same text capitalises it anywhere else.
    """
    positioned = _words_with_position(text)
    named = {word.casefold() for word, opens in positioned if _named_where_it_stands(word, opens)}
    seen: set[str] = set()
    names: list[str] = []
    for word, _ in positioned:
        folded = word.casefold()
        if folded in _STOPWORDS or folded in _CALENDAR:
            continue
        if folded not in named or not _is_candidate(word):
            continue
        if folded in seen:
            continue
        seen.add(folded)
        names.append(word)
    return names
