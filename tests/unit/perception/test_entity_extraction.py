"""The entity extractor is the leaf level of the semantic index, so what it can see
bounds what can ever be routed. Two properties matter: it must not be tied to one
script, and it must be the single definition of "entity" that the hot path and the
cold path both use -- two extractors would build two disagreeing indexes.
"""

from __future__ import annotations

from morgan_brain.modules.perception.text.entities import extract_entity_names


def test_extracts_latin_proper_nouns():
    assert extract_entity_names("I met Alice in Berlin") == ["Alice", "Berlin"]


def test_extracts_cyrillic_proper_nouns():
    """The reshape rebuilt the keyword index on FTS5 unicode61 so non-Latin scripts stop
    being dropped. An ASCII-only entity extractor reproduces exactly that bug one layer
    up, so this is the case the old `[A-Z][a-z]{2,}` regex fails."""
    names = extract_entity_names("Ромашка отправила образец в Кувшинку")
    assert "Ромашка" in names
    assert "Кувшинку" in names


def test_mixed_script_text_yields_both():
    names = extract_entity_names("Alice сохранила образец в Ромашке")
    assert {"Alice", "Ромашке"} <= set(names)


def test_drops_stopwords_and_calendar_words():
    names = extract_entity_names("Remind me about the meeting on Monday in March")
    assert "Remind" not in names
    assert "Monday" not in names
    assert "March" not in names


def test_drops_single_characters_and_pure_punctuation():
    assert extract_entity_names("A. B! -- ?") == []


def test_is_deterministic_and_order_preserving_without_duplicates():
    text = "Berlin then Alice then Berlin again"
    first = extract_entity_names(text)
    assert first == extract_entity_names(text)
    assert first.count("Berlin") == 1
    assert first.index("Berlin") < first.index("Alice")


def test_acronyms_are_entities():
    assert "GDPR" in extract_entity_names("does GDPR apply here")


def test_camelcase_brands_are_entities():
    """`"GitLab".istitle()` is False -- an internal capital breaks it -- and a technical
    corpus is mostly these. Missing them left the co-retrieval log half empty."""
    names = extract_entity_names("the GitLab pipeline calls PyPI from an iPhone")
    assert {"GitLab", "PyPI", "iPhone"} <= set(names)


def test_lowercase_words_are_still_not_entities():
    """The CamelCase rule must stay narrow: ordinary prose has no interior capitals."""
    assert extract_entity_names("the pipeline calls the registry") == []


def test_empty_text_is_not_an_error():
    assert extract_entity_names("") == []
