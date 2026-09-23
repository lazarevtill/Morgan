"""The secret gate's rules, one by one: each catches the fixture it assembles at runtime, none
catches another family's, the checksums accept a valid synthetic identifier and reject one digit
off, the keyword comes before the numeric exclusions, the entropy rule exempts what is a digest,
and the anchored assignment rule reads six secret forms and none of sixteen non-secret ones.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
import types
import uuid
from datetime import MAXYEAR

import pytest
from pydantic import ValidationError

from morgan_brain.config import Settings
from morgan_brain.memory.secrets.rules import (
    ASSIGNMENT_RE,
    GATE_VERSION,
    PROVIDER_RULE_NAMES,
    RULE_NAMES,
    GateLimits,
    Rule,
    bin_matches,
    entropy_bits,
    inn_ok,
    is_epoch_like,
    limits_of,
    luhn_ok,
    matches,
    ogrn_ok,
    rules,
    snils_ok,
)

LIMITS = GateLimits.defaults()
RULES = {rule.name: rule for rule in rules(LIMITS)}


def _value(rule: Rule, text: str) -> str | None:
    """The first value *rule* finds in *text*, or None."""
    spans = matches(rule, text, LIMITS)
    return text[spans[0][0] : spans[0][1]] if spans else None


def _value_all(rule: Rule, text: str) -> list[str]:
    """Every value *rule* finds in *text*, in order."""
    return [text[s:e] for s, e in matches(rule, text, LIMITS)]


def test_there_are_thirty_one_rules_in_scan_order():
    assert len(RULES) == 31
    assert tuple(RULES) == RULE_NAMES
    assert RULE_NAMES.index("anthropic_key") < RULE_NAMES.index("openai_key")
    assert RULE_NAMES[:21] == (
        "aws_access_key",
        "github_token",
        "gitlab_token",
        "slack_token",
        "slack_webhook",
        "anthropic_key",
        "openai_key",
        "google_key",
        "stripe_key",
        "telegram_bot",
        "jwt",
        "private_key",
        "huggingface_token",
        "vault_token",
        "tailscale_key",
        "npm_token",
        "sendgrid_key",
        "docker_pat",
        "azure_storage_key",
        "grafana_token",
        "yandex_token",
    )  # The 21 provider names, left to right in scan order.
    assert frozenset(RULE_NAMES[:21]) == PROVIDER_RULE_NAMES
    assert RULE_NAMES[21:25] == ("assignment", "bearer", "url_userinfo", "entropy")
    assert RULE_NAMES[25:] == ("inn", "snils", "ogrn", "bank_card", "passport_rf", "phone_rf")
    assert GATE_VERSION == 1


@pytest.mark.parametrize("name", RULE_NAMES)
def test_every_rule_catches_its_own_fixture(name):
    rule = RULES[name]
    fixture = rule.fixture()
    assert matches(rule, fixture, LIMITS), f"{name} misses its own fixture"
    # A fixture is assembled on each call, so no test file holds a literal token.
    assert rule.fixture() == fixture


@pytest.mark.parametrize("name", RULE_NAMES)
def test_a_fixture_matches_no_other_rule_of_its_family(name):
    """The families are the three kinds. ``anthropic_key`` runs before ``openai_key``, whose
    lookahead excludes it; an identifier fixture carries only its own keyword."""
    rule = RULES[name]
    fixture = rule.fixture()
    for other in RULES.values():
        if other is rule or other.kind != rule.kind:
            continue
        assert not matches(other, fixture, LIMITS), f"{other.name} also matches {name}'s fixture"


def test_private_key_catches_pgp_armor_and_an_encrypted_key():
    """Real PGP armor ends "PRIVATE KEY BLOCK-----", not "PRIVATE KEY-----" like every other
    variant, and an "ENCRYPTED PRIVATE KEY" header is also caught. Each fixture is assembled
    from parts."""
    pgp = (
        "-----BEGIN "
        + "PGP PRIVATE KEY BLOCK-----\n"
        + "A" * 40
        + "\n-----END "
        + "PGP PRIVATE KEY BLOCK-----"
    )
    assert _value(RULES["private_key"], pgp) == pgp
    encrypted = (
        "-----BEGIN "
        + "ENCRYPTED PRIVATE KEY-----\n"
        + "A" * 40
        + "\n-----END "
        + "ENCRYPTED PRIVATE KEY-----"
    )
    assert _value(RULES["private_key"], encrypted) == encrypted


def test_the_effects_by_class():
    for name in PROVIDER_RULE_NAMES:
        assert (RULES[name].kind, RULES[name].effect) == ("provider", "redact")
    for name in ("assignment", "bearer", "url_userinfo", "entropy"):
        assert (RULES[name].kind, RULES[name].effect) == ("generic", "redact")
    for name in ("inn", "snils", "ogrn", "bank_card"):
        assert (RULES[name].kind, RULES[name].effect) == ("identifier", "redact")
    for name in ("passport_rf", "phone_rf"):
        assert (RULES[name].kind, RULES[name].effect) == ("identifier", "flag")


# --- independent checksum reference helpers ----------------------------------------------------
#
# Every expected check digit below is computed here, by small helpers that write out the
# published weights themselves. None of them calls the rule module's own checksum functions, so
# a test failure here cannot be masked by a bug shared between the check and its test. Every base
# below, and every base the rule module's own fixtures use, is synthetic: none reproduces a real
# entity's number.


def _inn10_control_digit(nine: str) -> int:
    """INN-10: weights (2,4,10,3,5,9,4,6,8) over the first 9 digits, mod 11 mod 10."""
    weights = (2, 4, 10, 3, 5, 9, 4, 6, 8)
    return sum(int(d) * w for d, w in zip(nine, weights, strict=True)) % 11 % 10


def _inn12_control_digits(ten: str) -> tuple[int, int]:
    """INN-12: the 11th digit's weights (7,2,4,10,3,5,9,4,6,8) over the first 10 digits, the
    12th's (3,7,2,4,10,3,5,9,4,6,8) over the first 11 (the base plus the 11th), each mod 11
    mod 10."""
    eleventh = (
        sum(int(d) * w for d, w in zip(ten, (7, 2, 4, 10, 3, 5, 9, 4, 6, 8), strict=True)) % 11 % 10
    )
    eleven = ten + str(eleventh)
    twelfth = (
        sum(int(d) * w for d, w in zip(eleven, (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8), strict=True))
        % 11
        % 10
    )
    return eleventh, twelfth


def _ogrn13_control_digit(twelve: str) -> int:
    """OGRN-13: int(first 12) % 11 % 10."""
    return int(twelve) % 11 % 10


def _ogrnip15_control_digit(fourteen: str) -> int:
    """OGRNIP-15: int(first 14) % 13 % 10."""
    return int(fourteen) % 13 % 10


def _snils_control_digits(nine: str) -> int:
    """SNILS: the sum of digit*(9..1) over the first 9 digits, then the standard <100 /
    100-101 / mod-101 rule."""
    total = sum(int(d) * (9 - i) for i, d in enumerate(nine))
    if total < 100:
        return total
    if total in (100, 101):
        return 0
    remainder = total % 101
    return 0 if remainder == 100 else remainder


def _luhn_control_digit(body: str) -> int:
    """The Luhn check digit that completes *body*, found by trying each of the ten digits."""
    for candidate in range(10):
        digits = body + str(candidate)
        total = 0
        for index, char in enumerate(reversed(digits)):
            n = int(char)
            if index % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        if total % 10 == 0:
            return candidate
    raise AssertionError("unreachable: one of ten digits completes a Luhn number")


def _non_bin_card() -> str:
    """A Luhn-valid card whose issuer (900000) no network assigns, outside every default BIN --
    so a test using it is decided by the keyword alone, never by the BIN gate the default
    (Visa-range) fixture would also satisfy."""
    base = "9" + "0" * 7 + "1234567"
    return base + str(_luhn_control_digit(base))


def _generated_card(index: int) -> str:
    """A deterministic Luhn-valid card number in the same 900000 issuer range as
    ``_non_bin_card``, varied by *index* rather than drawn at random, so a set of many cases
    covers many digit combinations without being random."""
    body = "9" + "0" * 5 + f"{index:09d}"
    return body + str(_luhn_control_digit(body))


def _spaced(card: str) -> str:
    return " ".join(card[i : i + 4] for i in range(0, len(card), 4))


def _digits_covered(text: str, value: str, value_start: int) -> bool:
    """Whether every digit of *value* (found at *value_start* in *text*) lies inside some span
    ``matches`` returns for the ``bank_card`` rule -- not necessarily the same span, since a
    card's digits can be split across a merged span and an adjoining one."""
    spans = matches(RULES["bank_card"], text, LIMITS)
    for offset, char in enumerate(value):
        if not char.isdigit():
            continue
        pos = value_start + offset
        if not any(start <= pos < end for start, end in spans):
            return False
    return True


# --- checksums ---------------------------------------------------------------------------------


def test_inn_accepts_a_valid_synthetic_number_and_rejects_one_digit_off():
    ten = RULES["inn"].fixture().split()[-1]
    assert len(ten) == 10 and inn_ok(ten)
    assert not inn_ok(ten[:-1] + str((int(ten[-1]) + 1) % 10))
    assert int(ten[-1]) == _inn10_control_digit(ten[:9])

    base = "00" + "00" + "123456"  # region 00: no tax office assigns it, so no INN can carry it
    eleventh, twelfth = _inn12_control_digits(base)
    twelve = f"{base}{eleventh}{twelfth}"
    assert inn_ok(twelve)
    assert not inn_ok(twelve[:-1] + str((int(twelve[-1]) + 1) % 10))
    assert not inn_ok("12345")


def test_snils_accepts_a_valid_synthetic_number_and_rejects_one_digit_off():
    digits = RULES["snils"].fixture().split()[-2:]
    joined = "".join(digits).replace("-", "")
    assert len(joined) == 11 and snils_ok(joined)
    assert not snils_ok(joined[:-1] + str((int(joined[-1]) + 1) % 10))
    assert int(joined[9:]) == _snils_control_digits(joined[:9])


def test_ogrn_accepts_a_valid_synthetic_number_and_rejects_one_digit_off():
    thirteen = RULES["ogrn"].fixture().split()[-1]
    assert len(thirteen) == 13 and ogrn_ok(thirteen)
    assert not ogrn_ok(thirteen[:-1] + str((int(thirteen[-1]) + 1) % 10))
    assert int(thirteen[-1]) == _ogrn13_control_digit(thirteen[:12])

    base = "3" + "12" + "00" + "000000001"  # region 00: no OGRNIP can carry it
    fifteen = f"{base}{_ogrnip15_control_digit(base)}"
    assert ogrn_ok(fifteen)
    assert not ogrn_ok(fifteen[:-1] + str((int(fifteen[-1]) + 1) % 10))


def test_luhn_accepts_a_valid_synthetic_card_and_rejects_one_digit_off():
    card = RULES["bank_card"].fixture().split()[-1]
    assert luhn_ok(card)
    assert not luhn_ok(card[:-1] + str((int(card[-1]) + 1) % 10))
    assert int(card[-1]) == _luhn_control_digit(card[:-1])


# --- the keyword comes first --------------------------------------------------------------------


def test_a_region_16_inn_beside_its_keyword_is_an_identifier_and_alone_is_a_timestamp():
    """A region-16 INN (tax office 00 is a regional directorate, never an inspection that
    assigns numbers; serial 00000) parses as epoch seconds within the gate's epoch window --
    September 2020 -- and passes the INN-10 check: the keyword decides. The rule's own
    region-00 OGRN fixture is already epoch milliseconds. The INN's control digit is computed
    above, independently of the rule module; the OGRN reuses the rule's own fixture rather
    than a second synthetic base."""
    inn_base = "16" + "00" + "00000"
    inn = inn_base + str(_inn10_control_digit(inn_base))
    assert inn_ok(inn) and is_epoch_like(inn, LIMITS)
    assert _value(RULES["inn"], f"ИНН {inn}") == inn
    assert _value(RULES["inn"], f"inn: {inn}") == inn
    assert _value(RULES["inn"], f"started at {inn}") is None
    ogrn = RULES["ogrn"].fixture().split()[-1]
    assert ogrn_ok(ogrn) and is_epoch_like(ogrn, LIMITS)
    assert _value(RULES["ogrn"], f"ОГРН {ogrn}") == ogrn
    assert _value(RULES["ogrn"], f"ts={ogrn}") is None


def test_an_identifier_stored_as_a_json_number_is_redacted_beside_its_keyword():
    inn_base = "16" + "00" + "00000"  # region 16, tax office 00: see the test above
    inn = inn_base + str(_inn10_control_digit(inn_base))
    assert _value(RULES["inn"], f'{{"inn": {inn}, "name": "x"}}') == inn
    assert _value(RULES["inn"], f'{{"count": {inn}}}') is None


def test_the_keyword_must_sit_on_the_same_line_within_the_context():
    inn = RULES["inn"].fixture().split()[-1]
    far = "x" * (LIMITS.context_chars + 1)
    assert _value(RULES["inn"], f"ИНН {far} {inn}") is None
    assert _value(RULES["inn"], f"ИНН\n{inn}") is None
    assert _value(RULES["inn"], f"{inn} — это ИНН") == inn


def test_a_keyword_glued_to_a_preceding_letter_does_not_gate_a_number():
    """ "INN" inside "dinner", "card" inside "discard" and "тел" inside "пользователь" (user)
    must not gate a nearby number: a keyword matches only where no letter precedes it.
    ``bank_card`` also has the BIN gate, which redacts its default (BIN-4) fixture whether or
    not a keyword is nearby -- that would mask a keyword-regex bug, so its case here uses a
    card outside every default BIN, decided by the keyword alone."""
    inn = RULES["inn"].fixture().split()[-1]
    assert _value(RULES["inn"], f"dinner {inn}") is None

    card = _non_bin_card()
    assert _value(RULES["bank_card"], f"discard {card}") is None
    assert _value(RULES["bank_card"], f"card {card}") == card

    phone_fixture = RULES["phone_rf"].fixture()
    phone = _value(RULES["phone_rf"], phone_fixture)
    assert _value(RULES["phone_rf"], f"пользователь {phone}") is None
    assert _value(RULES["phone_rf"], phone_fixture) == phone


def test_master_alone_does_not_gate_a_card():
    """ "Master" is not a card keyword: "Mastercard" is what is actually written beside a card
    number, and "master" alone is also a common git branch name."""
    card = _non_bin_card()
    assert _value(RULES["bank_card"], f"master {card}") is None


def test_a_digit_or_underscore_is_not_a_letter_so_the_keyword_still_gates():
    """A keyword preceded by a digit or an underscore still gates: neither is a letter --
    "user_phone" and "customer_inn" both still gate their number."""
    inn = RULES["inn"].fixture().split()[-1]
    assert _value(RULES["inn"], f"customer_inn {inn}") == inn
    assert _value(RULES["inn"], f"5inn {inn}") == inn

    phone_fixture = RULES["phone_rf"].fixture()
    phone = _value(RULES["phone_rf"], phone_fixture)
    assert _value(RULES["phone_rf"], f"user_phone {phone}") == phone


def test_an_ascii_keyword_ends_at_a_words_end_with_an_optional_plural():
    """An ASCII keyword also ends at a word's end, with an optional plural "s": "inner",
    "innodb", "panel", "pandas", "mirror" and "cardinal" do not gate; "cards", "phones",
    "phone_number" and "Mastercard" do. Each card case uses a card outside every default BIN,
    so only the keyword decides."""
    inn = RULES["inn"].fixture().split()[-1]
    assert _value(RULES["inn"], f"inner {inn}") is None
    assert _value(RULES["inn"], f"innodb {inn}") is None

    card = _non_bin_card()
    for word in ("panel", "pandas", "mirror", "cardinal"):
        assert _value(RULES["bank_card"], f"{word} {card}") is None, word
    assert _value(RULES["bank_card"], f"cards {card}") == card
    assert _value(RULES["bank_card"], f"Mastercard {card}") == card

    phone_fixture = RULES["phone_rf"].fixture()
    phone = _value(RULES["phone_rf"], phone_fixture)
    assert _value(RULES["phone_rf"], f"phones {phone}") == phone
    assert _value(RULES["phone_rf"], f"phone_number {phone}") == phone


def test_a_cyrillic_keyword_stays_open_on_the_right_for_inflection():
    """Russian inflects a stem's ending, so a Cyrillic keyword is not right-anchored: "карт"
    also matches "карта" and "карты"."""
    card = _non_bin_card()
    assert _value(RULES["bank_card"], f"карта {card}") == card
    assert _value(RULES["bank_card"], f"карты {card}") == card

    phone_fixture = RULES["phone_rf"].fixture()
    phone = _value(RULES["phone_rf"], phone_fixture)
    assert _value(RULES["phone_rf"], f"телефона {phone}") == phone


def test_the_keyword_window_sees_the_real_text_past_either_edge():
    """A slice at the window's edge leaves a lookaround blind to the real neighbouring
    character on the far side of the cut -- both edges must be searched as bounded positions
    in the real text, never a copy. At the left edge, "Finn" would read as "inn" at the start
    of a slice; at the right, "inn" ending exactly at the window's edge would read as
    standalone if a naive ``endpos`` cut off its own lookahead."""
    inn = RULES["inn"].fixture().split()[-1]
    assert _value(RULES["inn"], "Finn" + " " * 37 + inn) is None
    assert _value(RULES["inn"], inn + " " * 37 + "inner") is None


@pytest.mark.parametrize(("gap", "hit"), [(37, True), (38, False), (39, False)])
def test_the_window_after_the_value_hits_at_the_same_gaps_as_the_window_before_it(gap, hit):
    """The keyword's whole match must lie inside the window on either side of the value, not
    merely start inside it: a keyword just past ``context_chars`` still gates if only its start
    is checked, which widens the window on that side by up to the keyword's own length."""
    inn = RULES["inn"].fixture().split()[-1]
    before = _value(RULES["inn"], "INN" + " " * gap + inn) is not None
    after = _value(RULES["inn"], inn + " " * gap + "INN") is not None
    assert before == hit
    assert after == hit


def test_a_card_is_redacted_by_keyword_or_by_bin_but_not_as_a_timestamp():
    card = RULES["bank_card"].fixture().split()[-1]  # a Visa-range number: BIN 4
    assert _value(RULES["bank_card"], f"pay with {card}") == card
    spaced = " ".join(card[i : i + 4] for i in range(0, len(card), 4))
    assert _value(RULES["bank_card"], f"card {spaced}") == spaced
    assert bin_matches(card, LIMITS.card_bins)
    assert not bin_matches("3700000000000002", LIMITS.card_bins)
    # A Luhn-valid, BIN-2200 number that is also epoch milliseconds (~2039) stays a number,
    # whether it is written bare (the epoch branch) or as a JSON value (the JSON-number branch).
    # Narrowing the epoch window to 2000-2001 turns the epoch branch off without touching the
    # JSON one, which isolates which branch excludes which text.
    epoch_ms = "220000000000" + str(_luhn_control_digit("220000000000"))
    assert luhn_ok(epoch_ms)
    assert is_epoch_like(epoch_ms, LIMITS)
    assert bin_matches(epoch_ms, LIMITS.card_bins)
    assert _value(RULES["bank_card"], f'{{"ts": {epoch_ms}}}') is None
    assert _value(RULES["bank_card"], f"logged at {epoch_ms}") is None

    narrow = dataclasses.replace(LIMITS, epoch_years=(2000, 2001))
    narrow_card = {rule.name: rule for rule in rules(narrow)}["bank_card"]
    assert not is_epoch_like(epoch_ms, narrow)
    # Epoch branch off: the bare value is no longer excluded, so it is now redacted.
    assert matches(narrow_card, f"logged at {epoch_ms}", narrow)
    # The JSON branch alone still excludes it, unaffected by the epoch window.
    assert not matches(narrow_card, f'{{"ts": {epoch_ms}}}', narrow)


def test_the_json_number_exclusion_does_not_cross_a_line():
    """Only spaces and tabs are trimmed on either side of the value; a line end -- LF, CRLF or
    a bare CR -- terminates the search rather than being crossed, so a number followed by one
    is excluded the same way whether the next line holds "}" or unrelated content. Uses the
    plain card fixture (Luhn-valid, BIN-4, sixteen digits) rather than an epoch-shaped value,
    so only the JSON-number logic decides, never the epoch branch."""
    card = RULES["bank_card"].fixture().split()[-1]
    assert not is_epoch_like(card, LIMITS)
    assert _value(RULES["bank_card"], f'{{"ts": {card}}}') is None
    assert _value(RULES["bank_card"], f'{{"ts": {card}\n}}') is None
    assert _value(RULES["bank_card"], f'{{"ts": {card}\r\n}}') is None
    assert _value(RULES["bank_card"], f'{{"ts": {card}') is None
    assert _value(RULES["bank_card"], f'{{"ts": {card}\nnext line') is None
    # Same-line-only cuts both ways: a colon several lines above is not this value's own key.
    assert _value(RULES["bank_card"], f'"ts":\n{card}') == card


def test_bank_card_lets_the_checksum_choose_the_span():
    """The pattern alone takes the longest digit run and Luhn rejects it afterwards; the
    checksum must choose among group-bounded sub-spans instead. Each combined span here is
    checked to fail Luhn -- "longest first" would otherwise, one time in ten, legitimately
    accept the longer combination instead of the card alone."""
    card = RULES["bank_card"].fixture().split()[-1]  # a Visa-range number: BIN 4
    groups = [card[i : i + 4] for i in range(0, len(card), 4)]
    spaced = " ".join(groups)

    before = "1" + card
    after = card + "1"
    assert not luhn_ok(before) and not luhn_ok(after)

    assert _value_all(RULES["bank_card"], "1 " + spaced) == [spaced]
    assert _value_all(RULES["bank_card"], spaced + " 1") == [spaced]
    assert _value_all(RULES["bank_card"], "2024-01-15 " + spaced) == [spaced]
    assert _value_all(RULES["bank_card"], f"{card} {card}") == [card, card]
    # A single unbroken run has only itself as a group -- no shorter, differently-bounded
    # sub-span to try -- so it matches nothing unless its own length is 13 to 19.
    assert _value_all(RULES["bank_card"], "card 000" + card + "000000") == []


def test_a_card_beside_its_keyword_is_never_left_partly_in_clear_after_a_preceding_date():
    """A digit-group run that starts with a date and ends with a card, all beside one "card"
    keyword, gates every group in the run: a Luhn-valid sub-span that starts inside the date
    can be accepted on its own, but the card's own group-bounded span is a separate, also
    accepted candidate, and the two merge -- so the union always covers every card digit, even
    when a spurious sub-span from the date is accepted too."""
    failures = []
    for i in range(200):
        card = _generated_card(i)
        spaced = _spaced(card)
        date = f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
        text = f"card {date} {spaced}"
        value_start = text.rindex(spaced)
        if not _digits_covered(text, spaced, value_start):
            failures.append(i)
    assert not failures, f"{len(failures)} of 200 cases left a card digit in clear: {failures}"


def test_a_card_beside_its_keyword_is_never_left_partly_in_clear_after_a_single_digit():
    """The same guarantee holds for the smallest possible preceding group: a single digit."""
    failures = []
    for i in range(200):
        card = _generated_card(i)
        text = f"card 7 {card}"
        value_start = text.rindex(card)
        if not _digits_covered(text, card, value_start):
            failures.append(i)
    assert not failures, f"{len(failures)} of 200 cases left a card digit in clear: {failures}"


def test_a_card_beside_its_keyword_is_never_left_partly_in_clear_before_a_following_date():
    """Groups that follow the card in the same run leak nothing either: the card is tried at
    its own start regardless of what comes after it."""
    for i in range(50):
        card = _generated_card(i + 1000)
        spaced = _spaced(card)
        date = f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
        text = f"card {spaced} {date}"
        assert _digits_covered(text, spaced, text.index(spaced))


def test_two_cards_in_one_gated_run_are_both_fully_covered():
    """Two distinct cards in the same digit run are each found, whole, as their own span (or
    spans merged with an overlapping candidate) -- the union-of-candidates fix does not fuse
    unrelated cards together or drop either one."""
    card1 = _generated_card(2000)
    card2 = _generated_card(2001)
    text = f"card {card1} {card2}"
    assert _digits_covered(text, card1, text.index(card1))
    assert _digits_covered(text, card2, text.rindex(card2))


def test_passport_and_phone_need_their_keyword_and_only_flag():
    assert _value(RULES["passport_rf"], "паспорт 00 00 123456") == "00 00 123456"
    assert _value(RULES["passport_rf"], "order 00 00 123456") is None
    assert _value(RULES["phone_rf"], "тел. +7 000 123-45-67") == "+7 000 123-45-67"
    assert _value(RULES["phone_rf"], "+7 000 123-45-67") is None


# --- entropy -----------------------------------------------------------------------------------


#: The UUID, 40-hex, 64-hex and ``sha256:`` cases measure below the *default* threshold
#: (3.92, 2.73, 4.00 and 4.14 bits -- verified below) and would pass with no exemption at all,
#: so the default threshold does not pin the exemption for them. Every case here is run
#: against a threshold below all six measured values, so only the exemption -- never entropy
#: alone -- can explain a miss.
_DIGEST_TOKENS = (
    "6f1a2b3c-4d5e-4f60-8a9b-0c1d2e3f4a5b",  # a UUID
    "a" * 20 + "0123456789abcdef" + "f" * 4,  # 40 hex
    "0123456789abcdef" * 4,  # 64 hex
    "sha256:" + "0123456789abcdef" * 4,
    "sha256-" + "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789+/=" + "AbCd",  # an SRI value
    "h1:" + "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789+/AB",  # a go.sum hash
)


def test_the_digest_tokens_measure_below_the_default_threshold():
    """Documents why the exemption, not entropy, must be what the next test pins."""
    assert [round(entropy_bits(t), 2) for t in _DIGEST_TOKENS[:4]] == [3.92, 2.73, 4.0, 4.14]
    assert all(entropy_bits(t) > 1.0 for t in _DIGEST_TOKENS)  # all clear the lowered threshold


@pytest.mark.parametrize("token", _DIGEST_TOKENS)
def test_digests_are_not_entropy_hits(token):
    lowered = dataclasses.replace(LIMITS, entropy_threshold=1.0)
    lowered_entropy = {rule.name: rule for rule in rules(lowered)}["entropy"]
    spans = matches(lowered_entropy, f"digest {token} recorded", lowered)
    assert not spans, f"{token!r} is an entropy hit even exempt"


def test_a_random_base64_token_is_an_entropy_hit():
    token = "abcdefghijkl" + "mnopqrstuvwx" + "yz0123456789" + "+/ABCDEF"
    assert len(token) == 44 and entropy_bits(token) > LIMITS.entropy_threshold
    assert _value(RULES["entropy"], f"token {token} here") == token
    assert entropy_bits("a" * 40) == 0.0
    assert _value(RULES["entropy"], "a" * 40) is None


def test_an_interior_equals_splits_the_entropy_token_from_its_key():
    """ "=" is trailing base64 padding only (0-2 of them), never part of the repeated class, so
    an interior "=" is a key/value seam: "request_id=<uuid>" scores the UUID alone, and its
    exemption applies to the UUID alone, never to the glued "request_id=<uuid>" as one token.
    200 deterministic values each, not one -- each derived from its own index, not a
    pseudo-random generator, so the run is reproducible without inviting a "not for
    cryptographic use" finding for a value that secures nothing."""
    for i in range(200):
        token = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"request-id-{i}"))
        assert _value(RULES["entropy"], f"request_id={token}") is None
    for i in range(200):
        token = hashlib.sha1(f"commit-{i}".encode(), usedforsecurity=False).hexdigest()
        assert _value(RULES["entropy"], f"commit={token}") is None


def test_trailing_base64_padding_stays_part_of_the_token():
    """Up to two trailing "=" are padding, not a seam: a padded token is still one entropy
    hit."""
    token = "abcdefghijkl" + "mnopqrstuvwx" + "yz0123456789" + "AB=="
    assert entropy_bits(token) > LIMITS.entropy_threshold
    assert _value(RULES["entropy"], f"token {token} here") == token


def test_the_digest_prefix_exemption_needs_a_non_word_character_before_it():
    """ "sha256:" and "h1:" count only after a non-word character or the start of the text: a
    bare suffix test would let "oauth1:" and "depth1:", which merely end in "h1:", exempt a
    token beside them too."""
    token = "abcdefghijkl" + "mnopqrstuvwx" + "yz0123456789" + "+/ABCDEF"  # the base64 hit
    assert _value(RULES["entropy"], f"sha256:{token}") is None  # start of text
    assert _value(RULES["entropy"], f"digest sha256:{token}") is None  # after a space
    assert _value(RULES["entropy"], f"oauth1:{token}") == token  # "t" precedes "h1:"
    assert _value(RULES["entropy"], f"depth1:{token}") == token  # "t" precedes "h1:" too
    assert _value(RULES["entropy"], f"x:{token}") == token  # not a recognised prefix at all


# --- the anchored assignment rule -----------------------------------------------------------

#: A synthetic value, assembled from parts: not AWS's documented example key.
_AWS_EXAMPLE_VALUE = "fake" + "Synthetic" + "Secret99"

SECRET_FORMS = [
    ("DB_PASSWORD=s3cr3tValue9", "s3cr3tValue9"),
    ("export OPENAI_API_KEY=abcdefghijklmnop", "abcdefghijklmnop"),
    ('"password": "hunter2hunter2"', "hunter2hunter2"),
    ("api_key: q1w2e3r4t5y6", "q1w2e3r4t5y6"),
    ("ACCESS_TOKEN2=abcdefgh12345", "abcdefgh12345"),
    (f"aws_secret_access_key = {_AWS_EXAMPLE_VALUE}", _AWS_EXAMPLE_VALUE),
]

#: None of these keys ends in a secret word -- eight keys that trail off in a lookalike word
#: (``secretsmanager``, ``total_tokens``, ``token_type``, ``token_count``, ``api_key_env``,
#: ``secret_name``, ``password_min_length``, ``access_token_id``) plus eight more of the same
#: shape -- so the assignment rule matches none of them at all. Every value is long and
#: non-numeric: a shorter or numeric value would also be excluded on its own and would not pin
#: the anchoring specifically; the numeric and placeholder exclusions are pinned separately
#: below, each on a key that *is* anchored.
NON_SECRET_FORMS = [
    '"total_tokens": "hunter2hunter2"',
    "max_tokens=hunter2hunter2",
    "tokenize = 'hunter2hunter2'",
    "secretsmanager: hunter2hunter2",
    'api_key_env: "hunter2hunter2"',
    "password_min_length: hunter2hunter2",
    '"token_type": "hunter2hunter2"',
    "user_id=hunter2hunter2",
    "token_count=hunter2hunter2",
    "secret_name=hunter2hunter2",
    "access_token_id=hunter2hunter2",
    "token_expiry=hunter2hunter2",
    "password_policy=hunter2hunter2",
    "secret_manager_name=hunter2hunter2",
    "client_secret_expiry=hunter2hunter2",
    "api_key_rotation_days=hunter2hunter2",
]


@pytest.mark.parametrize(("text", "value"), SECRET_FORMS)
def test_the_assignment_rule_reads_the_six_secret_forms(text, value):
    assert _value(RULES["assignment"], f"config: {text}") == value


@pytest.mark.parametrize("text", NON_SECRET_FORMS)
def test_the_assignment_rule_pins_its_anchoring_on_sixteen_non_secret_forms(text):
    """There is no suffix list: the key must end in a secret word, and none of these sixteen
    does, so none matches at all."""
    assert _value(RULES["assignment"], f"config: {text}") is None


def test_the_assignment_rule_excludes_a_numeric_value_up_front():
    """A value that is only digits and dots is never a secret -- a port, a version, a count.
    The key here is anchored (ends in "password"), so only the numeric exclusion explains the
    miss."""
    assert _value(RULES["assignment"], "config: PASSWORD=12345678") is None


@pytest.mark.parametrize(
    "text",
    [
        "PASSWORD=change-me",  # a placeholder value
        "SECRET=your-secret-here",  # a placeholder shape ("your-*-here")
        "TOKEN=${SOME_LONG_VARIABLE_NAME}",  # a placeholder prefix ("${")
        "API_KEY=<your-token>",  # a placeholder prefix ("<")
        "PASSWORD={{vault_password_long_name}}",  # a placeholder prefix ("{{")
    ],
)
def test_the_assignment_rule_excludes_each_placeholder_kind_up_front(text):
    """An anchored key beside a templated value is never a secret. Each key here is
    anchored and each value is long enough to clear the assignment value regex on its own, so
    only its placeholder kind explains the miss. The "{{" case has no space after the braces:
    with one, the value regex stops there and the length exclusion masks the placeholder check
    instead."""
    assert _value(RULES["assignment"], f"config: {text}") is None


def test_the_assignment_rule_reads_a_line_start_and_a_json_key():
    assert _value(RULES["assignment"], "PASSWORD=abcdefghij\nother=1") == "abcdefghij"
    assert _value(RULES["assignment"], '{"secret":"abcdefghij"}') == "abcdefghij"


def test_the_assignment_rule_finds_every_value_on_a_line():
    """``rest``'s up-to-64-character reach swallows the rest of the line, so a plain
    ``finditer`` resumes past a second assignment before it is ever tried; a short or
    placeholder first value swallows a real one that follows it on the same line. Resuming at
    an accepted value's end, or at a rejected one's own ``rest``, finds every value each of
    these shapes actually carries."""
    assert _value_all(
        RULES["assignment"], "DB_PASSWORD=s3cr3tValue9 REDIS_PASSWORD=an0therValue1"
    ) == ["s3cr3tValue9", "an0therValue1"]
    assert _value_all(
        RULES["assignment"], "docker run -e TOKEN=$TOKEN -e PASSWORD=realsecret12"
    ) == ["realsecret12"]
    assert _value_all(
        RULES["assignment"], '{"api_key": "${API_KEY}", "password": "realsecret12"}'
    ) == ["realsecret12"]
    assert _value_all(RULES["assignment"], "--token=abc --password=realsecret12") == [
        "realsecret12"
    ]


def test_assignment_re_does_not_backtrack_quadratically_on_trailing_whitespace():
    """``\\s*(?:["']\\s*)?[:=]`` accepts a separator with an optional quote and any amount of
    surrounding whitespace, using one ``\\s*`` run rather than two -- a two-run form backtracks
    quadratically when no "=" or ":" ever follows. A generous bound, not a benchmark: measured
    at ~0.007s for this length."""
    text = "token" + " " * 65536  # no separator ever follows
    started = time.monotonic()
    ASSIGNMENT_RE.search(text)
    assert time.monotonic() - started < 1.0


# --- the settings ------------------------------------------------------------------------------


def test_the_limits_come_from_the_settings_and_the_defaults_match_them():
    settings = Settings()
    limits = limits_of(settings)
    assert limits == LIMITS
    assert limits.epoch_years == (2000, 2100)
    assert limits.card_bins == ("4", "51-55", "2200-2204", "2221-2720")
    assert limits.keywords["inn"] == ("ИНН", "INN")
    assert limits.keywords["phone_rf"] == ("тел", "phone", "телефон", "моб")
    assert limits.placeholder_values == ("change-me", "changeme", "xxx")
    assert limits.placeholder_shapes == ("your-*-here",)
    assert limits.placeholder_prefixes == ("${", "<", "$(", "%(", "{{")


def test_a_lowered_threshold_reaches_the_rules(monkeypatch):
    monkeypatch.setenv("MORGAN_GATE_ENTROPY_THRESHOLD", "0.5")
    monkeypatch.setenv("MORGAN_GATE_KEYWORDS_INN", "tax-id")
    limits = limits_of(Settings())
    assert limits.entropy_threshold == 0.5
    assert limits.keywords["inn"] == ("tax-id",)
    lowered = {rule.name: rule for rule in rules(limits)}
    # matches() takes the same *limits* the rules were built from: _value's global LIMITS
    # would silently carry the default entropy_threshold and keywords instead.
    assert matches(lowered["entropy"], "ab" * 12, limits)
    inn = lowered["inn"].fixture().split()[-1]
    assert matches(lowered["inn"], f"tax-id {inn}", limits)
    assert not matches(lowered["inn"], f"ИНН {inn}", limits)


def test_gate_limits_keywords_cannot_be_mutated():
    """``keywords`` is a read-only mapping: a caller cannot quietly change what a rule set was
    built from after the fact."""
    assert isinstance(LIMITS.keywords, types.MappingProxyType)
    with pytest.raises(TypeError):
        LIMITS.keywords["inn"] = ("changed",)  # type: ignore[index]


def test_the_card_fixture_uses_the_configured_first_keyword():
    """All six identifier fixtures read their keyword from the settings, the same way; none is
    a literal string."""
    assert RULES["bank_card"].fixture().startswith(LIMITS.keywords["bank_card"][0] + " ")


@pytest.mark.parametrize(
    "field",
    [
        "gate_keywords_inn",
        "gate_keywords_snils",
        "gate_keywords_ogrn",
        "gate_keywords_card",
        "gate_keywords_passport",
        "gate_keywords_phone",
    ],
)
def test_an_empty_keyword_setting_refuses_at_load_naming_the_field(field, monkeypatch):
    """ "No keyword" would silently switch an identifier rule off, or turn the card gate into
    BIN-only, and the gate report could not tell that apart from a rule that simply finds
    nothing."""
    monkeypatch.setenv(f"MORGAN_{field.upper()}", "")
    with pytest.raises(ValidationError) as excinfo:
        Settings()
    assert field in str(excinfo.value)


def test_an_empty_card_bins_setting_means_bank_card_redacts_by_keyword_only():
    """Unlike the keyword lists, an empty ``gate_card_bins`` is not refused: it means bank_card
    has no BIN gate to fall back on, ever."""
    limits = dataclasses.replace(LIMITS, card_bins=())
    card_rule = {rule.name: rule for rule in rules(limits)}["bank_card"]
    card = card_rule.fixture().split()[-1]
    assert matches(card_rule, f"card {card}", limits)
    assert not matches(card_rule, f"pay with {card}", limits)


def test_empty_placeholder_settings_mean_no_placeholder_exclusion():
    """An anchored key beside a value that is normally excluded as a placeholder is redacted
    once the placeholder lists are empty."""
    limits = dataclasses.replace(
        LIMITS, placeholder_values=(), placeholder_shapes=(), placeholder_prefixes=()
    )
    assignment_rule = {rule.name: rule for rule in rules(limits)}["assignment"]
    assert matches(RULES["assignment"], "config: PASSWORD=change-me", LIMITS) == []
    assert matches(assignment_rule, "config: PASSWORD=change-me", limits)
    assert matches(RULES["assignment"], "config: SECRET=your-api-here", LIMITS) == []
    assert matches(assignment_rule, "config: SECRET=your-api-here", limits)
    assert matches(RULES["assignment"], "config: API_KEY=<a-generated-token>", LIMITS) == []
    assert matches(assignment_rule, "config: API_KEY=<a-generated-token>", limits)
    placeholder_braces = "config: PASSWORD={{vault_password_long_name}}"
    assert matches(RULES["assignment"], placeholder_braces, LIMITS) == []
    assert matches(assignment_rule, placeholder_braces, limits)


@pytest.mark.parametrize(
    "value",
    [
        "2000–2100",  # an en dash, not a hyphen
        "abcd-2100",  # non-digit
        "2100-2000",  # reversed
        "0-2100",  # below datetime.MINYEAR
        "2000-9999",  # is_epoch_like builds high + 1, one more than datetime.MAXYEAR allows
        "2000-99999",  # far above datetime.MAXYEAR
    ],
)
def test_a_malformed_epoch_years_setting_refuses_at_load(value, monkeypatch):
    """Named at load, not left to raise a bare ``ValueError`` from inside a scan."""
    monkeypatch.setenv("MORGAN_GATE_EPOCH_YEARS", value)
    with pytest.raises(ValidationError) as excinfo:
        Settings()
    assert "gate_epoch_years" in str(excinfo.value)


def test_the_largest_epoch_year_leaves_room_for_is_epoch_likes_high_plus_one(monkeypatch):
    """``is_epoch_like`` builds its exclusive upper bound from ``datetime(high + 1, 1, 1)``, so
    the largest year the setting accepts is one less than ``datetime.MAXYEAR`` -- the largest
    year ``high + 1`` can still represent."""
    monkeypatch.setenv("MORGAN_GATE_EPOCH_YEARS", f"2000-{MAXYEAR - 1}")
    limits = limits_of(Settings())
    assert limits.epoch_years == (2000, MAXYEAR - 1)
    assert is_epoch_like("1700000000", limits)  # 2023, well inside the window


@pytest.mark.parametrize(
    "value", ["5x", "51-550", "55-51"]
)  # non-digit; mismatched width; reversed
def test_a_malformed_card_bins_entry_refuses_at_load(value, monkeypatch):
    """Named at load, not left to raise a bare ``ValueError`` or be silently skipped from
    inside a scan."""
    monkeypatch.setenv("MORGAN_GATE_CARD_BINS", value)
    with pytest.raises(ValidationError) as excinfo:
        Settings()
    assert "gate_card_bins" in str(excinfo.value)
