"""The secret gate's rules, one by one: each catches the fixture it assembles at runtime, none
catches another family's, the checksums accept a valid synthetic identifier and reject one digit
off, the keyword comes before the numeric exclusions, the entropy rule exempts what is a digest,
and the anchored assignment rule reads six secret forms and none of fourteen non-secret ones.
"""

from __future__ import annotations

import pytest

from morgan_brain.config import Settings
from morgan_brain.memory.secrets.rules import (
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


def test_there_are_thirty_one_rules_in_scan_order():
    assert len(RULES) == 31
    assert tuple(RULES) == RULE_NAMES
    assert RULE_NAMES.index("anthropic_key") < RULE_NAMES.index("openai_key")
    assert RULE_NAMES[:21] == tuple(sorted(RULE_NAMES[:21], key=RULE_NAMES.index))
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
# is synthetic (sequential digits); no real-format identifier literal appears in this file.


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


# --- checksums ---------------------------------------------------------------------------------


def test_inn_accepts_a_valid_synthetic_number_and_rejects_one_digit_off():
    ten = RULES["inn"].fixture().split()[-1]
    assert len(ten) == 10 and inn_ok(ten)
    assert not inn_ok(ten[:-1] + str((int(ten[-1]) + 1) % 10))
    assert int(ten[-1]) == _inn10_control_digit(ten[:9])

    base = "1234567890"  # a synthetic 10-digit base: no real-format INN appears here
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

    base = "31234567890123"  # a synthetic 14-digit base: no real-format OGRNIP appears here
    fifteen = f"{base}{_ogrnip15_control_digit(base)}"
    assert ogrn_ok(fifteen)
    assert not ogrn_ok(fifteen[:-1] + str((int(fifteen[-1]) + 1) % 10))


def test_luhn_accepts_a_valid_synthetic_card_and_rejects_one_digit_off():
    card = RULES["bank_card"].fixture().split()[-1]
    assert luhn_ok(card)
    assert not luhn_ok(card[:-1] + str((int(card[-1]) + 1) % 10))
    assert int(card[-1]) == _luhn_control_digit(card[:-1])


# --- the keyword comes first (N2) ---------------------------------------------------------------


def test_a_region_16_inn_beside_its_keyword_is_an_identifier_and_alone_is_a_timestamp():
    """A region-16 INN parses as epoch seconds within the gate's epoch window, and passes the
    INN-10 check: the keyword decides. The same holds for a 1-prefixed OGRN, which is epoch
    milliseconds. Both bases are synthetic; their control digits are computed above,
    independently of the rule module."""
    inn = "165012345" + str(_inn10_control_digit("165012345"))
    assert inn_ok(inn) and is_epoch_like(inn, LIMITS)
    assert _value(RULES["inn"], f"ИНН {inn}") == inn
    assert _value(RULES["inn"], f"inn: {inn}") == inn
    assert _value(RULES["inn"], f"started at {inn}") is None
    ogrn = "123456789012" + str(_ogrn13_control_digit("123456789012"))
    assert ogrn_ok(ogrn) and is_epoch_like(ogrn, LIMITS)
    assert _value(RULES["ogrn"], f"ОГРН {ogrn}") == ogrn
    assert _value(RULES["ogrn"], f"ts={ogrn}") is None


def test_an_identifier_stored_as_a_json_number_is_redacted_beside_its_keyword():
    inn = "165012345" + str(_inn10_control_digit("165012345"))
    assert _value(RULES["inn"], f'{{"inn": {inn}, "name": "x"}}') == inn
    assert _value(RULES["inn"], f'{{"count": {inn}}}') is None


def test_the_keyword_must_sit_on_the_same_line_within_the_context():
    inn = RULES["inn"].fixture().split()[-1]
    far = "x" * (LIMITS.context_chars + 1)
    assert _value(RULES["inn"], f"ИНН {far} {inn}") is None
    assert _value(RULES["inn"], f"ИНН\n{inn}") is None
    assert _value(RULES["inn"], f"{inn} — это ИНН") == inn


def test_a_card_is_redacted_by_keyword_or_by_bin_but_not_as_a_timestamp():
    card = RULES["bank_card"].fixture().split()[-1]  # a Visa-range number: BIN 4
    assert _value(RULES["bank_card"], f"pay with {card}") == card
    spaced = " ".join(card[i : i + 4] for i in range(0, len(card), 4))
    assert _value(RULES["bank_card"], f"card {spaced}") == spaced
    assert bin_matches(card, LIMITS.card_bins)
    assert not bin_matches("3700000000000002", LIMITS.card_bins)
    # A Luhn-valid, BIN-2200 number that is also epoch milliseconds (~2039) stays a number: the
    # JSON-number shape excludes it, not its checksum.
    epoch_ms = "220000000000" + str(_luhn_control_digit("220000000000"))
    assert luhn_ok(epoch_ms)
    assert is_epoch_like(epoch_ms, LIMITS)
    assert bin_matches(epoch_ms, LIMITS.card_bins)
    assert _value(RULES["bank_card"], f'{{"ts": {epoch_ms}}}') is None


def test_passport_and_phone_need_their_keyword_and_only_flag():
    assert _value(RULES["passport_rf"], "паспорт 45 07 123456") == "45 07 123456"
    assert _value(RULES["passport_rf"], "order 45 07 123456") is None
    assert _value(RULES["phone_rf"], "тел. +7 916 123-45-67") == "+7 916 123-45-67"
    assert _value(RULES["phone_rf"], "+7 916 123-45-67") is None


# --- entropy -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    [
        "6f1a2b3c-4d5e-4f60-8a9b-0c1d2e3f4a5b",  # a UUID
        "a" * 20 + "0123456789abcdef" + "f" * 4,  # 40 hex
        "0123456789abcdef" * 4,  # 64 hex
        "sha256:" + "0123456789abcdef" * 4,
        "sha256-" + "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789+/=" + "AbCd",  # an SRI value
        "h1:" + "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789+/AB",  # a go.sum hash
    ],
)
def test_digests_are_not_entropy_hits(token):
    assert _value(RULES["entropy"], f"digest {token} recorded") is None


def test_a_random_base64_token_is_an_entropy_hit():
    token = "abcdefghijkl" + "mnopqrstuvwx" + "yz0123456789" + "+/ABCDEF"
    assert len(token) == 44 and entropy_bits(token) > LIMITS.entropy_threshold
    assert _value(RULES["entropy"], f"token {token} here") == token
    assert entropy_bits("a" * 40) == 0.0
    assert _value(RULES["entropy"], "a" * 40) is None


# --- the anchored assignment rule (D27, N15, S3) -----------------------------------------------

SECRET_FORMS = [
    ("DB_PASSWORD=s3cr3tValue9", "s3cr3tValue9"),
    ("export OPENAI_API_KEY=abcdefghijklmnop", "abcdefghijklmnop"),
    ('"password": "hunter2hunter2"', "hunter2hunter2"),
    ("api_key: q1w2e3r4t5y6", "q1w2e3r4t5y6"),
    ("ACCESS_TOKEN2=abcdefgh12345", "abcdefgh12345"),
    ("aws_secret_access_key = wJalrXUtnFEMI", "wJalrXUtnFEMI"),
]

NON_SECRET_FORMS = [
    '"total_tokens": 1234',
    "max_tokens=4096",
    "tokenize = 'unicode61 remove_diacritics 2'",
    "secretsmanager:GetSecretValue",
    'api_key_env: "OPENAI_API_KEY"',
    "password_min_length: 12",
    '"token_type": "bearer"',
    "PASSWORD=change-me",
    "TOKEN=${TOKEN}",
    "API_KEY=<your-token>",
    "SECRET=your-secret-here",
    "user_id=12345678",
    "token_count=1234",
    "PASSWORD={{ vault_password }}",
]


@pytest.mark.parametrize(("text", "value"), SECRET_FORMS)
def test_the_assignment_rule_reads_the_six_secret_forms(text, value):
    assert _value(RULES["assignment"], f"config: {text}") == value


@pytest.mark.parametrize("text", NON_SECRET_FORMS)
def test_the_assignment_rule_pins_its_anchoring_on_the_fourteen_non_secret_forms(text):
    """The key ends in its secret word: ``secretsmanager``, ``total_tokens``, ``token_type``,
    ``api_key_env`` and ``password_min_length`` never match; a numeric value and a placeholder
    are excluded up front."""
    assert _value(RULES["assignment"], f"config: {text}") is None


def test_the_assignment_rule_reads_a_line_start_and_a_json_key():
    assert _value(RULES["assignment"], "PASSWORD=abcdefghij\nother=1") == "abcdefghij"
    assert _value(RULES["assignment"], '{"secret":"abcdefghij"}') == "abcdefghij"


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
    assert _value(lowered["entropy"], "ab" * 12) == "ab" * 12
    inn = RULES["inn"].fixture().split()[-1]
    assert _value(lowered["inn"], f"tax-id {inn}") == inn
    assert _value(lowered["inn"], f"ИНН {inn}") is None
