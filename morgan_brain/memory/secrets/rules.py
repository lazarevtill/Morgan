"""The secret gate's rules: what a secret looks like, as data.

Three classes of rule, in scan order. A *provider* rule names a token by its shape (a GitHub
token, an AWS key). A *generic* rule reads the context a secret sits in -- an assignment, a bearer
header, a URL's userinfo -- or its randomness, the entropy rule. A *Russian identifier* rule reads
a number with a passing checksum beside its keyword: an INN, a SNILS, an OGRN, a bank card, a
passport, a phone. The scanner (``scan.py``) runs them in order over the text and applies the
verdict; this module describes each rule and says, with ``matches``, where it applies.

Every threshold, keyword list, BIN range, epoch window and placeholder is a setting
(``GateLimits``), never a literal here. Every rule carries a ``fixture``: a value it catches,
assembled from parts on each call, so no provider-shaped literal sits in this repository for a
scanner to find; the tests and the gate report's canaries both scan those fixtures.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fnmatch import fnmatch
from typing import Literal

from morgan_brain.config import Settings

RuleKind = Literal["provider", "generic", "identifier"]
Effect = Literal["redact", "flag"]

#: The rule set's version, stored on every captured session as ``gate_version``: a session
#: scanned under an older set is told apart from one scanned under this one.
GATE_VERSION = 1

#: What precedes an entropy-shaped token that is a digest, not a secret.
ENTROPY_EXEMPT_PREFIXES: tuple[str, ...] = ("sha256:", "h1:")


@dataclass(frozen=True)
class GateLimits:
    """Every number, keyword and shape the rules read, taken from the settings once."""

    entropy_min_length: int
    entropy_threshold: float
    context_chars: int
    card_bins: tuple[str, ...]
    epoch_years: tuple[int, int]
    #: By rule name: ``inn``, ``snils``, ``ogrn``, ``bank_card``, ``passport_rf``, ``phone_rf``.
    keywords: Mapping[str, tuple[str, ...]]
    placeholder_values: tuple[str, ...]
    placeholder_shapes: tuple[str, ...]
    placeholder_prefixes: tuple[str, ...]

    @classmethod
    def defaults(cls) -> GateLimits:
        """The settings' defaults, for a caller that builds no ``Settings``: the values of the
        fields themselves, read without validation and without the environment."""
        return limits_of(Settings.model_construct())


def limits_of(settings: Settings) -> GateLimits:
    """The gate's limits as *settings* has them."""
    low, _, high = settings.gate_epoch_years.partition("-")
    return GateLimits(
        entropy_min_length=settings.gate_entropy_min_length,
        entropy_threshold=settings.gate_entropy_threshold,
        context_chars=settings.gate_context_chars,
        card_bins=tuple(settings.gate_card_bins),
        epoch_years=(int(low.strip()), int((high or low).strip())),
        keywords={
            "inn": tuple(settings.gate_keywords_inn),
            "snils": tuple(settings.gate_keywords_snils),
            "ogrn": tuple(settings.gate_keywords_ogrn),
            "bank_card": tuple(settings.gate_keywords_card),
            "passport_rf": tuple(settings.gate_keywords_passport),
            "phone_rf": tuple(settings.gate_keywords_phone),
        },
        placeholder_values=tuple(settings.gate_placeholder_values),
        placeholder_shapes=tuple(settings.gate_placeholder_shapes),
        placeholder_prefixes=tuple(settings.gate_placeholder_prefixes),
    )


#: Where a match's value sits inside it: (start, end) in the text, or ``None`` to pass it over.
ValueSpan = Callable[[re.Match[str], GateLimits], "tuple[int, int] | None"]


@dataclass(frozen=True)
class Rule:
    """One rule. *pattern* finds candidates; *value_span* says which part of a candidate is the
    value (and may reject it); *check* is a checksum or the entropy test over that value;
    *exempt* are the shapes the value must not have; *keywords* is the context gate an
    identifier needs within ``context_chars`` on its line, and *bin_gate* whether a card BIN
    passes it instead; *effect* is what a hit does to the text; *fixture* builds a value the
    rule catches, from parts, on each call."""

    name: str
    kind: RuleKind
    pattern: re.Pattern[str]
    effect: Effect
    value_span: ValueSpan
    fixture: Callable[[], str]
    keywords: re.Pattern[str] | None = None
    check: Callable[[str], bool] | None = None
    exempt: tuple[re.Pattern[str], ...] = ()
    bin_gate: bool = False


def _whole(match: re.Match[str], limits: GateLimits) -> tuple[int, int]:
    return match.start(), match.end()


def _group_one(match: re.Match[str], limits: GateLimits) -> tuple[int, int]:
    return match.start(1), match.end(1)


# --- the assignment rule: a key that ends in a secret word, then its value -----------------------

ASSIGNMENT_RE = re.compile(
    r"""(?i)(?:^|[\s"'{,(\[])(?:export\s+)?"""
    r"""(?P<key>[A-Za-z0-9_.-]*?(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key"""
    r"""|secret[_-]?key|client[_-]?secret|private[_-]?key)\d*)"""
    r"""\s*["']?\s*[:=]\s*(?P<rest>.{0,64})""",
    re.MULTILINE,
)
ASSIGNMENT_VALUE_RE = re.compile(r"""^["']?([^\s"',;)}\]]{8,})""")


def is_numeric_value(value: str) -> bool:
    """Digits and dots only: a port, a version, a count -- never a secret."""
    return bool(value) and all(c.isdigit() or c == "." for c in value)


def is_placeholder(rest: str, value: str, limits: GateLimits) -> bool:
    """A template rather than a value: *rest* (its leading quote dropped) starts with a
    placeholder prefix, or the value is a placeholder value or matches a placeholder shape,
    case aside."""
    bare = rest.lstrip("\"'")
    if any(bare.startswith(prefix) for prefix in limits.placeholder_prefixes):
        return True
    lowered = value.lower()
    if lowered in {known.lower() for known in limits.placeholder_values}:
        return True
    return any(fnmatch(lowered, shape.lower()) for shape in limits.placeholder_shapes)


def _assignment_span(match: re.Match[str], limits: GateLimits) -> tuple[int, int] | None:
    rest = match.group("rest")
    found = ASSIGNMENT_VALUE_RE.match(rest)
    if found is None:
        return None
    value = found.group(1)
    if is_numeric_value(value) or is_placeholder(rest, value, limits):
        return None
    start = match.start("rest") + found.start(1)
    return start, start + len(value)


# --- entropy ----------------------------------------------------------------------------------


def entropy_bits(token: str) -> float:
    """Shannon entropy of *token*, in bits per character; 0.0 for an empty token."""
    if not token:
        return 0.0
    counts = Counter(token)
    total = len(token)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


_ENTROPY_EXEMPT: tuple[re.Pattern[str], ...] = (
    re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"),
    re.compile(r"[0-9a-fA-F]{40}"),
    re.compile(r"[0-9a-fA-F]{64}"),
    re.compile(r"sha(?:256|384|512)-[A-Za-z0-9+/=]+"),
)


# --- the Russian identifiers -------------------------------------------------------------------


def _digits(value: str) -> str:
    return "".join(c for c in value if c.isdigit())


def luhn_ok(digits: str) -> bool:
    """The Luhn check over *digits* (digits only)."""
    if not digits.isdigit() or len(digits) < 2:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        n = int(char)
        if index % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


_INN_10 = (2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_12_FIRST = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
_INN_12_SECOND = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)


def _inn_control(digits: str, weights: tuple[int, ...]) -> int:
    return sum(int(d) * w for d, w in zip(digits, weights, strict=True)) % 11 % 10


def inn_ok(digits: str) -> bool:
    """A 10-digit INN's control digit, or a 12-digit INN's two, as the tax service computes
    them."""
    if not digits.isdigit():
        return False
    if len(digits) == 10:
        return _inn_control(digits[:9], _INN_10) == int(digits[9])
    if len(digits) == 12:
        return _inn_control(digits[:10], _INN_12_FIRST) == int(digits[10]) and _inn_control(
            digits[:11], _INN_12_SECOND
        ) == int(digits[11])
    return False


def _snils_control(nine: str) -> int:
    total = sum(int(d) * (9 - i) for i, d in enumerate(nine))
    if total < 100:
        return total
    if total in (100, 101):
        return 0
    remainder = total % 101
    return 0 if remainder == 100 else remainder


def snils_ok(digits: str) -> bool:
    """An 11-digit SNILS: the sum of digit × (9 − position) over the first nine, the 100/101
    rule, against the last two."""
    if not digits.isdigit() or len(digits) != 11:
        return False
    return _snils_control(digits[:9]) == int(digits[9:])


def ogrn_ok(digits: str) -> bool:
    """A 13-digit OGRN (the first 12 mod 11) or a 15-digit OGRNIP (the first 14 mod 13), the
    result's last digit against the final one."""
    if not digits.isdigit():
        return False
    if len(digits) == 13:
        return int(digits[:12]) % 11 % 10 == int(digits[12])
    if len(digits) == 15:
        return int(digits[:14]) % 13 % 10 == int(digits[14])
    return False


def is_epoch_like(digits: str, limits: GateLimits) -> bool:
    """Whether *digits* parse as epoch seconds or milliseconds inside ``epoch_years``."""
    if not digits.isdigit():
        return False
    low, high = limits.epoch_years
    start = datetime(low, 1, 1, tzinfo=UTC).timestamp()
    end = datetime(high + 1, 1, 1, tzinfo=UTC).timestamp()
    value = int(digits)
    return start <= value < end or start * 1000 <= value < end * 1000


def bin_matches(digits: str, bins: Sequence[str]) -> bool:
    """Whether *digits* start with a listed BIN prefix, or one inside a listed ``a-b`` range."""
    for entry in bins:
        low, _, high = entry.partition("-")
        low, high = low.strip(), (high.strip() or low.strip())
        width = len(low)
        if len(high) != width or len(digits) < width or not digits[:width].isdigit():
            continue
        if int(low) <= int(digits[:width]) <= int(high):
            return True
    return False


def _is_json_number(text: str, start: int, end: int) -> bool:
    """A bare number in the value position of a JSON member."""
    if start > 0 and text[start - 1] == '"':
        return False
    before = text[:start].rstrip()
    after = text[end:].lstrip()
    return before.endswith(":") and (after == "" or after[0] in ",}]\n")


def _same_line_context(text: str, start: int, end: int, chars: int) -> tuple[str, str]:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end == -1 else line_end
    return text[max(line_start, start - chars) : start], text[end : min(line_end, end + chars)]


def _gated_in(rule: Rule, text: str, start: int, end: int, limits: GateLimits) -> bool:
    """The keyword comes first (D27, N2): a number beside its keyword is an identifier whatever
    its shape; only a keyword-less card reaching the BIN gate is dropped when it is a JSON
    number or an epoch timestamp; any other keyword-less number matches nothing."""
    before, after = _same_line_context(text, start, end, limits.context_chars)
    if rule.keywords is not None and (rule.keywords.search(before) or rule.keywords.search(after)):
        return True
    if not rule.bin_gate:
        return False
    digits = _digits(text[start:end])
    if not bin_matches(digits, limits.card_bins):
        return False
    return not (_is_json_number(text, start, end) or is_epoch_like(digits, limits))


def matches(rule: Rule, text: str, limits: GateLimits) -> list[tuple[int, int]]:
    """The (start, end) spans *rule* applies to in *text*, after its own checks."""
    found: list[tuple[int, int]] = []
    for match in rule.pattern.finditer(text):
        span = rule.value_span(match, limits)
        if span is None:
            continue
        start, end = span
        value = text[start:end]
        checked = _digits(value) if rule.kind == "identifier" else value
        if rule.check is not None and not rule.check(checked):
            continue
        if any(pattern.fullmatch(value) for pattern in rule.exempt):
            continue
        if rule.name == "entropy" and text[:start].endswith(ENTROPY_EXEMPT_PREFIXES):
            continue
        if rule.kind == "identifier" and not _gated_in(rule, text, start, end, limits):
            continue
        found.append((start, end))
    return found


# --- fixtures, assembled from parts on each call -----------------------------------------------


def _with_inn_control(nine: str) -> str:
    return nine + str(_inn_control(nine, _INN_10))


def _with_snils_control(nine: str) -> str:
    return f"{nine[:3]}-{nine[3:6]}-{nine[6:]} {_snils_control(nine):02d}"


def _with_ogrn_control(twelve: str) -> str:
    return twelve + str(int(twelve) % 11 % 10)


def _with_luhn_control(body: str) -> str:
    for last in "0123456789":
        if luhn_ok(body + last):
            return body + last
    raise AssertionError("unreachable: one of ten digits completes a Luhn number")


def _keywords(names: Sequence[str]) -> re.Pattern[str]:
    """Anchored to a word's start only (``(?<!\\w)``): "INN" inside "dinner" or "card" inside
    "discard" must not gate a number. Not anchored on the right, so a stem like "карт" still
    matches the start of a longer word ("карта")."""
    return re.compile(
        "(?<!\\w)(?:" + "|".join(re.escape(name) for name in names) + ")", re.IGNORECASE
    )


_PROVIDER_PATTERNS: tuple[tuple[str, str, Callable[[], str]], ...] = (
    ("aws_access_key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", lambda: "AKIA" + "A" * 16),
    (
        "github_token",
        r"\bgh[pousr]_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{22,}\b",
        lambda: "gh" + "p_" + "A" * 36,
    ),
    (
        "gitlab_token",
        r"\bgl(?:pat|rt|ptt|dt|cbt|ft|oas)-[A-Za-z0-9_-]{20,}",
        lambda: "gl" + "pat-" + "A" * 20,
    ),
    ("slack_token", r"\bxox[abposr]-[0-9A-Za-z-]{10,}", lambda: "xox" + "b-" + "1" * 12),
    (
        "slack_webhook",
        r"hooks\.slack\.com/services/T[A-Za-z0-9]+/B[A-Za-z0-9]+/[A-Za-z0-9]+",
        lambda: "hooks.slack" + ".com/services/T" + "A" * 8 + "/B" + "A" * 8 + "/" + "A" * 24,
    ),
    ("anthropic_key", r"\bsk-ant-[A-Za-z0-9_-]{20,}", lambda: "sk-" + "ant-" + "A" * 24),
    (
        "openai_key",
        r"\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{20,}",
        lambda: "sk-" + "proj-" + "A" * 24,
    ),
    ("google_key", r"\bAIza[0-9A-Za-z_-]{35}\b", lambda: "AIza" + "A" * 35),
    ("stripe_key", r"\b[sr]k_live_[0-9a-zA-Z]{24,}", lambda: "sk_" + "live_" + "A" * 24),
    ("telegram_bot", r"\b[0-9]{8,10}:[A-Za-z0-9_-]{35}\b", lambda: "1" * 9 + ":" + "A" * 35),
    (
        "jwt",
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
        lambda: "eyJ" + "A" * 12 + "." + "B" * 12 + "." + "C" * 12,
    ),
    (
        "private_key",
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"
        r"(?:.|\n)*?(?:-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----|\Z)",
        lambda: (
            "-----BEGIN " + "PRIVATE KEY-----\n" + "A" * 40 + "\n-----END " + "PRIVATE KEY-----"
        ),
    ),
    ("huggingface_token", r"\bhf_[A-Za-z0-9]{30,}", lambda: "hf_" + "A" * 30),
    ("vault_token", r"\bhvs\.[A-Za-z0-9_-]{20,}", lambda: "hvs." + "A" * 20),
    ("tailscale_key", r"\btskey-[a-z]+-[A-Za-z0-9-]{10,}", lambda: "tskey-" + "auth-" + "A" * 12),
    ("npm_token", r"\bnpm_[A-Za-z0-9]{36}\b", lambda: "npm_" + "A" * 36),
    (
        "sendgrid_key",
        r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}",
        lambda: "SG." + "A" * 22 + "." + "B" * 43,
    ),
    ("docker_pat", r"\bdckr_pat_[A-Za-z0-9_-]{20,}", lambda: "dckr_" + "pat_" + "A" * 20),
    ("azure_storage_key", r"AccountKey=[A-Za-z0-9+/=]{80,}", lambda: "AccountKey=" + "A" * 80),
    ("grafana_token", r"\bgl(?:sa|c)_[A-Za-z0-9_=-]{32,}", lambda: "gl" + "sa_" + "A" * 32),
    ("yandex_token", r"\b(?:y[0-3]_|AQVN)[A-Za-z0-9_-]{30,}", lambda: "y0_" + "A" * 30),
)

#: The provider rules' names: what ``gate_provider_hits`` and an import's count are made of.
PROVIDER_RULE_NAMES: frozenset[str] = frozenset(name for name, _, _ in _PROVIDER_PATTERNS)


def _identifier(
    name: str,
    pattern: str,
    *,
    check: Callable[[str], bool],
    effect: Effect,
    fixture: Callable[[], str],
    limits: GateLimits,
    bin_gate: bool = False,
) -> Rule:
    return Rule(
        name=name,
        kind="identifier",
        pattern=re.compile(pattern),
        effect=effect,
        value_span=_whole,
        fixture=fixture,
        keywords=_keywords(limits.keywords[name]),
        check=check,
        bin_gate=bin_gate,
    )


def rules(limits: GateLimits) -> tuple[Rule, ...]:
    """The 31 rules in scan order: the provider rules, ``anthropic_key`` before ``openai_key``;
    the four generic rules; the six Russian identifiers."""
    providers = tuple(
        Rule(
            name=name,
            kind="provider",
            pattern=re.compile(pattern),
            effect="redact",
            value_span=_whole,
            fixture=fixture,
        )
        for name, pattern, fixture in _PROVIDER_PATTERNS
    )
    entropy_pattern = re.compile(rf"[A-Za-z0-9+/=_\-]{{{limits.entropy_min_length},}}")
    generic = (
        Rule(
            name="assignment",
            kind="generic",
            pattern=ASSIGNMENT_RE,
            effect="redact",
            value_span=_assignment_span,
            # "=" sits in the entropy token class, so a "KEY=value" fixture is one 24-char
            # token at ~4.33 bits/char -- above the 4.2 threshold; ":" does not. The "=" form
            # stays covered by SECRET_FORMS' DB_PASSWORD= entry in the test.
            fixture=lambda: "DB_PASSWORD: " + "s3cr" + "3tValue9",
        ),
        Rule(
            name="bearer",
            kind="generic",
            pattern=re.compile(r"(?i)\bbearer\s+([A-Za-z0-9_.\-]{20,})"),
            effect="redact",
            value_span=_group_one,
            fixture=lambda: "Bearer " + "A" * 24,
        ),
        Rule(
            name="url_userinfo",
            kind="generic",
            pattern=re.compile(r"://[^/\s:@]+:([^/\s@]+)@"),
            effect="redact",
            value_span=_group_one,
            fixture=lambda: "https://user:" + "p4ssw" + "0rd@example.invalid/x",
        ),
        Rule(
            name="entropy",
            kind="generic",
            pattern=entropy_pattern,
            effect="redact",
            value_span=_whole,
            check=lambda token: entropy_bits(token) >= limits.entropy_threshold,
            exempt=_ENTROPY_EXEMPT,
            fixture=lambda: "abcdefghijkl" + "mnopqrstuvwx" + "yz0123456789" + "+/ABCDEF",
        ),
    )
    inn_keyword = limits.keywords["inn"][0]
    snils_keyword = limits.keywords["snils"][0]
    ogrn_keyword = limits.keywords["ogrn"][0]
    passport_keyword = limits.keywords["passport_rf"][0]
    phone_keyword = limits.keywords["phone_rf"][0]
    identifiers = (
        _identifier(
            "inn",
            r"(?<!\d)(?:\d{12}|\d{10})(?!\d)",
            check=inn_ok,
            effect="redact",
            fixture=lambda: f"{inn_keyword} " + _with_inn_control("00" + "00" + "12345"),
            limits=limits,
        ),
        _identifier(
            "snils",
            r"\b\d{3}-\d{3}-\d{3} \d{2}\b|(?<!\d)\d{11}(?!\d)",
            check=snils_ok,
            effect="redact",
            fixture=lambda: f"{snils_keyword} " + _with_snils_control("112" + "233445"),
            limits=limits,
        ),
        _identifier(
            "ogrn",
            r"(?<!\d)(?:\d{15}|\d{13})(?!\d)",
            check=ogrn_ok,
            effect="redact",
            fixture=lambda: (
                f"{ogrn_keyword} " + _with_ogrn_control("1" + "15" + "00" + "00" + "00001")
            ),
            limits=limits,
        ),
        _identifier(
            "bank_card",
            r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)",
            check=luhn_ok,
            effect="redact",
            fixture=lambda: "card " + _with_luhn_control("4123" + "45678901234"),
            limits=limits,
            bin_gate=True,
        ),
        _identifier(
            "passport_rf",
            r"\b\d{2} ?\d{2} ?\d{6}\b",
            check=lambda digits: len(digits) == 10,
            effect="flag",
            fixture=lambda: f"{passport_keyword} " + "00 00 " + "123456",
            limits=limits,
        ),
        _identifier(
            "phone_rf",
            r"(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}",
            check=lambda digits: len(digits) == 11,
            effect="flag",
            fixture=lambda: f"{phone_keyword}. " + "+7 000 " + "123-45-67",
            limits=limits,
        ),
    )
    return providers + generic + identifiers


#: The rules' names in scan order, read from the defaults: the names do not vary with the limits.
RULE_NAMES: tuple[str, ...] = tuple(rule.name for rule in rules(GateLimits.defaults()))
