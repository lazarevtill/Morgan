"""The scanner applies the rules under a verdict. Decoded JSON values are scanned so a token after
a newline is found; every position lands on the stored text; a long text is scanned in overlapping
windows, a match longer than the overlap still whole; a token at the write path's truncation
boundary is whole because the scan runs on the whole text; a placeholder already in a text --
including the scanner's own, from a tool call's decoded pass -- is guarded rather than re-scanned;
a flag-only hit found in a tool call's decoded pass is still carried to the stored text; nothing
the scanner produces -- an exception, a result, a log line -- carries a value.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError
from structlog.testing import capture_logs

from morgan_brain.config import Settings
from morgan_brain.memory.secrets import (
    Scanner,
    ScanResult,
    SecretRefused,
    build_scanner,
    strip_userinfo,
)
from morgan_brain.memory.secrets.rules import GateLimits, rules
from morgan_brain.models import tool_call_text

LIMITS = GateLimits.defaults()
RULES = {rule.name: rule for rule in rules(LIMITS)}
#: A token in no fixture and no dictionary: the value that must appear nowhere.
MARKER = "zq" + "xj" + "vgatemarker"


def _scanner(**overrides: int) -> Scanner:
    return Scanner(replace(LIMITS, **overrides))


def _github_token() -> str:
    return "gh" + "p_" + MARKER + "A" * (36 - len(MARKER))


def _placeholders(result: ScanResult) -> list[str]:
    return [result.text[h.start : h.start + h.length] for h in result.redactions]


# --- the two verdicts --------------------------------------------------------------------------


def test_refuse_names_the_rule_offset_and_length_and_never_the_value():
    scanner = _scanner()
    token = _github_token()
    text = f"deploy with {token} tonight"
    with pytest.raises(SecretRefused) as raised:
        scanner.scan(text, verdict="refuse")
    exc = raised.value
    assert (exc.rule, exc.start, exc.length) == ("github_token", text.index(token), len(token))
    assert str(exc) == (
        f"refused: the text contains a github_token at offset {exc.start} ({len(token)} "
        "characters); remove it and try again"
    )
    assert MARKER not in str(exc) and MARKER not in repr(exc)


def test_redact_replaces_a_provider_token_and_counts_it():
    scanner = _scanner()
    token = _github_token()
    result = scanner.scan(f"deploy with {token} tonight", verdict="redact")
    assert result.text == "deploy with [redacted:github_token] tonight"
    assert result.provider_hits == 1
    assert result.redacted_rules() == ["github_token"]
    assert result.flagged_rules() == []
    assert result.text_verdict == "redacted"
    assert _placeholders(result) == ["[redacted:github_token]"]
    assert json.loads(result.redactions_json()) == [
        {"rule": "github_token", "start": 12, "length": len("[redacted:github_token]")}
    ]
    assert MARKER not in result.text and MARKER not in result.redactions_json()


def test_a_generic_hit_is_redacted_and_flagged_under_both_verdicts():
    scanner = _scanner()
    for verdict in ("refuse", "redact"):
        result = scanner.scan(f"DB_PASSWORD={MARKER}99 in the compose file", verdict=verdict)
        assert result.text == "DB_PASSWORD=[redacted:assignment] in the compose file"
        assert result.redacted_rules() == ["assignment"]
        assert result.flagged_rules() == ["assignment"]
        assert result.provider_hits == 0
        assert result.flags[0].start == result.redactions[0].start


def test_an_identifier_with_a_checksum_is_redacted_and_a_keyword_only_pattern_is_flagged():
    scanner = _scanner()
    inn = RULES["inn"].fixture()  # "<keyword> <digits with their control digit>"
    redacted = scanner.scan(f"клиент: {inn}", verdict="redact")
    assert redacted.text == f"клиент: {inn.split()[0]} [redacted:inn]"
    assert redacted.redacted_rules() == ["inn"] and redacted.flagged_rules() == ["inn"]
    flagged = scanner.scan("паспорт 00 00 123456 выдан", verdict="redact")
    assert flagged.text == "паспорт 00 00 123456 выдан"
    assert flagged.redactions == () and flagged.flagged_rules() == ["passport_rf"]
    assert flagged.text_verdict == "flagged"
    assert flagged.text[flagged.flags[0].start :][: flagged.flags[0].length] == "00 00 123456"


def test_a_clean_text_is_returned_as_it_was():
    result = _scanner().scan("the mirror blocked the deploy", verdict="refuse")
    assert (result.text, result.redactions, result.flags, result.provider_hits) == (
        "the mirror blocked the deploy",
        (),
        (),
        0,
    )
    assert result.text_verdict == "clean"


def test_a_provider_hit_is_never_also_an_entropy_hit():
    scanner = _scanner()
    token = "sk-" + "ant-" + "AbCdEfGhIjKlMnOpQrStUvWxYz012345"
    result = scanner.scan(f"key {token}", verdict="redact")
    assert result.redacted_rules() == ["anthropic_key"]
    assert result.text == "key [redacted:anthropic_key]"


# --- decoded JSON values -------------------------------------------------------------------------


def test_a_token_right_after_a_newline_in_a_tool_call_is_found_on_the_decoded_value():
    scanner = _scanner()
    token = _github_token()
    text = tool_call_text("Bash", {"command": "cat token\n" + token + "\necho done"})
    assert "\\n" + token in text  # JSON spells the newline as two characters: no word boundary
    plain = scanner.scan(text, verdict="redact")
    assert MARKER in plain.text  # the whole-text pass alone misses it
    turn = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert MARKER not in turn.text
    assert turn.redacted_rules() == ["github_token"]
    assert turn.provider_hits == 1
    assert json.loads(turn.text) == {
        "input": {"command": "cat token\n[redacted:github_token]\necho done"},
        "tool": "Bash",
    }
    assert _placeholders(turn) == ["[redacted:github_token]"]


def test_cyrillic_stays_cyrillic_in_a_tool_call_and_an_identifier_is_found_there():
    scanner = _scanner()
    inn = RULES["inn"].fixture()
    text = tool_call_text("Bash", {"command": f"echo {inn}", "note": "проверка"})
    assert "проверка" in text and "\\u" not in text
    turn = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert "[redacted:inn]" in turn.text and "проверка" in turn.text
    assert _placeholders(turn) == ["[redacted:inn]"]


def test_a_tool_result_and_a_text_that_does_not_decode_are_scanned_as_text():
    scanner = _scanner()
    token = _github_token()
    result = scanner.scan_turn(f"output:\n{token}", role="tool_result", verdict="redact")
    assert result.text == "output:\n[redacted:github_token]"
    not_json = scanner.scan_turn("{not json " + token, role="tool_call", verdict="redact")
    assert not_json.text == "{not json [redacted:github_token]"


# --- positions, windows, and the whole text --------------------------------------------------


def test_every_position_lands_on_the_stored_text():
    scanner = _scanner()
    one, two = _github_token(), "gh" + "o_" + "B" * 36
    text = f"a {one} b DB_PASSWORD={MARKER}77 c {two} d паспорт 00 00 123456"
    result = scanner.scan(text, verdict="redact")
    for hit in result.redactions:
        assert result.text[hit.start : hit.start + hit.length] == f"[redacted:{hit.rule}]"
    passport = [h for h in result.flags if h.rule == "passport_rf"]
    assert len(passport) == 1
    assert result.text[passport[0].start :][: passport[0].length] == "00 00 123456"
    assert result.text.count("[redacted:") == 3
    assert MARKER not in result.text


def test_a_long_text_is_scanned_in_overlapping_windows_and_a_straddling_token_is_found_once():
    scanner = _scanner(window_chars=200, window_overlap_chars=80)
    token = _github_token()
    text = "x" * 170 + " " + token + " " + "y" * 400
    result = scanner.scan(text, verdict="redact")
    assert result.text.count("[redacted:github_token]") == 1
    assert result.provider_hits == 1
    assert MARKER not in result.text


def test_a_match_longer_than_the_overlap_straddling_a_boundary_is_seen_whole():
    """A PEM private key (longer than the overlap) placed so that it is truncated in one window
    and would be skipped past entirely by a naive fixed-step next window: the next window must
    start at the carried span's own start, not merely the usual overlap before the boundary."""
    scanner = _scanner(window_chars=200, window_overlap_chars=80)
    key = RULES["private_key"].fixture()
    text = "x" * 115 + key + "y" * 300
    result = scanner.scan(text, verdict="redact")
    assert result.text.count("[redacted:private_key]") == 1
    assert "-----BEGIN" not in result.text
    assert "A" * 40 not in result.text
    assert result.provider_hits == 1


def test_a_token_at_the_truncation_boundary_is_whole_because_the_scan_precedes_the_cap():
    """The write path caps a turn at 6,000 + 2,000 characters after it is scanned; the scanner
    itself sees the whole text, so a token at the boundary is found before any truncation."""
    scanner = _scanner()
    token = _github_token()
    text = "t" * 5_990 + " " + token + " " + "u" * 3_000
    result = scanner.scan(text, verdict="redact")
    assert result.provider_hits == 1 and MARKER not in result.text


# --- placeholders already in the text: guarded, not re-scanned --------------------------------


def test_scanning_a_redacted_text_again_leaves_it_unchanged_and_finds_nothing():
    """The redacted text is ``DB_PASSWORD: [redacted:assignment]``: without the guard, the
    ``assignment`` rule reads ``[redacted:assignment`` (no closing bracket) as a new value on a
    second pass, corrupting the text and losing the record of the first redaction."""
    scanner = _scanner()
    first = scanner.scan(RULES["assignment"].fixture(), verdict="redact")
    second = scanner.scan(first.text, verdict="redact")
    assert second.text == first.text
    assert second.redactions == () and second.flags == ()
    assert second.text_verdict == "clean"


def test_a_bracketed_span_that_is_not_a_real_rule_name_is_not_a_guard_and_is_still_redacted():
    """The guard is the exact alternation of the gate's own rule names, not any ``[a-z_]+``
    span: an all-lowercase token wrapped in brackets would fully match a loose ``[a-z_]+``
    guard and be skipped by mistake; the exact-name guard does not mistake it for one of its
    own placeholders, and the provider token inside is still found."""
    scanner = _scanner()
    token = RULES["github_token"].fixture().lower()
    result = scanner.scan(f"[redacted:{token}]", verdict="redact")
    assert result.redacted_rules() == ["github_token"]
    assert result.text == "[redacted:[redacted:github_token]]"


def test_a_tool_calls_provider_token_beside_a_key_that_also_reads_as_an_assignment():
    """After the decoded pass redacts the token, the stored text reads
    ``GITLAB_TOKEN=[redacted:gitlab_token]``, which itself looks like ``KEY=value`` to the
    ``assignment`` rule. Without the placeholder guard, the whole-text pass reads
    ``[redacted:gitlab_token`` (no closing bracket) as a new assignment value and renames the
    hit, corrupting the text with a stray ``]``; guarded, the placeholder is left alone and the
    redaction still names only the provider rule that made it."""
    scanner = _scanner()
    token = RULES["gitlab_token"].fixture()
    text = tool_call_text("Bash", {"command": f"export GITLAB_TOKEN={token} && make"})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert json.loads(result.text) == {
        "input": {"command": "export GITLAB_TOKEN=[redacted:gitlab_token] && make"},
        "tool": "Bash",
    }
    assert result.redacted_rules() == ["gitlab_token"]
    assert result.provider_hits == 1


def test_a_placeholder_already_in_a_tool_calls_argument_is_neither_rescanned_nor_counted():
    scanner = _scanner()
    text = tool_call_text("Write", {"content": "see [redacted:github_token] in the log"})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.redactions == ()
    assert result.text_verdict != "redacted"
    assert json.loads(result.text) == {
        "input": {"content": "see [redacted:github_token] in the log"},
        "tool": "Write",
    }


def test_a_tool_calls_argument_already_holding_a_redaction_sentinel_is_scanned_as_plain_text():
    """A decoded value already holding what looks like the scanner's own redaction marker
    falls back to a plain-text scan of the whole turn: without the fallback, the pre-existing
    marker is itself read back as a second, spurious hit at the reconciliation step (its index
    coincides with the real hit's), doubling the redaction count and consuming the raw marker
    text; with it, the raw marker characters are untouched, ordinary text and only the token
    is redacted."""
    scanner = _scanner()
    token = _github_token()
    text = tool_call_text("Bash", {"command": "0 " + token})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.redacted_rules() == ["github_token"]
    assert len(result.redactions) == 1
    assert result.provider_hits == 1
    assert MARKER not in result.text
    assert "" in result.text  # the pre-existing marker survives, untouched, as plain text


@pytest.mark.parametrize(
    "marker",
    [
        "7",  # a complete fake flag-open/close pair, index unrelated to any hit
        "",  # a bare flag-close marker with nothing open
    ],
    ids=["fake-open-close-pair", "bare-close-marker"],
)
def test_a_tool_calls_argument_already_holding_a_flag_marker_is_scanned_as_plain_text(marker):
    """Without the fallback, the outer parser reads the pre-existing marker as if the decoded
    pass had written it: a fake open/close pair looks up an out-of-range index in ``written``,
    and a bare close marker pops from an empty ``opened`` list -- both raise. The fallback
    avoids both by never running the marker-writing pass at all."""
    scanner = _scanner()
    token = _github_token()
    text = tool_call_text("Bash", {"command": f"weird{marker}data {token}"})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.redacted_rules() == ["github_token"]
    assert MARKER not in result.text


# --- a flag-only hit in a tool call's decoded pass is carried to the stored text --------------


def test_a_flag_only_hit_after_an_escaped_newline_in_a_tool_call_is_carried_to_the_stored_text():
    """The whole-text pass alone cannot find this: the keyword follows the ``n`` of the escaped
    ``\\n``, which is a word character, so it never gates. Only the decoded pass, reading the
    real newline, finds it -- and that hit must survive into the stored text."""
    scanner = _scanner()
    passport = RULES["passport_rf"].fixture()
    digits = passport.split(" ", 1)[1]
    text = tool_call_text("Bash", {"command": "echo start\n" + passport})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.redactions == ()
    assert result.flagged_rules() == ["passport_rf"]
    assert json.loads(result.text) == {
        "input": {"command": "echo start\n" + passport},
        "tool": "Bash",
    }
    flag = result.flags[0]
    assert result.text[flag.start : flag.start + flag.length] == digits


def test_the_same_for_a_phone_number_after_an_escaped_newline():
    scanner = _scanner()
    phone = RULES["phone_rf"].fixture()
    value = phone.split(". ", 1)[1]
    text = tool_call_text("Bash", {"command": "echo start\n" + phone})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.redactions == ()
    assert result.flagged_rules() == ["phone_rf"]
    flag = result.flags[0]
    assert result.text[flag.start : flag.start + flag.length] == value


def test_the_fixture_appearing_twice_in_one_decoded_value_gives_two_flags():
    """Each occurrence sits right after its own escaped newline, so the whole-text pass finds
    neither on its own (the keyword follows the ``n`` of ``\\n``); only the decoded pass, and
    only with both hits carried, finds both."""
    scanner = _scanner()
    passport = RULES["passport_rf"].fixture()
    text = tool_call_text("Bash", {"command": f"echo a\n{passport}\n" + "x" * 50 + f"\n{passport}"})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert len(result.flags) == 2
    assert result.flagged_rules() == ["passport_rf"]


def test_the_fixture_after_a_space_gives_exactly_one_flag_not_two():
    """After a plain space the whole-text pass can also find this hit; the decoded pass's own
    flag and the whole-text pass's must not both be counted."""
    scanner = _scanner()
    passport = RULES["passport_rf"].fixture()
    text = tool_call_text("Bash", {"command": f"note {passport} end"})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert len(result.flags) == 1
    assert result.flagged_rules() == ["passport_rf"]


def test_a_decoded_value_with_both_a_flag_and_a_redaction_records_both_correctly():
    """The passport sits right after an escaped newline, so only the decoded pass finds it; the
    token follows on the same line, in reach of the whole-text pass either way."""
    scanner = _scanner()
    passport = RULES["passport_rf"].fixture()
    digits = passport.split(" ", 1)[1]
    token = _github_token()
    text = tool_call_text("Bash", {"command": f"{token}\n{passport}"})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.flagged_rules() == ["passport_rf"]
    assert result.redacted_rules() == ["github_token"]
    assert result.provider_hits == 1
    flag = result.flags[0]
    assert result.text[flag.start : flag.start + flag.length] == digits
    redaction = result.redactions[0]
    assert (
        result.text[redaction.start : redaction.start + redaction.length]
        == "[redacted:github_token]"
    )


# --- nothing carries a value ------------------------------------------------------------------


def test_no_value_reaches_a_result_or_a_log_line():
    scanner = _scanner()
    with capture_logs() as logs:
        result = scanner.scan(f"secret {_github_token()}", verdict="redact")
        with pytest.raises(SecretRefused) as raised:
            scanner.scan(f"secret {_github_token()}", verdict="refuse")
    for piece in (result.text, result.redactions_json(), result.flags_json(), str(raised.value)):
        assert MARKER not in piece
    assert all(MARKER not in json.dumps(entry, default=str) for entry in logs)


# --- helpers and the settings ----------------------------------------------------------------


def test_strip_userinfo_drops_the_credential_and_keeps_the_rest():
    assert (
        strip_userinfo("https://user:" + MARKER + "@git.example/team/repo.git")
        == "https://git.example/team/repo.git"
    )
    assert strip_userinfo("git@git.example:team/repo.git") == "git@git.example:team/repo.git"
    assert (
        strip_userinfo("https://git.example/team/repo.git") == "https://git.example/team/repo.git"
    )


def test_build_scanner_reads_the_window_settings(monkeypatch):
    monkeypatch.setenv("MORGAN_GATE_WINDOW_CHARS", "2048")
    monkeypatch.setenv("MORGAN_GATE_WINDOW_OVERLAP_CHARS", "128")
    scanner = build_scanner(Settings())
    assert (scanner.limits.window_chars, scanner.limits.window_overlap_chars) == (2048, 128)
    assert build_scanner(Settings.model_construct()).limits == LIMITS


def test_an_overlap_at_or_above_the_window_refuses_at_load_naming_both_settings(monkeypatch):
    monkeypatch.setenv("MORGAN_GATE_WINDOW_CHARS", "1024")
    monkeypatch.setenv("MORGAN_GATE_WINDOW_OVERLAP_CHARS", "1024")
    with pytest.raises(ValidationError) as excinfo:
        Settings()
    assert "gate_window_chars" in str(excinfo.value)
    assert "gate_window_overlap_chars" in str(excinfo.value)
