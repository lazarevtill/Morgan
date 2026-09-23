"""The scanner applies the rules under a verdict. Decoded JSON values are scanned so a token after
a newline is found, for a document that is exactly a tool call's canonical text, through marker
characters that document does not already hold; every position lands on the stored text; a long
text is scanned in overlapping windows, a match longer than the overlap still whole and one longer
than a window continued into one span; a token at the write path's truncation boundary is whole
because the scan runs on the whole text; a placeholder already in a text -- including the
scanner's own, from a tool call's decoded pass -- is guarded rather than re-scanned; a flag-only
hit found in a tool call's decoded pass is still carried to the stored text; nothing the scanner
produces -- an exception, a result, a log line -- carries a value; neither the scanner nor these
tests hold a raw private-use character.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
import time
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError
from structlog.testing import capture_logs

import morgan_brain.memory.secrets.scan as scan_module
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
#: The first five private-use code points, U+E000 to U+E004: the marker characters the decoded
#: pass takes from a document that holds none of them.
FIRST_FIVE = "\ue000\ue001\ue002\ue003\ue004"


def _scanner(**overrides: int) -> Scanner:
    return Scanner(replace(LIMITS, **overrides))


def _github_token() -> str:
    return "gh" + "p_" + MARKER + "A" * (36 - len(MARKER))


def _placeholders(result: ScanResult) -> list[str]:
    return [result.text[h.start : h.start + h.length] for h in result.redactions]


def _seeded_run(length: int, seed: str) -> str:
    """A reproducible run of letters and digits, high in entropy: SHA-256 over the seed and a
    counter, each byte taken onto the 62 characters."""
    alphabet = string.ascii_letters + string.digits
    chars: list[str] = []
    counter = 0
    while len(chars) < length:
        digest = hashlib.sha256(f"{seed}-{counter}".encode()).digest()
        chars.extend(alphabet[byte % len(alphabet)] for byte in digest)
        counter += 1
    return "".join(chars[:length])


def _long_private_key(body_chars: int) -> str:
    """The private-key rule's own fixture with its body line repeated until the body holds at
    least *body_chars* characters."""
    head, body, foot = RULES["private_key"].fixture().split("\n")
    lines = -(-body_chars // (len(body) + 1))
    return "\n".join([head, *[body] * lines, foot])


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


def test_an_entropy_run_longer_than_a_window_is_redacted_whole_as_one_span():
    """A match that fills its window while the text goes on is continued: its rule is matched
    again from the span's start over the text that follows until the match ends, and the whole
    run is one redaction, with no character of it left in the stored text -- the result an
    unwindowed scan gives."""
    scanner = _scanner()
    size = LIMITS.window_chars
    run = _seeded_run(size + size // 10, "entropy")
    text = "before " + run + " after"
    result = scanner.scan(text, verdict="redact")
    assert result.text == "before [redacted:entropy] after"
    assert [(h.rule, h.start) for h in result.redactions] == [("entropy", len("before "))]
    assert result == _scanner(window_chars=len(text) + 1).scan(text, verdict="redact")


def test_a_private_key_block_longer_than_a_window_is_redacted_whole():
    """The block's pattern ends at its END line, or at the end of the text it is given: cut at a
    window's end, it would match up to that edge and leave the rest of the body and the END line
    in clear. Continued, the whole block is one redaction."""
    scanner = _scanner(window_chars=1024, window_overlap_chars=64)
    key = _long_private_key(1_500)
    assert len(key) > 1024
    text = "key file:\n" + key + "\nend of file " + "z " * 400
    result = scanner.scan(text, verdict="redact")
    assert result.text == text.replace(key, "[redacted:private_key]")
    assert [(h.rule, h.start) for h in result.redactions] == [("private_key", text.index(key))]
    assert result == _scanner(window_chars=len(text) + 1).scan(text, verdict="redact")


def test_a_long_match_its_rule_rejects_when_seen_whole_gives_way_to_the_span_it_displaced():
    """Cut at a window's edge, the provider-prefixed run below ends on a word boundary and the
    provider rule takes the whole window, displacing the entropy span over the same run. Seen
    whole, the run ends in ``_x`` and has no word boundary to end on, so the provider rule makes
    no match there: the window is resolved again without it, the entropy span it displaced comes
    back and is continued, and the run is redacted whole -- the result an unwindowed scan
    gives."""
    scanner = _scanner()
    size = LIMITS.window_chars
    run = "gh" + "p_" + _seeded_run(size + size // 10, "rejected") + "_x"
    text = "before " + run + " after"
    result = scanner.scan(text, verdict="redact")
    assert result.text == "before [redacted:entropy] after"
    assert result.provider_hits == 0
    assert result == _scanner(window_chars=len(text) + 1).scan(text, verdict="redact")


def test_the_refusal_of_a_private_key_longer_than_a_window_names_its_start_and_whole_length():
    scanner = _scanner(window_chars=1024, window_overlap_chars=64)
    key = _long_private_key(1_500)
    text = "key file:\n" + key + "\nend of file " + "z " * 400
    with pytest.raises(SecretRefused) as raised:
        scanner.scan(text, verdict="refuse")
    exc = raised.value
    assert (exc.rule, exc.start, exc.length) == ("private_key", text.index(key), len(key))


def test_ten_windows_of_runs_each_longer_than_a_window_are_all_redacted_in_bounded_time():
    """A generous bound, not a benchmark: a continuation re-matches only the rule that reached
    the window's edge, over a reach that doubles, so the work stays linear in the text."""
    scanner = _scanner()
    size = LIMITS.window_chars
    runs: list[str] = []
    while sum(len(run) + 1 for run in runs) < 10 * size:
        runs.append(_seeded_run(size + size // 10, f"run-{len(runs)}"))
    text = " ".join(runs)
    started = time.perf_counter()
    result = scanner.scan(text, verdict="redact")
    elapsed = time.perf_counter() - started
    assert result.text == " ".join(["[redacted:entropy]"] * len(runs))
    assert elapsed < 10


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


def test_a_tool_calls_argument_holding_a_redaction_shaped_marker_is_kept_and_never_read_as_one():
    """A value holding U+E000, a digit and U+E001 -- the shape of a redaction marker whose index
    is the real hit's -- is text, not a marker: the decoded pass writes markers the document does
    not hold, so the value is stored as it was, and the token beside it is redacted once."""
    scanner = _scanner()
    token = _github_token()
    text = tool_call_text("Bash", {"command": FIRST_FIVE[0] + "0" + FIRST_FIVE[1] + " " + token})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.redacted_rules() == ["github_token"]
    assert len(result.redactions) == 1
    assert result.provider_hits == 1
    assert MARKER not in result.text
    assert result.text == text.replace(token, "[redacted:github_token]")


@pytest.mark.parametrize(
    "shape",
    [FIRST_FIVE[2] + "7" + FIRST_FIVE[3], FIRST_FIVE[4]],
    ids=["flag-open-and-close-with-an-unwritten-index", "flag-end-with-nothing-open"],
)
def test_a_tool_calls_argument_holding_flag_shaped_markers_is_kept_and_never_read_as_one(shape):
    """Read as markers, an open/close pair would look up an index the pass never wrote, and a
    bare flag end would close a flag never opened. The decoded pass writes markers the document
    does not hold, so neither is read as one: the value is stored as it was and the token beside
    it is redacted."""
    scanner = _scanner()
    token = _github_token()
    text = tool_call_text("Bash", {"command": f"weird{shape}data {token}"})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.redacted_rules() == ["github_token"]
    assert MARKER not in result.text
    assert result.text == text.replace(token, "[redacted:github_token]")


#: A tool call holding U+E000 to U+E004 in each place a document's text can hold them, with a
#: token right after an escaped newline in a value.
_HOLDING_THE_FIRST_FIVE = {
    "in-a-value": lambda token: tool_call_text("Bash", {"command": FIRST_FIVE + "\n" + token}),
    "in-a-key": lambda token: tool_call_text(
        "Bash", {"command": "echo start\n" + token, FIRST_FIVE: "x"}
    ),
    "in-the-tool-name": lambda token: tool_call_text(
        "Bash" + FIRST_FIVE, {"command": "echo start\n" + token}
    ),
}


@pytest.mark.parametrize("where", list(_HOLDING_THE_FIRST_FIVE), ids=list(_HOLDING_THE_FIRST_FIVE))
def test_a_tool_call_already_holding_the_first_five_markers_still_takes_the_decoded_pass(where):
    """The decoded pass's five marker characters are chosen per document: the first five
    private-use code points it does not hold. A document already holding U+E000 to U+E004 -- in
    a value, a key or the tool's name -- gets the next five, so a token right after an escaped
    newline is still found on the decoded value, and the document's own characters are stored
    as they were."""
    scanner = _scanner()
    token = RULES["github_token"].fixture()
    text = _HOLDING_THE_FIRST_FIVE[where](token)
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.text == text.replace(token, "[redacted:github_token]")
    assert [(h.rule, result.text[h.start : h.start + h.length]) for h in result.redactions] == [
        ("github_token", "[redacted:github_token]")
    ]
    assert result.provider_hits == 1


def test_a_flag_only_hit_is_carried_when_the_document_already_holds_the_first_five_markers():
    scanner = _scanner()
    passport = RULES["passport_rf"].fixture()
    digits = passport.split(" ", 1)[1]
    text = tool_call_text("Bash", {"command": FIRST_FIVE + "\n" + passport})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.text == text
    assert result.redactions == ()
    assert [(h.rule, result.text[h.start : h.start + h.length]) for h in result.flags] == [
        ("passport_rf", digits)
    ]


def test_a_document_leaving_exactly_five_private_use_code_points_free_takes_the_decoded_pass():
    """U+E000 to U+F8FF holds 6,400 code points: a document holding 6,395 of them leaves five,
    and the decoded pass writes those."""
    scanner = _scanner()
    held = "".join(map(chr, range(0xE000, 0xF8FB)))
    assert len(held) == 6_395
    token = RULES["github_token"].fixture()
    text = tool_call_text("Write", {"content": held + "\n" + token})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.text == text.replace(token, "[redacted:github_token]")
    assert result.provider_hits == 1


def test_a_document_leaving_fewer_than_five_private_use_code_points_free_is_scanned_as_text():
    """With four code points free there is no set of five markers: the turn is scanned as plain
    text, without error, and a token after a space is still redacted."""
    scanner = _scanner()
    held = "".join(map(chr, range(0xE000, 0xF8FC)))
    token = RULES["github_token"].fixture()
    text = tool_call_text("Write", {"content": held + " " + token})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.text == text.replace(token, "[redacted:github_token]")
    assert result.provider_hits == 1


# --- only a canonical tool call takes the decoded path -----------------------------------------


def test_a_tool_call_document_with_another_key_is_stored_with_that_key_intact():
    """Only the exact text ``tool_call_text`` writes takes the decoded path, which re-serialises
    the document: a document with a key beside ``input`` and ``tool`` is scanned as plain text,
    so nothing of it is dropped -- and a token in it is still redacted."""
    scanner = _scanner()
    token = RULES["github_token"].fixture()
    text = json.dumps(
        {"extra": "keep me", "input": {"command": "echo " + token}, "tool": "Bash"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.text == text.replace(token, "[redacted:github_token]")
    assert json.loads(result.text)["extra"] == "keep me"


def test_a_tool_call_document_whose_tool_is_not_a_string_is_stored_as_given():
    scanner = _scanner()
    text = json.dumps({"input": {"a": 1}, "tool": 5}, sort_keys=True, separators=(",", ":"))
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.text == text


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
    assert result.text == text
    assert result.redactions == ()
    assert len(result.flags) == 1
    flag = result.flags[0]
    assert flag.rule == "passport_rf"
    assert result.text[flag.start : flag.start + flag.length] == digits


def test_the_same_for_a_phone_number_after_an_escaped_newline():
    scanner = _scanner()
    phone = RULES["phone_rf"].fixture()
    value = phone.split(". ", 1)[1]
    text = tool_call_text("Bash", {"command": "echo start\n" + phone})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.text == text
    assert result.redactions == ()
    assert len(result.flags) == 1
    flag = result.flags[0]
    assert flag.rule == "phone_rf"
    assert result.text[flag.start : flag.start + flag.length] == value


def test_the_fixture_appearing_twice_in_one_decoded_value_gives_two_flags():
    """Each occurrence sits right after its own escaped newline, so the whole-text pass finds
    neither on its own (the keyword follows the ``n`` of ``\\n``); only the decoded pass, and
    only with both hits carried, finds both."""
    scanner = _scanner()
    passport = RULES["passport_rf"].fixture()
    digits = passport.split(" ", 1)[1]
    text = tool_call_text("Bash", {"command": f"echo a\n{passport}\n" + "x" * 50 + f"\n{passport}"})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.text == text
    assert [(h.rule, result.text[h.start : h.start + h.length]) for h in result.flags] == [
        ("passport_rf", digits),
        ("passport_rf", digits),
    ]
    first = text.index(digits)
    assert [h.start for h in result.flags] == [first, text.index(digits, first + 1)]


def test_the_fixture_after_a_space_gives_exactly_one_flag_not_two():
    """After a plain space the whole-text pass can also find this hit; the decoded pass's own
    flag and the whole-text pass's must not both be counted."""
    scanner = _scanner()
    passport = RULES["passport_rf"].fixture()
    digits = passport.split(" ", 1)[1]
    text = tool_call_text("Bash", {"command": f"note {passport} end"})
    result = scanner.scan_turn(text, role="tool_call", verdict="redact")
    assert result.text == text
    assert [(h.rule, result.text[h.start : h.start + h.length]) for h in result.flags] == [
        ("passport_rf", digits)
    ]


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
    assert len(result.flags) == 1 and len(result.redactions) == 1
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


def test_the_scanner_and_these_tests_hold_no_raw_private_use_character():
    """A private-use character is invisible in a diff or an editor: written raw, a marker reads
    as an empty string and the marker protocol cannot be reviewed. Every one is an escape or is
    computed from its code point."""
    private_use = re.compile("[" + chr(0xE000) + "-" + chr(0xF8FF) + "]")
    for path in (Path(scan_module.__file__), Path(__file__)):
        found = private_use.search(path.read_bytes().decode("utf-8"))
        assert found is None, f"{path.name} holds U+{ord(found.group()):04X}" if found else ""


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
