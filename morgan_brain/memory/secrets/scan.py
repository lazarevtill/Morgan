"""The scanner: the rules applied to a text, under a verdict.

Two verdicts, by who can still act. *refuse* is for text the caller can rephrase: a provider hit
raises ``SecretRefused`` by rule, offset and length, and nothing is stored or sent. *redact* is
for history nobody can rephrase: a provider hit is replaced by ``[redacted:<rule>]`` and counted.
Under both, a generic hit is redacted and flagged, an identifier with its checksum and its
keyword is redacted and flagged, and a keyword-gated pattern without a checksum (``passport_rf``,
``phone_rf``) is flagged with the text kept.

The rules run in order (``rules.rules``): a span an earlier rule took is not offered to a later
one, which is how a provider token is never also an entropy hit -- and a placeholder already in
the text, a replayed row or a quoted redaction among them, is guarded the same way, so a text
scanned twice keeps its redaction and its record rather than being corrupted by a second pass.
A text longer than ``window_chars`` is scanned in windows overlapping by
``window_overlap_chars``. Each window also reads that much of the text before its start, so the
keyword or prefix before a span -- ``Bearer``, an identifier's keyword, a digest's ``sha256:``
-- is read with it, and takes only the spans that start inside it. A hit that ends inside its
window is recorded, and the overlaps between recorded spans are resolved once, over the whole
text, as an unwindowed scan resolves them. A rule's check is judged on a whole run, never on one
a window's edge cuts: a span that edge cuts -- a hit, or a run its rule's pattern matches to the
edge whatever its cut's check says -- is undecided, takes no hit's place, and is carried into
the next window, which starts at the span's own start (or the usual overlap, whichever is
earlier), so a match longer than the overlap is still seen whole. A span that fills a whole
window from its start while the text goes on is continued: its rule's pattern alone is matched
again from that start over a reach that doubles, until the run ends inside the reach or the text
ends, and only then is the rule's check judged. A run the rule accepts is one span, nothing of
it left in clear, and scanning resumes the overlap before its end, or earlier, at a cut span of
an earlier rule inside it. A run the rule rejects seen whole is dropped, as an unwindowed scan
drops it, and that rule is offered nothing more inside it, so no later window reads it again.

A ``tool_call`` turn's text is one JSON document (``models.tool_call_text``). When the text is
exactly what ``tool_call_text`` writes for the document it decodes to, its string values are
scanned decoded first -- so a word boundary holds after a newline that JSON spells as two
characters -- each hit written as a marker (a redaction as one, a flag-only hit as its kept text
between two others), the document re-serialised, every marker replaced or removed at its
position on the stored text, and the whole text scanned once more with every placeholder guarded
and every placed span excluded so nothing counts twice. The five marker characters are chosen
per document, the first five private-use code points it does not hold, so a marker found in the
stored text is always one this pass wrote. Any other text -- one that does not decode, a
document with another key or a tool that is not a string, one serialised another way -- is
scanned as plain text and keeps its shape, and so is a document that leaves fewer than five
private-use code points free. Capture scans turns under ``redact``, where every recorded
position is on the stored text; under ``refuse`` a hit inside a decoded value is reported at its
offset within that value, since the caller rephrases before anything is stored. Nothing here
carries a value out: not the exception, not a log line, not a result.
"""

from __future__ import annotations

import bisect
import json
import re
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import islice, pairwise
from typing import Any, Literal

from morgan_brain.config import Settings
from morgan_brain.memory.secrets.rules import (
    RULE_NAMES,
    Effect,
    GateLimits,
    Rule,
    RuleKind,
    limits_of,
    matches,
    rules,
)
from morgan_brain.models import tool_call_text

Verdict = Literal["refuse", "redact"]
TextVerdict = Literal["clean", "flagged", "redacted"]

#: What stands in the stored text for a redacted span.
REDACTION = "[redacted:{rule}]"
#: The placeholders the gate itself writes, and only those: built from the rule names at import,
#: so a guard never covers bracketed text the gate did not write.
_PLACEHOLDER_RE = re.compile(
    r"\[redacted:(?:" + "|".join(re.escape(name) for name in RULE_NAMES) + r")\]"
)
#: The Private Use Area of the Basic Multilingual Plane, where a tool call's decoded pass takes
#: its five marker characters from: the first five code points, walking up from the first, that
#: the document does not hold -- U+E000 to U+E004 for a document that holds none. JSON leaves
#: every one as it is, so each is found again, at its own position, in the re-serialised text.
_PRIVATE_USE_FIRST = 0xE000
_PRIVATE_USE_LAST = 0xF8FF


def placeholder_spans(text: str) -> list[tuple[int, int]]:
    """Where each placeholder the gate writes stands in *text*, as ``(start, end)`` in text
    order -- a replayed row's, a quoted redaction's, the decoded pass's own. The scanner guards
    them: no rule takes a span that overlaps one, and none is counted as a hit."""
    return [(m.start(), m.end()) for m in _PLACEHOLDER_RE.finditer(text)]


@dataclass(frozen=True)
class _Markers:
    """The five characters one tool call's decoded pass writes, none of which its document
    holds: a redaction is ``redaction_open``, the hit's index, ``redaction_close``; a flag-only
    hit is ``flag_open``, its index, ``flag_close``, the kept text, ``flag_end``."""

    redaction_open: str
    redaction_close: str
    flag_open: str
    flag_close: str
    flag_end: str

    @classmethod
    def free_in(cls, text: str) -> _Markers | None:
        """The first five private-use code points *text* does not hold, or ``None`` when fewer
        than five are free. *text* is a canonical tool call, written with ``ensure_ascii`` off:
        JSON escapes only a quote, a backslash and a control character, so every private-use
        character of every decoded string -- a value, a key, the tool's name -- stands in it as
        itself, and a code point the text does not hold is in no decoded string either."""
        held = set(text)
        candidates = map(chr, range(_PRIVATE_USE_FIRST, _PRIVATE_USE_LAST + 1))
        free = list(islice((char for char in candidates if char not in held), 5))
        if len(free) < 5:
            return None
        return cls(*free)

    def pattern(self) -> re.Pattern[str]:
        """What finds a marker again in the re-serialised text: group 1 is a redaction's index,
        group 2 a flag's, and a bare ``flag_end`` closes the flag last opened."""
        e = re.escape
        return re.compile(
            f"{e(self.redaction_open)}(\\d+){e(self.redaction_close)}"
            f"|{e(self.flag_open)}(\\d+){e(self.flag_close)}|{e(self.flag_end)}"
        )


@dataclass(frozen=True)
class Hit:
    """One redaction or flag, positioned in the stored text. A redacted hit's ``length`` is the
    placeholder's; a flagged one's is the kept text's."""

    rule: str
    kind: RuleKind
    start: int
    length: int
    effect: Effect

    def as_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "start": self.start, "length": self.length}


@dataclass(frozen=True)
class ScanResult:
    """The text as it may be stored, and what the scan did to it."""

    text: str
    redactions: tuple[Hit, ...]
    flags: tuple[Hit, ...]
    provider_hits: int

    def redactions_json(self) -> str:
        return json.dumps([hit.as_dict() for hit in self.redactions])

    def flags_json(self) -> str:
        return json.dumps([hit.as_dict() for hit in self.flags])

    def redacted_rules(self) -> list[str]:
        return sorted({hit.rule for hit in self.redactions})

    def flagged_rules(self) -> list[str]:
        return sorted({hit.rule for hit in self.flags})

    @property
    def text_verdict(self) -> TextVerdict:
        if self.redactions:
            return "redacted"
        return "flagged" if self.flags else "clean"


class SecretRefused(Exception):
    """A provider token in text the caller can rephrase. Carries the rule, the offset and the
    length -- never the value -- and its message is what both surfaces show."""

    def __init__(self, *, rule: str, start: int, length: int) -> None:
        self.rule = rule
        self.start = start
        self.length = length
        super().__init__(
            f"refused: the text contains a {rule} at offset {start} ({length} characters); "
            "remove it and try again"
        )


@dataclass(frozen=True)
class _Span:
    rule: Rule
    start: int
    end: int


def _starts_inside(span: _Span, run: tuple[int, int] | None) -> bool:
    """Whether *span* starts inside *run*, a ``(start, end)`` range."""
    return run is not None and run[0] <= span.start < run[1]


def _judges_its_matches(rule: Rule) -> bool:
    """Whether *rule*'s scan can pass over what its pattern matches -- a check, an exemption or
    a keyword gate -- so that a run cut at a window's edge may fail where the whole run passes.
    A rule with its own span finder does not scan by its pattern alone, and its spans are no
    longer than the smallest overlap the settings allow, which the next window sees whole."""
    if rule.find_spans is not None:
        return False
    return (
        rule.check is not None
        or bool(rule.exempt)
        or bool(rule.exempt_prefixes)
        or rule.keywords is not None
    )


def _flagged(rule: Rule) -> bool:
    """Every hit but a provider's is flagged: a generic hit and a checksummed identifier beside
    their redaction, a keyword-only pattern on its own."""
    return rule.kind != "provider"


class Scanner:
    """The rules, built once from the limits, applied to one text at a time."""

    def __init__(self, limits: GateLimits) -> None:
        self.limits = limits
        self._rules = rules(limits)
        self._by_name = {rule.name: rule for rule in self._rules}
        self._order = {rule.name: index for index, rule in enumerate(self._rules)}
        self._judging = tuple(rule for rule in self._rules if _judges_its_matches(rule))

    def scan(self, text: str, *, verdict: Verdict, known: Sequence[Hit] = ()) -> ScanResult:
        """Scan *text* under *verdict*: raise on a provider hit under ``refuse``, redact and
        count it under ``redact``; redact and flag the rest as each rule says.

        *known* are the hits a scan of a longer text made inside *text*, positioned on it: a
        piece cut from a text scanned whole carries them. *text* is still scanned in full, and
        each known hit is recorded beside this scan's own, once per rule and position, where it
        stands in the result. A known redaction's placeholder is already in *text*, guarded, so
        this scan never takes it again, and it holds no value to refuse. A known flag whose kept
        text overlaps a span this scan takes gives way to it, so a hit passed in never keeps a
        secret from being redacted. A known redaction whose position does not hold its
        placeholder, a hit of no rule or outside *text*, and two known hits that overlap raise
        ``ValueError``: a caller's error, named by rule and position."""
        if not known:
            return _apply(text, self._spans(text), verdict)
        own = self._spans(text)
        placed = [
            span
            for span in self._known_spans(text, known)
            if span.rule.effect == "redact"
            or not any(span.start < o.end and o.start < span.end for o in own)
        ]
        spans = sorted([*own, *placed], key=lambda s: s.start)
        return _apply(text, spans, verdict, placed=placed)

    def _known_spans(self, text: str, known: Sequence[Hit]) -> list[_Span]:
        """*known* as spans on *text*, one per rule and position, in text order: a redaction as
        its placeholder -- which a flag of the same rule shares -- and a flag-only hit as its
        kept text. Raises ``ValueError`` for a hit of no rule, one outside *text*, a redaction
        whose position does not hold its placeholder, and two spans that overlap."""
        spans: dict[tuple[int, str], _Span] = {}
        for hit in known:
            rule = self._by_name.get(hit.rule)
            if rule is None:
                raise ValueError(f"a known hit names no rule of the gate's: {hit.rule!r}")
            end = hit.start + hit.length
            if hit.start < 0 or hit.length <= 0 or end > len(text):
                raise ValueError(
                    f"a known {rule.name} hit at offset {hit.start} ({hit.length} characters) "
                    f"lies outside the text ({len(text)} characters)"
                )
            if rule.effect == "redact" and text[hit.start : end] != REDACTION.format(
                rule=rule.name
            ):
                raise ValueError(
                    f"a known {rule.name} redaction at offset {hit.start} does not stand on its "
                    "placeholder"
                )
            spans.setdefault((hit.start, rule.name), _Span(rule, hit.start, end))
        ordered = sorted(spans.values(), key=lambda s: s.start)
        for before, after in pairwise(ordered):
            if after.start < before.end:
                raise ValueError(
                    f"known hits overlap: {before.rule.name} at offset {before.start} and "
                    f"{after.rule.name} at offset {after.start}"
                )
        return ordered

    def scan_turn(self, text: str, *, role: str, verdict: Verdict) -> ScanResult:
        """``scan``, JSON-aware for a ``tool_call`` turn whose text is exactly what
        ``tool_call_text`` writes for the document it decodes to: its string values are scanned
        decoded, each hit written as a marker -- a redaction as one, a flag-only hit as its kept
        text between two others, in characters the document does not hold -- the document
        re-serialised, every marker replaced or removed at its position on the stored text, and
        the whole scanned once more with every placeholder guarded and every placed span
        excluded so nothing is counted twice. Any other text, and a document that leaves fewer
        than five private-use code points free, is scanned as plain text, so the stored text
        keeps its shape. Capture scans turns under ``redact``; under ``refuse`` a hit inside a
        decoded value is reported at its offset within that value."""
        if role != "tool_call":
            return self.scan(text, verdict=verdict)
        try:
            document = json.loads(text)
        except ValueError:
            return self.scan(text, verdict=verdict)
        # Only the canonical text is decoded: re-serialising any other would drop a key the
        # document carries, coerce a value or change its spacing.
        tool = document.get("tool") if isinstance(document, dict) else None
        if (
            not isinstance(tool, str)
            or "input" not in document
            or tool_call_text(tool, document["input"]) != text
        ):
            return self.scan(text, verdict=verdict)
        markers = _Markers.free_in(text)
        if markers is None:
            return self.scan(text, verdict=verdict)
        written: list[str] = []
        marked = tool_call_text(
            tool, self._scan_values(document["input"], verdict, written, markers)
        )
        pieces: list[str] = []
        placed: list[_Span] = []
        flagged: list[_Span] = []
        opened: list[tuple[Rule, int]] = []
        cursor = length = 0
        for found in markers.pattern().finditer(marked):
            before = marked[cursor : found.start()]
            pieces.append(before)
            length += len(before)
            cursor = found.end()
            if found.group(1) is not None:
                rule = self._by_name[written[int(found.group(1))]]
                placeholder = REDACTION.format(rule=rule.name)
                pieces.append(placeholder)
                placed.append(_Span(rule, length, length + len(placeholder)))
                length += len(placeholder)
            elif found.group(2) is not None:
                opened.append((self._by_name[written[int(found.group(2))]], length))
            else:
                rule, start = opened.pop()
                flagged.append(_Span(rule, start, length))
        pieces.append(marked[cursor:])
        stored = "".join(pieces)
        # The whole-text pass sees every placed placeholder as a guard (they are real rule
        # names) and so never re-matches them; a whole-text span that overlaps an already-placed
        # flag is dropped, so that hit is never counted twice.
        whole = [
            span
            for span in self._spans(stored)
            if not any(span.start < f.end and f.start < span.end for f in flagged)
        ]
        spans = sorted([*whole, *placed, *flagged], key=lambda s: s.start)
        return _apply(stored, spans, verdict)

    def _scan_values(
        self, value: object, verdict: Verdict, written: list[str], markers: _Markers
    ) -> object:
        """*value* with every string in it scanned under *verdict*; each redaction becomes one
        marker and a flag-only hit keeps its text between two others, each hit's rule appended to
        *written* in the order its marker carries. Keys are not scanned: a key is a name."""
        if isinstance(value, str):
            result = self.scan(value, verdict=verdict)
            marks = sorted(
                [(h.start, h.length, h.rule, True) for h in result.redactions]
                + [
                    (h.start, h.length, h.rule, False)
                    for h in result.flags
                    if self._by_name[h.rule].effect == "flag"
                ]
            )
            pieces: list[str] = []
            cursor = 0
            for start, length, rule_name, redacted in marks:
                pieces.append(result.text[cursor:start])
                index = str(len(written))
                if redacted:
                    pieces += [markers.redaction_open, index, markers.redaction_close]
                else:
                    kept = result.text[start : start + length]
                    pieces += [markers.flag_open, index, markers.flag_close, kept, markers.flag_end]
                written.append(rule_name)
                cursor = start + length
            pieces.append(result.text[cursor:])
            return "".join(pieces)
        if isinstance(value, dict):
            return {
                key: self._scan_values(item, verdict, written, markers)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._scan_values(item, verdict, written, markers) for item in value]
        return value

    def _spans(self, text: str) -> list[_Span]:
        """Every span a rule takes in *text*, in text order, an earlier rule winning an overlap
        and no span overlapping a placeholder already there. A long text is scanned in windows
        (``_window``); each records the spans it decides, and the overlaps between them are
        resolved once, over the whole text, as an unwindowed scan resolves them."""
        limits = self.limits
        guards = placeholder_spans(text)
        if len(text) <= limits.window_chars:
            return self._without_overlaps(self._matches_in(text, 0, 0), guards)
        found: dict[tuple[int, str], _Span] = {}
        # By rule name, the last run that rule rejected seen whole.
        rejected: dict[str, tuple[int, int]] = {}
        offset = 0
        while True:
            end = offset + limits.window_chars
            last = end >= len(text)
            decided, resume = self._window(text, offset, end, guards, rejected, last=last)
            for span in decided:
                found.setdefault((span.start, span.rule.name), span)
            if last:
                break
            # The settings refuse an overlap at or above the window, and every carried span
            # starts after the window's start, so this always advances.
            offset = max(offset + 1, resume)
        return self._without_overlaps(found.values(), guards)

    def _window(
        self,
        text: str,
        offset: int,
        end: int,
        guards: Sequence[tuple[int, int]],
        rejected: dict[str, tuple[int, int]],
        *,
        last: bool,
    ) -> tuple[list[_Span], int]:
        """The spans the window ``text[offset:end]`` decides, and where the next window starts.

        The window reads ``window_overlap_chars`` of the text before its start as well, so a
        keyword or a prefix before a span is seen with it, and takes only the spans that start
        inside it: one that starts before was the previous window's. No span is taken that
        starts inside a run its rule rejected seen whole (*rejected*, by rule name).

        A hit that ends before the window's end is decided, whatever it overlaps: the overlaps
        are resolved over the whole text. A span the window's end cuts -- a hit, or a run a
        rule's pattern matches to that end whatever its cut's check says (``_open_runs``) -- is
        undecided and takes no decided hit's place. One that starts after the window's start is
        carried: the next window starts at its start, or ``window_overlap_chars`` before the
        window's end, whichever is earlier, so a match longer than the overlap is seen whole.
        One that fills the window from its start is continued, in rule order, unless a decided
        hit of an earlier rule or a placeholder already there takes its place: accepted, it is
        one span, the next window starts ``window_overlap_chars`` before its end, and every cut
        span of an earlier rule inside it is carried, so the whole-text resolution chooses
        between them; rejected, its run is added to *rejected* and the next is tried."""
        limits = self.limits
        seen_from = max(0, offset - limits.window_overlap_chars)
        view = text[seen_from:end]
        hits = [
            span
            for span in self._matches_in(view, seen_from, offset)
            if not _starts_inside(span, rejected.get(span.rule.name))
        ]
        if last:
            return hits, len(text)
        seen = {(span.rule.name, span.start, span.end) for span in hits}
        runs = [
            run
            for run in self._open_runs(view, seen_from, offset)
            if (run.rule.name, run.start, run.end) not in seen
            and not _starts_inside(run, rejected.get(run.rule.name))
        ]
        decided = [span for span in hits if span.end < end]
        cut = [span for span in [*hits, *runs] if span.end >= end]
        filling = sorted(
            (span for span in cut if span.start == offset), key=lambda s: self._order[s.rule.name]
        )
        for filled in filling:
            if filled not in self._without_overlaps([*decided, filled], guards):
                continue
            whole, run_end = self._continued(text, filled)
            if whole is not None:
                resume = whole.end - limits.window_overlap_chars
                for span in cut:
                    if (
                        span.start > offset
                        and self._order[span.rule.name] < self._order[whole.rule.name]
                    ):
                        resume = min(resume, span.start)
                return [whole, *decided], resume
            if run_end is not None:
                rejected[filled.rule.name] = (filled.start, run_end)
        resume = end - limits.window_overlap_chars
        for span in cut:
            if span.start > offset:
                resume = min(resume, span.start)
        return decided, resume

    def _continued(self, text: str, span: _Span) -> tuple[_Span | None, int | None]:
        """For a span or cut run that filled its window while the text goes on: the whole match
        of its rule at its start, and where the run its rule's pattern matches there ends. The
        text is read from ``window_overlap_chars`` before the span, as its window read it, so a
        value after a keyword -- a bearer token -- is matched with that keyword. The pattern
        alone is matched again over a reach that doubles -- at least a window further on every
        pass, and under four times the run's length in all, so the work stays linear -- until
        its match ends inside the reach or the reach is the end of the text. Only then is the
        rule's check judged, once, on the whole run: never on a reach that cuts it. The whole
        match is ``None`` when the rule rejects the run seen whole; the run's end is ``None``
        when the pattern makes no match there, and the rule is judged on the reach read so
        far."""
        rule = span.rule
        seen_from = max(0, span.start - self.limits.window_overlap_chars)
        at = span.start - seen_from
        reach = span.end - span.start
        while True:
            reach *= 2
            stop = min(span.start + reach, len(text))
            view = text[seen_from:stop]
            run = self._run_at(rule, view, at)
            if run is None or run.end() < len(view) or stop == len(text):
                break
        ends = [end for start, end in matches(rule, view, self.limits) if start == at]
        whole = _Span(rule, span.start, seen_from + ends[0]) if ends else None
        return whole, (None if run is None else seen_from + run.end())

    def _run_at(self, rule: Rule, view: str, at: int) -> re.Match[str] | None:
        """The match of *rule*'s pattern in *view* whose value starts at *at*: the one its scan
        of *view* makes there. Only a rule without a span finder fills a window -- a span
        finder's spans are no longer than the smallest overlap the settings allow -- so the
        pattern's match and its value are the rule's own."""
        for match in rule.pattern.finditer(view):
            if match.start() > at:
                return None
            value = rule.value_span(match, self.limits)
            if value is not None and value[0] == at:
                return match
        return None

    def _open_runs(self, view: str, seen_from: int, offset: int) -> list[_Span]:
        """For each rule whose scan can pass over what its pattern matches, the run its pattern
        matches in *view* (the text from *seen_from* to the window's end) up to the window's
        end, when that run starts inside the window, at or after *offset*. That run is cut, and
        a check is never judged on a cut: whatever its cut's check says, the run is carried into
        the next window or, filling this one, continued, like a hit."""
        runs: list[_Span] = []
        for rule in self._judging:
            final = deque(rule.pattern.finditer(view), maxlen=1)
            if final and final[0].end() == len(view) and seen_from + final[0].start() >= offset:
                runs.append(_Span(rule, seen_from + final[0].start(), seen_from + len(view)))
        return runs

    def _matches_in(self, view: str, seen_from: int, start: int) -> list[_Span]:
        """Every rule's spans in *view*, the text from *seen_from* on, that start at or after
        *start*, positioned on the text."""
        return [
            _Span(rule, span_start + seen_from, span_end + seen_from)
            for rule in self._rules
            for span_start, span_end in matches(rule, view, self.limits)
            if span_start + seen_from >= start
        ]

    def _without_overlaps(
        self, spans: Iterable[_Span], guards: Sequence[tuple[int, int]]
    ) -> list[_Span]:
        """Spans in rule order, then text order, each dropped when it overlaps a guard or one
        already kept: the provider rules come first, so a provider token is never also an
        entropy hit. The kept intervals never overlap, so only the neighbours of a new span's
        position need checking (bisect), not every kept span."""
        starts: list[int] = []
        ends: list[int] = []
        for start, end in sorted(guards):
            starts.append(start)
            ends.append(end)
        kept: list[_Span] = []
        for span in sorted(spans, key=lambda s: (self._order[s.rule.name], s.start)):
            i = bisect.bisect_right(starts, span.start)
            if i > 0 and ends[i - 1] > span.start:
                continue
            if i < len(starts) and starts[i] < span.end:
                continue
            starts.insert(i, span.start)
            ends.insert(i, span.end)
            kept.append(span)
        return sorted(kept, key=lambda s: s.start)


def _apply(
    text: str, spans: list[_Span], verdict: Verdict, *, placed: Sequence[_Span] = ()
) -> ScanResult:
    """*text* with the spans redacted or flagged as their rules say, positions on the result.
    A span in *placed* is a hit a scan of a longer text already made -- a placeholder in *text*
    or a flag's kept text -- so it is recorded but holds no value to refuse."""
    if verdict == "refuse":
        for span in spans:
            if span.rule.kind == "provider" and span not in placed:
                raise SecretRefused(
                    rule=span.rule.name, start=span.start, length=span.end - span.start
                )
    pieces: list[str] = []
    cursor = 0
    delta = 0
    redactions: list[Hit] = []
    flags: list[Hit] = []
    provider_hits = 0
    for span in spans:
        rule = span.rule
        length = span.end - span.start
        placed_at = span.start + delta
        if rule.kind == "provider":
            provider_hits += 1
        if rule.effect == "redact":
            placeholder = REDACTION.format(rule=rule.name)
            pieces.append(text[cursor : span.start])
            pieces.append(placeholder)
            cursor = span.end
            delta += len(placeholder) - length
            redactions.append(Hit(rule.name, rule.kind, placed_at, len(placeholder), "redact"))
            if _flagged(rule):
                flags.append(Hit(rule.name, rule.kind, placed_at, len(placeholder), "flag"))
        else:
            flags.append(Hit(rule.name, rule.kind, placed_at, length, "flag"))
    pieces.append(text[cursor:])
    return ScanResult(
        text="".join(pieces),
        redactions=tuple(redactions),
        flags=tuple(flags),
        provider_hits=provider_hits,
    )


def build_scanner(settings: Settings) -> Scanner:
    """The one scanner a process builds, from its settings."""
    return Scanner(limits_of(settings))


_USERINFO_RE = re.compile(r"://[^/@\s]+@")


def strip_userinfo(url: str) -> str:
    """*url* with ``user:token@`` removed, unchanged when it carries none: a credential is not
    part of what a repository is."""
    return _USERINFO_RE.sub("://", url)
