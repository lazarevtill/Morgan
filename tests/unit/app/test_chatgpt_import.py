"""Seeding the brain from a ChatGPT export, with a holdout the optimizer can never mine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from morgan_brain.app.chatgpt_import import (
    ARCHIVE_PROJECT,
    HOLDOUT_PROJECT,
    MAX_EMBED_CHARS,
    import_chatgpt,
    is_held_out,
    split_for_embedding,
)
from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.secrets.rules import GateLimits, rules
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import MemoryQuery, MemorySource

#: Conversation ids on each side of the holdout rule, chosen by asking the rule itself
#: rather than by hoping a made-up id lands where the test wants it.
KEPT = next(c for c in (f"c{i}" for i in range(50)) if not is_held_out(c))
HELD = next(c for c in (f"c{i}" for i in range(50)) if is_held_out(c))

#: A token in no fixture and no dictionary: the value that must appear nowhere.
MARKER = "zq" + "xj" + "vgatemarker"


def _conversation(conv_id: str, *turns: tuple[str, str], title: str = "t") -> dict:
    """A conversation in the export's own shape: a mapping of nodes keyed by message id."""
    mapping = {}
    for i, (role, text) in enumerate(turns):
        mid = f"{conv_id}-m{i}"
        mapping[mid] = {
            "id": mid,
            "message": {
                "id": mid,
                "author": {"role": role},
                "create_time": 1700000000.0 + i,
                "content": {"content_type": "text", "parts": [text]},
            },
        }
    return {"conversation_id": conv_id, "id": conv_id, "title": title, "mapping": mapping}


def _export(path: Path, conversations: list[dict]) -> Path:
    path.write_text(json.dumps(conversations), encoding="utf-8")
    return path


@pytest.fixture
def gate(tmp_path):
    conn = open_db(str(tmp_path / "morgan.db"))
    module = build_memory_module(conn=conn, embedder=FakeEmbedder(dim=8), dim=8)
    return MemoryGate(module)


async def _contents(gate: MemoryGate, project: str) -> list[str]:
    found = (
        await gate.recall(MemoryQuery(user_id="owner", project=project, text="", top_k=100))
    ).memories
    return [m.content for m in found]


async def test_user_and_assistant_turns_keep_their_attribution(gate, tmp_path):
    """The actor-attribution invariant is the whole reason the import is not a bulk insert:
    an assistant's answer is an inference and must never be stored as the owner's statement.
    """
    path = _export(
        tmp_path / "c.json",
        [_conversation(KEPT, ("user", "I prefer terse answers"), ("assistant", "Understood"))],
    )

    await import_chatgpt(path, gate=gate, user_id="owner")

    stored = await gate.recall(
        MemoryQuery(user_id="owner", project=ARCHIVE_PROJECT, text="", top_k=100)
    )
    by_content = {m.content: m.source for m in stored.memories}
    assert by_content["I prefer terse answers"] is MemorySource.USER_STATED
    assert by_content["Understood"] is MemorySource.AGENT_INFERRED


async def test_tool_turns_and_empty_parts_are_not_memories(gate, tmp_path):
    path = _export(
        tmp_path / "c.json",
        [
            _conversation(
                KEPT,
                ("user", "a real message"),
                ("tool", "some tool output"),
                ("assistant", "   "),
            )
        ],
    )

    report = await import_chatgpt(path, gate=gate, user_id="owner")

    assert await _contents(gate, ARCHIVE_PROJECT) == ["a real message"]
    assert report.memories == 1


async def test_the_holdout_is_a_separate_project_so_scoping_enforces_the_firewall(gate, tmp_path):
    """Golden items are drawn from held-out conversations, and consolidation must never see
    them -- otherwise the eval measures memorisation. Project scoping already refuses to
    cross that line on every read and write, so the holdout is a project rather than a flag
    some future caller has to remember to check.
    """
    held, kept = HELD, KEPT
    path = _export(
        tmp_path / "c.json",
        [
            _conversation(held, ("user", "held out message")),
            _conversation(kept, ("user", "ordinary message")),
        ],
    )

    report = await import_chatgpt(path, gate=gate, user_id="owner")

    assert await _contents(gate, ARCHIVE_PROJECT) == ["ordinary message"]
    assert await _contents(gate, HOLDOUT_PROJECT) == ["held out message"]
    assert report.conversations == 1
    assert report.held_out == 1


def test_holdout_membership_is_decided_by_the_conversation_id_alone():
    """Reproducible without stored state: the same export re-imported selects the same
    holdout, and no seed has to survive between runs for the split to mean anything.
    """
    ids = [f"conversation-{i}" for i in range(400)]
    held = [c for c in ids if is_held_out(c)]

    assert [is_held_out(c) for c in ids] == [is_held_out(c) for c in ids]
    assert 0.1 < len(held) / len(ids) < 0.3


async def test_reimport_updates_in_place_rather_than_duplicating(gate, tmp_path):
    path = _export(tmp_path / "c.json", [_conversation(KEPT, ("user", "only once"))])

    await import_chatgpt(path, gate=gate, user_id="owner")
    await import_chatgpt(path, gate=gate, user_id="owner")

    assert await _contents(gate, ARCHIVE_PROJECT) == ["only once"]


def test_a_turn_within_budget_is_left_whole():
    text = "a paragraph\n\nand another"

    assert split_for_embedding(text, budget=100) == [(0, text)]


def test_a_long_turn_is_split_without_losing_or_duplicating_a_character():
    """The export's longest turn is 87,000 characters, and an embedding server refuses
    anything over its context. Truncating would leave the memory searchable by keyword and
    invisible to the vector signal -- the exact "visible to one signal, not another" failure
    the one-write-path invariant exists to prevent. So it is split, and nothing is dropped.
    """
    text = "\n\n".join(f"paragraph {i} " + "word " * 200 for i in range(6))

    pieces = split_for_embedding(text, budget=1000)
    chunks = [chunk for _, chunk in pieces]

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert "".join("".join(c.split()) for c in chunks) == "".join(text.split())
    assert all(text[offset : offset + len(chunk)] == chunk for offset, chunk in pieces)


def test_a_split_prefers_a_paragraph_boundary_over_a_word_boundary():
    text = "first paragraph here\n\n" + "second paragraph " * 40

    pieces = split_for_embedding(text, budget=60)

    assert pieces[0] == (0, "first paragraph here")
    assert all(text[offset : offset + len(chunk)] == chunk for offset, chunk in pieces)


def test_an_unbroken_run_longer_than_the_budget_is_still_split():
    """Log dumps and base64 blobs arrive as one unbroken run. A splitter that only cuts on
    whitespace would hand the server a chunk over its context and fail the whole import."""
    pieces = split_for_embedding("x" * 5000, budget=1000)
    chunks = [chunk for _, chunk in pieces]

    assert chunks and all(len(c) <= 1000 for c in chunks)
    assert "".join(chunks) == "x" * 5000
    assert [offset for offset, _ in pieces] == [0, 1000, 2000, 3000, 4000]


def test_each_piece_carries_its_offset_in_the_text_it_was_cut_from():
    """The whitespace a split strips is counted, so an offset is where the piece stands in the
    text as given: the hits of a scan of that text are shifted by it onto the piece."""
    text = "   alpha beta\n\n   gamma"

    assert split_for_embedding(text, budget=12) == [(3, "alpha beta"), (18, "gamma")]


def test_a_cut_that_would_fall_inside_a_placeholder_moves_to_its_start():
    """A placeholder cut in two is neither one: each half is ordinary text to the next scan,
    and the piece it lands in records nothing where the gate redacted something."""
    placeholder = "[redacted:github_token]"
    text = "x" * 990 + placeholder + "y" * 500

    pieces = split_for_embedding(text, budget=1000)

    assert pieces[0] == (0, "x" * 990)
    assert pieces[1][0] == 990 and pieces[1][1].startswith(placeholder)
    assert all(text[offset : offset + len(chunk)] == chunk for offset, chunk in pieces)


async def test_a_long_turn_is_stored_as_several_memories_and_reimport_does_not_duplicate(
    gate, tmp_path
):
    long_turn = "\n\n".join(f"section {i} " + "word " * 400 for i in range(8))
    path = _export(tmp_path / "c.json", [_conversation(KEPT, ("user", long_turn))])

    first = await import_chatgpt(path, gate=gate, user_id="owner")
    second = await import_chatgpt(path, gate=gate, user_id="owner")

    assert first.memories > 1
    # The second run writes nothing because everything is already there, which is what makes
    # an interrupted import resumable. What must not change is the stored result.
    assert second.memories == 0
    assert await _contents(gate, ARCHIVE_PROJECT) == await _contents(gate, ARCHIVE_PROJECT)
    assert len(await _contents(gate, ARCHIVE_PROJECT)) == first.memories


# --- the gate on an import: history nobody can rephrase ---------------------------------------


def _rows(gate: MemoryGate) -> list[tuple[str, str]]:
    """Every stored memory's content and recorded redactions."""
    conn = gate._store._conn
    return [(r["content"], r["redactions"]) for r in conn.execute("SELECT * FROM memories")]


def _placed_rules(content: str, redactions: str) -> list[str]:
    """The rules *redactions* records, each checked to stand at its placeholder in *content*."""
    hits = json.loads(redactions)
    for hit in hits:
        placed = content[hit["start"] : hit["start"] + hit["length"]]
        assert placed == f"[redacted:{hit['rule']}]"
    return sorted(hit["rule"] for hit in hits)


def _straddling(before: str, across: str, after: str = " and the rest") -> str:
    """A turn whose first ``MAX_EMBED_CHARS`` characters end five characters into *across*:
    words and single spaces, then *before* (ending in a space), then *across*, then *after*.
    No line break comes before *across*, so a split of the raw turn cuts at the space right
    before it."""
    start = MAX_EMBED_CHARS - 5
    head = ("word " * (start // 5 + 1))[-(start - len(before)) :] + before
    turn = head + across + after
    window = turn[:MAX_EMBED_CHARS]
    assert len(head) == start and "\n" not in window and window.rfind(" ") == start - 1
    return turn


def _fixture(rule_name: str) -> str:
    return next(rule for rule in rules(GateLimits.defaults()) if rule.name == rule_name).fixture()


async def test_an_imported_message_is_redacted_and_counted_never_refused(gate, tmp_path):
    """A provider token in a ChatGPT export is redacted and counted on the report, and the
    import goes on."""
    token = "gh" + "p_" + MARKER + "A" * (36 - len(MARKER))
    path = _export(
        tmp_path / "c.json",
        [
            _conversation(
                KEPT,
                ("user", f"my token is {token}"),
                ("assistant", f"DB_PASSWORD={MARKER}99 is what compose reads"),
                ("user", "a clean turn"),
            )
        ],
    )

    report = await import_chatgpt(path, gate=gate, user_id="owner")

    assert (report.memories, report.redacted_pieces, report.provider_hits) == (3, 2, 1)
    contents = await _contents(gate, ARCHIVE_PROJECT)
    assert "my token is [redacted:github_token]" in contents
    assert "DB_PASSWORD=[redacted:assignment] is what compose reads" in contents
    assert all(MARKER not in c for c in contents)


async def test_a_redacted_turn_is_stored_once_and_a_rerun_passes_it(gate, tmp_path):
    """A resumed import finds the piece it stored redacted and passes it, without an error and
    without storing it again."""
    token = "gh" + "p_" + MARKER + "A" * (36 - len(MARKER))
    path = _export(tmp_path / "c.json", [_conversation(KEPT, ("user", f"deploy {token} now"))])

    first = await import_chatgpt(path, gate=gate, user_id="owner")
    second = await import_chatgpt(path, gate=gate, user_id="owner")

    assert await _contents(gate, ARCHIVE_PROJECT) == ["deploy [redacted:github_token] now"]
    assert (first.memories, first.redacted_pieces, first.provider_hits) == (1, 1, 1)
    assert (second.memories, second.skipped_turns) == (0, 1)


async def test_an_assignment_cut_from_its_value_is_redacted_and_recorded_where_it_lands(
    gate, tmp_path
):
    """The turn is scanned whole before it is split: a cut between ``DB_PASSWORD =`` and its
    value leaves no value behind, and the piece that holds the placeholder records the rule."""
    turn = _straddling("DB_PASSWORD = ", f"{MARKER}99")
    path = _export(tmp_path / "c.json", [_conversation(KEPT, ("user", turn))])

    report = await import_chatgpt(path, gate=gate, user_id="owner")

    rows = _rows(gate)
    assert all(MARKER not in content for content, _ in rows)
    [holding] = [row for row in rows if "[redacted:assignment]" in row[0]]
    assert _placed_rules(*holding) == ["assignment"]
    assert (report.memories, report.redacted_pieces, report.provider_hits) == (2, 1, 0)


async def test_a_private_key_that_straddles_a_cut_is_redacted_whole(gate, tmp_path):
    body = MARKER + "A" * 25
    turn = _straddling("-----BEGIN PRIVATE ", f"KEY-----\n{body}\n-----END PRIVATE KEY-----")
    path = _export(tmp_path / "c.json", [_conversation(KEPT, ("user", turn))])

    report = await import_chatgpt(path, gate=gate, user_id="owner")

    rows = _rows(gate)
    assert all(MARKER not in content and "PRIVATE" not in content for content, _ in rows)
    [holding] = [row for row in rows if "[redacted:private_key]" in row[0]]
    assert _placed_rules(*holding) == ["private_key"]
    assert (report.redacted_pieces, report.provider_hits) == (1, 1)


async def test_a_quoted_placeholder_in_a_turn_is_not_counted(gate, tmp_path):
    turn = "the log shows [redacted:github_token] where the key was"
    path = _export(tmp_path / "c.json", [_conversation(KEPT, ("user", turn))])

    report = await import_chatgpt(path, gate=gate, user_id="owner")

    assert _rows(gate) == [(turn, "[]")]
    assert (report.memories, report.redacted_pieces, report.provider_hits) == (1, 0, 0)


async def test_an_identifier_whose_keyword_ends_one_piece_is_redacted_in_the_next(gate, tmp_path):
    keyword, number = _fixture("inn").split(" ")
    turn = _straddling(f"{keyword} ", number)
    path = _export(tmp_path / "c.json", [_conversation(KEPT, ("user", turn))])

    report = await import_chatgpt(path, gate=gate, user_id="owner")

    rows = _rows(gate)
    assert all(number not in content for content, _ in rows)
    [holding] = [row for row in rows if "[redacted:inn]" in row[0]]
    assert holding[0].startswith("[redacted:inn]")
    assert _placed_rules(*holding) == ["inn"]
    assert (report.redacted_pieces, report.provider_hits) == (1, 0)
