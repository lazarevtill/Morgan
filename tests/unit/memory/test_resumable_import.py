"""Re-running an import must not redo work it already did.

Every piece costs an embedding call, and a real export is thousands of them -- hours on a CPU
model. An import that starts from zero each time is one that never finishes on a machine that
gets interrupted, which this one did three times.
"""

from __future__ import annotations

import json

import pytest

from morgan_brain.app.chatgpt_import import ARCHIVE_PROJECT, MAX_EMBED_CHARS, import_chatgpt
from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import Embedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.secrets.rules import GateLimits, rules
from morgan_brain.memory.store.db import open_db


class CountingEmbedder(Embedder):
    """A hash embedder that says how often it was asked."""

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [float(len(text) % 7), 1.0, 0.0, 0.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


def _export(path, turns):
    conversations = [
        {
            "conversation_id": "c2",  # not in the holdout
            "title": "t",
            "mapping": {
                f"m{i}": {
                    "id": f"m{i}",
                    "message": {
                        "id": f"m{i}",
                        "author": {"role": "user"},
                        "create_time": 1700000000.0 + i,
                        "content": {"content_type": "text", "parts": [text]},
                    },
                }
                for i, text in enumerate(turns)
            },
        }
    ]
    path.write_text(json.dumps(conversations), encoding="utf-8")
    return path


@pytest.fixture
def gate_and_embedder(tmp_path):
    conn = open_db(str(tmp_path / "morgan.db"))
    embedder = CountingEmbedder()
    return MemoryGate(build_memory_module(conn=conn, embedder=embedder, dim=4)), embedder


async def test_a_second_run_does_not_re_embed_what_is_already_stored(gate_and_embedder, tmp_path):
    gate, embedder = gate_and_embedder
    path = _export(tmp_path / "c.json", ["first turn", "second turn"])

    first = await import_chatgpt(path, gate=gate, user_id="owner")
    after_first = embedder.calls
    second = await import_chatgpt(path, gate=gate, user_id="owner")

    assert first.memories == 2
    assert embedder.calls == after_first, "re-embedded work it had already done"
    assert second.skipped_turns >= 2


async def test_a_piece_its_own_scan_redacts_further_is_passed_on_a_rerun_and_not_embedded(
    gate_and_embedder, tmp_path
):
    """A piece is stored as its own scan leaves it, and that scan can redact more than the scan
    of the whole turn: a high-entropy run after a long run of one letter is part of one
    low-entropy token in the turn, and a token of its own once the split cuts it off. A re-run
    compares the stored piece with that same form, so it passes the piece and embeds nothing."""
    gate, embedder = gate_and_embedder
    run = next(rule for rule in rules(GateLimits.defaults()) if rule.name == "entropy").fixture()
    path = _export(tmp_path / "c.json", ["a" * MAX_EMBED_CHARS + run])

    first = await import_chatgpt(path, gate=gate, user_id="owner")
    after_first = embedder.calls
    second = await import_chatgpt(path, gate=gate, user_id="owner")

    assert (first.memories, first.redacted_pieces) == (2, 1)
    assert embedder.calls == after_first, "re-embedded a piece it had already stored"
    assert (second.memories, second.skipped_turns, second.redacted_pieces) == (0, 2, 0)


async def test_an_edited_turn_is_rewritten_rather_than_skipped(gate_and_embedder, tmp_path):
    """Skipping by id alone would freeze a correction out of the brain for good."""
    gate, _ = gate_and_embedder
    path = _export(tmp_path / "c.json", ["the original wording"])
    await import_chatgpt(path, gate=gate, user_id="owner")

    _export(path, ["the corrected wording"])
    await import_chatgpt(path, gate=gate, user_id="owner")

    found = await gate.recall(
        __import__("morgan_brain.models", fromlist=["MemoryQuery"]).MemoryQuery(
            user_id="owner", project=ARCHIVE_PROJECT, text="wording", top_k=5
        )
    )
    assert [m.content for m in found.memories] == ["the corrected wording"]
