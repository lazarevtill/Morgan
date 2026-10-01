"""Re-running an import must not redo work it already did.

Every piece costs an embedding call, and a real export is thousands of them -- hours on a CPU
model. An import that starts from zero each time is one that never finishes on a machine that
gets interrupted, which this one did three times.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from morgan_brain.app.chatgpt_import import ARCHIVE_PROJECT, _memory_id, import_chatgpt
from morgan_brain.composition import build_memory_module
from morgan_brain.memory.embedder import Embedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store.db import open_db
from morgan_brain.memory.store.episodic import EventIdentityConflict
from morgan_brain.models import Memory, MemorySource, OriginKind


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


async def test_an_edited_turn_refuses_overwrite_and_preserves_original(gate_and_embedder, tmp_path):
    """Changed source payloads need explicit new identity; existing assertions stay immutable."""
    gate, embedder = gate_and_embedder
    path = _export(tmp_path / "c.json", ["the original wording"])
    await import_chatgpt(path, gate=gate, user_id="owner")
    before_calls = embedder.calls
    conn = gate._store._conn
    before_db = conn.serialize()

    _export(path, ["the corrected wording"])
    with pytest.raises(EventIdentityConflict, match="content"):
        await import_chatgpt(path, gate=gate, user_id="owner")
    assert embedder.calls == before_calls
    assert conn.serialize() == before_db

    found = await gate.recall(
        __import__("morgan_brain.models", fromlist=["MemoryQuery"]).MemoryQuery(
            user_id="owner", project=ARCHIVE_PROJECT, text="wording", top_k=5
        )
    )
    assert [m.content for m in found.memories] == ["the original wording"]


async def test_new_assistant_import_reports_assistant_author(gate_and_embedder, tmp_path):
    gate, _ = gate_and_embedder
    path = _export(tmp_path / "assistant.json", ["Synthetic assistant claim"])
    export = json.loads(path.read_text(encoding="utf-8"))
    export[0]["mapping"]["m0"]["message"]["author"]["role"] = "assistant"
    path.write_text(json.dumps(export), encoding="utf-8")
    await import_chatgpt(path, gate=gate, user_id="owner")
    stored = await gate.get(_memory_id("m0", 0), user_id="owner")
    assert stored.source is MemorySource.AGENT_INFERRED
    assert stored.author_id == "chatgpt:assistant"


async def test_legacy_assistant_exact_replay_keeps_historical_author(gate_and_embedder, tmp_path):
    gate, embedder = gate_and_embedder
    path = _export(tmp_path / "assistant.json", ["Legacy assistant claim"])
    export = json.loads(path.read_text(encoding="utf-8"))
    export[0]["mapping"]["m0"]["message"]["author"]["role"] = "assistant"
    path.write_text(json.dumps(export), encoding="utf-8")
    identity = _memory_id("m0", 0)
    await gate.store(
        Memory(
            id=identity,
            user_id="owner",
            project=ARCHIVE_PROJECT,
            content="Legacy assistant claim",
            source=MemorySource.AGENT_INFERRED,
            author_id="owner",
            origin_kind=OriginKind.IMPORT,
            client="cli",
            cwd="old location",
            created_at=datetime.fromtimestamp(1700000000.0, tz=UTC),
        )
    )
    conn = gate._store._conn
    before_db = conn.serialize()
    before_calls = embedder.calls
    report = await import_chatgpt(path, gate=gate, user_id="owner")
    assert report.skipped_turns == 1 and report.memories == 0
    assert embedder.calls == before_calls
    assert conn.serialize() == before_db
    assert (await gate.get(identity, user_id="owner")).author_id == "owner"


@pytest.mark.parametrize(
    ("field", "value", "conflict_field"),
    [("author", {"role": "assistant"}, "source"), ("create_time", 1800000000.0, "created_at")],
)
async def test_same_text_changed_source_identity_refuses_before_embedding(
    gate_and_embedder, tmp_path, field, value, conflict_field
):
    gate, embedder = gate_and_embedder
    path = _export(tmp_path / "c.json", ["unchanged words"])
    await import_chatgpt(path, gate=gate, user_id="owner")
    before_calls = embedder.calls
    conn = gate._store._conn
    before_db = conn.serialize()
    export = json.loads(path.read_text(encoding="utf-8"))
    export[0]["mapping"]["m0"]["message"][field] = value
    path.write_text(json.dumps(export), encoding="utf-8")
    with pytest.raises(EventIdentityConflict) as exc:
        await import_chatgpt(path, gate=gate, user_id="owner")
    assert conflict_field in exc.value.fields
    assert embedder.calls == before_calls
    assert conn.serialize() == before_db


@pytest.mark.parametrize("role", ["user", "assistant"])
async def test_version_three_import_replay_after_migration_keeps_evidence(tmp_path, role):
    from morgan_brain.composition import migration_stores
    from morgan_brain.memory import migrations
    from tests.unit.memory.test_provenance_columns import _a_version_three_database_with

    conn = _a_version_three_database_with(tmp_path, projects=[ARCHIVE_PROJECT])
    identity = _memory_id("m0", 0)
    source = "user_stated" if role == "user" else "agent_inferred"
    timestamp = datetime.fromtimestamp(1700000000.0, tz=UTC).isoformat()
    conn.execute(
        "UPDATE memories SET id = ?, source = ?, content = ?, created_at = ?",
        (identity, source, "Legacy imported claim", timestamp),
    )
    conn.commit()
    migrations.migrate(conn, migration_stores(conn))
    embedder = CountingEmbedder()
    gate = MemoryGate(build_memory_module(conn=conn, embedder=embedder, dim=4))
    path = _export(tmp_path / "legacy.json", ["Legacy imported claim"])
    export = json.loads(path.read_text(encoding="utf-8"))
    export[0]["mapping"]["m0"]["message"]["author"]["role"] = role
    path.write_text(json.dumps(export), encoding="utf-8")
    before = conn.serialize()
    result = await import_chatgpt(path, gate=gate, user_id="u")
    assert result.memories == 0 and result.skipped_turns == 1
    assert embedder.calls == 0
    assert conn.serialize() == before
    stored = await gate.get(identity, user_id="u")
    assert stored.author_id == "u" and stored.client == ""
    assert stored.recorded_at is None
