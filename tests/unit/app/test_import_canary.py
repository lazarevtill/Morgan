"""A model that is still having a bad moment when a canary runs is caught within one stretch,
not at the end.

Two hours of embedding produced one wrong vector on 2026-09-19 and nothing noticed. The canary
bounds how many memories a *sustained* bad stretch can reach: every ``canary_every`` memories
the importer actually stores, and once more at the end for whatever was stored since the last
good check, it re-sends the five fingerprint strings in their own small call and compares them
against the recorded fingerprint -- never short-circuited by a process's own "already checked"
cache. It does not catch a vector that was wrong only in between two checks and has since
recovered; ``morgan doctor --vectors``'s full re-embed sample is the check for that.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from morgan_brain.app.chatgpt_import import ImportStopped, _memory_id, import_chatgpt
from morgan_brain.composition import build_memory_module
from morgan_brain.config import Settings
from morgan_brain.memory import fingerprint
from morgan_brain.memory.checked_embedder import CheckedEmbedder
from morgan_brain.memory.embedder import Embedder
from morgan_brain.memory.gate import MemoryGate
from morgan_brain.memory.store import spaces
from morgan_brain.memory.store.db import open_db
from morgan_brain.providers.wire import EmbeddingSpaceMismatch

#: Where the embedder under test is addressed, and by which setting -- distinct from every
#: other test file's constant so a leaked ``_checked`` entry (cleared by the autouse
#: ``_a_fresh_process`` fixture anyway) could never be mistaken for this one's.
_URL = "http://import-canary.test/v1"
_SETTING = "MORGAN_EMBEDDING_ENDPOINT"


class _DriftingEmbedder(Embedder):
    """A deterministic embedder that answers correctly until *drift_after* memory pieces have
    gone through it, then answers every input -- fingerprint strings included -- with the same
    wrong vector, the way a model having a bad moment does: it does not know which text is the
    canary's, it just answers badly from then on.

    Counts only texts that are not one of the five fingerprint strings, matching what the
    importer itself counts toward its interval: a canary's own small call never moves the
    drift point, and neither does a piece the importer skipped because it was already stored.
    """

    def __init__(self, *, dim: int, drift_after: int) -> None:
        self._dim = dim
        self.drift_after = drift_after
        self.memories_seen = 0

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            if text not in fingerprint.STRINGS:
                self.memories_seen += 1
            out.append(self._vector(text))
        return out

    def _vector(self, text: str) -> list[float]:
        seed = text if self.memories_seen <= self.drift_after else "\x00drifted\x00"
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self._dim)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]


def _build_gate(
    tmp_path: Path, *, drift_after: int
) -> tuple[MemoryGate, _DriftingEmbedder, sqlite3.Connection]:
    """A real ``MemoryGate`` over a real ``MemoryModule`` over a real ``CheckedEmbedder`` --
    the path an import actually runs, not a stand-in for it."""
    conn = open_db(str(tmp_path / "m.db"))
    inner = _DriftingEmbedder(dim=4, drift_after=drift_after)
    settings = Settings(data_dir=str(tmp_path))
    checked = CheckedEmbedder(inner, conn=conn, settings=settings, endpoint=_URL, setting=_SETTING)
    module = build_memory_module(
        conn=conn, embedder=checked, dim=4, clock=lambda: datetime(2026, 9, 21, tzinfo=UTC)
    )
    spaces.register(
        conn,
        model=settings.embedding_model,
        dims=4,
        table_name="vec_items",
        clock=lambda: datetime(2026, 9, 21, tzinfo=UTC),
    )
    return MemoryGate(module), inner, conn


def _turns_export(path: Path, *, conversations: int, turns_each: int) -> Path:
    """*conversations* conversations of *turns_each* distinct single-turn "user" messages
    each, in a fixed order -- so the Nth memory the importer ever stores is always the same
    conversation and turn across a run, and the test's ordinals mean what they say."""
    convos = []
    for c in range(conversations):
        mapping = {}
        for t in range(turns_each):
            mid = f"c{c}-m{t}"
            mapping[mid] = {
                "id": mid,
                "message": {
                    "id": mid,
                    "author": {"role": "user"},
                    "create_time": 1_700_000_000.0 + t,
                    "content": {"content_type": "text", "parts": [f"conversation {c} turn {t}"]},
                },
            }
        convos.append(
            {"conversation_id": f"conv-{c}", "id": f"conv-{c}", "title": "t", "mapping": mapping}
        )
    path.write_text(json.dumps(convos), encoding="utf-8")
    return path


def _expected_ids(first: int, last: int, *, turns_each: int = 10) -> list[str]:
    """The memory ids a ``_turns_export`` export produces for the 1-based global ordinals
    *first* through *last* (inclusive), independent of any test's own drift or resume logic --
    computed straight from the same ``_memory_id`` the importer itself calls, so an assertion
    against this proves identity, not just count."""
    ids = []
    for n in range(first, last + 1):
        c, t = divmod(n - 1, turns_each)
        ids.append(_memory_id(f"c{c}-m{t}", 0))
    return ids


async def test_a_failed_canary_stops_the_import_and_names_the_suspects(tmp_path):
    """20 conversations of 10 turns each is 200 memories. The model answers correctly through
    memory 50's canary and then has a bad moment: the canary at 100 is the first chance to
    notice, and it must stop the import before memory 101 is ever trusted."""
    export = _turns_export(tmp_path / "export.json", conversations=20, turns_each=10)
    gate, _inner, _conn = _build_gate(tmp_path, drift_after=50)

    with pytest.raises(ImportStopped) as exc:
        await import_chatgpt(export, gate=gate, user_id="u", canary_every=50)

    assert exc.value.first == 51
    assert exc.value.last == 100
    # Identity, not just count: these are the ids this run actually stored for that range,
    # not just any 50 ids.
    assert exc.value.suspect_ids == _expected_ids(51, 100)
    assert exc.value.setting == _SETTING
    message = str(exc.value)
    assert "51" in message and "100" in message
    assert exc.value.detail in message, "the mismatch's own detail (cosine, tolerance) is lost"
    assert exc.value.suspect_ids[0] in message and exc.value.suspect_ids[-1] in message
    assert "doctor --vectors" in message
    assert _SETTING in message
    # Plainly not repaired: stays stored, a re-run skips rather than fixes it.
    assert "stay stored" in message
    assert "skips them" in message
    assert "does not repair" in message
    assert isinstance(exc.value.__cause__, EmbeddingSpaceMismatch)


async def test_a_rerun_after_the_model_is_fixed_stores_the_rest_and_rewrites_nothing(tmp_path):
    """The canary must not cost the resume: an import stopped at memory 100 and re-run once
    the model is sound again stores the remaining 100 and re-embeds none of the first 100."""
    export = _turns_export(tmp_path / "export.json", conversations=20, turns_each=10)
    gate, inner, conn = _build_gate(tmp_path, drift_after=50)

    with pytest.raises(ImportStopped):
        await import_chatgpt(export, gate=gate, user_id="u", canary_every=50)
    after_first = inner.memories_seen

    # `morgan doctor --vectors` confirmed the model is sound again.
    inner.drift_after = 10_000
    second = await import_chatgpt(export, gate=gate, user_id="u", canary_every=50)

    assert after_first == 100
    assert second.memories == 100
    assert second.skipped_turns == 100, "the 100 already-stored pieces, and nothing else"
    assert inner.memories_seen == 200, "re-embedded a memory the first run had already stored"
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 200


async def test_a_resumed_run_that_itself_stops_names_only_its_own_suspects(tmp_path):
    """Ruling 4: a piece skipped because it is already stored costs no embedding and must not
    count toward the interval. Prime 60 memories with a model that never drifts, then resume
    over the full 200-memory export with a model that starts drifting globally after memory
    110 -- the first 60 are skipped (not embedded), so this run's own canary at its 50th store
    (global memory 110) still passes, and its canary at its 100th store (global 160) is first
    to fail. The range and ids it names must be this run's own, not the export's numbering."""
    small = _turns_export(tmp_path / "small.json", conversations=6, turns_each=10)  # 60 memories
    full = _turns_export(tmp_path / "full.json", conversations=20, turns_each=10)  # 200 memories
    gate, inner, conn = _build_gate(tmp_path, drift_after=10_000)  # nothing drifts yet

    priming = await import_chatgpt(small, gate=gate, user_id="u", canary_every=50)
    assert priming.memories == 60
    assert inner.memories_seen == 60

    inner.drift_after = 110  # global memory 110 is the last good one

    with pytest.raises(ImportStopped) as exc:
        await import_chatgpt(full, gate=gate, user_id="u", canary_every=50)

    assert exc.value.first == 51
    assert exc.value.last == 100
    assert exc.value.suspect_ids == _expected_ids(60 + 51, 60 + 100)
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 160


async def test_a_drift_after_the_last_full_interval_is_still_caught_at_the_end(tmp_path):
    """70 memories, canary every 50: the only interval boundary inside the run is 50, and the
    model is still good there. Nothing would ever re-check memories 51-70 unless one more
    canary runs once the loop itself ends."""
    export = _turns_export(tmp_path / "export.json", conversations=7, turns_each=10)
    gate, _inner, _conn = _build_gate(tmp_path, drift_after=50)

    with pytest.raises(ImportStopped) as exc:
        await import_chatgpt(export, gate=gate, user_id="u", canary_every=50)

    assert exc.value.first == 51
    assert exc.value.last == 70
    assert exc.value.suspect_ids == _expected_ids(51, 70)


def test_settings_refuses_a_canary_interval_below_one(tmp_path):
    """M6: the settings validation, for a caller that goes through ``Settings`` at all."""
    with pytest.raises(ValidationError, match="import_canary_every"):
        Settings(data_dir=str(tmp_path), import_canary_every=0)


async def test_import_chatgpt_refuses_a_canary_interval_below_one(tmp_path):
    """M6: ``import_chatgpt`` guards its own parameter too, by name, for a caller that builds
    no ``Settings`` and would otherwise divide by zero on the very first store."""
    export = _turns_export(tmp_path / "export.json", conversations=1, turns_each=1)
    gate, _inner, _conn = _build_gate(tmp_path, drift_after=10_000)

    with pytest.raises(ValueError, match="canary_every"):
        await import_chatgpt(export, gate=gate, user_id="u", canary_every=0)
