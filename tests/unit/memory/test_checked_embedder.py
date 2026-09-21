"""The first embedding call a process makes also proves the model is the one that wrote the rows.

Trust on first use is closed by the sample: a NULL fingerprint is recorded only when freshly
embedded stored rows match the vectors already in the database.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from structlog.testing import capture_logs

from morgan_brain.config import Settings
from morgan_brain.memory import checked_embedder, fingerprint
from morgan_brain.memory.checked_embedder import CheckedEmbedder
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.module import MemoryModule
from morgan_brain.memory.store import spaces, vectors
from morgan_brain.models import Memory, MemoryKind
from morgan_brain.providers.embeddings import OpenAICompatEmbedder
from morgan_brain.providers.factory import retry_budget_of
from morgan_brain.providers.wire import EmbeddingSpaceMismatch
from tests.fakes import model_server
from tests.unit.memory.conftest import build_memory_module


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    return build_memory_module(str(tmp_path / "m.db"))._conn


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=str(tmp_path), embedding_backend="hash")


async def test_the_first_call_carries_the_strings_and_later_calls_do_not(conn, settings):
    inner = _recording_embedder(dims=4)
    embedder = CheckedEmbedder(
        inner, conn=conn, settings=settings, endpoint=_URL, setting=_SETTING, sample=lambda n: []
    )

    await embedder.embed("a real query")
    await embedder.embed("another")

    assert len(inner.calls[0]) == 6  # the query plus five strings
    assert len(inner.calls[1]) == 1


async def test_a_same_width_model_that_answers_differently_is_named(conn, settings):
    space = _a_space_with_a_recorded_fingerprint(conn, dims=4)
    embedder = CheckedEmbedder(
        _other_model(dims=4),
        conn=conn,
        settings=settings,
        endpoint=_URL,
        setting=_SETTING,
        sample=lambda n: [],
    )

    with pytest.raises(EmbeddingSpaceMismatch) as exc:
        await embedder.embed("a real query")

    message = str(exc.value)
    assert space.model in message and "4096" not in message
    assert "MORGAN_EMBEDDING_MODEL" in message and "0.995" in message


async def test_a_null_fingerprint_is_recorded_only_when_the_stored_sample_matches(conn, settings):
    spaces.register(conn, model="m", dims=4, table_name="vec_items", clock=_clock)
    stored = [("id-1 text", [1.0, 0.0, 0.0, 0.0])]
    embedder = CheckedEmbedder(
        _embedder_returning({"id-1 text": [1.0, 0.0, 0.0, 0.0]}),
        conn=conn,
        settings=settings,
        endpoint=_URL,
        setting=_SETTING,
        sample=lambda n: stored,
    )

    await embedder.embed("a real query")

    assert spaces.active(conn).fingerprint is not None


async def test_a_sample_that_does_not_match_refuses_and_records_nothing(conn, settings):
    spaces.register(conn, model="m", dims=4, table_name="vec_items", clock=_clock)
    embedder = CheckedEmbedder(
        _embedder_returning({"id-1 text": [0.0, 1.0, 0.0, 0.0]}),
        conn=conn,
        settings=settings,
        endpoint=_URL,
        setting=_SETTING,
        sample=lambda n: [("id-1 text", [1.0, 0.0, 0.0, 0.0])],
    )

    with pytest.raises(EmbeddingSpaceMismatch):
        await embedder.embed("a real query")

    assert spaces.active(conn).fingerprint is None


async def test_a_width_change_is_still_caught(conn, settings):
    spaces.register(conn, model="m", dims=4, table_name="vec_items", clock=_clock)
    embedder = CheckedEmbedder(
        _recording_embedder(dims=3),
        conn=conn,
        settings=settings,
        endpoint=_URL,
        setting=_SETTING,
        sample=lambda n: [],
    )

    with pytest.raises(EmbeddingSpaceMismatch, match="3-dimensional"):
        await embedder.embed("a real query")


async def test_with_no_active_space_it_embeds_and_logs_once(conn, settings):
    # A database below version 6 has no space registered until `morgan migrate`; it is
    # read-only meanwhile, and recall must keep working on it.
    inner = _recording_embedder(dims=4)

    with capture_logs() as logs:
        first = await CheckedEmbedder(
            inner, conn=conn, settings=settings, endpoint=_URL, setting=_SETTING, sample=_no_sample
        ).embed("a real query")
        second = await CheckedEmbedder(
            inner, conn=conn, settings=settings, endpoint=_URL, setting=_SETTING, sample=_no_sample
        ).embed("another")

    assert first == _unit("model-a", "a real query", 4)
    assert second == _unit("model-a", "another", 4)
    events = [entry["event"] for entry in logs]
    assert events.count("embedding-space.none-registered") == 1


async def test_exactly_the_five_string_vectors_are_compared(conn, settings, monkeypatch):
    _a_space_with_a_recorded_fingerprint(conn, dims=4)
    compared: list[tuple[list[list[float]], list[list[float]]]] = []
    real_compare = fingerprint.compare

    def spy(fresh: list[list[float]], stored: list[list[float]]) -> fingerprint.Comparison:
        compared.append((fresh, stored))
        return real_compare(fresh, stored)

    monkeypatch.setattr(fingerprint, "compare", spy)
    embedder = CheckedEmbedder(
        _recording_embedder(dims=4),
        conn=conn,
        settings=settings,
        endpoint=_URL,
        setting=_SETTING,
        sample=_no_sample,
    )

    got = await embedder.embed_batch(["one", "two", "three"])

    [(fresh, stored)] = compared
    assert len(fresh) == len(stored) == len(fingerprint.STRINGS) == 5
    assert fresh == [_unit("model-a", s, 4) for s in fingerprint.STRINGS]
    # The caller gets its own vectors back, in order, and none of the check's.
    assert got == [_unit("model-a", t, 4) for t in ("one", "two", "three")]


async def test_the_recorded_fingerprint_is_the_five_strings_alone(conn, settings):
    spaces.register(conn, model="m", dims=4, table_name="vec_items", clock=_clock)
    stored = [(t, _unit("model-a", t, 4)) for t in ("row one", "row two")]
    embedder = CheckedEmbedder(
        _recording_embedder(dims=4),
        conn=conn,
        settings=settings,
        endpoint=_URL,
        setting=_SETTING,
        sample=lambda n: stored,
    )

    got = await embedder.embed_batch(["query one", "query two"])

    recorded = spaces.unpack(spaces.active(conn).fingerprint, dims=4)
    assert len(recorded) == 5
    for vector, text in zip(recorded, fingerprint.STRINGS, strict=True):
        assert vector == pytest.approx(_unit("model-a", text, 4))
    assert got == [_unit("model-a", t, 4) for t in ("query one", "query two")]


async def test_the_sample_asks_for_the_configured_number_of_rows(conn, settings):
    spaces.register(conn, model="m", dims=4, table_name="vec_items", clock=_clock)
    asked: list[int] = []

    def sample(n: int) -> list[tuple[str, list[float]]]:
        asked.append(n)
        return []

    tuned = settings.model_copy(update={"embedding_fingerprint_sample_rows": 7})
    await CheckedEmbedder(
        _recording_embedder(dims=4),
        conn=conn,
        settings=tuned,
        endpoint=_URL,
        setting=_SETTING,
        sample=sample,
    ).embed("q")

    assert asked == [7]


async def test_once_per_process_means_a_second_instance_sends_no_strings(conn, settings):
    _a_space_with_a_recorded_fingerprint(conn, dims=4)
    first_inner = _recording_embedder(dims=4)
    second_inner = _recording_embedder(dims=4)

    await CheckedEmbedder(
        first_inner,
        conn=conn,
        settings=settings,
        endpoint=_URL,
        setting=_SETTING,
        sample=_no_sample,
    ).embed("a")
    await CheckedEmbedder(
        second_inner,
        conn=conn,
        settings=settings,
        endpoint=_URL,
        setting=_SETTING,
        sample=_no_sample,
    ).embed("b")

    assert len(first_inner.calls[0]) == 6
    assert second_inner.calls == [["b"]]


async def test_a_mismatch_is_raised_again_on_the_next_call(conn, settings):
    # A refused check is not remembered as done: the next call would otherwise hand back a
    # vector from the wrong model, silently -- the failure this check exists to prevent.
    _a_space_with_a_recorded_fingerprint(conn, dims=4)
    embedder = CheckedEmbedder(
        _other_model(dims=4),
        conn=conn,
        settings=settings,
        endpoint=_URL,
        setting=_SETTING,
        sample=_no_sample,
    )

    with pytest.raises(EmbeddingSpaceMismatch):
        await embedder.embed("first")
    with pytest.raises(EmbeddingSpaceMismatch):
        await embedder.embed("second")


async def test_a_fresh_space_with_no_rows_records_its_fingerprint_directly(conn, settings):
    space = spaces.register(conn, model="m", dims=4, table_name="vec_items", clock=_clock)

    with capture_logs() as logs:
        await CheckedEmbedder(
            _recording_embedder(dims=4),
            conn=conn,
            settings=settings,
            endpoint=_URL,
            setting=_SETTING,
            sample=_no_sample,
        ).embed("q")

    assert spaces.active(conn).fingerprint is not None
    [recorded] = [e for e in logs if e["event"] == "embedding-space.fingerprint-recorded"]
    assert recorded["space_id"] == space.id and recorded["model"] == "m"


async def test_verify_sends_only_the_check_and_once_proven_sends_nothing(conn, settings):
    """``morgan migrate`` checks the space with no text of its own: the request carries the
    five strings alone, and the fingerprint is recorded as a first call would record it."""
    spaces.register(conn, model="m", dims=4, table_name="vec_items", clock=_clock)
    inner = _recording_embedder(dims=4)
    embedder = CheckedEmbedder(
        inner, conn=conn, settings=settings, endpoint=_URL, setting=_SETTING, sample=_no_sample
    )

    await embedder.verify()
    await embedder.verify()

    assert inner.calls == [list(fingerprint.STRINGS)]
    assert spaces.active(conn).fingerprint is not None


async def test_verify_with_no_active_space_sends_nothing(conn, settings):
    inner = _recording_embedder(dims=4)
    embedder = CheckedEmbedder(
        inner, conn=conn, settings=settings, endpoint=_URL, setting=_SETTING, sample=_no_sample
    )

    await embedder.verify()

    assert inner.calls == []


async def test_check_sends_only_the_five_strings_and_never_records(conn, settings):
    """Task 26's import canary. Unlike ``verify()``, ``check()`` is never short-circuited by
    ``_checked`` and never records -- a full round trip against a space that already has a
    fingerprint, called twice, sends the five strings both times."""
    _a_space_with_a_recorded_fingerprint(conn, dims=4)
    before = spaces.active(conn).fingerprint
    inner = _recording_embedder(dims=4)
    embedder = CheckedEmbedder(inner, conn=conn, settings=settings, endpoint=_URL, setting=_SETTING)

    await embedder.check()
    await embedder.check()

    assert inner.calls == [list(fingerprint.STRINGS), list(fingerprint.STRINGS)]
    assert spaces.active(conn).fingerprint == before


async def test_check_refuses_a_wrong_answer_count(conn, settings):
    """I1: a canary answer that cannot even be lined up against the fingerprint string by
    string must not reach ``fingerprint.compare``'s bare ``ValueError`` -- the import that
    sent it only catches ``EmbeddingSpaceMismatch``, so a short answer must be one."""
    _a_space_with_a_recorded_fingerprint(conn, dims=4)
    embedder = CheckedEmbedder(
        _short_answering_model(dims=4, short_by=2),
        conn=conn,
        settings=settings,
        endpoint=_URL,
        setting=_SETTING,
    )

    with pytest.raises(EmbeddingSpaceMismatch, match="3 vectors for 5 inputs") as exc:
        await embedder.check()

    assert exc.value.setting == _SETTING


async def test_check_refuses_a_non_finite_answer(conn, settings):
    """I1's "at least one non-cosine path": ``_require_answers`` reused through ``check()``,
    not only through ``embed()``/``_first_call``."""
    _a_space_with_a_recorded_fingerprint(conn, dims=4)
    embedder = CheckedEmbedder(
        _nan_model(dims=4), conn=conn, settings=settings, endpoint=_URL, setting=_SETTING
    )

    with pytest.raises(EmbeddingSpaceMismatch, match="non-finite"):
        await embedder.check()


async def test_check_refuses_when_no_fingerprint_is_recorded_yet(conn, settings):
    """M2: a real, registered space with nothing recorded gets its own honest detail, naming
    the space that is really there."""
    space = spaces.register(conn, model="m", dims=4, table_name="vec_items", clock=_clock)
    embedder = CheckedEmbedder(
        _recording_embedder(dims=4), conn=conn, settings=settings, endpoint=_URL, setting=_SETTING
    )

    with pytest.raises(EmbeddingSpaceMismatch, match="no fingerprint is recorded") as exc:
        await embedder.check()

    assert exc.value.space_id == space.id
    assert exc.value.setting == _SETTING


async def test_check_refuses_when_no_space_is_active(conn, settings):
    """M2: no space at all must get its own detail too, and must not claim a fake one --
    "embedding space 0 (..., 0 dims)" would name a space that does not exist."""
    embedder = CheckedEmbedder(
        _recording_embedder(dims=4), conn=conn, settings=settings, endpoint=_URL, setting=_SETTING
    )

    with pytest.raises(EmbeddingSpaceMismatch, match="no embedding space is registered") as exc:
        await embedder.check()

    assert "embedding space 0" not in str(exc.value)
    assert exc.value.setting == _SETTING


async def test_the_stored_sample_pairs_each_memory_with_its_stored_vector(tmp_path):
    module = build_memory_module(str(tmp_path / "m.db"))
    texts = ["the first memory", "the second memory", "the third memory"]
    for text in texts:
        await _store(module, text)

    pairs = vectors.stored_sample(module._conn, table_name="vec_items", n=2)

    assert len(pairs) == 2
    for text, vector in pairs:
        assert text in texts
        assert vector == pytest.approx(await FakeEmbedder(dim=4).embed(text))
    assert len(vectors.stored_sample(module._conn, table_name="vec_items", n=10)) == 3


def test_the_stored_sample_of_an_empty_database_is_empty(conn):
    assert vectors.stored_sample(conn, table_name="vec_items", n=5) == []


async def test_by_default_the_sample_is_read_from_the_database(tmp_path, settings):
    # The production sample: the memories' own text, and the vectors stored beside them.
    module = build_memory_module(str(tmp_path / "m.db"))
    await _store(module, "a memory the stored model embedded")
    conn = module._conn
    spaces.register(conn, model="m", dims=4, table_name="vec_items", clock=_clock)

    await CheckedEmbedder(
        FakeEmbedder(dim=4), conn=conn, settings=settings, endpoint=_URL, setting=_SETTING
    ).embed("q")

    assert spaces.active(conn).fingerprint is not None


async def test_by_default_a_stored_row_from_another_model_refuses(tmp_path, settings):
    module = build_memory_module(str(tmp_path / "m.db"))
    await _store(module, "a memory the stored model embedded")
    conn = module._conn
    spaces.register(conn, model="m", dims=4, table_name="vec_items", clock=_clock)

    with pytest.raises(EmbeddingSpaceMismatch):
        await CheckedEmbedder(
            _other_model(dims=4), conn=conn, settings=settings, endpoint=_URL, setting=_SETTING
        ).embed("q")

    assert spaces.active(conn).fingerprint is None


async def test_a_nan_answer_against_a_recorded_fingerprint_is_refused_and_not_cached(
    conn, settings
):
    _a_space_with_a_recorded_fingerprint(conn, dims=4)
    embedder = CheckedEmbedder(
        _nan_model(dims=4),
        conn=conn,
        settings=settings,
        endpoint=_URL,
        setting=_SETTING,
        sample=_no_sample,
    )

    with pytest.raises(EmbeddingSpaceMismatch, match="non-finite"):
        await embedder.embed("a real query")

    assert checked_embedder._checked == set()
    with pytest.raises(EmbeddingSpaceMismatch, match="non-finite"):
        await embedder.embed("again")


async def test_a_nan_answer_over_a_stored_sample_records_nothing(conn, settings):
    spaces.register(conn, model="m", dims=4, table_name="vec_items", clock=_clock)
    embedder = CheckedEmbedder(
        _nan_model(dims=4),
        conn=conn,
        settings=settings,
        endpoint=_URL,
        setting=_SETTING,
        sample=lambda n: [("id-1 text", [1.0, 0.0, 0.0, 0.0])],
    )

    with pytest.raises(EmbeddingSpaceMismatch, match="non-finite"):
        await embedder.embed("a real query")

    assert spaces.active(conn).fingerprint is None
    assert checked_embedder._checked == set()


def test_a_sample_of_zero_rows_is_refused_by_the_settings():
    # Zero rows would record whatever model answered first, unverified.
    with pytest.raises(ValidationError):
        Settings(embedding_fingerprint_sample_rows=0)


async def test_an_empty_sample_over_stored_rows_is_refused_and_records_nothing(tmp_path, settings):
    # Only a space with no stored vectors at all may record its fingerprint unverified.
    module = build_memory_module(str(tmp_path / "m.db"))
    await _store(module, "a memory the stored model embedded")
    conn = module._conn
    spaces.register(conn, model="m", dims=4, table_name="vec_items", clock=_clock)
    embedder = CheckedEmbedder(
        _other_model(dims=4),
        conn=conn,
        settings=settings,
        endpoint=_URL,
        setting=_SETTING,
        sample=_no_sample,
    )

    with pytest.raises(EmbeddingSpaceMismatch, match="cannot be verified"):
        await embedder.embed("q")
    with pytest.raises(EmbeddingSpaceMismatch, match="cannot be verified"):
        await embedder.embed("q")

    assert spaces.active(conn).fingerprint is None
    assert checked_embedder._checked == set()


async def test_the_stored_sample_draws_past_a_memory_whose_vector_is_missing(tmp_path):
    module = build_memory_module(str(tmp_path / "m.db"))
    for text in ("kept", "vector lost one", "vector lost two"):
        await _store(module, text)
    conn = module._conn
    conn.execute(
        "DELETE FROM vec_items WHERE rowid IN "
        "(SELECT rowid FROM vec_meta WHERE id IN "
        "(SELECT id FROM memories WHERE content LIKE 'vector lost%'))"
    )
    conn.commit()

    # Random order: a sample that stopped at the first n candidates would mostly come back
    # empty here, because two of the three have no vector to pair with.
    for _ in range(20):
        pairs = vectors.stored_sample(conn, table_name="vec_items", n=1)
        assert [text for text, _ in pairs] == ["kept"]


async def test_a_server_that_reorders_its_answers_still_passes_the_check(conn, tmp_path):
    spaces.register(conn, model="m", dims=4, table_name="vec_items", clock=_clock)
    with model_server(embedding_dim=4) as url:
        in_order = Settings(data_dir=str(tmp_path), embedding_endpoint=url)
        expected = await _adapter(url, in_order).embed("a real query")
        await CheckedEmbedder(
            _adapter(url, in_order), conn=conn, settings=in_order, endpoint=url, setting=_SETTING
        ).embed("q")
    assert spaces.active(conn).fingerprint is not None
    checked_embedder._checked.clear()  # a new process

    with model_server(embedding_dim=4, reorder=True) as url:
        reordered = Settings(data_dir=str(tmp_path), embedding_endpoint=url)
        embedder = CheckedEmbedder(
            _adapter(url, reordered), conn=conn, settings=reordered, endpoint=url, setting=_SETTING
        )
        got = await embedder.embed("a real query")

    assert got == expected


# --- helpers ---------------------------------------------------------------------------------

#: Where the embedder under test is addressed, and by which setting: the factory decides both
#: in production and hands them in.
_URL = "http://embed.test/v1"
_SETTING = "MORGAN_EMBEDDING_ENDPOINT"


def _clock() -> datetime:
    return datetime(2026, 9, 21, tzinfo=UTC)


def _no_sample(n: int) -> list[tuple[str, list[float]]]:
    return []


def _unit(model: str, text: str, dims: int) -> list[float]:
    """A deterministic unit vector per (model, text): same model, same answer; another model,
    a direction that has nothing to do with the first. Signed components, so two models'
    answers are far apart rather than all crowded into the positive orthant."""
    digest = hashlib.sha256(f"{model}\0{text}".encode()).digest()
    raw = [digest[i % len(digest)] / 127.5 - 1.0 for i in range(dims)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class _RecordingEmbedder:
    """An embedding model that records every request it is sent, one list of inputs each."""

    def __init__(
        self,
        *,
        dims: int,
        model: str,
        answers: dict[str, list[float]] | None = None,
        nan: bool = False,
        short_by: int = 0,
    ) -> None:
        self._dims = dims
        self._model = model
        self._answers = answers or {}
        self._nan = nan
        self._short_by = short_by
        self.calls: list[list[str]] = []

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self._nan:
            return [[math.nan] * self._dims for _ in texts]
        answered = [self._answers.get(t) or _unit(self._model, t, self._dims) for t in texts]
        return answered[: len(answered) - self._short_by] if self._short_by else answered


def _recording_embedder(*, dims: int) -> _RecordingEmbedder:
    return _RecordingEmbedder(dims=dims, model="model-a")


def _other_model(*, dims: int) -> _RecordingEmbedder:
    return _RecordingEmbedder(dims=dims, model="model-b")


def _nan_model(*, dims: int) -> _RecordingEmbedder:
    """A broken model whose every component is NaN -- which Python's JSON parser accepts from a
    bare `NaN` token, so the live adapter can hand one back too."""
    return _RecordingEmbedder(dims=dims, model="model-a", nan=True)


def _short_answering_model(*, dims: int, short_by: int) -> _RecordingEmbedder:
    """A model that drops the last *short_by* vectors of every answer -- a malformed reply a
    count check must catch before anything tries to line it up against the fingerprint."""
    return _RecordingEmbedder(dims=dims, model="model-a", short_by=short_by)


def _adapter(url: str, settings: Settings) -> OpenAICompatEmbedder:
    return OpenAICompatEmbedder(
        url,
        settings.embedding_model,
        budget=retry_budget_of(settings, "interactive"),
        setting="MORGAN_EMBEDDING_ENDPOINT",
        key_setting="MORGAN_LLM_API_KEY",
    )


def _embedder_returning(answers: dict[str, list[float]]) -> _RecordingEmbedder:
    return _RecordingEmbedder(dims=4, model="model-a", answers=answers)


def _a_space_with_a_recorded_fingerprint(
    conn: sqlite3.Connection, *, dims: int
) -> spaces.EmbeddingSpace:
    space = spaces.register(conn, model="model-a", dims=dims, table_name="vec_items", clock=_clock)
    spaces.record_fingerprint(
        conn, space.id, [_unit("model-a", s, dims) for s in fingerprint.STRINGS], clock=_clock
    )
    return space


async def _store(module: MemoryModule, text: str) -> None:
    await module.store(Memory(user_id="u1", kind=MemoryKind.EPISODIC, content=text))
