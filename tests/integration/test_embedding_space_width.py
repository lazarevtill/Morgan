"""The width of a database's vectors is read from the database, never asked of the model.

The active embedding space records the model and width that write the stored vectors. Opening
memory registers one on a writable database that has none -- the settings' model and width,
its fingerprint left for the first embedding call to record -- and refuses a space whose width
disagrees with ``MORGAN_EMBEDDING_DIM``, naming both. Neither sends a request. A model that
answers at another width is refused at its first answer instead, by the setting that addresses
it: the embedding endpoint whenever one is configured, not the chat endpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from morgan_brain.composition import build_memory_context, build_memory_module, sqlite_path, utcnow
from morgan_brain.config import Settings
from morgan_brain.memory import migrations
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.store import spaces
from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Memory
from morgan_brain.providers.wire import EmbeddingSpaceMismatch
from tests.fakes import counting_model_server, model_server


def test_a_fresh_database_registers_the_settings_space_without_embedding(tmp_path):
    with counting_model_server() as (url, calls):
        for _ in range(2):  # the second open finds the space and adds none
            build_memory_context(_settings(tmp_path, url)).conn.close()
        assert calls.total == 0

    [space] = _spaces(tmp_path)
    assert (space["model"], space["dims"], space["status"]) == ("a-model", 1024, "active")
    assert (space["table_name"], space["fingerprint"]) == ("vec_items", None)


def test_a_space_another_process_registered_first_is_not_registered_twice(tmp_path, monkeypatch):
    """Two opens of one fresh file race to register. The look that finds no space is taken
    again under the write lock; without it the loser inserts a second active space and dies on
    the partial unique index."""
    _a_database_with_no_space(tmp_path, dims=1024)
    other_process = open_db(_db(tmp_path))
    real_active = spaces.active
    looks: list[int] = []

    def the_other_process_registers_after_the_first_look(conn):
        looks.append(1)
        if len(looks) == 1:
            spaces.register(
                other_process, model="a-model", dims=1024, table_name="vec_items", clock=utcnow
            )
            return None
        return real_active(conn)

    monkeypatch.setattr(spaces, "active", the_other_process_registers_after_the_first_look)
    try:
        build_memory_context(_settings(tmp_path, "http://embed.invalid/v1")).conn.close()
    finally:
        other_process.close()

    assert len(_spaces(tmp_path)) == 1


def test_a_database_waiting_for_migrate_registers_nothing(tmp_path, monkeypatch):
    _a_database_with_no_space(tmp_path, dims=1024)
    heavy = migrations.Step(len(migrations._STEPS) + 1, "a heavy step", True, lambda c, s: None)
    monkeypatch.setattr(migrations, "_STEPS", (*migrations._STEPS, heavy))

    with counting_model_server() as (url, calls):
        ctx = build_memory_context(_settings(tmp_path, url))
        try:
            with pytest.raises(migrations.DatabaseNeedsMigration):
                ctx.gate.require_writable()
        finally:
            ctx.conn.close()
        assert calls.total == 0

    assert _spaces(tmp_path) == []


async def test_a_vector_table_of_another_width_is_refused_and_no_space_is_registered(tmp_path):
    """Every store built at one width and no space registered: what an open leaves on a
    database waiting for `morgan migrate`, which registers none. Registering the settings'
    width over that table would record a width the table does not have, and the space-width
    check would then approve it: every store fails on sqlite-vec's raw dimension error, and the
    open at the table's own width is refused with advice that is false."""
    _a_database_with_no_space(tmp_path, dims=4)
    assert _spaces(tmp_path) == []

    with counting_model_server(embedding_dim=8) as (url, calls):
        with pytest.raises(RuntimeError) as info:
            build_memory_context(_settings(tmp_path, url, embedding_dim=8))
        assert calls.total == 0

    message = str(info.value)
    assert "vec_items" in message and "4 wide" in message
    assert "MORGAN_EMBEDDING_DIM is 8" in message
    assert _spaces(tmp_path) == []  # no space, so no fingerprint either

    # At the table's own width the database opens, registers that width, and stores.
    with model_server(embedding_dim=4) as url:
        ctx = build_memory_context(_settings(tmp_path, url, embedding_dim=4))
        try:
            await ctx.gate.store(
                Memory(user_id=ctx.settings.owner_user_id, project="p", content="stored at 4")
            )
        finally:
            ctx.conn.close()

    [space] = _spaces(tmp_path)
    assert space["dims"] == 4 and space["fingerprint"] is not None


def test_the_hash_backend_registers_no_space(tmp_path):
    # No model writes its vectors, so there is no model to record, and nothing checks one.
    settings = Settings(data_dir=str(tmp_path), embedding_backend="hash", embedding_dim=1024)
    build_memory_context(settings).conn.close()

    assert _spaces(tmp_path) == []


def test_a_space_of_another_width_is_refused_naming_both_and_the_file_is_let_go(tmp_path):
    _a_database_with_no_space(tmp_path, dims=4096)
    conn = open_db(_db(tmp_path))
    spaces.register(conn, model="a-wide-model", dims=4096, table_name="vec_items", clock=utcnow)
    conn.close()

    with counting_model_server() as (url, calls):
        with pytest.raises(RuntimeError) as info:
            build_memory_context(_settings(tmp_path, url, embedding_dim=1024))
        assert calls.total == 0

    message = str(info.value)
    assert "embedding space 1 (a-wide-model" in message
    assert "4096" in message and "MORGAN_EMBEDDING_DIM" in message and "1024" in message
    # The refusal closed the connection it opened: nothing holds the file.
    Path(_db(tmp_path)).unlink()


async def test_a_model_of_another_width_is_refused_at_its_first_answer_by_its_own_setting(
    tmp_path,
):
    with model_server(embedding_dim=3) as url:
        ctx = build_memory_context(_settings(tmp_path, url, embedding_dim=1024))
        try:
            with pytest.raises(EmbeddingSpaceMismatch, match="3-dimensional") as info:
                await ctx.embedder.embed("a real query")
        finally:
            ctx.conn.close()

    assert "MORGAN_EMBEDDING_ENDPOINT" in str(info.value)
    assert "MORGAN_LLM_ENDPOINT" not in str(info.value)


def test_the_width_refusal_names_a_new_database_for_a_model_changed_on_purpose(tmp_path):
    """The setting says 1024 and the database was written at 4096: either the setting is wrong,
    or the model was changed on purpose, and this database keeps the model it was written with."""
    _a_database_with_no_space(tmp_path, dims=4096)
    conn = open_db(_db(tmp_path))
    spaces.register(conn, model="a-wide-model", dims=4096, table_name="vec_items", clock=utcnow)
    conn.close()

    with pytest.raises(RuntimeError) as info:
        build_memory_context(_settings(tmp_path, "http://embed.invalid/v1", embedding_dim=1024))

    message = str(info.value)
    assert "set MORGAN_EMBEDDING_DIM=4096, the width this database was written at" in message
    assert "point MORGAN_DATA_DIR at a new database if the model was changed on purpose" in message


# --- helpers ---------------------------------------------------------------------------------


def _settings(tmp_path: Path, url: str, *, embedding_dim: int = 1024) -> Settings:
    return Settings(
        data_dir=str(tmp_path),
        llm_endpoint="http://chat.invalid/v1",
        embedding_endpoint=url,
        embedding_backend="provider",
        embedding_model="a-model",
        embedding_dim=embedding_dim,
    )


def _db(tmp_path: Path) -> str:
    return sqlite_path(Settings(data_dir=str(tmp_path)).temporal_db_url)


def _a_database_with_no_space(tmp_path: Path, *, dims: int) -> None:
    """Every store, at the code's version, and no embedding space: what the stores alone
    leave, without the open that registers one."""
    conn = open_db(_db(tmp_path))
    try:
        build_memory_module(conn, embedder=FakeEmbedder(dim=dims), dim=dims)
    finally:
        conn.close()


def _spaces(tmp_path: Path) -> list[dict[str, object]]:
    """Every embedding space the database holds, whatever its status."""
    conn = open_db(_db(tmp_path))
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM embedding_spaces ORDER BY id")]
    finally:
        conn.close()
