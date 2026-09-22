"""Every table with a project column is in the one registry forget reads, and forget erases
every table in it.

A table that keys a project and is not listed outlives every forget, holding the owner's words
for as long as the database exists. So would a listed table forget had no deleter for; forget
refuses one by name before it erases anything.
"""

from __future__ import annotations

import struct
from datetime import UTC, datetime

import pytest

from morgan_brain.memory.store import projects as projects_store
from morgan_brain.memory.store import spaces, tables
from morgan_brain.memory.store.history import SessionHistoryStore
from morgan_brain.memory.store.tables import NAME_KEYED_PROJECT_TABLES, project_tables
from morgan_brain.models import Entity, Memory, Message, Role, TemporalFact
from tests.unit.memory.conftest import build_memory_module


def _project_keyed(conn) -> set[str]:
    """Every real table with a project column. vec0 keeps shadow tables of its own
    (``vec_items_chunks`` and friends); they are the virtual table's own storage, not tables a
    caller writes, and forget reaches them by dropping from the virtual table."""
    found = set()
    for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'vec_items_%'"
    ):
        name = row["name"]
        if any(c["name"] == "project" for c in conn.execute(f"PRAGMA table_info({name})")):
            found.add(name)
    return found


def _full_stack_conn(tmp_path):
    """``build_memory_module`` alone never opens ``session_history`` -- only the composition
    root's ``build_memory_context`` does, because ``MemoryModule`` itself has no use for it.
    Both registry checks below care whether a *registered* table is ever gone from the schema,
    so the connection needs every store that owns a project-keyed table opened on it, the same
    way ``test_forget.py::test_forget_does_not_report_present_tables_as_skipped`` does."""
    conn = build_memory_module(str(tmp_path / "m.db"))._conn
    SessionHistoryStore(conn, clock=lambda: datetime.now(UTC))
    return conn


def test_every_project_keyed_table_is_registered(tmp_path):
    conn = _full_stack_conn(tmp_path)
    unregistered = _project_keyed(conn) - set(project_tables(conn))
    assert not unregistered, f"project-keyed but unregistered: {sorted(unregistered)}"


def test_the_registry_names_no_table_that_is_gone(tmp_path):
    conn = _full_stack_conn(tmp_path)
    for name in project_tables(conn):
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (name,)).fetchone(), (
            f"{name} is registered and does not exist"
        )


def test_the_name_keyed_registry_also_names_no_table_that_is_gone(tmp_path):
    """``projects`` is keyed by ``name`` -- the project's own name is its primary key, not a
    ``project`` column -- so ``_project_keyed`` above never finds it. It has its own registry,
    ``NAME_KEYED_PROJECT_TABLES``, checked here the same way."""
    conn = _full_stack_conn(tmp_path)
    for name in NAME_KEYED_PROJECT_TABLES:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (name,)).fetchone(), (
            f"{name} is registered and does not exist"
        )


async def test_forget_removes_the_projects_row(tmp_path):
    """The remote URL and the root path a name-keyed row carries are the owner's data -- a
    project's own repository and where it lives on disk -- so ``forget`` erases that row like
    any other, even though it never joins the id-based deletes above."""
    module = build_memory_module(str(tmp_path / "m.db"))
    await module.store(Memory(user_id="u", project="p", content="harbor mirror secret"))
    projects_store.seed(module._conn, clock=lambda: datetime.now(UTC))
    assert projects_store.get(module._conn, "p") is not None

    await module.forget(user_id="u", project="p")

    assert projects_store.get(module._conn, "p") is None


def _clock() -> datetime:
    return datetime.now(UTC)


def _rows(conn, table: str, project: str) -> int:
    """Rows of *project* in *table*, read through the table's own SELECT -- a vec0 table
    answers a plain ``WHERE project = ?`` on its metadata column too."""
    # `table` comes from the registry or from this module's literals, never from data.
    sql = f"SELECT COUNT(*) FROM {table} WHERE project = ?"  # noqa: S608
    return int(conn.execute(sql, (project,)).fetchone()[0])


async def _write_every_table(module, history: SessionHistoryStore, project: str) -> None:
    """Rows for *project* in every registered table, each through its store's real write
    path: a memory with an entity (``memories``, ``vec_meta``, ``vec_items``,
    ``fts_memories``, ``memory_entities``), a fact, a session-history row. ``projects`` is
    seeded from these by the caller."""
    await module.store(
        Memory(
            user_id="u",
            project=project,
            content=f"harbor mirror secret of {project}",
            entities=[Entity(name="harbor", type="place")],
        )
    )
    await module.upsert_fact(
        TemporalFact(user_id="u", project=project, subject="user", predicate="likes", object="tea")
    )
    history.append(
        f"u:{project}",
        Message(user_id="u", role=Role.USER, content=f"harbor mirror of {project}"),
        project=project,
    )


async def test_forget_erases_every_registered_table_and_leaves_other_projects(tmp_path):
    """Walks the registry rather than a list of its own, so a table registered tomorrow is
    checked here the day it is registered -- and the precondition fails first if nothing in
    this test writes to it."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    history = SessionHistoryStore(conn, clock=_clock)
    for project in ("p", "q"):
        await _write_every_table(module, history, project)
    projects_store.seed(conn, clock=_clock)
    registered = project_tables(conn)
    before_q = {table: _rows(conn, table, "q") for table in registered}
    for table in registered:
        assert _rows(conn, table, "p") > 0, f"nothing in this test writes {table} for p"
        assert before_q[table] > 0, f"nothing in this test writes {table} for q"
    kept = projects_store.get(conn, "q")
    assert projects_store.get(conn, "p") is not None
    assert kept is not None

    report = await module.forget(user_id="u", project="p")

    assert (report.memories, report.facts, report.history) == (1, 1, 1)
    assert report.tables_skipped == []
    for table in registered:
        assert _rows(conn, table, "p") == 0, f"{table} still holds p after forget"
        assert _rows(conn, table, "q") == before_q[table], f"forget of p touched q in {table}"
    assert projects_store.get(conn, "p") is None
    assert projects_store.get(conn, "q") == kept


#: A second embedding space's vec0 table, at the DDL ``vec_items`` has. Named outside
#: ``vec_items_%``, the prefix of vec0's own shadow tables that the registry test skips.
_SECOND_SPACE = "vec_second_space"
_SECOND_SPACE_DDL = f"""CREATE VIRTUAL TABLE {_SECOND_SPACE} USING vec0(
    embedding float[4] distance_metric=cosine,
    user_id TEXT,
    project TEXT,
    status TEXT,
    scope TEXT,
    author_id TEXT
)"""


def _vectors_in(conn, table: str) -> list[tuple[int, str, str]]:
    """Every row of a vec0 table, as (rowid, user, project), read through its own SELECT."""
    # `table` is one of this module's literals, never data.
    sql = f"SELECT rowid, user_id, project FROM {table} ORDER BY rowid"  # noqa: S608
    return [(r["rowid"], r["user_id"], r["project"]) for r in conn.execute(sql)]


async def test_forget_erases_a_second_embedding_spaces_vectors(tmp_path):
    """A second space's table is erased by its own ``user_id`` and ``project`` columns. Its
    rows are written here at rowids of their own, ``q``'s first, the order a re-embed walking
    the archive its own way would give them: ``vec_meta`` has ``p`` at rowid 1 and ``q`` at 2,
    this table the other way round. ``q``'s vector is kept, ``p``'s goes, and another user's
    vector in ``p`` stays."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    for project in ("p", "q"):
        await module.store(Memory(user_id="u", project=project, content=f"harbor of {project}"))
    meta = {r["project"]: r["rowid"] for r in conn.execute("SELECT rowid, project FROM vec_meta")}
    assert meta == {"p": 1, "q": 2}
    spaces.register(
        conn,
        model="second-model",
        dims=4,
        table_name=_SECOND_SPACE,
        status="shadow",
        clock=_clock,
    )
    conn.execute(_SECOND_SPACE_DDL)
    for rowid, user_id, project in [(1, "u", "q"), (2, "u", "p"), (3, "v", "p")]:
        conn.execute(
            f"INSERT INTO {_SECOND_SPACE} "  # noqa: S608
            "(rowid, embedding, user_id, project, status, scope, author_id) "
            "VALUES (?, ?, ?, ?, 'stored', 'private', ?)",
            (rowid, struct.pack("4f", 0.5, 0.5, 0.5, 0.5), user_id, project, user_id),
        )
    conn.commit()
    assert _SECOND_SPACE in project_tables(conn)

    await module.forget(user_id="u", project="p")

    assert _vectors_in(conn, _SECOND_SPACE) == [(1, "u", "q"), (3, "v", "p")]


async def test_a_space_table_without_user_and_project_columns_stops_forget(tmp_path):
    """A space's table is erased by its own ``user_id`` and ``project`` columns, so a
    registered one without them cannot be erased: forget refuses it by name, and nothing
    of ``p`` is erased anywhere."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    await _write_every_table(module, SessionHistoryStore(conn, clock=_clock), "p")
    projects_store.seed(conn, clock=_clock)
    spaces.register(
        conn, model="bare-model", dims=4, table_name="vec_bare", status="shadow", clock=_clock
    )
    conn.execute("CREATE VIRTUAL TABLE vec_bare USING vec0(embedding float[4])")
    conn.execute(
        "INSERT INTO vec_bare (rowid, embedding) VALUES (1, ?)",
        (struct.pack("4f", 0.5, 0.5, 0.5, 0.5),),
    )
    conn.commit()
    keyed = [t for t in project_tables(conn) if t != "vec_bare"]
    before = {table: _rows(conn, table, "p") for table in keyed}
    assert all(before.values()), before

    with pytest.raises(RuntimeError, match="vec_bare"):
        await module.forget(user_id="u", project="p")

    assert {table: _rows(conn, table, "p") for table in keyed} == before
    assert conn.execute("SELECT COUNT(*) FROM vec_bare").fetchone()[0] == 1
    assert projects_store.get(conn, "p") is not None


async def test_a_registered_table_without_a_deleter_stops_forget_before_it_erases(
    tmp_path, monkeypatch
):
    """The day a store registers a table and forgets its deleter, forget refuses by the
    table's name, and every row of the project -- in that table and in every other -- is
    still there."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    await _write_every_table(module, SessionHistoryStore(conn, clock=_clock), "p")
    projects_store.seed(conn, clock=_clock)
    conn.execute("CREATE TABLE stray_notes (user_id TEXT, project TEXT, note TEXT)")
    conn.execute("INSERT INTO stray_notes VALUES ('u', 'p', 'harbor mirror secret')")
    conn.commit()
    monkeypatch.setattr(tables, "PROJECT_TABLES", (*tables.PROJECT_TABLES, "stray_notes"))
    before = {table: _rows(conn, table, "p") for table in project_tables(conn)}
    assert all(before.values()), before

    with pytest.raises(RuntimeError, match="stray_notes"):
        await module.forget(user_id="u", project="p")

    assert {table: _rows(conn, table, "p") for table in project_tables(conn)} == before
    assert projects_store.get(conn, "p") is not None


async def test_a_name_keyed_table_without_a_deleter_also_stops_forget(tmp_path, monkeypatch):
    """The same rule for ``NAME_KEYED_PROJECT_TABLES``: registered there with no deleter, a
    table stops forget by name, and every row of the project -- in that table and in every
    other -- is still there."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    await _write_every_table(module, SessionHistoryStore(conn, clock=_clock), "p")
    projects_store.seed(conn, clock=_clock)
    conn.execute("CREATE TABLE stray_names (name TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO stray_names VALUES ('p')")
    conn.commit()
    monkeypatch.setattr(
        tables, "NAME_KEYED_PROJECT_TABLES", (*tables.NAME_KEYED_PROJECT_TABLES, "stray_names")
    )
    before = {table: _rows(conn, table, "p") for table in project_tables(conn)}
    assert all(before.values()), before

    with pytest.raises(RuntimeError, match="stray_names"):
        await module.forget(user_id="u", project="p")

    assert {table: _rows(conn, table, "p") for table in project_tables(conn)} == before
    for table in tables.NAME_KEYED_PROJECT_TABLES:
        # `table` comes from the registry, never from data.
        sql = f"SELECT COUNT(*) FROM {table} WHERE name = 'p'"  # noqa: S608
        assert conn.execute(sql).fetchone()[0] == 1, f"{table} lost p's row"


#: The tables a memory is indexed in, each with a ``user_id`` and a ``project`` column.
_INDEXES = ("memories", "memory_entities", "fts_memories", "vec_meta", "vec_items")


def _owned(conn, table: str, user_id: str, project: str) -> int:
    # `table` is one of `_INDEXES`, never data.
    sql = f"SELECT COUNT(*) FROM {table} WHERE user_id = ? AND project = ?"  # noqa: S608
    return int(conn.execute(sql, (user_id, project)).fetchone()[0])


async def test_forget_erases_orphaned_index_rows_and_never_another_users(tmp_path):
    """An index row of ``p`` whose memory is gone -- its ``memories`` row, or its ``vec_meta``
    row -- is erased by its own ``user_id`` and ``project`` columns. Another user's rows in
    ``p``, and the owner's rows in ``q``, are left as they were."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    harbor = [Entity(name="harbor", type="place")]
    lost_row = Memory(user_id="u", project="p", content="harbor lost row", entities=harbor)
    lost_meta = Memory(user_id="u", project="p", content="harbor lost meta", entities=harbor)
    for memory in (
        lost_row,
        lost_meta,
        Memory(user_id="v", project="p", content="harbor of v", entities=harbor),
        Memory(user_id="u", project="q", content="harbor of q", entities=harbor),
    ):
        await module.store(memory)
    # Orphans: every index row of `lost_row` without its memory, and `lost_meta`'s vector
    # without the `vec_meta` row that addresses it.
    conn.execute("DELETE FROM memories WHERE id = ?", (lost_row.id,))
    conn.execute("DELETE FROM vec_meta WHERE id = ?", (lost_meta.id,))
    conn.commit()
    for table in _INDEXES[1:]:
        assert _owned(conn, table, "u", "p") > 0, f"no orphan of p in {table}"
    kept = {
        (table, user_id, project): _owned(conn, table, user_id, project)
        for table in _INDEXES
        for user_id, project in (("v", "p"), ("u", "q"))
    }
    assert all(kept.values()), kept

    await module.forget(user_id="u", project="p")

    for table in _INDEXES:
        assert _owned(conn, table, "u", "p") == 0, f"{table} still holds an orphan of p"
    assert {key: _owned(conn, *key) for key in kept} == kept
