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
from morgan_brain.memory.store import spaces, tables, vectors
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


def _audited(conn, project: str) -> list[str]:
    """The texts ``doctor --vectors`` reads for *project* from the second space's table."""
    sample = vectors.audit_sample(
        conn, table_name=_SECOND_SPACE, n=10, user_id="u", project=project, all_projects=False
    )
    return [text for _id, text, _vector in sample]


async def test_forget_erases_a_second_embedding_spaces_vectors(tmp_path):
    """A space registers its vec0 table in ``embedding_spaces``, and ``vectors.py`` reads any
    space's table at the rowid ``vec_meta`` gives a memory (``stored_sample``,
    ``audit_sample``). The second space's vectors are written at those rowids, and the audit
    reading them back for ``p`` proves they sit where the code looks for them."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    for project in ("p", "q"):
        await module.store(Memory(user_id="u", project=project, content=f"harbor of {project}"))
    spaces.register(
        conn,
        model="second-model",
        dims=4,
        table_name=_SECOND_SPACE,
        status="shadow",
        clock=_clock,
    )
    conn.execute(_SECOND_SPACE_DDL)
    for row in conn.execute("SELECT rowid, user_id, project FROM vec_meta").fetchall():
        conn.execute(
            f"INSERT INTO {_SECOND_SPACE} "  # noqa: S608
            "(rowid, embedding, user_id, project, status, scope, author_id) "
            "VALUES (?, ?, ?, ?, 'stored', 'private', ?)",
            (
                row["rowid"],
                struct.pack("4f", 0.5, 0.5, 0.5, 0.5),
                row["user_id"],
                row["project"],
                row["user_id"],
            ),
        )
    conn.commit()
    assert _SECOND_SPACE in project_tables(conn)
    assert _audited(conn, "p") == ["harbor of p"]

    await module.forget(user_id="u", project="p")

    assert _rows(conn, _SECOND_SPACE, "p") == 0
    assert _rows(conn, _SECOND_SPACE, "q") == 1
    assert _audited(conn, "q") == ["harbor of q"]


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
    table stops forget by name, and the project's own row in it is still there."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    await module.store(Memory(user_id="u", project="p", content="harbor mirror secret"))
    conn.execute("CREATE TABLE stray_names (name TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO stray_names VALUES ('p')")
    conn.commit()
    monkeypatch.setattr(
        tables, "NAME_KEYED_PROJECT_TABLES", (*tables.NAME_KEYED_PROJECT_TABLES, "stray_names")
    )

    with pytest.raises(RuntimeError, match="stray_names"):
        await module.forget(user_id="u", project="p")

    assert conn.execute("SELECT name FROM stray_names").fetchall()[0]["name"] == "p"
    assert _rows(conn, "memories", "p") == 1
