"""Every table with a project column is in the one registry forget reads, and forget erases
every table in it.

A table that keys a project and is not listed outlives every forget, holding the owner's words
for as long as the database exists. So would a listed table forget had no deleter for; forget
refuses one by name before it erases anything.
"""

from __future__ import annotations

import json
import struct
from datetime import UTC, datetime

import pytest

from morgan_brain.composition import build_memory_module as build_full_stack_module
from morgan_brain.memory.embedder import FakeEmbedder
from morgan_brain.memory.store import calls, digests, spaces, tables
from morgan_brain.memory.store import projects as projects_store
from morgan_brain.memory.store import sessions as sessions_store
from morgan_brain.memory.store.db import open_db, write_transaction
from morgan_brain.memory.store.digests import DigestLineRef, DigestRef, DigestRow
from morgan_brain.memory.store.history import SessionHistoryStore
from morgan_brain.memory.store.tables import ANSWERED_BY, NAME_KEYED_PROJECT_TABLES, project_tables
from morgan_brain.models import (
    CaptureCursor,
    Entity,
    Memory,
    Message,
    Role,
    Session,
    TemporalFact,
    Turn,
    session_id_of,
    utc_iso,
)
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
    """``build_memory_module`` opens every store, the history store included; the explicit
    ``SessionHistoryStore`` here is kept so the test reads as the composition root wires it.
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


NOW = "2026-09-22T10:00:00.000Z"


def _session(project: str, native_id: str, *, user_id: str = "u") -> Session:
    return Session(
        id=session_id_of("claude-code", native_id),
        user_id=user_id,
        project=project,
        harness="claude-code",
        native_id=native_id,
        source_path=f"/transcripts/{native_id}.jsonl",
        cwd=f"/src/{project}",
        project_source="git",
        started_at=NOW,
        reader_version=1,
        gate_version=1,
    )


def _archive_rows(
    conn, project: str, native_id: str, *, memory_id: str
) -> tuple[Session, list[int]]:
    """A session of *project* with two turns, a correction row, a link between them, a rating
    of that link, a digest quoting the second turn and the memory, a rating of the digest, a
    call, a pause and a cursor -- every archive table, through its store."""
    session = _session(project, native_id)
    digest_id = f"d-{project}-{native_id}"
    with write_transaction(conn):
        sessions_store.upsert_session(conn, session, now=NOW)
        ids = sessions_store.insert_turns(
            conn,
            [
                Turn(
                    session_id=session.id,
                    user_id="u",
                    project=project,
                    native_key="a:0",
                    role="user",
                    text=f"harbor mirror secret of {project}",
                    ts=NOW,
                ),
                Turn(
                    session_id=session.id,
                    user_id="u",
                    project=project,
                    native_key="b:0",
                    role="user",
                    text=f"no, the mirror of {project} is on the other host",
                    ts=NOW,
                ),
            ],
        )
        conn.execute(
            "INSERT INTO corrections_fts (rowid, norm, project) VALUES (?, ?, ?)",
            (ids[1], f"no the mirror of {project} is on the other host", project),
        )
        conn.execute("UPDATE turns SET is_correction = 1 WHERE id = ?", (ids[1],))
        conn.execute(
            "INSERT INTO turn_links (turn_id, earlier_turn_id, user_id, project, earlier_project, "
            "similarity, lexicon_version, computed_at) VALUES (?, ?, 'u', ?, ?, 0.8, 1, ?)",
            (ids[1], ids[0], project, project, NOW),
        )
        digests.rate_link(
            conn,
            turn_id=ids[1],
            earlier_turn_id=ids[0],
            user_id="u",
            project=project,
            rating="right",
            now=NOW,
        )
        digests.insert_digest(
            conn,
            DigestRow(
                id=digest_id,
                ts=NOW,
                user_id="u",
                project=project,
                harness="claude-code",
                native_session_id=native_id,
                source="startup",
                entrypoint="cli",
                entrypoint_source="env",
                first_for_session=True,
                text=f"| memory: harbor mirror secret of {project}\n",
                lines=(DigestLineRef(1, "memory", memory_id),),
                refs=(DigestRef("memory", memory_id), DigestRef("turn", str(ids[1]))),
                chars=40,
                ms=1,
            ),
        )
        digests.rate_line(conn, digest_id=digest_id, line_no=1, rating="right", now=NOW)
        calls.insert_call(
            conn,
            calls.CallRecord(
                ts=NOW,
                surface="cli",
                client="cli",
                native_session_id=native_id,
                command="recall",
                user_id="u",
                project=project,
                all_projects=False,
                outcome="ok",
                embed_outcome="ok",
                degraded=None,
                degrade_reason=None,
                embed_latency_ms=1.0,
                total_ms=2.0,
                query_language="en",
                reason=None,
            ),
        )
        sessions_store.pause_open(
            conn, user_id="u", project=project, paused_from=NOW, paused_until=None
        )
        sessions_store.cursor_put(
            conn,
            CaptureCursor(
                harness="claude-code",
                native_id=native_id,
                source_path=session.source_path,
                byte_offset=10,
                size=10,
                mtime_ns=1,
                identity="x",
                status="ok",
                last_read_at=NOW,
            ),
        )
    return session, ids


async def _write_every_table(module, history: SessionHistoryStore, project: str) -> None:
    """Rows for *project* in every registered table, each through its store's real write
    path: a memory with an entity (``memories``, ``vec_meta``, ``vec_items``,
    ``fts_memories``, ``memory_entities``), a fact, a session-history row, and the archive's
    rows (``_archive_rows``). ``projects`` is seeded from these by the caller."""
    memory_id = await module.store(
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
    _archive_rows(module._conn, project, f"s-{project}", memory_id=memory_id)


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
    assert (report.sessions, report.turns, report.digests) == (1, 2, 1)
    assert report.tables_skipped == []
    for table in registered:
        assert _rows(conn, table, "p") == 0, f"{table} still holds p after forget"
        assert _rows(conn, table, "q") == before_q[table], f"forget of p touched q in {table}"
    assert projects_store.get(conn, "p") is None
    assert projects_store.get(conn, "q") == kept
    assert sessions_store.cursor_get(conn, harness="claude-code", native_id="s-p") is None
    assert sessions_store.is_excluded(conn, harness="claude-code", native_id="s-p")
    assert sessions_store.cursor_get(conn, harness="claude-code", native_id="s-q") is not None
    assert not sessions_store.is_excluded(conn, harness="claude-code", native_id="s-q")


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


async def test_the_projects_row_stays_while_another_owner_has_data_in_the_project(tmp_path):
    """A project's row is shared by every owner with data in it, so one owner's forget keeps
    it while another's rows remain, and the last owner's forget removes it."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    history = SessionHistoryStore(conn, clock=_clock)
    for user_id in ("a", "b"):
        await module.store(Memory(user_id=user_id, project="p", content=f"harbor of {user_id}"))
        history.append(
            f"{user_id}:s", Message(user_id=user_id, role=Role.USER, content="tide"), project="p"
        )
    projects_store.seed(conn, clock=_clock)
    assert projects_store.get(conn, "p") is not None

    await module.forget(user_id="a", project="p")

    assert projects_store.get(conn, "p") is not None, "b's data is still in p"

    await module.forget(user_id="b", project="p")

    assert projects_store.get(conn, "p") is None


async def test_a_single_owners_forget_removes_the_projects_row_once_every_table_is_empty(
    tmp_path,
):
    """With every registered table holding the owner's rows for ``p``, the row goes: the check
    runs after the erasure, so it sees none of the rows the same forget deleted."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    await _write_every_table(module, SessionHistoryStore(conn, clock=_clock), "p")
    projects_store.seed(conn, clock=_clock)
    assert all(_rows(conn, table, "p") for table in project_tables(conn))

    await module.forget(user_id="u", project="p")

    assert projects_store.get(conn, "p") is None


#: A moment distinct from ``NOW``, the session's own ``started_at``/``updated_at`` literal, so
#: an exclusion's ``excluded_at`` equalling it cannot be a coincidence of a shared timestamp.
_ERASED_AT = datetime(2026, 9, 23, 12, 0, 0, tzinfo=UTC)


def _erasure_time_module(tmp_path):
    """A module whose clock always answers ``_ERASED_AT``, built through the composition root
    directly rather than the test conftest's wrapper, which hard-codes ``datetime.now(UTC)``."""
    conn = open_db(str(tmp_path / "m.db"))
    return build_full_stack_module(
        conn, embedder=FakeEmbedder(dim=4), dim=4, clock=lambda: _ERASED_AT
    )


async def test_a_project_grain_forget_stamps_the_exclusion_with_the_erasure_time(tmp_path):
    """The exclusion a project-grain forget writes for each session it erases whole carries
    the moment the erasure ran, not the session's own ``updated_at``."""
    module = _erasure_time_module(tmp_path)
    conn = module._conn
    await _write_every_table(module, SessionHistoryStore(conn, clock=lambda: _ERASED_AT), "p")

    await module.forget(user_id="u", project="p")

    [exclusion] = [e for e in sessions_store.exclusions(conn) if e.native_id == "s-p"]
    assert exclusion.excluded_at == utc_iso(_ERASED_AT)
    assert exclusion.excluded_at != NOW


async def test_a_session_grain_forget_stamps_the_exclusion_with_the_erasure_time(tmp_path):
    """The same, for ``forget_sessions``: the exclusion it writes carries the erasure's own
    moment, not the session's ``updated_at``."""
    module = _erasure_time_module(tmp_path)
    conn = module._conn
    await _write_every_table(module, SessionHistoryStore(conn, clock=lambda: _ERASED_AT), "p")

    await module.forget_sessions(
        user_id="u",
        project="p",
        session_ids=[session_id_of("claude-code", "s-p")],
        reason="forget",
    )

    [exclusion] = [e for e in sessions_store.exclusions(conn) if e.native_id == "s-p"]
    assert exclusion.excluded_at == utc_iso(_ERASED_AT)
    assert exclusion.excluded_at != NOW


async def test_the_session_grain_erases_one_sessions_rows_and_leaves_the_rest(tmp_path):
    """Two sessions of ``p``: the erased one's turns, both FTS rows, links and link ratings from
    either end, digests with their refs and ratings, call rows, cursor and lease go, and an
    exclusion keeps the next sweep out; the other session, the memories, the facts, the history
    and the pause stay."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    history = SessionHistoryStore(conn, clock=_clock)
    await _write_every_table(module, history, "p")
    memory_id = str(conn.execute("SELECT id FROM memories WHERE project = 'p'").fetchone()[0])
    other, other_ids = _archive_rows(conn, "p", "s-other", memory_id=memory_id)
    erased_session = session_id_of("claude-code", "s-p")
    before = {table: _rows(conn, table, "p") for table in project_tables(conn)}

    report = await module.forget_sessions(
        user_id="u", project="p", session_ids=[erased_session], reason="forget"
    )

    assert (report.sessions, report.turns, report.links, report.digests, report.excluded) == (
        1,
        2,
        1,
        1,
        1,
    )
    assert report.memories_reached is False
    assert sessions_store.get_session(conn, erased_session) is None
    assert sessions_store.get_session(conn, other.id) is not None
    assert [t.id for t in sessions_store.turns_of(conn, other.id)] == other_ids
    for table in (
        "turns",
        "turns_fts",
        "corrections_fts",
        "turn_links",
        "link_ratings",
        "digests",
        "call_log",
    ):
        assert _rows(conn, table, "p") == before[table] // 2, table
    assert conn.execute("SELECT COUNT(*) FROM digest_refs").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM digest_ratings").fetchone()[0] == 1
    for table in (
        "memories",
        "facts",
        "session_history",
        "capture_pauses",
        "memory_entities",
        "vec_meta",
        "fts_memories",
    ):
        assert _rows(conn, table, "p") == before[table], table
    assert sessions_store.cursor_get(conn, harness="claude-code", native_id="s-p") is None
    exclusion = [e for e in sessions_store.exclusions(conn) if e.native_id == "s-p"]
    assert [e.reason for e in exclusion] == ["forget"]
    assert sessions_store.cursor_get(conn, harness="claude-code", native_id="s-other") is not None
    assert not conn.in_transaction


async def test_a_session_of_another_owner_or_project_is_not_erased_by_name(tmp_path):
    """Naming a session in *session_ids* is not enough: ``forget_sessions`` also scopes by the
    caller's own ``user_id`` and ``project``, so a session of another project (``s-q``) and one
    of another owner in the same project (``s-v``) are both left alone."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    await _write_every_table(module, SessionHistoryStore(conn, clock=_clock), "p")
    await _write_every_table(module, SessionHistoryStore(conn, clock=_clock), "q")
    with write_transaction(conn):
        sessions_store.upsert_session(conn, _session("p", "s-v", user_id="v"), now=NOW)

    report = await module.forget_sessions(
        user_id="u",
        project="p",
        session_ids=[session_id_of("claude-code", "s-q"), session_id_of("claude-code", "s-v")],
        reason="forget",
    )

    assert (report.sessions, report.turns, report.excluded) == (0, 0, 0)
    assert sessions_store.get_session(conn, session_id_of("claude-code", "s-q")) is not None
    assert sessions_store.get_session(conn, session_id_of("claude-code", "s-v")) is not None


async def test_a_table_with_neither_a_session_deleter_nor_the_declaration_stops_the_erasure(
    tmp_path, monkeypatch
):
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    await _write_every_table(module, SessionHistoryStore(conn, clock=_clock), "p")
    conn.execute("CREATE TABLE stray_notes (user_id TEXT, project TEXT, note TEXT)")
    conn.execute("INSERT INTO stray_notes VALUES ('u', 'p', 'harbor mirror secret')")
    conn.commit()
    monkeypatch.setattr(tables, "PROJECT_TABLES", (*tables.PROJECT_TABLES, "stray_notes"))
    before = {table: _rows(conn, table, "p") for table in project_tables(conn)}

    with pytest.raises(RuntimeError, match="stray_notes"):
        await module.forget_sessions(
            user_id="u",
            project="p",
            session_ids=[session_id_of("claude-code", "s-p")],
            reason="forget",
        )

    assert {table: _rows(conn, table, "p") for table in project_tables(conn)} == before
    assert sessions_store.cursor_get(conn, harness="claude-code", native_id="s-p") is not None


async def test_a_second_embedding_spaces_table_holds_nothing_per_session(tmp_path):
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    await _write_every_table(module, SessionHistoryStore(conn, clock=_clock), "p")
    spaces.register(
        conn, model="second-model", dims=4, table_name=_SECOND_SPACE, status="shadow", clock=_clock
    )
    conn.execute(_SECOND_SPACE_DDL)
    conn.execute(
        f"INSERT INTO {_SECOND_SPACE} "  # noqa: S608
        "(rowid, embedding, user_id, project, status, scope, author_id) "
        "VALUES (1, ?, 'u', 'p', 'stored', 'private', 'u')",
        (struct.pack("4f", 0.5, 0.5, 0.5, 0.5),),
    )
    conn.commit()

    report = await module.forget_sessions(
        user_id="u", project="p", session_ids=[session_id_of("claude-code", "s-p")], reason="forget"
    )

    assert report.sessions == 1
    assert _vectors_in(conn, _SECOND_SPACE) == [(1, "u", "p")]


async def test_links_and_link_ratings_go_from_either_end_at_the_project_grain(tmp_path):
    """A link between two projects, its earlier turn in ``p`` and its later turn (and its own
    ``project``) in ``q``, goes with its rating when ``p`` is forgotten, because its
    ``earlier_project`` matches too; ``q``'s own link stays."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    await _write_every_table(module, SessionHistoryStore(conn, clock=_clock), "p")
    await _write_every_table(module, SessionHistoryStore(conn, clock=_clock), "q")
    p_ids = [t.id for t in sessions_store.turns_of(conn, session_id_of("claude-code", "s-p"))]
    q_ids = [t.id for t in sessions_store.turns_of(conn, session_id_of("claude-code", "s-q"))]
    with write_transaction(conn):
        conn.execute(
            "INSERT INTO turn_links (turn_id, earlier_turn_id, user_id, project, earlier_project, "
            "similarity, lexicon_version, computed_at) VALUES (?, ?, 'u', 'q', 'p', 0.9, 1, ?)",
            (q_ids[1], p_ids[0], NOW),
        )
        digests.rate_link(
            conn,
            turn_id=q_ids[1],
            earlier_turn_id=p_ids[0],
            user_id="u",
            project="q",
            rating="wrong",
            now=NOW,
        )

    await module.forget(user_id="u", project="p")

    links = conn.execute("SELECT turn_id, earlier_turn_id FROM turn_links").fetchall()
    assert [tuple(r) for r in links] == [(q_ids[1], q_ids[0])]
    ratings = conn.execute("SELECT turn_id, earlier_turn_id FROM link_ratings").fetchall()
    assert [tuple(r) for r in ratings] == [(q_ids[1], q_ids[0])]


async def test_links_and_link_ratings_go_from_either_end_at_the_session_grain(tmp_path):
    """Two sessions of the same project, ``s-p`` (erased) and ``s-other`` (kept), with a link
    from a later turn of ``s-p`` to an earlier turn of ``s-other``, and the mirror -- a later
    turn of ``s-other`` to an earlier turn of ``s-p``. Erasing ``s-p`` alone takes both links
    and both ratings, because either end sits in the erased session; ``s-other``'s own link
    (written by ``_archive_rows``) stays."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    history = SessionHistoryStore(conn, clock=_clock)
    await _write_every_table(module, history, "p")
    memory_id = str(conn.execute("SELECT id FROM memories WHERE project = 'p'").fetchone()[0])
    other, other_ids = _archive_rows(conn, "p", "s-other", memory_id=memory_id)
    p_ids = [t.id for t in sessions_store.turns_of(conn, session_id_of("claude-code", "s-p"))]
    with write_transaction(conn):
        conn.execute(
            "INSERT INTO turn_links (turn_id, earlier_turn_id, user_id, project, earlier_project, "
            "similarity, lexicon_version, computed_at) VALUES (?, ?, 'u', ?, ?, 0.9, 1, ?)",
            (p_ids[1], other_ids[0], "p", "p", NOW),
        )
        digests.rate_link(
            conn,
            turn_id=p_ids[1],
            earlier_turn_id=other_ids[0],
            user_id="u",
            project="p",
            rating="right",
            now=NOW,
        )
        conn.execute(
            "INSERT INTO turn_links (turn_id, earlier_turn_id, user_id, project, earlier_project, "
            "similarity, lexicon_version, computed_at) VALUES (?, ?, 'u', ?, ?, 0.9, 1, ?)",
            (other_ids[1], p_ids[0], "p", "p", NOW),
        )
        digests.rate_link(
            conn,
            turn_id=other_ids[1],
            earlier_turn_id=p_ids[0],
            user_id="u",
            project="p",
            rating="wrong",
            now=NOW,
        )

    await module.forget_sessions(
        user_id="u", project="p", session_ids=[session_id_of("claude-code", "s-p")], reason="forget"
    )

    links = {
        tuple(r) for r in conn.execute("SELECT turn_id, earlier_turn_id FROM turn_links").fetchall()
    }
    assert links == {(other_ids[1], other_ids[0])}
    ratings = {
        tuple(r)
        for r in conn.execute("SELECT turn_id, earlier_turn_id FROM link_ratings").fetchall()
    }
    assert ratings == {(other_ids[1], other_ids[0])}
    assert [t.id for t in sessions_store.turns_of(conn, other.id)] == other_ids


async def test_forget_sessions_refuses_to_run_inside_a_write_transaction(tmp_path):
    """``forget_sessions()`` checkpoints the write-ahead log once its erasure has committed,
    and SQLite refuses that checkpoint while any transaction is open, even the caller's own.
    Nested in a caller's ``write_transaction``, it would erase, fail at the checkpoint, and
    have the caller's block roll the erasure back behind an error that says nothing about
    nesting. It refuses up front instead, before touching anything, the same way ``forget()``
    already does for its own ``VACUUM``."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    await _write_every_table(module, SessionHistoryStore(conn, clock=_clock), "p")
    session_id = session_id_of("claude-code", "s-p")

    with pytest.raises(RuntimeError, match="write transaction"), write_transaction(conn):
        await module.forget_sessions(
            user_id="u", project="p", session_ids=[session_id], reason="forget"
        )

    assert sessions_store.get_session(conn, session_id) is not None
    assert sessions_store.cursor_get(conn, harness="claude-code", native_id="s-p") is not None
    assert [e for e in sessions_store.exclusions(conn) if e.native_id == "s-p"] == []


async def test_delete_sessions_refuses_an_empty_erasure_time(tmp_path):
    """``Erasure.erased_at`` defaults to ``""``, which is not an ISO timestamp: a caller that
    built one without filling it would persist an empty ``capture_exclusions.excluded_at``,
    sorting before every real time. ``delete_sessions`` raises instead, before touching the
    session, its cursor or writing an exclusion."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    await _write_every_table(module, SessionHistoryStore(conn, clock=_clock), "p")
    session_id = session_id_of("claude-code", "s-p")
    erasure = tables.Erasure(
        user_id="u",
        project="p",
        memory_ids="[]",
        vector_rowids="[]",
        grain="sessions",
        session_ids=json.dumps([session_id]),
        native_ids=json.dumps(["s-p"]),
    )

    with pytest.raises(ValueError, match="erased_at"), write_transaction(conn):
        sessions_store.delete_sessions(conn, erasure)

    assert sessions_store.get_session(conn, session_id) is not None
    assert sessions_store.cursor_get(conn, harness="claude-code", native_id="s-p") is not None
    assert [e for e in sessions_store.exclusions(conn) if e.native_id == "s-p"] == []


def test_the_reads_that_walk_the_registry_never_touch_an_fts_table(tmp_path):
    """``distinct_projects`` and ``_holds_rows_of`` skip ``turns_fts`` and ``corrections_fts``:
    ``turns`` answers for them, and a filter on an FTS5 table's UNINDEXED column is a full scan
    inside the lock."""
    module = build_memory_module(str(tmp_path / "m.db"))
    conn = module._conn
    SessionHistoryStore(conn, clock=_clock)
    assert ANSWERED_BY == {"turns_fts": "turns", "corrections_fts": "turns"}
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    module._episodics.distinct_projects("u")
    projects_store._holds_rows_of(conn, "p")
    conn.set_trace_callback(None)
    touched = [s for s in statements if "turns_fts" in s or "corrections_fts" in s]
    assert touched == []
    assert any("FROM turns " in s or "FROM turns\n" in s for s in statements)
