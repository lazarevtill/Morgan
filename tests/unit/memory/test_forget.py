"""forget() — cascading erasure across every store, in one transaction.

The "forget me" premise is only real if every table that carries this user's text is emptied
atomically. A recall-only assertion would pass with orphaned rows left behind in
``vec_items``, so the completeness test queries the underlying tables directly.
"""

from __future__ import annotations

from datetime import UTC, datetime

from morgan_brain.memory.history import SessionHistoryStore
from morgan_brain.models import Entity, Memory, MemoryQuery, Message, Role, TemporalFact
from tests.unit.memory.conftest import build_memory_module as _module


async def test_forget_removes_from_every_index(tmp_path):
    path = str(tmp_path / "m.db")
    m = _module(path)
    await m.store(Memory(user_id="u", project="p", content="harbor mirror secret"))
    report = await m.forget(user_id="u", project="p")
    assert report.memories == 1
    reopened = _module(path)
    assert await reopened.recall(MemoryQuery(user_id="u", project="p", text="harbor")) == []


async def test_forget_is_project_scoped(tmp_path):
    path = str(tmp_path / "m.db")
    m = _module(path)
    await m.store(Memory(user_id="u", project="acme", content="harbor"))
    await m.store(Memory(user_id="u", project="personal", content="harbor"))
    await m.forget(user_id="u", project="acme")
    left = await m.recall(MemoryQuery(user_id="u", text="harbor", all_projects=True))
    assert len(left) == 1


async def test_forget_is_idempotent(tmp_path):
    path = str(tmp_path / "m.db")
    m = _module(path)
    await m.store(Memory(user_id="u", project="p", content="harbor"))
    await m.forget(user_id="u", project="p")
    second = await m.forget(user_id="u", project="p")
    assert second.memories == 0


async def test_forget_reports_an_absent_table_as_skipped_not_zero(tmp_path):
    """No SessionHistoryStore has opened on this connection, so ``history`` stays 0
    (honest: nothing was erased) AND the table is named in ``tables_skipped`` (honest:
    there was nothing to erase FROM). The vector and index tables DO exist here, so they
    must not be reported as skipped."""
    m = _module(str(tmp_path / "m.db"))
    await m.store(Memory(user_id="u", project="p", content="harbor"))
    report = await m.forget(user_id="u", project="p")
    assert report.history == 0
    assert report.tables_skipped == ["session_history"]


async def test_forget_does_not_report_present_tables_as_skipped(tmp_path):
    m = _module(str(tmp_path / "m.db"))
    conn = m._episodics._conn  # test-only introspection
    SessionHistoryStore(conn, clock=lambda: datetime.now(UTC))  # creates session_history
    await m.store(Memory(user_id="u", project="p", content="harbor"))
    report = await m.forget(user_id="u", project="p")
    assert report.history == 0
    assert report.tables_skipped == []


async def test_forget_empties_every_underlying_table(tmp_path):
    """The completeness proof: query every table directly and assert zero rows in each --
    not just that recall stops returning the memory."""
    m = _module(str(tmp_path / "m.db"))
    await m.store(
        Memory(
            user_id="u",
            project="p",
            content="Harbor mirror secret",
            entities=[Entity(name="harbor", type="place")],
        )
    )
    await m.upsert_fact(
        TemporalFact(user_id="u", project="p", subject="user", predicate="likes", object="tea")
    )
    conn = m._episodics._conn  # test-only introspection
    history = SessionHistoryStore(conn, clock=lambda: datetime.now(UTC))
    history.append(
        "u:s1", Message(user_id="u", role=Role.USER, content="harbor mirror"), project="p"
    )

    report = await m.forget(user_id="u", project="p")
    assert report.memories == 1
    assert report.facts == 1
    assert report.history == 1
    assert report.index_entries > 0

    for table, where in [
        ("memories", "user_id = 'u' AND project = 'p'"),
        ("fts_memories", "user_id = 'u' AND project = 'p'"),
        ("memory_entities", "user_id = 'u' AND project = 'p'"),
        ("vec_meta", "user_id = 'u' AND project = 'p'"),
        ("facts", "user_id = 'u' AND project = 'p'"),
        ("session_history", "user_id = 'u' AND project = 'p'"),
        ("mem_entity_nodes", "user_id = 'u' AND project = 'p'"),
        ("mem_schemas", "user_id = 'u' AND project = 'p'"),
    ]:
        # The table name and predicate come from the literal list above, not from data.
        sql = f"SELECT COUNT(*) AS n FROM {table} WHERE {where}"  # noqa: S608
        row = conn.execute(sql).fetchone()
        assert row["n"] == 0, f"{table} still has rows after forget()"

    # vec_items carries no id column of its own -- it is reachable only by rowid through
    # vec_meta, so an empty vec_meta plus an empty vec_items proves nothing is orphaned.
    orphaned = conn.execute("SELECT COUNT(*) AS n FROM vec_items").fetchone()
    assert orphaned["n"] == 0
