"""forget() — cascading erasure across every store, in one transaction.

The "forget me" premise is only real if every table that carries this user's text is emptied
atomically. A recall-only assertion would pass even with orphaned rows left behind in
``vec_items`` or ``interaction_signals`` (the tables that hold each turn's raw ``query``,
``original_reply`` and ``user_edit`` text), so the completeness test below queries the
underlying tables directly instead of trusting recall alone.
"""

from __future__ import annotations

from datetime import UTC, datetime

from morgan_brain.learning.history import SessionHistoryStore
from morgan_brain.learning.signals import InteractionSignal, SignalStore
from morgan_brain.models.base import Entity
from morgan_brain.models.memory import Memory, MemoryQuery, TemporalFact
from morgan_brain.models.message import Message, Role
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


async def test_forget_erases_signal_text(tmp_path):
    """interaction_signals holds query/original_reply/user_edit -- the premise covers it."""
    path = str(tmp_path / "m.db")
    m = _module(path)
    conn = m._episodics._conn  # test-only introspection
    signals = SignalStore(conn, clock=lambda: datetime.now(UTC))
    await signals.record(
        InteractionSignal(
            user_id="u",
            project="p",
            session_id="s1",
            turn_id="t1",
            query="what is the mirror address?",
            original_reply="harbor.example.internal",
        )
    )
    report = await m.forget(user_id="u", project="p")
    assert report.signals == 1


async def test_forget_is_idempotent(tmp_path):
    path = str(tmp_path / "m.db")
    m = _module(path)
    await m.store(Memory(user_id="u", project="p", content="harbor"))
    await m.forget(user_id="u", project="p")
    second = await m.forget(user_id="u", project="p")
    assert second.memories == 0


async def test_forget_reports_absent_tables_as_skipped_not_zero(tmp_path):
    """A table that never existed (no SignalStore/SessionHistoryStore opened on this
    connection, and the sqlite vector backend never used) must be distinguishable from a
    table that existed and was already empty -- see the Task 14 review / Task 17 brief.
    ``signals`` and ``history`` stay 0 (honest: nothing was erased) AND the table names show
    up in ``tables_skipped`` (honest: there was nothing to erase FROM, because it never
    existed)."""
    path = str(tmp_path / "m.db")
    m = _module(path)
    await m.store(Memory(user_id="u", project="p", content="harbor"))
    report = await m.forget(user_id="u", project="p")
    assert report.signals == 0
    assert report.history == 0
    assert "interaction_signals" in report.tables_skipped
    assert "session_history" in report.tables_skipped
    # vec_items/vec_meta DO exist here (SqliteVectorIndex backs conftest's build_memory_module),
    # so they must NOT be reported as skipped.
    assert "vec_items" not in report.tables_skipped


async def test_forget_does_not_report_present_tables_as_skipped(tmp_path):
    """Once a SignalStore/SessionHistoryStore has opened on the shared connection, the tables
    exist -- forgetting a project with no signals/history in it must report a real 0, not a
    skip, because the table is present and genuinely empty for this project."""
    path = str(tmp_path / "m.db")
    m = _module(path)
    conn = m._episodics._conn  # test-only introspection
    SignalStore(conn, clock=lambda: datetime.now(UTC))  # creates interaction_signals
    SessionHistoryStore(conn, clock=lambda: datetime.now(UTC))  # creates session_history
    await m.store(Memory(user_id="u", project="p", content="harbor"))
    report = await m.forget(user_id="u", project="p")
    assert report.signals == 0
    assert report.history == 0
    assert report.tables_skipped == []


async def test_forget_champions_flagged_is_empty_by_design(tmp_path):
    """No PromptRegistry is wired into MemoryModule, so this stays empty rather than
    inventing a half-mechanism -- see the forget() docstring for why."""
    path = str(tmp_path / "m.db")
    m = _module(path)
    report = await m.forget(user_id="u", project="p")
    assert report.champions_flagged == []


async def test_forget_empties_every_underlying_table(tmp_path):
    """The completeness proof: query memories, vec_items/vec_meta, fts_memories,
    memory_entities, facts, interaction_signals and session_history directly and assert
    zero rows in each -- not just that recall stops returning the memory."""
    path = str(tmp_path / "m.db")
    m = _module(path)
    await m.store(
        Memory(
            user_id="u",
            project="p",
            content="harbor mirror secret",
            entities=[Entity(name="harbor", type="place")],
        )
    )
    await m.upsert_fact(
        TemporalFact(user_id="u", project="p", subject="user", predicate="likes", object="tea")
    )
    conn = m._episodics._conn  # test-only introspection

    signals = SignalStore(conn, clock=lambda: datetime.now(UTC))
    await signals.record(
        InteractionSignal(
            user_id="u",
            project="p",
            session_id="s1",
            turn_id="t1",
            query="what is the mirror address?",
            original_reply="harbor.example.internal",
        )
    )

    history = SessionHistoryStore(conn, clock=lambda: datetime.now(UTC))
    history.append(
        "u:s1", Message(user_id="u", role=Role.USER, content="harbor mirror"), project="p"
    )

    report = await m.forget(user_id="u", project="p")
    assert report.memories == 1
    assert report.facts == 1
    assert report.signals == 1
    assert report.history == 1
    assert report.champions_flagged == []

    for table, where in [
        ("memories", "user_id = 'u' AND project = 'p'"),
        ("fts_memories", "user_id = 'u' AND project = 'p'"),
        ("memory_entities", "user_id = 'u' AND project = 'p'"),
        ("vec_meta", "user_id = 'u' AND project = 'p'"),
        ("facts", "user_id = 'u' AND project = 'p'"),
        ("interaction_signals", "user_id = 'u' AND project = 'p'"),
        ("session_history", "user_id = 'u' AND project = 'p'"),
    ]:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {table} WHERE {where}").fetchone()
        assert row["n"] == 0, f"{table} still has rows after forget()"

    # vec_items carries no id column of its own -- it is reachable only by rowid through
    # vec_meta, so an empty vec_meta plus an empty vec_items proves nothing is orphaned.
    orphaned = conn.execute("SELECT COUNT(*) AS n FROM vec_items").fetchone()
    assert orphaned["n"] == 0
