"""Every table with a project column is in the one registry forget reads.

A table that keys a project and is not listed outlives every forget, holding the owner's words
for as long as the database exists.
"""

from __future__ import annotations

from datetime import UTC, datetime

from morgan_brain.memory.store.history import SessionHistoryStore
from morgan_brain.memory.store.tables import project_tables
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
