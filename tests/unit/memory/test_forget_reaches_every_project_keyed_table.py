"""Every table with a project column is in the one registry forget reads.

A table that keys a project and is not listed outlives every forget, holding the owner's words
for as long as the database exists.
"""

from __future__ import annotations

from datetime import UTC, datetime

from morgan_brain.memory.store import projects as projects_store
from morgan_brain.memory.store.history import SessionHistoryStore
from morgan_brain.memory.store.tables import NAME_KEYED_PROJECT_TABLES, project_tables
from morgan_brain.models import Memory
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
