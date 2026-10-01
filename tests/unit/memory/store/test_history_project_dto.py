"""History content was scoped correctly; returned messages must preserve that project."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from morgan_brain.memory.store.db import open_db
from morgan_brain.models import Message, Role
from tests.unit.memory.conftest import a_history_store


@pytest.mark.parametrize("owner", ["alice", "bob"])
@pytest.mark.parametrize("project", ["orchid", "personal"])
def test_reopened_history_preserves_project_and_scoped_tail(
    tmp_path: Path, owner: str, project: str
) -> None:
    path = str(tmp_path / "history.sqlite")
    clock = lambda: datetime(2026, 10, 1, tzinfo=UTC)  # noqa: E731
    conn = open_db(path)
    store = a_history_store(conn, clock=clock)
    # Deliberately collide the low-level key; owner/project filtering must precede limit.
    key = "shared-key"
    for sequence in range(3):
        for write_owner in ("alice", "bob"):
            for write_project in ("orchid", "personal"):
                store.append(
                    key,
                    Message(
                        user_id=write_owner,
                        role=Role.USER if sequence % 2 == 0 else Role.ASSISTANT,
                        content=f"{write_owner}:{write_project}:{sequence}",
                    ),
                    project=write_project,
                )
    conn.close()

    reopened = open_db(path)
    try:
        history = a_history_store(reopened, clock=clock)
        messages = history.recent(key, user_id=owner, project=project, limit=2)
        assert [m.content for m in messages] == [
            f"{owner}:{project}:1",
            f"{owner}:{project}:2",
        ]
        assert [m.role for m in messages] == [Role.ASSISTANT, Role.USER]
        assert all(m.user_id == owner for m in messages)
        assert all(m.project == project for m in messages)
        assert history.recent(key, user_id=owner, project="untouched", limit=2) == []
    finally:
        reopened.close()
