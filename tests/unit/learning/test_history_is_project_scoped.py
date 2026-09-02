"""Session history does not cross projects.

Rows carried a `project` from the day project scoping landed, but `recent()` read by
`session_id` alone -- and `session_key` falls back to a per-user `"<user>:default"` bucket
whenever the client sends no `session_id`, which the CLI and every default API call do. So one
history key served every project: a turn in one repository put the previous turn from another
straight into the prompt, and erasing a project did not stop its transcript reappearing under
the next one.
"""

from __future__ import annotations

from datetime import UTC, datetime

from morgan_brain.memory.history import SessionHistoryStore, session_key
from morgan_brain.models import Message, Role


def _store() -> SessionHistoryStore:
    return SessionHistoryStore(clock=lambda: datetime.now(UTC))


def _msg(text: str) -> Message:
    return Message(user_id="u", role=Role.USER, content=text)


def test_the_default_session_bucket_does_not_leak_between_projects() -> None:
    """The realistic case: no session_id, so every project shares one key."""
    store = _store()
    key = session_key("u", None)  # -> "u:default", identical for both projects
    store.append(key, _msg("the acme registry credentials rotate on Fridays"), project="acme")
    store.append(key, _msg("buy milk"), project="personal")

    acme = [m.content for m in store.recent(key, project="acme")]
    personal = [m.content for m in store.recent(key, project="personal")]

    assert acme == ["the acme registry credentials rotate on Fridays"]
    assert personal == ["buy milk"]


def test_an_explicit_session_id_is_still_project_scoped() -> None:
    store = _store()
    key = session_key("u", "s1")
    store.append(key, _msg("from acme"), project="acme")
    store.append(key, _msg("from personal"), project="personal")

    assert [m.content for m in store.recent(key, project="acme")] == ["from acme"]


def test_a_project_with_no_history_reads_empty_not_someone_elses() -> None:
    store = _store()
    key = session_key("u", None)
    store.append(key, _msg("from acme"), project="acme")

    assert store.recent(key, project="untouched") == []


def test_user_scoping_still_holds_within_a_project() -> None:
    """Both filters apply: the same project name under two owners must not merge."""
    store = _store()
    store.append(session_key("alice", "s1"), _msg("alice note"), project="shared-name")
    store.append(session_key("bob", "s1"), _msg("bob note"), project="shared-name")

    alice = store.recent(session_key("alice", "s1"), project="shared-name")
    assert [m.content for m in alice] == ["alice note"]
