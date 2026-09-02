"""Session history must be keyed by (user_id, session_id), never session_id alone.

Two users sharing the same client-supplied session_id must NOT see each other's
history — cross-user contamination of memory/learning is the cardinal invariant.
A missing session_id falls back to a *per-user* bucket, never a global one
(agreed with the Neural-Interface client in COORDINATION.md).
"""

from __future__ import annotations

from morgan_brain.memory.history import SessionHistoryStore, session_key
from morgan_brain.models import Message, Role


def test_session_key_namespaces_by_user() -> None:
    assert session_key("alice", "s1") == "alice:s1"
    assert session_key("bob", "s1") == "bob:s1"
    # different users with the same client session id must produce different keys
    assert session_key("alice", "s1") != session_key("bob", "s1")
    # missing/empty session_id → per-user default bucket, never a global "default"
    assert session_key("alice", None) == "alice:default"
    assert session_key("alice", "") == "alice:default"
    assert session_key("alice", None) != session_key("bob", None)


def test_history_is_isolated_per_user_for_same_session_id() -> None:
    store = SessionHistoryStore()  # :memory:
    store.append(
        session_key("alice", "s1"),
        Message(user_id="alice", role=Role.USER, content="I am Alice"),
    )
    store.append(
        session_key("bob", "s1"),
        Message(user_id="bob", role=Role.USER, content="I am Bob"),
    )

    alice = store.recent(session_key("alice", "s1"), project="default")
    bob = store.recent(session_key("bob", "s1"), project="default")

    assert [m.content for m in alice] == ["I am Alice"]
    assert [m.content for m in bob] == ["I am Bob"]
