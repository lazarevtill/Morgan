"""Unit tests for SessionHistoryStore (commit 2).

Covers append + recent, limit enforcement, chronological order, and session scoping.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.learning.history import SessionHistoryStore
from morgan_brain.models.message import Message, Role

CLOCK = lambda: datetime(2026, 1, 1)  # noqa: E731
USER_ID = "u1"


@pytest.fixture
def store() -> SessionHistoryStore:
    return SessionHistoryStore(clock=CLOCK)


def _user_msg(content: str) -> Message:
    return Message(user_id=USER_ID, role=Role.USER, content=content)


def _asst_msg(content: str) -> Message:
    return Message(user_id=USER_ID, role=Role.ASSISTANT, content=content)


# ---------------------------------------------------------------------------
# Basic append + recent
# ---------------------------------------------------------------------------


def test_recent_empty_session_returns_empty_list(store: SessionHistoryStore) -> None:
    assert store.recent("s1") == []


def test_append_and_recent_one_message(store: SessionHistoryStore) -> None:
    msg = _user_msg("hello")
    store.append("s1", msg)
    result = store.recent("s1")
    assert len(result) == 1
    assert result[0].content == "hello"
    assert result[0].role is Role.USER


def test_recent_returns_chronological_order(store: SessionHistoryStore) -> None:
    store.append("s1", _user_msg("first"))
    store.append("s1", _asst_msg("second"))
    store.append("s1", _user_msg("third"))

    result = store.recent("s1")
    assert [m.content for m in result] == ["first", "second", "third"]


def test_recent_limit_truncates_oldest(store: SessionHistoryStore) -> None:
    for i in range(5):
        store.append("s1", _user_msg(f"msg{i}"))

    result = store.recent("s1", limit=3)
    # limit=3 → returns last 3 by insertion order (chronological = tail)
    assert len(result) == 3
    assert result[0].content == "msg2"
    assert result[1].content == "msg3"
    assert result[2].content == "msg4"


def test_sessions_are_isolated(store: SessionHistoryStore) -> None:
    store.append("s1", _user_msg("session one"))
    store.append("s2", _asst_msg("session two"))

    assert store.recent("s1")[0].content == "session one"
    assert store.recent("s2")[0].content == "session two"
    assert len(store.recent("s1")) == 1
    assert len(store.recent("s2")) == 1


def test_default_limit_is_10(store: SessionHistoryStore) -> None:
    for i in range(15):
        store.append("s1", _user_msg(f"msg{i}"))
    result = store.recent("s1")
    assert len(result) == 10
    # Default limit=10 returns the 10 most recent (chronological tail)
    assert result[0].content == "msg5"
    assert result[-1].content == "msg14"
