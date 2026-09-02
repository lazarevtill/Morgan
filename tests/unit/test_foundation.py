"""The foundation contracts hold."""

from __future__ import annotations

import pytest

from morgan_brain.memory.gate import MemoryGate
from morgan_brain.models import MemoryQuery, MemorySource, TemporalFact


def test_everything_is_user_scoped() -> None:
    fact = TemporalFact(user_id="u1", subject="user", predicate="lives_in", object="Berlin")
    assert fact.user_id == "u1"
    assert fact.valid_to is None  # currently valid
    assert fact.source is MemorySource.USER_STATED


async def test_memory_gate_requires_user_id() -> None:
    class _NullStore:
        async def recall(self, query):  # type: ignore[no-untyped-def]
            return []

    gate = MemoryGate(_NullStore())  # type: ignore[arg-type]
    with pytest.raises(PermissionError):
        await gate.recall(MemoryQuery(user_id="", text="hi"))


async def test_memory_gate_requires_a_project() -> None:
    class _NullStore:
        async def forget(self, *, user_id, project):  # type: ignore[no-untyped-def]
            raise AssertionError("must not be reached")

    gate = MemoryGate(_NullStore())  # type: ignore[arg-type]
    with pytest.raises(PermissionError):
        await gate.forget(user_id="u", project="")
