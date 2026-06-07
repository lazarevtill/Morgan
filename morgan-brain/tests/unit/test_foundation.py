"""Phase 0 smoke tests — the foundation contracts hold."""
from __future__ import annotations

from morgan_brain.config import Settings
from morgan_brain.models.memory import MemorySource, TemporalFact
from morgan_brain.models.user import RelationshipStage, UserModel
from morgan_brain.security.memory_gate import MemoryGate
from morgan_brain.security.permissions import PermissionGate, PermissionMode


def test_settings_default_event_bus_is_inproc() -> None:
    assert Settings().event_bus == "inproc"


def test_everything_is_user_scoped() -> None:
    fact = TemporalFact(user_id="u1", subject="user", predicate="lives_in", object="Berlin")
    assert fact.user_id == "u1"
    assert fact.valid_to is None  # currently valid
    assert fact.source is MemorySource.USER_STATED


def test_user_model_starts_as_new_relationship() -> None:
    um = UserModel(user_id="u1")
    assert um.relationship_stage is RelationshipStage.NEW
    assert um.confidence == 0.0


def test_permission_gate_denies_explicitly() -> None:
    gate = PermissionGate(default=PermissionMode.ASK)
    gate.set("bash", PermissionMode.DENY)
    assert gate.allowed("bash") is False
    assert gate.allowed("calculator") is True  # default ASK is allowed


async def test_memory_gate_requires_user_id() -> None:
    class _NullStore:
        async def store(self, memory):  # type: ignore[no-untyped-def]
            return "id"

        async def recall(self, query):  # type: ignore[no-untyped-def]
            return []

        async def upsert_fact(self, fact):  # type: ignore[no-untyped-def]
            return "id"

        async def current_facts(self, *, user_id, subject=None):  # type: ignore[no-untyped-def]
            return []

    gate = MemoryGate(_NullStore())  # type: ignore[arg-type]
    from morgan_brain.models.memory import MemoryQuery

    try:
        await gate.recall(MemoryQuery(user_id="", text="hi"))
    except PermissionError:
        return
    raise AssertionError("MemoryGate should reject empty user_id")
