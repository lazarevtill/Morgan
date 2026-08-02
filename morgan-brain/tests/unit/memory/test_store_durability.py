"""Recall must survive a restart on all three signals."""

from __future__ import annotations

from morgan_brain.models.base import Entity
from morgan_brain.models.memory import Memory, MemoryQuery
from tests.unit.memory.conftest import build_memory_module as _module


async def test_keyword_recall_survives_restart(tmp_path):
    path = str(tmp_path / "m.db")
    await _module(path).store(Memory(user_id="u", content="the Harbor mirror blocked the deploy"))
    got = await _module(path).recall(MemoryQuery(user_id="u", text="Harbor"))
    assert any("Harbor" in m.content for m in got)


async def test_cyrillic_keyword_recall_survives_restart(tmp_path):
    path = str(tmp_path / "m.db")
    await _module(path).store(Memory(user_id="u", content="реестр Harbor заблокировал деплой"))
    got = await _module(path).recall(MemoryQuery(user_id="u", text="реестр"))
    assert any("реестр" in m.content for m in got)


async def test_entity_recall_survives_restart(tmp_path):
    path = str(tmp_path / "m.db")
    await _module(path).store(
        Memory(user_id="u", content="a note", entities=[Entity(name="Harbor", type="org")])
    )
    got = await _module(path).recall(MemoryQuery(user_id="u", text="harbor"))
    assert any(m.content == "a note" for m in got)


async def test_recall_is_user_scoped_after_restart(tmp_path):
    path = str(tmp_path / "m.db")
    await _module(path).store(Memory(user_id="u1", content="secret harbor note"))
    got = await _module(path).recall(MemoryQuery(user_id="u2", text="harbor"))
    assert got == []
