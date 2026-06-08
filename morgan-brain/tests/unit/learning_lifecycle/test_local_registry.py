"""Tests for LocalPromptRegistry (SQLite-backed, dependency-light)."""
from datetime import datetime

import pytest

from morgan_brain.learning_lifecycle.interfaces import PromptRegistry
from morgan_brain.learning_lifecycle.local import LocalPromptRegistry

_CLOCK = lambda: datetime(2026, 1, 1)  # noqa: E731


@pytest.fixture
def reg() -> LocalPromptRegistry:
    return LocalPromptRegistry(db_path=":memory:", clock=_CLOCK)


# ------------------------------------------------------------------
# Protocol structural check
# ------------------------------------------------------------------


def test_local_registry_is_prompt_registry(reg: LocalPromptRegistry) -> None:
    assert isinstance(reg, PromptRegistry)


# ------------------------------------------------------------------
# register — auto-increment versions
# ------------------------------------------------------------------


async def test_register_first_version(reg: LocalPromptRegistry) -> None:
    pv = await reg.register("system", "Be helpful.")
    assert pv.name == "system"
    assert pv.version == 1
    assert pv.body == "Be helpful."
    assert pv.created_at == datetime(2026, 1, 1)


async def test_register_increments_version(reg: LocalPromptRegistry) -> None:
    await reg.register("system", "v1 body")
    pv2 = await reg.register("system", "v2 body")
    assert pv2.version == 2


async def test_register_independent_names(reg: LocalPromptRegistry) -> None:
    a = await reg.register("alpha", "alpha body")
    b = await reg.register("beta", "beta body")
    assert a.version == 1
    assert b.version == 1  # independent counters


async def test_register_stores_metrics_and_commit_message(reg: LocalPromptRegistry) -> None:
    pv = await reg.register(
        "system",
        "body",
        commit_message="initial prompt",
        metrics={"accuracy": 0.9},
    )
    assert pv.commit_message == "initial prompt"
    assert pv.metrics["accuracy"] == pytest.approx(0.9)


# ------------------------------------------------------------------
# champion + set_champion roundtrip
# ------------------------------------------------------------------


async def test_no_champion_initially(reg: LocalPromptRegistry) -> None:
    await reg.register("system", "v1")
    assert await reg.champion("system") is None


async def test_set_champion_and_retrieve(reg: LocalPromptRegistry) -> None:
    await reg.register("system", "v1 body")
    await reg.set_champion("system", 1)

    champ = await reg.champion("system")
    assert champ is not None
    assert champ.version == 1
    assert champ.body == "v1 body"


async def test_champion_returns_none_for_unknown_name(reg: LocalPromptRegistry) -> None:
    assert await reg.champion("nonexistent") is None


# ------------------------------------------------------------------
# rollback — set_champion to an older version
# ------------------------------------------------------------------


async def test_rollback_to_older_version(reg: LocalPromptRegistry) -> None:
    await reg.register("system", "v1 body")
    await reg.register("system", "v2 body")
    await reg.set_champion("system", 2)

    # Now v2 is champion; roll back to v1.
    await reg.set_champion("system", 1)

    champ = await reg.champion("system")
    assert champ is not None
    assert champ.version == 1
    assert champ.body == "v1 body"


async def test_rollback_leaves_v2_in_history(reg: LocalPromptRegistry) -> None:
    await reg.register("system", "v1 body")
    await reg.register("system", "v2 body")
    await reg.set_champion("system", 2)
    await reg.set_champion("system", 1)

    versions = await reg.list_versions("system")
    assert len(versions) == 2


# ------------------------------------------------------------------
# set_champion raises for nonexistent version
# ------------------------------------------------------------------


async def test_set_champion_raises_for_missing_version(reg: LocalPromptRegistry) -> None:
    await reg.register("system", "v1")
    with pytest.raises(ValueError, match="version 99"):
        await reg.set_champion("system", 99)


# ------------------------------------------------------------------
# list_versions
# ------------------------------------------------------------------


async def test_list_versions_returns_all_oldest_first(reg: LocalPromptRegistry) -> None:
    for body in ("a", "b", "c"):
        await reg.register("system", body)

    versions = await reg.list_versions("system")
    assert [v.version for v in versions] == [1, 2, 3]
    assert [v.body for v in versions] == ["a", "b", "c"]


async def test_list_versions_empty_for_unknown_name(reg: LocalPromptRegistry) -> None:
    assert await reg.list_versions("ghost") == []


# ------------------------------------------------------------------
# prompts are user-agnostic (global, not per-user)
# ------------------------------------------------------------------


async def test_prompts_are_global_not_per_user(reg: LocalPromptRegistry) -> None:
    """Prompts belong to a named skill/system prompt, not a user_id."""
    pv = await reg.register("morgan-system", "global body")
    champ = await reg.champion("morgan-system")
    assert champ is None  # champion not set yet
    await reg.set_champion("morgan-system", pv.version)
    champ = await reg.champion("morgan-system")
    assert champ is not None
    assert champ.name == "morgan-system"
