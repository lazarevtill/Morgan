"""Unit tests for SkillRegistry.

All tests are deterministic and in-process — no network, no real filesystem side-effects
beyond loading the bundled skill files that ship with the package.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from morgan_brain.interfaces.skills import Skill
from morgan_brain.learning_lifecycle.local import LocalPromptRegistry
from morgan_brain.models.base import Entity
from morgan_brain.models.emotion import EmotionState, SentimentScore
from morgan_brain.models.perception import FusedPerception, Intent
from morgan_brain.modules.skills.registry import SkillRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perception(
    text: str,
    intent: str = "chat",
    entities: list[str] | None = None,
) -> FusedPerception:
    return FusedPerception(
        text=text,
        intent=Intent(name=intent, confidence=0.9),
        entities=[Entity(name=e) for e in (entities or [])],
        emotion=EmotionState(),
        sentiment=SentimentScore(),
    )


def _write_skill_file(directory: Path, name: str, content: str) -> None:
    (directory / f"{name}.md").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Bundled skills load
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bundled_skills_load() -> None:
    registry = SkillRegistry()
    # All 5 bundled skills must be present.
    for expected in ("conversation", "empathy", "research", "coding", "planning"):
        skill = await registry.get(expected)
        assert skill is not None, f"Bundled skill '{expected}' not loaded"
        assert skill.name == expected
        assert len(skill.triggers) > 0
        assert len(skill.body) > 0


@pytest.mark.asyncio
async def test_bundled_skills_have_bodies() -> None:
    registry = SkillRegistry()
    coding = await registry.get("coding")
    assert coding is not None
    assert len(coding.body) > 10  # has meaningful content


# ---------------------------------------------------------------------------
# select: intent match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_select_matches_by_intent() -> None:
    registry = SkillRegistry()
    # "coding" skill has trigger "code"; intent name "code" should match.
    results = await registry.select(_perception("help me", intent="code"))
    names = [s.name for s in results]
    assert "coding" in names


@pytest.mark.asyncio
async def test_select_matches_by_keyword_in_text() -> None:
    registry = SkillRegistry()
    results = await registry.select(_perception("I feel sad today"))
    names = [s.name for s in results]
    assert "empathy" in names


@pytest.mark.asyncio
async def test_select_matches_by_entity_name() -> None:
    registry = SkillRegistry()
    # "research" trigger is "research"; entity name matches.
    results = await registry.select(_perception("tell me about X", entities=["research"]))
    names = [s.name for s in results]
    assert "research" in names


@pytest.mark.asyncio
async def test_select_non_matching_returns_empty() -> None:
    registry = SkillRegistry()
    # Use a nonsense intent + text that matches no skill triggers.
    results = await registry.select(_perception("xyzzy quux frobnicate", intent="unknown_intent"))
    assert results == []


@pytest.mark.asyncio
async def test_select_ordering_deterministic() -> None:
    registry = SkillRegistry()
    # A text that matches both "coding" and "planning" triggers.
    results = await registry.select(_perception("I need to plan the code refactor"))
    names = [s.name for s in results]
    assert names == sorted(names), "Results must be sorted by name"


@pytest.mark.asyncio
async def test_select_case_insensitive_text() -> None:
    registry = SkillRegistry()
    results = await registry.select(_perception("I need to CODE something"))
    names = [s.name for s in results]
    assert "coding" in names


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_returns_skill() -> None:
    registry = SkillRegistry()
    skill = await registry.get("planning")
    assert skill is not None
    assert skill.name == "planning"
    assert skill.version == 1


@pytest.mark.asyncio
async def test_get_unknown_returns_none() -> None:
    registry = SkillRegistry()
    result = await registry.get("does_not_exist")
    assert result is None


# ---------------------------------------------------------------------------
# deploy + champion override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deploy_updates_in_memory_without_registry() -> None:
    registry = SkillRegistry()
    updated = Skill(name="coding", version=2, triggers=["code"], body="Updated body.")
    await registry.deploy(updated)
    skill = await registry.get("coding")
    assert skill is not None
    assert skill.body == "Updated body."


@pytest.mark.asyncio
async def test_deploy_with_prompt_registry_sets_champion() -> None:
    prom_reg = LocalPromptRegistry()
    registry = SkillRegistry(registry=prom_reg)

    new_body = "Champion empathy skill body."
    deployed = Skill(name="empathy", version=2, triggers=["sad"], body=new_body)
    await registry.deploy(deployed)

    # Champion should now be stored in the PromptRegistry.
    champ = await prom_reg.champion("empathy")
    assert champ is not None
    assert champ.body == new_body

    # get() must return the champion body.
    skill = await registry.get("empathy")
    assert skill is not None
    assert skill.body == new_body


@pytest.mark.asyncio
async def test_select_returns_champion_body() -> None:
    """After deploying a new champion, select() must return that body."""
    prom_reg = LocalPromptRegistry()
    registry = SkillRegistry(registry=prom_reg)

    champion_body = "New champion body for coding skill."
    deployed = Skill(name="coding", version=2, triggers=["code", "bug"], body=champion_body)
    await registry.deploy(deployed)

    results = await registry.select(_perception("fix the code bug"))
    coding_results = [s for s in results if s.name == "coding"]
    assert len(coding_results) == 1
    assert coding_results[0].body == champion_body


@pytest.mark.asyncio
async def test_champion_wins_over_file_body() -> None:
    """Champion body from PromptRegistry overrides the on-disk file body."""
    prom_reg = LocalPromptRegistry()
    # Pre-register a champion without going through deploy.
    pv = await prom_reg.register("conversation", "Pre-loaded champion body.")
    await prom_reg.set_champion("conversation", pv.version)

    registry = SkillRegistry(registry=prom_reg)

    skill = await registry.get("conversation")
    assert skill is not None
    assert skill.body == "Pre-loaded champion body."


# ---------------------------------------------------------------------------
# User-supplied skills_dir overrides bundled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_skills_dir_overrides_bundled() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_skill_file(
            Path(tmpdir),
            "coding",
            "---\nname: coding\ntriggers: [code]\nversion: 99\n---\nOverridden body.\n",
        )
        registry = SkillRegistry(skills_dir=tmpdir)
        skill = await registry.get("coding")
        assert skill is not None
        assert skill.version == 99
        assert "Overridden body." in skill.body


@pytest.mark.asyncio
async def test_user_skills_dir_adds_new_skill() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_skill_file(
            Path(tmpdir),
            "custom_skill",
            "---\nname: custom_skill\ntriggers: [custom, widget]\n---\nCustom body.\n",
        )
        registry = SkillRegistry(skills_dir=tmpdir)
        skill = await registry.get("custom_skill")
        assert skill is not None
        assert skill.name == "custom_skill"
        results = await registry.select(_perception("use the custom widget"))
        names = [s.name for s in results]
        assert "custom_skill" in names
