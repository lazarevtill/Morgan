"""SkillRegistry — load, select, and deploy markdown skills.

Skills are ``.md`` files with YAML-subset frontmatter
(name, triggers, tools, model, version).  The active body for a skill is:

1. The champion body stored in the ``PromptRegistry`` (if one is wired and a
   champion version exists) — this is the GEPA-trainable path.
2. Otherwise, the body parsed from the ``.md`` file.

``select(perception)`` matches skills whose ``triggers`` intersect the
current perception — checked against:
- ``perception.intent.name`` (exact match, case-folded)
- lowercase tokens of ``perception.text`` (whole-word substring)
- entity names (case-folded)

Results are sorted deterministically by skill name.

``deploy(skill)`` either registers the body in the ``PromptRegistry`` (and
sets it as champion) or, when no registry is wired, replaces the in-memory
skill directly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from morgan_brain.interfaces.skills import Skill, SkillEngine
from morgan_brain.learning_lifecycle.interfaces import PromptRegistry
from morgan_brain.models.perception import FusedPerception
from morgan_brain.modules.skills.frontmatter import parse_frontmatter

_BUNDLED_DIR = Path(__file__).parent / "bundled"


def _text_tokens(text: str) -> set[str]:
    """Return a set of lowercase word-boundary tokens from *text*."""
    return set(re.findall(r"[a-z]+", text.lower()))


def _load_skill_from_file(path: Path) -> Skill | None:
    """Parse a single ``.md`` file into a :class:`Skill`.

    Returns ``None`` when the file has no ``name`` in its frontmatter.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None

    meta, body = parse_frontmatter(raw)
    if "name" not in meta:
        return None

    name = str(meta["name"])

    raw_triggers = meta.get("triggers", [])
    if isinstance(raw_triggers, list):
        triggers = [str(t) for t in raw_triggers]
    else:
        triggers = [str(raw_triggers)]

    raw_tools = meta.get("tools", [])
    if isinstance(raw_tools, list):
        tools = [str(t) for t in raw_tools]
    else:
        tools = [str(raw_tools)]

    model: str | None = str(meta["model"]) if "model" in meta else None
    version_raw = meta.get("version", 1)
    version = int(version_raw) if isinstance(version_raw, (int, str)) else 1

    return Skill(
        name=name,
        version=version,
        triggers=triggers,
        tools=tools,
        model=model,
        body=body.strip(),
    )


def _load_skills_from_dir(directory: Path) -> dict[str, Skill]:
    """Load all ``.md`` files in *directory* and return a name → Skill dict."""
    skills: dict[str, Skill] = {}
    if not directory.is_dir():
        return skills
    for path in sorted(directory.glob("*.md")):
        skill = _load_skill_from_file(path)
        if skill is not None:
            skills[skill.name] = skill
    return skills


class SkillRegistry:
    """``SkillEngine`` implementation backed by markdown files and an optional
    ``PromptRegistry`` for champion-body versioning.

    Parameters
    ----------
    skills_dir:
        Optional path to an additional directory of ``.md`` skill files.
        Files here override bundled skills with the same name.
    registry:
        Optional ``PromptRegistry`` for champion-body resolution and
        ``deploy`` persistence.  When ``None``, skills are stored in memory
        only and champion versioning is unavailable.
    """

    def __init__(
        self,
        *,
        skills_dir: Optional[str] = None,
        registry: Optional[PromptRegistry] = None,
    ) -> None:
        self._registry = registry
        # Start from bundled skills, then overlay user-supplied dir.
        self._skills: dict[str, Skill] = _load_skills_from_dir(_BUNDLED_DIR)
        if skills_dir:
            user_skills = _load_skills_from_dir(Path(skills_dir))
            self._skills.update(user_skills)

    # ------------------------------------------------------------------
    # SkillEngine Protocol
    # ------------------------------------------------------------------

    def list_skills(self) -> list[Skill]:
        """Return all loaded skills as a flat list, sorted by name."""
        return [self._skills[name] for name in sorted(self._skills)]

    async def select(self, perception: FusedPerception) -> list[Skill]:
        """Return skills whose triggers match the current perception.

        Matching rules (any trigger in the skill's list must match at least
        one of the following):
        - Case-folded equality with ``perception.intent.name``.
        - Presence as a word-boundary token inside the lowercased
          ``perception.text``.
        - Case-folded equality with any entity name.

        Returns skills sorted deterministically by name.
        """
        intent_name = perception.intent.name.lower()
        text_tokens = _text_tokens(perception.text)
        entity_names = {e.name.lower() for e in perception.entities}

        matched: list[Skill] = []
        for name in sorted(self._skills):
            skill = await self.get(name)
            if skill is None:
                continue
            for trigger in skill.triggers:
                t = trigger.lower()
                if t == intent_name or t in text_tokens or t in entity_names:
                    matched.append(skill)
                    break  # one matching trigger is enough

        return matched

    async def get(self, name: str) -> Skill | None:
        """Return the skill by *name* with the champion body applied if available."""
        base = self._skills.get(name)
        if base is None:
            return None
        return await self._apply_champion(base)

    async def deploy(self, skill: Skill) -> None:
        """Install or replace a validated skill version.

        - If a ``PromptRegistry`` is wired: register the body and set it as
          champion so future calls to ``select``/``get`` return the new body.
        - Otherwise: replace the in-memory skill directly.
        """
        if self._registry is not None:
            pv = await self._registry.register(
                skill.name,
                skill.body,
                commit_message=f"deploy skill {skill.name} v{skill.version}",
            )
            await self._registry.set_champion(skill.name, pv.version)

        # Always keep the in-memory record current (metadata + body).
        self._skills[skill.name] = skill

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _apply_champion(self, skill: Skill) -> Skill:
        """Return *skill* with ``body`` replaced by the champion body if one exists."""
        if self._registry is None:
            return skill
        champ = await self._registry.champion(skill.name)
        if champ is None:
            return skill
        return skill.model_copy(update={"body": champ.body})


# Structural type-check: SkillRegistry must satisfy the SkillEngine Protocol at import time.
_: SkillEngine = SkillRegistry()
