"""Filing entities into the semantic upper index.

Every entity an indexed memory mentions gets a schema slot (``work``, ``health``, ...),
so ``SemanticIndex.route`` can narrow recall to the memories that share the query's
entities and slots. Classification is deterministic and keyword-based -- it reads the text
around the entity, since the name is usually a proper noun that says nothing about the slot.

**An entity is classified once.** Reclassifying on every write would let a slot flap, and
each flap rewrites what a query can route to. An early misclassification therefore persists;
the cost is precision on that one entity, never recall, because routing that finds nothing
useful says so and recall searches everything.
"""

from __future__ import annotations

from typing import Protocol

from morgan_brain.memory.semantic_index import SemanticIndex
from morgan_brain.models import Memory

#: Where an entity goes when its slot cannot be established. `knowledge` is the widest of
#: the presets, so a wrong guess here costs precision on one entity rather than filing it
#: somewhere that actively misroutes.
FALLBACK_SCHEMA = "knowledge"


class SchemaClassifier(Protocol):
    async def classify(
        self, names: list[str], *, schemas: list[str], samples: dict[str, str]
    ) -> dict[str, str]:
        """Return ``{entity_name: schema_name}``. Omissions are allowed; the caller
        fills them in. Must not raise -- an outage is a quality problem, not a job
        failure."""
        ...


class KeywordSchemaClassifier:
    """A deterministic classifier that needs no model. Deliberately crude: the first slot
    whose cues appear in the memory text wins, and an entity with no cue goes to the widest
    slot."""

    #: Ordered: the first slot whose cues appear wins, so the mapping is deterministic.
    _CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "work",
            (
                "deploy",
                "pipeline",
                "release",
                "ticket",
                "sprint",
                "repo",
                "review",
                "деплой",
                "релиз",
                "задача",
                "пайплайн",
            ),
        ),
        (
            "health",
            ("doctor", "dentist", "sleep", "gym", "pain", "appointment", "врач", "сон", "боль"),
        ),
        (
            "relationships",
            (
                "wife",
                "husband",
                "friend",
                "family",
                "colleague",
                "team",
                "жена",
                "муж",
                "друг",
                "семья",
                "коллега",
            ),
        ),
        ("goals", ("goal", "plan", "deadline", "want to", "aim", "цель", "план", "дедлайн")),
        (
            "daily_life",
            ("dinner", "commute", "shopping", "weekend", "home", "ужин", "дом", "выходные"),
        ),
    )

    async def classify(
        self, names: list[str], *, schemas: list[str], samples: dict[str, str]
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        for name in names:
            text = samples.get(name, "").lower()
            for slot, cues in self._CUES:
                if slot in schemas and any(cue in text for cue in cues):
                    out[name] = slot
                    break
            else:
                out[name] = FALLBACK_SCHEMA
        return out


class SemanticIndexBuilder:
    """Files new entities into the upper index and records their co-occurrence."""

    def __init__(self, *, semantic: SemanticIndex, classifier: SchemaClassifier) -> None:
        self._semantic = semantic
        self._classifier = classifier

    async def index(self, *, user_id: str, project: str, memories: list[Memory]) -> None:
        """Index *memories* for one ``(user_id, project)`` scope."""
        per_memory = [sorted({e.name.lower() for e in m.entities if e.name}) for m in memories]
        all_names = sorted({n for names in per_memory for n in names})
        if not all_names:
            return

        self._semantic.ensure_schemas(user_id=user_id, project=project)
        schemas = self._semantic.schemas(user_id=user_id, project=project)

        unfiled = [
            n
            for n in all_names
            if self._semantic.schema_of(user_id=user_id, project=project, entity=n) is None
        ]
        if unfiled:
            samples = {
                n: next(
                    (
                        m.content
                        for m, names in zip(memories, per_memory, strict=True)
                        if n in names
                    ),
                    "",
                )
                for n in unfiled
            }
            assigned = await self._classifier.classify(unfiled, schemas=schemas, samples=samples)
            known = set(schemas)
            for name in unfiled:
                slot = assigned.get(name, FALLBACK_SCHEMA)
                if slot not in known:
                    # An unknown slot. File the entity anyway: an unfiled entity is
                    # invisible to routing, which costs recall on every memory it touches.
                    slot = FALLBACK_SCHEMA
                self._semantic.assign(
                    user_id=user_id, project=project, entity=name, schema_name=slot
                )

        for names in per_memory:
            if len(names) > 1:
                # Co-occurrence is per memory, never across the batch: linking every
                # entity seen in one nightly run would connect everything to everything
                # and make one-hop expansion meaningless.
                self._semantic.observe_cooccurrence(user_id=user_id, project=project, names=names)
