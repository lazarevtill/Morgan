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

from dataclasses import dataclass
from typing import Protocol

from morgan_brain.memory.recall.semantic_index import PRESET_SCHEMAS, SemanticIndex
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


@dataclass(frozen=True)
class IndexPlan:
    """What indexing a batch will write, decided before anything is written.

    Classification is awaited -- a model call, once a model-backed classifier exists -- and
    nothing may await while the write lock is held, so it happens here, against a read of the
    index. ``SemanticIndexBuilder.apply`` re-reads under the lock before acting on it.
    """

    user_id: str
    project: str
    #: The entity names of each memory, lower-cased and sorted.
    per_memory: list[list[str]]
    #: The slot the classifier chose for each entity that was unfiled when the plan was made.
    assigned: dict[str, str]


class SemanticIndexBuilder:
    """Files new entities into the upper index and records their co-occurrence."""

    def __init__(self, *, semantic: SemanticIndex, classifier: SchemaClassifier) -> None:
        self._semantic = semantic
        self._classifier = classifier

    async def index(self, *, user_id: str, project: str, memories: list[Memory]) -> None:
        """Index *memories* for one ``(user_id, project)`` scope: ``plan``, then ``apply``."""
        self.apply(await self.plan(user_id=user_id, project=project, memories=memories))

    async def plan(self, *, user_id: str, project: str, memories: list[Memory]) -> IndexPlan:
        """Classify the entities of *memories* that are not filed yet. Reads only."""
        per_memory = [sorted({e.name.lower() for e in m.entities if e.name}) for m in memories]
        all_names = sorted({n for names in per_memory for n in names})
        unfiled = [
            n
            for n in all_names
            if self._semantic.schema_of(user_id=user_id, project=project, entity=n) is None
        ]
        assigned: dict[str, str] = {}
        if unfiled:
            # The slots apply() will have once it has ensured the presets exist.
            schemas = sorted(
                {*PRESET_SCHEMAS, *self._semantic.schemas(user_id=user_id, project=project)}
            )
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
        return IndexPlan(user_id=user_id, project=project, per_memory=per_memory, assigned=assigned)

    def apply(self, plan: IndexPlan) -> None:
        """Write *plan* into the index. Synchronous, so a caller can hold the write lock."""
        user_id, project = plan.user_id, plan.project
        all_names = sorted({n for names in plan.per_memory for n in names})
        if not all_names:
            return

        self._semantic.ensure_schemas(user_id=user_id, project=project)
        known = set(self._semantic.schemas(user_id=user_id, project=project))
        for name in all_names:
            if self._semantic.schema_of(user_id=user_id, project=project, entity=name):
                # Filed already -- perhaps by another process since the plan was made. An
                # entity is classified once.
                continue
            # An entity the plan saw as filed has no slot in it when an erasure has unfiled it
            # since; it takes the fallback like any other unknown slot.
            slot = plan.assigned.get(name, FALLBACK_SCHEMA)
            if slot not in known:
                # An unknown slot. File the entity anyway: an unfiled entity is
                # invisible to routing, which costs recall on every memory it touches.
                slot = FALLBACK_SCHEMA
            self._semantic.assign(user_id=user_id, project=project, entity=name, schema_name=slot)

        for names in plan.per_memory:
            if len(names) > 1:
                # Co-occurrence is per memory, never across the batch: linking every
                # entity seen in one nightly run would connect everything to everything
                # and make one-hop expansion meaningless.
                self._semantic.observe_cooccurrence(user_id=user_id, project=project, names=names)
