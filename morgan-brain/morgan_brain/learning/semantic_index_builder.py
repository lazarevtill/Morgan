"""Builds the semantic upper index — cold path only.

VoiceMem's index is maintained by an asynchronous updater that runs "off the critical
path" (§3.1, *Fast Update*), which is the same rule Morgan already enforces: the request
path reads learned knowledge, the learning-worker writes it. Nothing in this module is
reachable from a request.

Its job for each batch of new memories:

1. file every entity under exactly one schema, and
2. record which entities appeared together, so one-hop expansion has edges to follow.

**Classification is a model call, so it is treated as untrusted.** A classifier can be
unreachable, or can invent a slot that does not exist. Neither may corrupt the index or
fail the nightly run: an entity whose schema cannot be established is filed under
``FALLBACK_SCHEMA`` rather than dropped, because an unfiled entity is invisible to
routing and would silently shrink recall for everything it touches.

**An entity is classified once.** Reclassifying on every pass would let an entity's slot
flap between nights, and each flap rewrites which memories a query routes to -- recall
would change without anything the owner did changing. Re-filing is a deliberate act
(cluster emergence), not a side effect of the index being rebuilt.
"""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel

from morgan_brain.models.memory import Memory
from morgan_brain.modules.memory.retrieval.semantic_index import SemanticIndex
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import RoleRouter
from morgan_brain.providers.structured import generate_structured
from morgan_brain.providers.wire import ChatMessage

logger = logging.getLogger(__name__)

#: Where an entity goes when its slot cannot be established. `knowledge` is the widest of
#: the presets, so a wrong guess here costs precision on one entity rather than filing it
#: somewhere that actively misroutes.
FALLBACK_SCHEMA = "knowledge"


class Assignment(BaseModel):
    entity: str
    schema_name: str


class AssignmentBatch(BaseModel):
    """The schema passed to ``generate_structured``."""

    assignments: list[Assignment]


class SchemaClassifier(Protocol):
    async def classify(
        self, names: list[str], *, schemas: list[str], samples: dict[str, str]
    ) -> dict[str, str]:
        """Return ``{entity_name: schema_name}``. Omissions are allowed; the caller
        fills them in. Must not raise -- an outage is a quality problem, not a job
        failure."""
        ...


class KeywordSchemaClassifier:
    """A deterministic classifier that needs no model.

    Deliberately crude: it exists so the index builds on a box with no reachable model,
    on the offline/dev path, and in tests, and so ``LLMSchemaClassifier`` has something
    to degrade *to* rather than degrading to nothing. It reads the memory text around the
    entity, not the entity name, because the name is usually a proper noun that says
    nothing about which slot it belongs in.
    """

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


class LLMSchemaClassifier:
    """Classifies a whole batch of entities in one call on the ``reflection`` role.

    One call per batch rather than per entity: the nightly run can see hundreds of new
    entities, and per-entity calls would make the index's cost scale with the corpus.
    """

    def __init__(
        self,
        *,
        router: RoleRouter,
        capability_registry: CapabilityRegistry,
        role: str = "reflection",
        fallback: SchemaClassifier | None = None,
    ) -> None:
        self._router = router
        self._reg = capability_registry
        self._role = role
        self._fallback = fallback if fallback is not None else KeywordSchemaClassifier()

    async def classify(
        self, names: list[str], *, schemas: list[str], samples: dict[str, str]
    ) -> dict[str, str]:
        try:
            client, model = self._router.chat_for(self._role, needs_json_schema=True)
        except LookupError:
            try:
                client, model = self._router.chat_for(self._role)
            except LookupError:
                logger.info(
                    "semantic-index: no binding for role %r; using the keyword classifier",
                    self._role,
                )
                return await self._fallback.classify(names, schemas=schemas, samples=samples)

        provider = model.split("/", 1)[0] if "/" in model else "fake"
        listing = "\n".join(f"- {n}: {samples.get(n, '')[:200]}" for n in names)
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You file entities into a fixed set of memory slots. "
                    "Choose exactly one slot per entity, from this list and no other: "
                    + ", ".join(schemas)
                    + ". Judge from the surrounding text, not from the entity name."
                ),
            ),
            ChatMessage(role="user", content=f"Entities and the text they appeared in:\n{listing}"),
        ]
        try:
            batch = await generate_structured(
                client,
                messages,
                model=model,
                schema=AssignmentBatch,
                descriptor=self._reg.get(provider, model),
            )
        except Exception:
            # An outage or an unparseable answer costs index quality for this batch. It
            # must not take the nightly run down with it -- but it is logged with the
            # traceback, because silently degrading to keyword classification every night
            # is indistinguishable from the model never having been configured.
            logger.exception("semantic-index: classification failed; using the keyword classifier")
            return await self._fallback.classify(names, schemas=schemas, samples=samples)
        return {a.entity.lower(): a.schema_name for a in batch.assignments}


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
                    # An invented slot. File the entity anyway: an unfiled entity is
                    # invisible to routing, which costs recall on every memory it touches.
                    logger.info(
                        "semantic-index: classifier proposed unknown slot %r for %r; "
                        "filing under %r",
                        slot,
                        name,
                        FALLBACK_SCHEMA,
                    )
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
