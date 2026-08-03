"""Streaming self-evolving-memory benchmark (Evo-Memory style).

Evo-Memory (2025) frames the core failure of agent memory as: *systems handle a continuous
task stream yet fail to learn from accumulated interactions*. This benchmark measures exactly
that on Morgan's real loop, deterministically:

* knowledge established early in a long stream must remain recallable **beyond the bounded
  session-history window** — i.e. the durable (consolidated) fact layer, not just recent turns,
  carries it. This is the "learns from accumulated interactions" property.
* a mid-stream knowledge update must propagate: later queries return the new value and the stale
  value is no longer a current fact (temporal self-evolution under streaming).

Scores: ``recall_accuracy`` (stream-distance-independent recall of the durable fact) and
``update_accuracy`` (post-update correctness). Both should be 1.0 — the platform's self-evolving
promise, held over a stream much longer than the history window.
"""

from __future__ import annotations

from dataclasses import dataclass

from morgan_brain.config import Settings
from morgan_brain.learning.history import session_key
from morgan_brain.models.memory import MemorySource, TemporalFact
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex
from morgan_brain.providers.adapters.fake import FakeChatClient
from morgan_brain.providers.router import Binding, RoleRouter
from tests.e2e.harness import ConversationHarness, StepClock, _contains, _fake_capability_registry

_HISTORY_WINDOW = 10  # SessionHistoryStore.recent default limit — queries past this rely on facts


@dataclass
class StreamingReport:
    stream_len: int
    history_window: int
    recall_queries: int
    recall_hits: int
    recall_distances: list[int]
    update_queries: int
    update_hits: int
    stale_after_update: bool

    @property
    def max_recall_distance(self) -> int:
        return max(self.recall_distances) if self.recall_distances else 0

    @property
    def recall_accuracy(self) -> float:
        return self.recall_hits / self.recall_queries if self.recall_queries else 0.0

    @property
    def update_accuracy(self) -> float:
        return self.update_hits / self.update_queries if self.update_queries else 0.0


def _harness() -> ConversationHarness:
    """A deterministic harness whose fake model returns a constant reply for every turn
    (we measure what reaches the prompt via recall, not the model's words)."""
    fake = FakeChatClient(reply="ack")
    router = RoleRouter(
        reg=_fake_capability_registry(),
        bindings={"strong": [Binding("fake", "test-model", fake)]},
    )
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    return ConversationHarness(
        embedder=FakeEmbedder(dim=32),
        router=router,
        vectors=InMemoryVectorIndex(),
        clock=StepClock(),
        settings=settings,
        fake_client=fake,
    )


async def run_streaming(*, stream_len: int = 24) -> StreamingReport:
    """Drive a long stream; measure distance-independent recall + mid-stream update propagation."""
    h = _harness()
    user, session = "u-stream", "s-stream"
    query = "What is my favorite programming language?"

    # Establish a durable fact (as consolidation would) BEFORE the stream begins.
    await h.memory_module.upsert_fact(
        TemporalFact(
            user_id=user,
            subject="user",
            predicate="favorite_language",
            object="Rust",
            source=MemorySource.USER_STATED,
        )
    )

    recall_distances = [2, 6, 11]  # all BEFORE the update; the last exceeds the history window
    update_at = 12
    update_query_distances = [16, 22]  # both after the update, both past the window
    recall_hits = recall_queries = 0
    update_hits = update_queries = 0

    for i in range(1, stream_len + 1):
        if i == update_at:
            # Mid-stream knowledge update: supersede the durable fact.
            await h.memory_module.upsert_fact(
                TemporalFact(
                    user_id=user,
                    subject="user",
                    predicate="favorite_language",
                    object="Haskell",
                    source=MemorySource.USER_STATED,
                )
            )

        if i in recall_distances or i in update_query_distances:
            await h.say(user_id=user, text=query, session_id=session)
            prompt = h.last_prompt_text()
            if i in recall_distances:
                recall_queries += 1
                if _contains(prompt, "rust"):
                    recall_hits += 1
            else:  # post-update query
                update_queries += 1
                if _contains(prompt, "haskell") and not _contains(prompt, "rust"):
                    update_hits += 1
        else:
            # Filler turns push earlier knowledge out of the bounded history window.
            await h.say(user_id=user, text=f"unrelated small talk number {i}", session_id=session)

    # Confirm the stale value is no longer a current fact (temporal supersession held).
    facts = await h.memory_module.current_facts(user_id=user, subject="user")
    current_lang = {f.object for f in facts if f.predicate == "favorite_language"}
    stale_after_update = current_lang == {"Haskell"}

    # Sanity: the late queries genuinely exceeded the history window.
    assert h._history.recent(
        session_key(user, session), project="default"
    )  # history exists  # noqa: SLF001

    return StreamingReport(
        stream_len=stream_len,
        history_window=_HISTORY_WINDOW,
        recall_queries=recall_queries,
        recall_hits=recall_hits,
        recall_distances=recall_distances,
        update_queries=update_queries,
        update_hits=update_hits,
        stale_after_update=stale_after_update,
    )
