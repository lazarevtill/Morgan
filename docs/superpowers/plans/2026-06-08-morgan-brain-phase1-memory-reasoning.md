# Morgan Brain — Phase 1 (Memory + Reasoning) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Phase 0 skeleton into a working text assistant that recalls prior turns and facts — `POST /api/chat` perceives the message, recalls relevant memory (vector + BM25 + entity, plus currently-valid temporal facts), generates a reply via the local LLM, and stores the turn off the response path.

**Architecture:** Implement the `Memory` and `Reasoning` modules behind their existing Protocols, plus a minimal text `Perception` and pass-through `Personalizer`/`SkillEngine`/`Learner` so the `Orchestrator` loop runs end-to-end. All external services (Qdrant, Ollama) sit behind small Protocols with in-memory fakes, so the whole loop is unit-testable with zero infrastructure; real adapters are smoke-tested separately. The turn is stored by an event subscriber on `RESPONSE_GENERATED`, honoring "hot path reads, cold path writes."

**Tech Stack:** Python 3.12, pydantic v2, FastAPI, httpx (Ollama OpenAI-compat), qdrant-client, SQLite (stdlib `sqlite3`), pytest + pytest-asyncio (`asyncio_mode=auto`).

---

## Design decisions locked for Phase 1

- **Single rerank layer = Reciprocal Rank Fusion (RRF)** over the vector / BM25 / entity rankings. CrossEncoder reranking is deferred (design spec §13 degradation ladder; Phase 1 uses the cheapest rung).
- **Embeddings + LLM** target the Ollama OpenAI-compatible API (`/v1/embeddings`, `/v1/chat/completions`).
- **Temporal facts** live in SQLite (`MORGAN_TEMPORAL_DB_URL`); episodic/semantic memories live in the vector store. Phase 1 does not yet *extract* facts from chat (that's Phase 2 Learning) — but `upsert_fact`/`current_facts` are fully implemented and tested so Phase 2 can call them.
- **Timestamps** are injected (a `now: datetime` parameter / `clock` callable), never read implicitly — keeps tests deterministic (matches `models/base.py`).
- **Two new internal Protocols** are introduced (`Embedder`, `VectorIndex`, `LLMClient`) to isolate I/O. They live next to their implementations, not in `interfaces/` (which holds only the cross-module contracts).

## File structure (created/modified in this plan)

```
morgan-brain/morgan_brain/
  modules/perception/text/analyzer.py        # TextPerception (Perception)
  modules/memory/
    indexing/embedder.py                     # Embedder protocol + OllamaEmbedder + FakeEmbedder
    stores/vector.py                          # VectorIndex protocol + InMemoryVectorIndex + QdrantVectorIndex
    stores/temporal.py                        # SqliteTemporalStore (bi-temporal facts)
    retrieval/bm25.py                         # Bm25Index (pure)
    retrieval/fusion.py                       # reciprocal_rank_fusion (pure)
    store.py                                  # MemoryModule (implements MemoryStore)
  modules/reasoning/
    llm/client.py                             # LLMClient protocol + OllamaLLMClient + FakeLLMClient
    context/builder.py                        # build_messages (pure)
    reasoner.py                               # ReasoningModule (implements Reasoner)
  modules/personalization/passthrough.py      # PassthroughPersonalizer (Personalizer)
  modules/skills/noop.py                      # NoopSkillEngine (SkillEngine)
  modules/learning/minimal.py                 # MinimalLearner (Learner)
  composition.py                              # build_orchestrator(settings) wiring + turn-storage subscriber
  apps/brain_api/app.py                       # MODIFY: wire /api/chat to the orchestrator
morgan-brain/tests/
  unit/test_text_perception.py
  unit/test_embedder.py
  unit/test_vector_index.py
  unit/test_temporal_store.py
  unit/test_bm25.py
  unit/test_fusion.py
  unit/test_memory_module.py
  unit/test_context_builder.py
  unit/test_reasoner.py
  unit/test_passthrough_personalizer.py
  unit/test_minimal_learner.py
  integration/test_chat_loop.py
  memory_quality/conftest.py                  # fixtures + scorer
  memory_quality/test_recall_quality.py
```

---

### Task 0: Environment setup & baseline

**Files:** none (verification only)

- [ ] **Step 1: Install dev dependencies**

Run (from `morgan-brain/`):
```bash
pip install -e ".[dev]"
```
Expected: completes; installs fastapi, httpx, qdrant-client, redis, structlog, pytest, pytest-asyncio.

- [ ] **Step 2: Run the Phase 0 foundation tests**

Run:
```bash
pytest tests/unit/test_foundation.py -v
```
Expected: PASS (5 tests). If `pytest-asyncio` is missing, `pip install pytest-asyncio` and confirm `asyncio_mode = "auto"` is set in `pyproject.toml` (it is).

- [ ] **Step 3: Commit nothing (baseline only)** — proceed to Task 1.

---

### Task 1: Text Perception

Produces a `FusedPerception` from text. Phase 1 keeps it minimal: intent label + naive entity extraction (capitalized tokens). Emotion/sentiment stay at defaults (Phase 2 enriches; the audio path is Phase 5).

**Files:**
- Create: `morgan_brain/modules/perception/text/__init__.py`
- Create: `morgan_brain/modules/perception/text/analyzer.py`
- Test: `tests/unit/test_text_perception.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_text_perception.py
from morgan_brain.modules.perception.text.analyzer import TextPerception
from morgan_brain.models.perception import Modality


async def test_returns_fused_perception_for_text():
    p = TextPerception()
    out = await p.analyze(user_id="u1", text="Remind me about Berlin on Monday")
    assert out.text == "Remind me about Berlin on Monday"
    assert out.modalities_used == [Modality.TEXT]
    assert out.intent.name in {"chat", "command", "question"}


async def test_extracts_capitalized_entities():
    p = TextPerception()
    out = await p.analyze(user_id="u1", text="I met Alice in Berlin")
    names = {e.name for e in out.entities}
    assert "Alice" in names and "Berlin" in names


async def test_question_intent_detected():
    p = TextPerception()
    out = await p.analyze(user_id="u1", text="What time is it?")
    assert out.intent.name == "question"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_text_perception.py -v`
Expected: FAIL with `ModuleNotFoundError: ...perception.text.analyzer`.

- [ ] **Step 3: Write minimal implementation**

```python
# morgan_brain/modules/perception/text/__init__.py
"""Text perception (Phase 1)."""
```

```python
# morgan_brain/modules/perception/text/analyzer.py
"""Minimal text perception. Implements interfaces.Perception for the text modality.

Phase 1 scope: intent classification by simple heuristics and capitalized-token entity
extraction. Emotion/sentiment remain at defaults until Phase 2; audio/vision are Phase 5.
"""
from __future__ import annotations

import re

from morgan_brain.models.base import Entity
from morgan_brain.models.perception import FusedPerception, Intent, Modality

_CAP_TOKEN = re.compile(r"\b([A-Z][a-z]{2,})\b")
_STOPWORDS = {"I", "The", "A", "An", "What", "When", "Where", "Why", "How", "Remind", "Monday",
              "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}


class TextPerception:
    async def analyze(
        self, *, user_id: str, text: str, audio: bytes | None = None, image: bytes | None = None
    ) -> FusedPerception:
        intent_name = self._classify_intent(text)
        entities = [
            Entity(name=m.group(1))
            for m in _CAP_TOKEN.finditer(text)
            if m.group(1) not in _STOPWORDS
        ]
        # de-duplicate by name, preserve order
        seen: set[str] = set()
        unique = [e for e in entities if not (e.name in seen or seen.add(e.name))]
        return FusedPerception(
            text=text,
            intent=Intent(name=intent_name, confidence=0.6),
            entities=unique,
            modalities_used=[Modality.TEXT],
        )

    @staticmethod
    def _classify_intent(text: str) -> str:
        stripped = text.strip()
        if stripped.endswith("?") or re.match(r"^(what|when|where|why|how|who|is|are|do|does)\b",
                                               stripped, re.IGNORECASE):
            return "question"
        if re.match(r"^(remind|create|add|delete|set|schedule|run)\b", stripped, re.IGNORECASE):
            return "command"
        return "chat"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_text_perception.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add morgan_brain/modules/perception/text tests/unit/test_text_perception.py
git commit -m "feat(perception): minimal text perception producing FusedPerception"
```

---

### Task 2: Embedder

An `Embedder` Protocol with an Ollama-backed implementation and a deterministic fake for tests.

**Files:**
- Create: `morgan_brain/modules/memory/indexing/__init__.py`
- Create: `morgan_brain/modules/memory/indexing/embedder.py`
- Test: `tests/unit/test_embedder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_embedder.py
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder


async def test_fake_embedder_is_deterministic_and_fixed_dim():
    emb = FakeEmbedder(dim=16)
    a = await emb.embed("hello world")
    b = await emb.embed("hello world")
    c = await emb.embed("different text")
    assert len(a) == 16
    assert a == b           # deterministic
    assert a != c           # content-sensitive


async def test_fake_embedder_batch():
    emb = FakeEmbedder(dim=8)
    out = await emb.embed_batch(["a", "b", "c"])
    assert len(out) == 3 and all(len(v) == 8 for v in out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_embedder.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# morgan_brain/modules/memory/indexing/__init__.py
"""Embedding indexing."""
```

```python
# morgan_brain/modules/memory/indexing/embedder.py
"""Embedder: text -> vector. OllamaEmbedder hits the OpenAI-compatible /v1/embeddings endpoint;
FakeEmbedder is a deterministic, dependency-free stand-in for tests."""
from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

import httpx


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """Deterministic hash-based embeddings. Not semantically meaningful, but stable and
    content-sensitive — enough to test storage, retrieval plumbing, and ranking determinism."""

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    async def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self._dim)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class OllamaEmbedder:
    def __init__(self, endpoint: str, model: str, timeout: float = 30.0) -> None:
        self._url = endpoint.rstrip("/") + "/embeddings"
        self._model = model
        self._timeout = timeout

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._url, json={"model": self._model, "input": texts})
            resp.raise_for_status()
            data = resp.json()["data"]
        return [item["embedding"] for item in data]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_embedder.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add morgan_brain/modules/memory/indexing tests/unit/test_embedder.py
git commit -m "feat(memory): Embedder protocol with Ollama + fake implementations"
```

---

### Task 3: Vector index

A `VectorIndex` Protocol with an in-memory cosine implementation (tests) and a Qdrant adapter (runtime).

**Files:**
- Create: `morgan_brain/modules/memory/stores/__init__.py`
- Create: `morgan_brain/modules/memory/stores/vector.py`
- Test: `tests/unit/test_vector_index.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_vector_index.py
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex, VectorRecord


async def test_upsert_and_search_returns_nearest_first():
    idx = InMemoryVectorIndex()
    await idx.upsert(VectorRecord(id="a", user_id="u1", vector=[1.0, 0.0], payload={"content": "A"}))
    await idx.upsert(VectorRecord(id="b", user_id="u1", vector=[0.0, 1.0], payload={"content": "B"}))
    hits = await idx.search(user_id="u1", vector=[0.9, 0.1], top_k=2)
    assert [h.id for h in hits] == ["a", "b"]
    assert hits[0].score >= hits[1].score


async def test_search_is_user_scoped():
    idx = InMemoryVectorIndex()
    await idx.upsert(VectorRecord(id="a", user_id="u1", vector=[1.0, 0.0], payload={}))
    await idx.upsert(VectorRecord(id="b", user_id="u2", vector=[1.0, 0.0], payload={}))
    hits = await idx.search(user_id="u1", vector=[1.0, 0.0], top_k=5)
    assert [h.id for h in hits] == ["a"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_vector_index.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# morgan_brain/modules/memory/stores/__init__.py
"""Memory stores."""
```

```python
# morgan_brain/modules/memory/stores/vector.py
"""VectorIndex: user-scoped vector storage + cosine search. InMemoryVectorIndex for tests;
QdrantVectorIndex for runtime. Both satisfy the same Protocol."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class VectorRecord:
    id: str
    user_id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorHit:
    id: str
    score: float
    payload: dict[str, Any]


@runtime_checkable
class VectorIndex(Protocol):
    async def upsert(self, record: VectorRecord) -> None: ...
    async def search(self, *, user_id: str, vector: list[float], top_k: int) -> list[VectorHit]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    async def upsert(self, record: VectorRecord) -> None:
        self._records[record.id] = record

    async def search(self, *, user_id: str, vector: list[float], top_k: int) -> list[VectorHit]:
        scored = [
            VectorHit(id=r.id, score=_cosine(vector, r.vector), payload=r.payload)
            for r in self._records.values()
            if r.user_id == user_id
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]


class QdrantVectorIndex:
    """Runtime adapter. Uses a single collection with a `user_id` payload filter for scoping.
    Smoke-tested against a live Qdrant, not in unit tests."""

    def __init__(self, url: str, collection: str = "morgan_memories", dim: int = 1024) -> None:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.http import models as qm

        self._client = AsyncQdrantClient(url=url)
        self._collection = collection
        self._dim = dim
        self._qm = qm

    async def ensure_collection(self) -> None:
        qm = self._qm
        existing = await self._client.get_collections()
        names = {c.name for c in existing.collections}
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
            )

    async def upsert(self, record: VectorRecord) -> None:
        qm = self._qm
        await self._client.upsert(
            collection_name=self._collection,
            points=[qm.PointStruct(
                id=record.id,
                vector=record.vector,
                payload={**record.payload, "user_id": record.user_id},
            )],
        )

    async def search(self, *, user_id: str, vector: list[float], top_k: int) -> list[VectorHit]:
        qm = self._qm
        res = await self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=top_k,
            query_filter=qm.Filter(
                must=[qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))]
            ),
        )
        return [VectorHit(id=str(p.id), score=p.score, payload=dict(p.payload or {})) for p in res]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_vector_index.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add morgan_brain/modules/memory/stores tests/unit/test_vector_index.py
git commit -m "feat(memory): VectorIndex protocol with in-memory + Qdrant implementations"
```

---

### Task 4: Temporal fact store (bi-temporal)

SQLite-backed facts with validity intervals. `upsert_fact` supersedes the conflicting currently-valid fact (evolution, not overwrite). This is the core invariant of the design.

**Files:**
- Create: `morgan_brain/modules/memory/stores/temporal.py`
- Test: `tests/unit/test_temporal_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_temporal_store.py
from datetime import datetime

from morgan_brain.models.memory import TemporalFact
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore


def _fact(obj: str, **kw) -> TemporalFact:
    return TemporalFact(user_id="u1", subject="user", predicate="lives_in", object=obj, **kw)


async def test_upsert_then_current_returns_fact():
    store = SqliteTemporalStore(":memory:")
    await store.upsert_fact(_fact("Berlin"), now=datetime(2026, 1, 1))
    current = await store.current_facts(user_id="u1")
    assert len(current) == 1 and current[0].object == "Berlin"
    assert current[0].valid_to is None


async def test_conflicting_fact_supersedes_not_overwrites():
    store = SqliteTemporalStore(":memory:")
    first_id = await store.upsert_fact(_fact("Berlin"), now=datetime(2026, 1, 1))
    await store.upsert_fact(_fact("Munich"), now=datetime(2026, 6, 1))

    current = await store.current_facts(user_id="u1")
    assert len(current) == 1 and current[0].object == "Munich"

    history = await store.history(user_id="u1", subject="user", predicate="lives_in")
    assert len(history) == 2
    old = next(f for f in history if f.id == first_id)
    assert old.valid_to == datetime(2026, 6, 1)
    assert old.superseded_by is not None


async def test_user_scoped():
    store = SqliteTemporalStore(":memory:")
    await store.upsert_fact(_fact("Berlin"), now=datetime(2026, 1, 1))
    assert await store.current_facts(user_id="u2") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_temporal_store.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# morgan_brain/modules/memory/stores/temporal.py
"""Bi-temporal fact store (SQLite). A fact is currently valid when valid_to IS NULL. Asserting a
new value for the same (user, subject, predicate) closes the old interval (sets valid_to = now,
superseded_by = new id) instead of deleting it — so history stays queryable and recall is never
confidently stale."""
from __future__ import annotations

import sqlite3
from datetime import datetime

from morgan_brain.models.memory import MemorySource, TemporalFact

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    superseded_by TEXT,
    last_confirmed TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts_current
    ON facts (user_id, subject, predicate) WHERE valid_to IS NULL;
"""


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class SqliteTemporalStore:
    def __init__(self, path: str = ":memory:") -> None:
        # check_same_thread=False so it can be used from the async server's threadpool.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _row_to_fact(self, row: sqlite3.Row) -> TemporalFact:
        return TemporalFact(
            id=row["id"], user_id=row["user_id"], subject=row["subject"],
            predicate=row["predicate"], object=row["object"],
            source=MemorySource(row["source"]), confidence=row["confidence"],
            valid_from=_dt(row["valid_from"]), valid_to=_dt(row["valid_to"]),
            superseded_by=row["superseded_by"], last_confirmed=_dt(row["last_confirmed"]),
        )

    async def upsert_fact(self, fact: TemporalFact, *, now: datetime) -> str:
        cur = self._conn.execute(
            "SELECT id FROM facts WHERE user_id=? AND subject=? AND predicate=? AND valid_to IS NULL",
            (fact.user_id, fact.subject, fact.predicate),
        )
        existing = [r["id"] for r in cur.fetchall()]
        if fact.valid_from is None:
            fact.valid_from = now
        fact.last_confirmed = now
        self._conn.execute(
            "INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (fact.id, fact.user_id, fact.subject, fact.predicate, fact.object,
             fact.source.value, fact.confidence, _iso(fact.valid_from), _iso(fact.valid_to),
             fact.superseded_by, _iso(fact.last_confirmed)),
        )
        for old_id in existing:
            self._conn.execute(
                "UPDATE facts SET valid_to=?, superseded_by=? WHERE id=?",
                (_iso(now), fact.id, old_id),
            )
        self._conn.commit()
        return fact.id

    async def current_facts(
        self, *, user_id: str, subject: str | None = None
    ) -> list[TemporalFact]:
        sql = "SELECT * FROM facts WHERE user_id=? AND valid_to IS NULL"
        params: list[object] = [user_id]
        if subject is not None:
            sql += " AND subject=?"
            params.append(subject)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_fact(r) for r in rows]

    async def history(
        self, *, user_id: str, subject: str, predicate: str
    ) -> list[TemporalFact]:
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE user_id=? AND subject=? AND predicate=? ORDER BY valid_from",
            (user_id, subject, predicate),
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_temporal_store.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add morgan_brain/modules/memory/stores/temporal.py tests/unit/test_temporal_store.py
git commit -m "feat(memory): bi-temporal SQLite fact store (supersede, never overwrite)"
```

---

### Task 5: BM25 keyword index

**Files:**
- Create: `morgan_brain/modules/memory/retrieval/__init__.py`
- Create: `morgan_brain/modules/memory/retrieval/bm25.py`
- Test: `tests/unit/test_bm25.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_bm25.py
from morgan_brain.modules.memory.retrieval.bm25 import Bm25Index


def test_ranks_documents_by_keyword_overlap():
    idx = Bm25Index()
    idx.add("d1", "the cat sat on the mat")
    idx.add("d2", "dogs run in the park")
    idx.add("d3", "a cat and a dog")
    ranked = idx.search("cat", top_k=3)
    ids = [doc_id for doc_id, _ in ranked]
    assert ids[0] in {"d1", "d3"}
    assert "d2" not in ids[:1]


def test_empty_query_returns_nothing():
    idx = Bm25Index()
    idx.add("d1", "hello world")
    assert idx.search("", top_k=5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_bm25.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# morgan_brain/modules/memory/retrieval/__init__.py
"""Retrieval: keyword, fusion, search orchestration."""
```

```python
# morgan_brain/modules/memory/retrieval/bm25.py
"""Tiny in-memory BM25. Sufficient for a single user's memory volume in Phase 1; swappable for a
real index later without touching callers."""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Bm25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._docs: dict[str, list[str]] = {}
        self._df: Counter[str] = Counter()

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._docs:
            for term in set(self._docs[doc_id]):
                self._df[term] -= 1
        tokens = _tokenize(text)
        self._docs[doc_id] = tokens
        for term in set(tokens):
            self._df[term] += 1

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        q_terms = _tokenize(query)
        if not q_terms or not self._docs:
            return []
        n = len(self._docs)
        avgdl = sum(len(d) for d in self._docs.values()) / n
        scores: dict[str, float] = {}
        for doc_id, tokens in self._docs.items():
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                df = max(self._df.get(term, 0), 1)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                freq = tf[term]
                denom = freq + self._k1 * (1 - self._b + self._b * dl / avgdl)
                score += idf * (freq * (self._k1 + 1)) / denom
            if score > 0:
                scores[doc_id] = score
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_bm25.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add morgan_brain/modules/memory/retrieval tests/unit/test_bm25.py
git commit -m "feat(memory): in-memory BM25 keyword index"
```

---

### Task 6: Reciprocal Rank Fusion (the single rerank layer)

**Files:**
- Create: `morgan_brain/modules/memory/retrieval/fusion.py`
- Test: `tests/unit/test_fusion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_fusion.py
from morgan_brain.modules.memory.retrieval.fusion import reciprocal_rank_fusion


def test_item_ranked_high_across_lists_wins():
    vector = ["a", "b", "c"]
    bm25 = ["b", "a", "d"]
    entity = ["a", "e"]
    fused = reciprocal_rank_fusion([vector, bm25, entity])
    assert fused[0] == "a"   # top or near-top in all three


def test_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_list_preserves_order():
    assert reciprocal_rank_fusion([["x", "y", "z"]]) == ["x", "y", "z"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_fusion.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# morgan_brain/modules/memory/retrieval/fusion.py
"""Reciprocal Rank Fusion — combine several ranked id-lists into one. This is Phase 1's single
rerank layer (CrossEncoder reranking is deferred per the degradation ladder)."""
from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return [item_id for item_id, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_fusion.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add morgan_brain/modules/memory/retrieval/fusion.py tests/unit/test_fusion.py
git commit -m "feat(memory): reciprocal rank fusion (single rerank layer)"
```

---

### Task 7: Memory module (implements `MemoryStore`)

Combines embedder + vector index + bm25 + entity match + fusion + temporal store into the
`interfaces.MemoryStore` Protocol.

**Files:**
- Create: `morgan_brain/modules/memory/store.py`
- Test: `tests/unit/test_memory_module.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_memory_module.py
from datetime import datetime

from morgan_brain.models.base import Entity
from morgan_brain.models.memory import Memory, MemoryKind, MemoryQuery, TemporalFact
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.store import MemoryModule


def _module() -> MemoryModule:
    return MemoryModule(
        embedder=FakeEmbedder(dim=16),
        vectors=InMemoryVectorIndex(),
        temporal=SqliteTemporalStore(":memory:"),
        clock=lambda: datetime(2026, 1, 1),
    )


async def test_store_then_recall_finds_memory():
    m = _module()
    await m.store(Memory(user_id="u1", kind=MemoryKind.EPISODIC, content="I love hiking in Berlin"))
    hits = await m.recall(MemoryQuery(user_id="u1", text="hiking Berlin", top_k=5))
    assert any("hiking" in h.content for h in hits)


async def test_recall_is_user_scoped():
    m = _module()
    await m.store(Memory(user_id="u1", kind=MemoryKind.EPISODIC, content="secret note"))
    hits = await m.recall(MemoryQuery(user_id="u2", text="secret", top_k=5))
    assert hits == []


async def test_entity_overlap_boosts_recall():
    m = _module()
    await m.store(Memory(user_id="u1", content="met Alice yesterday",
                         entities=[Entity(name="Alice")]))
    await m.store(Memory(user_id="u1", content="random unrelated text"))
    hits = await m.recall(MemoryQuery(user_id="u1", text="Alice", top_k=2))
    assert hits and hits[0].content == "met Alice yesterday"


async def test_facts_delegate_to_temporal_store():
    m = _module()
    await m.upsert_fact(TemporalFact(user_id="u1", subject="user", predicate="lives_in",
                                     object="Berlin"))
    facts = await m.current_facts(user_id="u1")
    assert len(facts) == 1 and facts[0].object == "Berlin"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_memory_module.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# morgan_brain/modules/memory/store.py
"""MemoryModule — the interfaces.MemoryStore implementation.

Recall is multi-signal: vector (semantic) + BM25 (keyword) + entity overlap, combined with
reciprocal rank fusion (the single rerank layer). Facts are delegated to the bi-temporal store.
All access is user-scoped; callers reach it only through the MemoryGate.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable

from morgan_brain.models.memory import Memory, MemoryQuery, TemporalFact
from morgan_brain.modules.memory.indexing.embedder import Embedder
from morgan_brain.modules.memory.retrieval.bm25 import Bm25Index
from morgan_brain.modules.memory.retrieval.fusion import reciprocal_rank_fusion
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex, VectorHit, VectorRecord


class MemoryModule:
    def __init__(
        self,
        *,
        embedder: Embedder,
        vectors: InMemoryVectorIndex | object,
        temporal: SqliteTemporalStore,
        clock: Callable[[], datetime],
    ) -> None:
        self._embedder = embedder
        self._vectors = vectors  # VectorIndex
        self._temporal = temporal
        self._clock = clock
        self._bm25 = Bm25Index()
        self._by_id: dict[str, Memory] = {}
        self._entities: dict[str, set[str]] = {}  # memory_id -> lowercased entity names

    async def store(self, memory: Memory) -> str:
        if memory.created_at is None:
            memory.created_at = self._clock()
        vector = await self._embedder.embed(memory.content)
        memory.embedding = vector
        await self._vectors.upsert(VectorRecord(
            id=memory.id, user_id=memory.user_id, vector=vector,
            payload={"content": memory.content, "user_id": memory.user_id},
        ))
        self._bm25.add(memory.id, memory.content)
        self._by_id[memory.id] = memory
        self._entities[memory.id] = {e.name.lower() for e in memory.entities}
        return memory.id

    async def recall(self, query: MemoryQuery) -> list[Memory]:
        q_vector = await self._embedder.embed(query.text)
        vec_hits: list[VectorHit] = await self._vectors.search(
            user_id=query.user_id, vector=q_vector, top_k=query.top_k * 2
        )
        vector_ranking = [h.id for h in vec_hits]

        bm25_ranking = [
            mid for mid, _ in self._bm25.search(query.text, top_k=query.top_k * 2)
            if self._owned(mid, query.user_id)
        ]

        q_terms = {t.lower() for t in query.text.split()}
        entity_ranking = [
            mid for mid in self._by_id
            if self._owned(mid, query.user_id) and (self._entities.get(mid, set()) & q_terms)
        ]

        fused_ids = reciprocal_rank_fusion([vector_ranking, bm25_ranking, entity_ranking])
        results = [self._by_id[mid] for mid in fused_ids if mid in self._by_id]
        return results[: query.top_k]

    def _owned(self, memory_id: str, user_id: str) -> bool:
        mem = self._by_id.get(memory_id)
        return mem is not None and mem.user_id == user_id

    async def upsert_fact(self, fact: TemporalFact) -> str:
        return await self._temporal.upsert_fact(fact, now=self._clock())

    async def current_facts(
        self, *, user_id: str, subject: str | None = None
    ) -> list[TemporalFact]:
        return await self._temporal.current_facts(user_id=user_id, subject=subject)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_memory_module.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add morgan_brain/modules/memory/store.py tests/unit/test_memory_module.py
git commit -m "feat(memory): MemoryModule with multi-signal recall + temporal facts"
```

---

### Task 8: LLM client

**Files:**
- Create: `morgan_brain/modules/reasoning/llm/__init__.py`
- Create: `morgan_brain/modules/reasoning/llm/client.py`
- Test: `tests/unit/test_llm_client.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_llm_client.py
from morgan_brain.modules.reasoning.llm.client import FakeLLMClient, ChatMessage


async def test_fake_llm_echoes_scripted_reply():
    llm = FakeLLMClient(reply="hello back")
    out = await llm.complete([ChatMessage(role="user", content="hi")], model="m")
    assert out == "hello back"


async def test_fake_llm_records_last_messages():
    llm = FakeLLMClient(reply="ok")
    msgs = [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="q")]
    await llm.complete(msgs, model="m")
    assert llm.last_messages == msgs
    assert llm.last_model == "m"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_llm_client.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# morgan_brain/modules/reasoning/llm/__init__.py
"""LLM client."""
```

```python
# morgan_brain/modules/reasoning/llm/client.py
"""LLMClient: chat completion. OllamaLLMClient hits the OpenAI-compatible /v1/chat/completions
endpoint; FakeLLMClient returns a scripted reply and records inputs for assertions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx


@dataclass
class ChatMessage:
    role: str  # system | user | assistant
    content: str


@runtime_checkable
class LLMClient(Protocol):
    async def complete(self, messages: list[ChatMessage], *, model: str) -> str: ...


class FakeLLMClient:
    def __init__(self, reply: str = "ok") -> None:
        self._reply = reply
        self.last_messages: list[ChatMessage] | None = None
        self.last_model: str | None = None

    async def complete(self, messages: list[ChatMessage], *, model: str) -> str:
        self.last_messages = messages
        self.last_model = model
        return self._reply


class OllamaLLMClient:
    def __init__(self, endpoint: str, timeout: float = 120.0) -> None:
        self._url = endpoint.rstrip("/") + "/chat/completions"
        self._timeout = timeout

    async def complete(self, messages: list[ChatMessage], *, model: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_llm_client.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add morgan_brain/modules/reasoning/llm tests/unit/test_llm_client.py
git commit -m "feat(reasoning): LLMClient protocol with Ollama + fake implementations"
```

---

### Task 9: Context builder

Pure function assembling chat messages from a `ReasoningRequest`.

**Files:**
- Create: `morgan_brain/modules/reasoning/context/__init__.py`
- Create: `morgan_brain/modules/reasoning/context/builder.py`
- Test: `tests/unit/test_context_builder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_context_builder.py
from morgan_brain.interfaces.personalization import PersonalizedContext
from morgan_brain.interfaces.reasoning import ReasoningRequest
from morgan_brain.models.memory import Memory, MemoryKind
from morgan_brain.models.perception import FusedPerception
from morgan_brain.modules.reasoning.context.builder import build_messages


def _request(**kw) -> ReasoningRequest:
    base = dict(
        user_id="u1",
        perception=FusedPerception(text="where do I live?"),
        personalization=PersonalizedContext(system_fragment="User prefers terse replies."),
        memories=[Memory(user_id="u1", kind=MemoryKind.SEMANTIC, content="User lives in Berlin")],
        history=[],
        skill_prompt="",
    )
    base.update(kw)
    return ReasoningRequest(**base)


def test_system_message_includes_personalization_and_memories():
    msgs = build_messages(_request())
    system = msgs[0]
    assert system.role == "system"
    assert "terse" in system.content
    assert "Berlin" in system.content


def test_last_message_is_the_user_query():
    msgs = build_messages(_request())
    assert msgs[-1].role == "user"
    assert msgs[-1].content == "where do I live?"


def test_skill_prompt_included_when_present():
    msgs = build_messages(_request(skill_prompt="ALWAYS cite memories."))
    assert any("cite memories" in m.content for m in msgs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_context_builder.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# morgan_brain/modules/reasoning/context/__init__.py
"""Context assembly."""
```

```python
# morgan_brain/modules/reasoning/context/builder.py
"""Assemble the LLM message list from a ReasoningRequest: a system message carrying
personalization signals + the active skill + recalled memories, the prior history, then the
current user turn. Pure and deterministic."""
from __future__ import annotations

from morgan_brain.interfaces.reasoning import ReasoningRequest
from morgan_brain.modules.reasoning.llm.client import ChatMessage

_BASE_SYSTEM = (
    "You are Morgan, a personal assistant that knows the user well. "
    "Use the provided memories when relevant. If a memory conflicts with general knowledge, "
    "prefer the memory. Be helpful and concise."
)


def build_messages(request: ReasoningRequest) -> list[ChatMessage]:
    parts = [_BASE_SYSTEM]
    if request.personalization.system_fragment:
        parts.append("About the user: " + request.personalization.system_fragment)
    if request.skill_prompt:
        parts.append("Active skill:\n" + request.skill_prompt)
    if request.memories:
        rendered = "\n".join(f"- {m.content}" for m in request.memories)
        parts.append("Relevant memories:\n" + rendered)

    messages = [ChatMessage(role="system", content="\n\n".join(parts))]
    for msg in request.history:
        messages.append(ChatMessage(role=msg.role.value, content=msg.content))
    messages.append(ChatMessage(role="user", content=request.perception.text))
    return messages
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_context_builder.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add morgan_brain/modules/reasoning/context tests/unit/test_context_builder.py
git commit -m "feat(reasoning): context builder assembling system+memory+history messages"
```

---

### Task 10: Reasoner (implements `Reasoner`)

**Files:**
- Create: `morgan_brain/modules/reasoning/reasoner.py`
- Test: `tests/unit/test_reasoner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_reasoner.py
from morgan_brain.interfaces.personalization import PersonalizedContext
from morgan_brain.interfaces.reasoning import ReasoningRequest
from morgan_brain.models.memory import Memory
from morgan_brain.models.perception import FusedPerception
from morgan_brain.modules.reasoning.llm.client import FakeLLMClient
from morgan_brain.modules.reasoning.reasoner import ReasoningModule


def _request() -> ReasoningRequest:
    return ReasoningRequest(
        user_id="u1",
        perception=FusedPerception(text="hi"),
        personalization=PersonalizedContext(),
        memories=[Memory(user_id="u1", content="user is called Sam")],
        history=[],
        skill_prompt="",
    )


async def test_generate_returns_llm_reply_and_model():
    llm = FakeLLMClient(reply="Hello Sam!")
    r = ReasoningModule(llm=llm, model="qwen2.5:7b", fast_model="qwen2.5:7b")
    result = await r.generate(_request())
    assert result.text == "Hello Sam!"
    assert result.model_used == "qwen2.5:7b"


async def test_generate_passes_memories_into_context():
    llm = FakeLLMClient(reply="ok")
    r = ReasoningModule(llm=llm, model="m", fast_model="m")
    await r.generate(_request())
    system = llm.last_messages[0]
    assert "Sam" in system.content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_reasoner.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Write minimal implementation**

```python
# morgan_brain/modules/reasoning/reasoner.py
"""ReasoningModule — interfaces.Reasoner. Phase 1: build context, route to a model (only the
strong model is used until planning lands), call the LLM, return the reply. Streaming and
tool-calls arrive in later phases."""
from __future__ import annotations

from typing import AsyncIterator

from morgan_brain.interfaces.reasoning import ReasoningRequest, ReasoningResult
from morgan_brain.modules.reasoning.context.builder import build_messages
from morgan_brain.modules.reasoning.llm.client import LLMClient


class ReasoningModule:
    def __init__(self, *, llm: LLMClient, model: str, fast_model: str) -> None:
        self._llm = llm
        self._model = model
        self._fast_model = fast_model

    async def generate(self, request: ReasoningRequest) -> ReasoningResult:
        messages = build_messages(request)
        text = await self._llm.complete(messages, model=self._model)
        return ReasoningResult(text=text, model_used=self._model, tools_invoked=[])

    async def stream(self, request: ReasoningRequest) -> AsyncIterator[str]:
        result = await self.generate(request)
        yield result.text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_reasoner.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add morgan_brain/modules/reasoning/reasoner.py tests/unit/test_reasoner.py
git commit -m "feat(reasoning): ReasoningModule wiring context builder to the LLM client"
```

---

### Task 11: Minimal Personalizer / SkillEngine / Learner

The loop needs all collaborators. Phase 1 ships trivial-but-correct versions; Phase 2/3 replace
the Personalizer/Learner/SkillEngine with real implementations.

**Files:**
- Create: `morgan_brain/modules/personalization/passthrough.py`
- Create: `morgan_brain/modules/skills/noop.py`
- Create: `morgan_brain/modules/learning/minimal.py`
- Test: `tests/unit/test_passthrough_personalizer.py`
- Test: `tests/unit/test_minimal_learner.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_passthrough_personalizer.py
from morgan_brain.models.perception import FusedPerception
from morgan_brain.models.user import CommunicationPrefs, UserModel
from morgan_brain.modules.personalization.passthrough import PassthroughPersonalizer


async def test_passthrough_reflects_comm_prefs_in_fragment():
    p = PassthroughPersonalizer()
    um = UserModel(user_id="u1", comm_prefs=CommunicationPrefs(tone="warm", length="terse"))
    ctx = await p.build(user_model=um, perception=FusedPerception(text="hi"))
    assert "terse" in ctx.system_fragment
    assert ctx.tone == "warm"


async def test_passthrough_empty_for_blank_user_model():
    p = PassthroughPersonalizer()
    ctx = await p.build(user_model=UserModel(user_id="u1"), perception=FusedPerception(text="hi"))
    assert isinstance(ctx.system_fragment, str)
```

```python
# tests/unit/test_minimal_learner.py
from datetime import datetime

from morgan_brain.models.message import Conversation, Message, Role
from morgan_brain.models.user import RelationshipStage
from morgan_brain.modules.learning.minimal import MinimalLearner


class _RecordingMemory:
    def __init__(self):
        self.stored = []

    async def store(self, memory):
        self.stored.append(memory)
        return memory.id

    async def recall(self, query):
        return []

    async def upsert_fact(self, fact):
        return fact.id

    async def current_facts(self, *, user_id, subject=None):
        return []


async def test_user_model_defaults_to_new():
    learner = MinimalLearner(memory=_RecordingMemory(), clock=lambda: datetime(2026, 1, 1))
    um = await learner.user_model("u1")
    assert um.user_id == "u1"
    assert um.relationship_stage is RelationshipStage.NEW


async def test_process_session_stores_each_message_as_episodic():
    mem = _RecordingMemory()
    learner = MinimalLearner(memory=mem, clock=lambda: datetime(2026, 1, 1))
    convo = Conversation(user_id="u1", session_id="s1", messages=[
        Message(user_id="u1", role=Role.USER, content="hello"),
        Message(user_id="u1", role=Role.ASSISTANT, content="hi there"),
    ])
    await learner.process_session(convo)
    assert len(mem.stored) == 2
    assert {m.content for m in mem.stored} == {"hello", "hi there"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_passthrough_personalizer.py tests/unit/test_minimal_learner.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Write minimal implementations**

```python
# morgan_brain/modules/personalization/passthrough.py
"""Phase 1 Personalizer: renders the user's communication preferences into a short system
fragment. No trait selection yet (Phase 2) — it simply surfaces what little the UserModel holds."""
from __future__ import annotations

from morgan_brain.interfaces.personalization import PersonalizedContext
from morgan_brain.models.perception import FusedPerception
from morgan_brain.models.user import UserModel


class PassthroughPersonalizer:
    async def build(
        self, *, user_model: UserModel, perception: FusedPerception
    ) -> PersonalizedContext:
        prefs = user_model.comm_prefs
        bits = []
        if prefs.length != "balanced":
            bits.append(f"prefers {prefs.length} replies")
        if prefs.tone != "neutral":
            bits.append(f"tone: {prefs.tone}")
        if prefs.code_vs_prose != "balanced":
            bits.append(prefs.code_vs_prose.replace("_", " "))
        fragment = "; ".join(bits)
        return PersonalizedContext(system_fragment=fragment, tone=prefs.tone)
```

```python
# morgan_brain/modules/skills/noop.py
"""Phase 1 SkillEngine: selects nothing. Real skill selection + SkillOpt arrive in Phase 3."""
from __future__ import annotations

from morgan_brain.interfaces.skills import Skill
from morgan_brain.models.perception import FusedPerception


class NoopSkillEngine:
    async def select(self, perception: FusedPerception) -> list[Skill]:
        return []

    async def get(self, name: str) -> Skill | None:
        return None

    async def deploy(self, skill: Skill) -> None:
        return None
```

```python
# morgan_brain/modules/learning/minimal.py
"""Phase 1 Learner: returns a default UserModel and persists each session message as an episodic
memory (so recall works across turns). Real trait/preference extraction + UserModel maintenance
arrive in Phase 2, running in the learning-worker."""
from __future__ import annotations

from datetime import datetime
from typing import Callable

from morgan_brain.interfaces.memory import MemoryStore
from morgan_brain.models.memory import Memory, MemoryKind, MemorySource
from morgan_brain.models.message import Conversation, Role
from morgan_brain.models.user import UserModel


class MinimalLearner:
    def __init__(self, *, memory: MemoryStore, clock: Callable[[], datetime]) -> None:
        self._memory = memory
        self._clock = clock

    async def process_session(self, conversation: Conversation) -> None:
        for msg in conversation.messages:
            source = MemorySource.USER_STATED if msg.role is Role.USER else MemorySource.AGENT_INFERRED
            await self._memory.store(Memory(
                user_id=conversation.user_id,
                kind=MemoryKind.EPISODIC,
                content=msg.content,
                source=source,
                created_at=self._clock(),
            ))

    async def user_model(self, user_id: str) -> UserModel:
        return UserModel(user_id=user_id)

    async def consolidate(self, user_id: str) -> None:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_passthrough_personalizer.py tests/unit/test_minimal_learner.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add morgan_brain/modules/personalization/passthrough.py morgan_brain/modules/skills/noop.py \
        morgan_brain/modules/learning/minimal.py \
        tests/unit/test_passthrough_personalizer.py tests/unit/test_minimal_learner.py
git commit -m "feat: minimal Personalizer, SkillEngine, Learner to complete the Phase 1 loop"
```

---

### Task 12: Composition root + wire `/api/chat`

Build the orchestrator from settings, register a turn-storage subscriber on `RESPONSE_GENERATED`
(cold-path write), and make `/api/chat` drive the loop.

**Files:**
- Create: `morgan_brain/composition.py`
- Modify: `morgan_brain/apps/brain_api/app.py`
- Test: `tests/integration/test_chat_loop.py`
- Create: `tests/integration/__init__.py`

- [ ] **Step 1: Write the failing integration test**

```python
# tests/integration/__init__.py
```

```python
# tests/integration/test_chat_loop.py
from datetime import datetime

from morgan_brain.composition import build_orchestrator_for_test


async def test_chat_loop_generates_and_recalls():
    orch, mem = build_orchestrator_for_test(reply="Nice to meet you!", clock=lambda: datetime(2026, 1, 1))

    first = await orch.handle_turn(user_id="u1", text="My name is Sam", session_id="s1")
    assert first.text == "Nice to meet you!"

    # the RESPONSE_GENERATED subscriber should have stored the turn as episodic memory
    hits = await mem.recall_raw(user_id="u1", text="Sam")
    assert any("Sam" in h.content for h in hits)


async def test_chat_loop_is_user_scoped():
    orch, mem = build_orchestrator_for_test(reply="ok", clock=lambda: datetime(2026, 1, 1))
    await orch.handle_turn(user_id="u1", text="secret for u1", session_id="s1")
    other = await mem.recall_raw(user_id="u2", text="secret")
    assert other == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_chat_loop.py -v`
Expected: FAIL with import error (`build_orchestrator_for_test`).

- [ ] **Step 3: Write the composition root**

```python
# morgan_brain/composition.py
"""Composition root — assemble the Orchestrator from settings (production) or fakes (tests).

Also registers the cold-path turn-storage subscriber: on RESPONSE_GENERATED, the just-finished
turn is persisted as episodic memory via the Learner. With the in-process bus this runs after the
reply is produced; with the Redis bus (later phases) it runs in the learning-worker, off-path.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from morgan_brain.config import Settings, get_settings
from morgan_brain.core.orchestrator import Orchestrator
from morgan_brain.interfaces.events import Event, EventType
from morgan_brain.models.message import Conversation, Message, Role
from morgan_brain.modules.learning.minimal import MinimalLearner
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder, OllamaEmbedder
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex
from morgan_brain.modules.perception.text.analyzer import TextPerception
from morgan_brain.modules.personalization.passthrough import PassthroughPersonalizer
from morgan_brain.modules.reasoning.llm.client import FakeLLMClient, OllamaLLMClient
from morgan_brain.modules.reasoning.reasoner import ReasoningModule
from morgan_brain.modules.skills.noop import NoopSkillEngine
from morgan_brain.security.memory_gate import MemoryGate
from morgan_brain.bus.inproc import InProcessBus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _register_turn_storage(bus: InProcessBus, learner: MinimalLearner) -> None:
    async def _store_turn(event: Event) -> None:
        payload = event.payload
        convo = Conversation(
            user_id=event.user_id,
            session_id=payload.get("session_id") or "default",
            messages=[
                Message(user_id=event.user_id, role=Role.USER, content=payload["request"]),
                Message(user_id=event.user_id, role=Role.ASSISTANT, content=payload["response"]),
            ],
        )
        await learner.process_session(convo)

    bus.subscribe(EventType.RESPONSE_GENERATED, _store_turn)


def _assemble(
    *,
    embedder,
    llm,
    settings: Settings,
    clock: Callable[[], datetime],
    temporal_path: str,
) -> tuple[Orchestrator, MemoryModule]:
    memory_module = MemoryModule(
        embedder=embedder,
        vectors=InMemoryVectorIndex(),
        temporal=SqliteTemporalStore(temporal_path),
        clock=clock,
    )
    gate = MemoryGate(memory_module)
    learner = MinimalLearner(memory=gate, clock=clock)
    bus = InProcessBus()
    _register_turn_storage(bus, learner)
    orch = Orchestrator(
        perception=TextPerception(),
        personalizer=PassthroughPersonalizer(),
        memory=gate,
        skills=NoopSkillEngine(),
        reasoner=ReasoningModule(
            llm=llm, model=settings.llm_model, fast_model=settings.llm_fast_model
        ),
        learner=learner,
        bus=bus,
    )
    return orch, memory_module


def build_orchestrator(settings: Settings | None = None) -> Orchestrator:
    """Production wiring (Ollama + in-memory vectors for Phase 1; Qdrant swaps in later)."""
    settings = settings or get_settings()
    embedder = OllamaEmbedder(settings.llm_endpoint, settings.embedding_model)
    llm = OllamaLLMClient(settings.llm_endpoint)
    orch, _ = _assemble(
        embedder=embedder, llm=llm, settings=settings, clock=_utcnow,
        temporal_path=_sqlite_path(settings.temporal_db_url),
    )
    return orch


def build_orchestrator_for_test(
    *, reply: str, clock: Callable[[], datetime]
):
    """Test wiring: fake embedder + fake LLM + in-memory stores. Returns (orchestrator, memory
    handle) where the memory handle exposes recall_raw() for assertions."""
    settings = Settings(llm_model="test-model", llm_fast_model="test-model")
    orch, memory_module = _assemble(
        embedder=FakeEmbedder(dim=16), llm=FakeLLMClient(reply=reply),
        settings=settings, clock=clock, temporal_path=":memory:",
    )

    class _Handle:
        async def recall_raw(self, *, user_id: str, text: str):
            from morgan_brain.models.memory import MemoryQuery
            return await memory_module.recall(MemoryQuery(user_id=user_id, text=text, top_k=10))

    return orch, _Handle()


def _sqlite_path(url: str) -> str:
    """Turn a sqlite:/// URL into a filesystem path; pass through ':memory:'."""
    prefix = "sqlite:///"
    return url[len(prefix):] if url.startswith(prefix) else url
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_chat_loop.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Wire `/api/chat` in the FastAPI app**

Replace the body of `morgan_brain/apps/brain_api/app.py` with:

```python
# morgan_brain/apps/brain_api/app.py
"""FastAPI app factory for brain-api. Phase 1: /api/chat drives the cognitive loop."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from morgan_brain import __version__
from morgan_brain.composition import build_orchestrator
from morgan_brain.config import get_settings


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    user_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    model_used: str


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="morgan-brain", version=__version__)
    orchestrator = build_orchestrator(settings)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__, "event_bus": settings.event_bus}

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        user_id = req.user_id or settings.owner_user_id
        result = await orchestrator.handle_turn(
            user_id=user_id, text=req.message, session_id=req.session_id
        )
        return ChatResponse(response=result.text, model_used=result.model_used)

    return app


app = create_app()
```

- [ ] **Step 6: Verify the app imports and the health route still works**

Run:
```bash
PYTHONPATH=. python -c "from morgan_brain.apps.brain_api.app import create_app; app = create_app(); print('app ok', [r.path for r in app.routes if hasattr(r, 'path')])"
```
Expected: prints `app ok [...] '/health' ... '/api/chat'` (no exceptions; `build_orchestrator` constructs Ollama clients lazily, so no network call happens at import).

- [ ] **Step 7: Commit**

```bash
git add morgan_brain/composition.py morgan_brain/apps/brain_api/app.py \
        tests/integration/__init__.py tests/integration/test_chat_loop.py
git commit -m "feat: composition root + /api/chat driving the cognitive loop"
```

---

### Task 13: Memory-quality regression harness

A small LoCoMo/LongMemEval-style scorer so memory changes are measured. Phase 1 ships
single-hop, knowledge-update, and temporal cases.

**Files:**
- Create: `tests/memory_quality/__init__.py`
- Create: `tests/memory_quality/conftest.py`
- Create: `tests/memory_quality/test_recall_quality.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/memory_quality/__init__.py
```

```python
# tests/memory_quality/conftest.py
"""Fixtures + scorer for the memory-quality suite (design spec §13)."""
from __future__ import annotations

from datetime import datetime

import pytest

from morgan_brain.models.memory import Memory, MemoryKind, MemoryQuery
from morgan_brain.modules.memory.indexing.embedder import FakeEmbedder
from morgan_brain.modules.memory.store import MemoryModule
from morgan_brain.modules.memory.stores.temporal import SqliteTemporalStore
from morgan_brain.modules.memory.stores.vector import InMemoryVectorIndex


@pytest.fixture
def memory() -> MemoryModule:
    return MemoryModule(
        embedder=FakeEmbedder(dim=32),
        vectors=InMemoryVectorIndex(),
        temporal=SqliteTemporalStore(":memory:"),
        clock=lambda: datetime(2026, 1, 1),
    )


async def seed(memory: MemoryModule, user_id: str, contents: list[str]) -> None:
    for c in contents:
        await memory.store(Memory(user_id=user_id, kind=MemoryKind.EPISODIC, content=c))


def recall_at_k(results: list[Memory], expected_substring: str, k: int) -> float:
    """1.0 if any of the top-k recalled memories contains the expected substring, else 0.0."""
    return 1.0 if any(expected_substring.lower() in m.content.lower() for m in results[:k]) else 0.0
```

```python
# tests/memory_quality/test_recall_quality.py
"""Recall-quality regression. Keep these GREEN; if a memory change drops a score, that's the
signal the change hurt recall. Thresholds are intentionally strict for the fake embedder
(deterministic), and document the categories that matter."""
from __future__ import annotations

from morgan_brain.models.memory import MemoryQuery, TemporalFact
from tests.memory_quality.conftest import recall_at_k, seed


async def test_single_hop_recall(memory):
    await seed(memory, "u1", [
        "User's favorite programming language is Python",
        "User enjoys mountain biking on weekends",
        "User works as a data engineer",
    ])
    results = await memory.recall(MemoryQuery(user_id="u1", text="favorite programming language", top_k=3))
    assert recall_at_k(results, "Python", k=3) == 1.0


async def test_knowledge_update_latest_fact_wins(memory):
    # facts evolve: city changed from Berlin to Munich
    await memory.upsert_fact(TemporalFact(user_id="u1", subject="user", predicate="lives_in",
                                          object="Berlin"))
    await memory.upsert_fact(TemporalFact(user_id="u1", subject="user", predicate="lives_in",
                                          object="Munich"))
    current = await memory.current_facts(user_id="u1", subject="user")
    objs = {f.object for f in current}
    assert objs == {"Munich"}            # latest wins
    assert "Berlin" not in objs          # stale value does not leak


async def test_temporal_history_is_queryable(memory):
    await memory.upsert_fact(TemporalFact(user_id="u1", subject="user", predicate="lives_in",
                                          object="Berlin"))
    await memory.upsert_fact(TemporalFact(user_id="u1", subject="user", predicate="lives_in",
                                          object="Munich"))
    history = await memory._temporal.history(user_id="u1", subject="user", predicate="lives_in")
    assert {f.object for f in history} == {"Berlin", "Munich"}   # past is retained, not deleted
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/memory_quality/ -v`
Expected: FAIL initially only if earlier tasks incomplete; otherwise these should pass once Tasks 4 & 7 exist. If `tests/memory_quality/README.md` previously existed, leave it — it documents the suite.

- [ ] **Step 3: (No new implementation needed)** — the harness exercises Task 4 & 7 code.

- [ ] **Step 4: Run the full suite**

Run: `pytest -v`
Expected: PASS (all unit + integration + memory_quality tests).

- [ ] **Step 5: Commit**

```bash
git add tests/memory_quality/__init__.py tests/memory_quality/conftest.py \
        tests/memory_quality/test_recall_quality.py
git commit -m "test(memory): LoCoMo-style recall-quality regression harness"
```

---

### Task 14: Full verification + optional live smoke test

**Files:** none (verification)

- [ ] **Step 1: Run the whole suite**

Run: `pytest -v`
Expected: ALL PASS. If any fail, fix before proceeding (do not move to Phase 2 on red).

- [ ] **Step 2: Type + lint**

Run:
```bash
ruff check .
mypy morgan_brain
```
Expected: clean (or only pre-agreed ignores). Fix real issues.

- [ ] **Step 3: (Optional) live smoke test against Ollama + Qdrant**

Only if Ollama is running locally with the configured models:
```bash
docker compose up -d redis qdrant
python -m morgan_brain.apps.brain_api &
sleep 2
curl -s localhost:8080/health
curl -s -X POST localhost:8080/api/chat -H 'content-type: application/json' \
     -d '{"message":"My name is Sam and I live in Berlin"}'
curl -s -X POST localhost:8080/api/chat -H 'content-type: application/json' \
     -d '{"message":"What is my name?"}'
```
Expected: second reply references "Sam" (cross-turn recall via stored episodic memory).
Note: Phase 1 uses `InMemoryVectorIndex` even in production wiring; swapping to `QdrantVectorIndex`
is a Phase 1.5 follow-up (see "Deferred" below). Memory does not survive a process restart yet.

- [ ] **Step 4: Final commit (if any fixes were made)**

```bash
git add -A
git commit -m "chore: Phase 1 verification fixes"
```

---

## Deferred to a Phase 1.5 follow-up (explicitly out of scope here)
- Swap production wiring from `InMemoryVectorIndex` to `QdrantVectorIndex` (persistence across
  restarts) + a `MemoryModule` that rebuilds its BM25/entity indexes from the vector store on boot.
- Streaming responses over WebSocket.
- CrossEncoder reranking rung above RRF.
- Redis Streams bus + moving turn-storage into the learning-worker (true off-path writes).

These are noted so the next plan can pick them up; Phase 1's goal (a working, recall-capable text
assistant, fully unit-tested) is met without them.

## Self-review notes (completed by plan author)
- **Spec coverage:** §6 cognitive loop → Task 12 orchestrator wiring; §7 memory (multi-signal,
  bi-temporal, actor attribution, single rerank) → Tasks 3–7, 13; §9 Personalization (minimal) →
  Task 11; reasoning → Tasks 8–10; §13 testing/memory-quality → Task 13. Learning extraction,
  proactivity, skills, MCP, audio are later phases by design.
- **Placeholders:** none — every code step is complete.
- **Type consistency:** method names match the Protocols in `morgan_brain/interfaces/`
  (`store`/`recall`/`upsert_fact`/`current_facts`, `generate`/`stream`, `build`, `select`/`get`/
  `deploy`, `process_session`/`user_model`/`consolidate`). `ChatMessage`, `VectorRecord/VectorHit`,
  `Embedder/LLMClient/VectorIndex` are defined before use.
