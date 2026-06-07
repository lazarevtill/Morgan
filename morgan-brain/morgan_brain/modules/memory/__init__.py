"""Memory module — implements ``interfaces.MemoryStore`` (accessed only via MemoryGate).

Responsibility: store and retrieve episodic/semantic/procedural memory. Multi-signal retrieval
(vector + BM25 + entity), bi-temporal facts (evolution, not overwrite), actor attribution,
single rerank layer. Never decides what to learn or how to apply it.
Service: brain-api. Phase: 1.

Planned files: stores/vector.py (Qdrant), stores/temporal.py (SQLite→PG bi-temporal facts),
stores/workspace.py (SOUL.md/MEMORY.md), retrieval/search.py, retrieval/reranker.py,
indexing/embedder.py (port from legacy services/embeddings).
"""
