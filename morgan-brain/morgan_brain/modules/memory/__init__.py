"""Memory module — implements ``interfaces.MemoryStore`` (accessed only via MemoryGate).

Responsibility: store and retrieve episodic/semantic/procedural memory. Multi-signal retrieval
(vector + BM25 + entity), bi-temporal facts (evolution, not overwrite), actor attribution.
Never decides what to learn or how to apply it.
Service: brain-api (wired into the request path via MemoryGate; read on the hot path, written on
the cold path). Built.

Files: store.py (MemoryStore impl), stores/vector.py (in-memory + Qdrant backends),
stores/temporal.py (SQLite bi-temporal facts), retrieval/bm25.py + retrieval/fusion.py
(multi-signal fusion), indexing/embedder.py.
"""
