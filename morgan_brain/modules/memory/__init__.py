"""Memory module — implements ``interfaces.MemoryStore`` (accessed only via MemoryGate).

Responsibility: store and retrieve episodic/semantic/procedural memory. Multi-signal retrieval
(vector + FTS5 keyword + entity), bi-temporal facts (evolution, not overwrite), actor
attribution. Never decides what to learn or how to apply it.
Service: brain-api (wired into the request path via MemoryGate; read on the hot path, written on
the cold path). Built.

Files: store.py (MemoryStore impl), stores/db.py (shared sqlite-vec connection),
stores/sqlite_vector.py + stores/vector.py (persistent + in-memory/Qdrant vector backends),
stores/temporal.py (SQLite bi-temporal facts), stores/episodic.py (durable episodic records),
retrieval/fts.py + retrieval/entities.py + retrieval/fusion.py (multi-signal fusion),
indexing/embedder.py.
"""
