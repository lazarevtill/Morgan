"""The memory core: one SQLite database, project-scoped, behind one gate.

``gate.MemoryGate`` is the only door. Behind it, ``module.MemoryModule`` writes each memory
to every index in one place (episodic rows, sqlite-vec vectors, FTS5, the entity index, and
the semantic upper index) and fuses the three retrieval signals on recall.
"""
