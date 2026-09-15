"""The memory core: one SQLite database, project-scoped, behind one gate.

``gate.MemoryGate`` is the only door. Behind it, ``module.MemoryModule`` writes each memory
to every index in one place (episodic rows, sqlite-vec vectors, FTS5 and the entity index)
and fuses the retrieval signals on recall. ``migrations`` brings a database written by an
older version up to date when it is opened.
"""
