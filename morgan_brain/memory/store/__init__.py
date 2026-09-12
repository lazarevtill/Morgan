"""Persistence: one file per table family, all over the one SQLite connection.

Nothing here decides what to store or how to rank it. Each store owns its schema, its
migrations and its own queries, and every one of them is project-scoped because the row is
where the scope lives. ``db.py`` opens the single connection the rest share.
"""
