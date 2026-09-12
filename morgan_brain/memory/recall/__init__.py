"""Turning a query into ranked memories.

``semantic_index`` narrows the search to a candidate pool -- coarsely by schema, concretely
by entity -- and returns ``None`` rather than an empty pool whenever it has nothing useful
to say, because routing may cost precision and must never cost recall. ``fusion`` merges the
vector, keyword and entity rankings by reciprocal rank. Rank-only: scores do not survive
fusion, so any relevance threshold belongs on a signal before it reaches here.
"""
