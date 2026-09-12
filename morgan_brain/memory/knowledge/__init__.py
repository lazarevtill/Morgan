"""Turning stored memories into structured knowledge.

``extract`` finds the entities a memory mentions, ``schema_classifier`` files them into the
upper index's slots, and ``consolidation`` asks a model to turn recent episodics into
valid-time facts. This is the work that costs a model call or a full pass over a project: it
runs when asked, never inside a recall.
"""
