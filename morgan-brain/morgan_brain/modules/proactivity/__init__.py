"""Proactivity module — consent-gated autonomy.

Responsibility: initiate, but only via approved rules and gated by consent (default-deny).
Built. Enabled via MORGAN_ENABLE_PROACTIVITY. Delivery via channels (Telegram seam).

Implementation lives in sibling top-level packages, not here:
``morgan_brain.proactivity`` (engine.py, consent.py) and ``morgan_brain.scheduling``
(cron.py, heartbeat.py, learning_jobs.py — the worker's nightly consolidation + optimizer).
Optional dependency: morgan-brain[scheduling] (APScheduler; an in-process scheduler is the default).
"""
