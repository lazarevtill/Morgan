"""Proactivity module — consent-gated autonomy.

Responsibility: initiate, but only via approved rules and gated by UserModel.relationship_stage.
Heartbeat (jittered tick), cron (scheduled jobs), pattern-triggered (from mined behavioral
patterns). Delivery via channels (Telegram/Discord).
Service: learning-worker (triggers) + brain-api (delivery). Phase: 4.
Optional dependency: morgan-brain[scheduling].

Planned files: heartbeat.py, cron.py, triggers.py, consent.py.
"""
