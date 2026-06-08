"""Scheduling package — CronService, InProcessScheduler, HeartbeatManager.

APScheduler is entirely optional; if absent (or ``enable_scheduling=False``),
:class:`InProcessScheduler` is used instead.  Tests always use
``InProcessScheduler`` via deterministic ``tick`` calls — no real sleeping.
"""
from morgan_brain.scheduling.cron import (
    CronService,
    InProcessScheduler,
    Job,
    Scheduler,
)
from morgan_brain.scheduling.heartbeat import HeartbeatManager

__all__ = [
    "CronService",
    "HeartbeatManager",
    "InProcessScheduler",
    "Job",
    "Scheduler",
]
