"""Scheduling package — CronService and InProcessScheduler.

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
from morgan_brain.scheduling.learning_jobs import LearningScheduler

__all__ = [
    "CronService",
    "InProcessScheduler",
    "Job",
    "LearningScheduler",
    "Scheduler",
]
