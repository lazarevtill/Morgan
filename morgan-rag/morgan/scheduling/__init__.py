"""
Scheduling Module for Morgan AI Assistant.

Provides two complementary scheduling patterns ported from OpenClaw:

- **Cron**: Exact timing, isolated subprocess, specific schedule (via APScheduler
  when available, graceful fallback otherwise).
- **Heartbeat**: Approximate interval with jitter, batches 2-3 checks per beat,
  conversational / lightweight.
"""

from morgan.scheduling.jobs import CronJob, HeartbeatCheck
from morgan.scheduling.cron_service import CronService
from morgan.scheduling.heartbeat import HeartbeatManager

__all__ = [
    "CronJob",
    "HeartbeatCheck",
    "CronService",
    "HeartbeatManager",
]
