"""
CronService — manages cron-scheduled jobs with optional APScheduler backend.

If APScheduler is installed, ``start()`` will register real triggers.  Otherwise
jobs are stored in memory (and optionally persisted to disk) but a warning is
logged indicating that automatic scheduling is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

from morgan.scheduling.jobs import CronJob

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detect APScheduler availability
# ---------------------------------------------------------------------------
_HAS_APSCHEDULER = False
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
    from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]

    _HAS_APSCHEDULER = True
except ImportError:
    pass

_PERSISTENCE_VERSION = 1


class CronService:
    """Manage cron-scheduled jobs.

    Parameters:
        persistence_path: Optional filesystem path for JSON persistence.
            When ``None``, jobs live only in memory.
    """

    def __init__(self, persistence_path: str | None = None) -> None:
        self._jobs: dict[str, CronJob] = {}
        self._persistence_path = persistence_path
        self._job_handler: Callable[..., Any] | None = None
        self._apscheduler_available: bool = _HAS_APSCHEDULER
        self._scheduler: Any = None  # AsyncIOScheduler when available
        self._started: bool = False

    # -- CRUD ---------------------------------------------------------------

    def add_job(self, job: CronJob) -> None:
        """Register a new cron job.  Raises ``ValueError`` on duplicate ``job_id``."""
        if job.job_id in self._jobs:
            raise ValueError(f"Job '{job.job_id}' already exists")
        self._jobs[job.job_id] = job

    def remove_job(self, job_id: str) -> bool:
        """Remove a job by id.  Returns ``True`` if found and removed."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            # Also remove from APScheduler if running
            if self._scheduler is not None:
                try:
                    self._scheduler.remove_job(job_id)
                except Exception:
                    pass
            return True
        return False

    def get_job(self, job_id: str) -> CronJob | None:
        """Return a job by id, or ``None`` if not found."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[CronJob]:
        """Return all registered jobs."""
        return list(self._jobs.values())

    def set_job_handler(self, handler: Callable[..., Any]) -> None:
        """Set the callback invoked when a cron job fires.

        The handler receives a single ``CronJob`` argument.
        """
        self._job_handler = handler

    # -- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Start scheduling.

        If APScheduler is available, creates an ``AsyncIOScheduler`` and
        registers all jobs.  Otherwise logs a warning.
        """
        if self._started:
            return

        if not self._apscheduler_available:
            logger.warning(
                "APScheduler not installed — cron jobs are stored but will "
                "NOT be automatically triggered.  Install apscheduler to "
                "enable real scheduling."
            )
            self._started = True
            return

        # APScheduler path
        self._scheduler = AsyncIOScheduler()
        for job in self._jobs.values():
            self._register_ap_job(job)
        self._scheduler.start()
        self._started = True
        logger.info("CronService started with APScheduler (%d jobs)", len(self._jobs))

    async def stop(self) -> None:
        """Stop scheduling and clean up."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
        self._started = False

    # -- Persistence --------------------------------------------------------

    def save(self) -> None:
        """Persist all jobs to disk as JSON."""
        if self._persistence_path is None:
            return
        data: dict[str, Any] = {
            "version": _PERSISTENCE_VERSION,
            "jobs": {jid: job.to_dict() for jid, job in self._jobs.items()},
        }
        # Atomic-ish write via temp file + rename
        tmp_path = self._persistence_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self._persistence_path)

    def load(self) -> None:
        """Load jobs from disk.  No-op if the file does not exist."""
        if self._persistence_path is None:
            return
        if not os.path.exists(self._persistence_path):
            return
        try:
            with open(self._persistence_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load cron persistence file: %s", exc)
            return

        version = data.get("version", 0)
        if version != _PERSISTENCE_VERSION:
            logger.warning(
                "Cron persistence version mismatch (got %s, expected %s) — skipping load",
                version,
                _PERSISTENCE_VERSION,
            )
            return

        for jid, jdata in data.get("jobs", {}).items():
            self._jobs[jid] = CronJob.from_dict(jdata)

    # -- Internal -----------------------------------------------------------

    def _register_ap_job(self, job: CronJob) -> None:
        """Register a single job with the APScheduler backend."""
        if self._scheduler is None:
            return

        parts = job.schedule.split()
        trigger = CronTrigger(
            minute=parts[0] if len(parts) > 0 else "*",
            hour=parts[1] if len(parts) > 1 else "*",
            day=parts[2] if len(parts) > 2 else "*",
            month=parts[3] if len(parts) > 3 else "*",
            day_of_week=parts[4] if len(parts) > 4 else "*",
        )

        def _fire(j: CronJob = job) -> None:
            if self._job_handler is not None:
                self._job_handler(j)

        self._scheduler.add_job(_fire, trigger, id=job.job_id)
