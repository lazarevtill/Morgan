"""
Tests for Scheduling Module (Heartbeat + Cron)

Tests cron job management, heartbeat batching, disk persistence,
and graceful APScheduler fallback.
"""

import asyncio
import json
import os
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from morgan.scheduling.jobs import CronJob, HeartbeatCheck
from morgan.scheduling.cron_service import CronService
from morgan.scheduling.heartbeat import HeartbeatManager


# ---------------------------------------------------------------------------
# CronJob dataclass
# ---------------------------------------------------------------------------
class TestCronJob(unittest.TestCase):
    """Test the CronJob dataclass."""

    def test_create_basic(self):
        job = CronJob(
            job_id="daily-summary",
            schedule="0 9 * * *",
            prompt="Summarise today's agenda",
            channel="slack-general",
        )
        self.assertEqual(job.job_id, "daily-summary")
        self.assertEqual(job.schedule, "0 9 * * *")
        self.assertEqual(job.prompt, "Summarise today's agenda")
        self.assertEqual(job.channel, "slack-general")
        # defaults
        self.assertEqual(job.model, "default")
        self.assertFalse(job.isolated)
        self.assertEqual(job.metadata, {})

    def test_to_dict_roundtrip(self):
        job = CronJob(
            job_id="j1",
            schedule="*/5 * * * *",
            prompt="ping",
            channel="ch",
            model="gpt-4",
            isolated=True,
            metadata={"key": "val"},
        )
        d = job.to_dict()
        restored = CronJob.from_dict(d)
        self.assertEqual(job, restored)

    def test_from_dict_missing_optional(self):
        d = {
            "job_id": "j2",
            "schedule": "0 * * * *",
            "prompt": "hello",
            "channel": "ch",
        }
        job = CronJob.from_dict(d)
        self.assertEqual(job.model, "default")
        self.assertFalse(job.isolated)
        self.assertEqual(job.metadata, {})


# ---------------------------------------------------------------------------
# HeartbeatCheck dataclass
# ---------------------------------------------------------------------------
class TestHeartbeatCheck(unittest.TestCase):
    """Test the HeartbeatCheck dataclass."""

    def test_create(self):
        fn = lambda: "ok"
        check = HeartbeatCheck(name="mem-check", fn=fn, priority=5)
        self.assertEqual(check.name, "mem-check")
        self.assertEqual(check.priority, 5)
        self.assertEqual(check.last_run, 0.0)
        self.assertIs(check.fn, fn)

    def test_sort_key(self):
        """Checks sort by (last_run, -priority) so least-recent + highest priority first."""
        c1 = HeartbeatCheck(name="a", fn=lambda: None, priority=1, last_run=100.0)
        c2 = HeartbeatCheck(name="b", fn=lambda: None, priority=10, last_run=50.0)
        c3 = HeartbeatCheck(name="c", fn=lambda: None, priority=5, last_run=50.0)

        ordered = sorted([c1, c2, c3], key=lambda c: (c.last_run, -c.priority))
        self.assertEqual([c.name for c in ordered], ["b", "c", "a"])


# ---------------------------------------------------------------------------
# CronService — in-memory operations
# ---------------------------------------------------------------------------
class TestCronServiceInMemory(unittest.TestCase):
    """Test CronService CRUD without disk persistence."""

    def setUp(self):
        self.svc = CronService(persistence_path=None)

    def test_add_and_get_job(self):
        job = CronJob(
            job_id="j1", schedule="0 9 * * *", prompt="hi", channel="ch"
        )
        self.svc.add_job(job)
        self.assertEqual(self.svc.get_job("j1"), job)

    def test_add_duplicate_raises(self):
        job = CronJob(job_id="j1", schedule="0 9 * * *", prompt="hi", channel="ch")
        self.svc.add_job(job)
        with self.assertRaises(ValueError):
            self.svc.add_job(job)

    def test_remove_job(self):
        job = CronJob(job_id="j1", schedule="0 9 * * *", prompt="hi", channel="ch")
        self.svc.add_job(job)
        removed = self.svc.remove_job("j1")
        self.assertTrue(removed)
        self.assertIsNone(self.svc.get_job("j1"))

    def test_remove_missing_returns_false(self):
        self.assertFalse(self.svc.remove_job("nope"))

    def test_list_jobs(self):
        for i in range(3):
            self.svc.add_job(
                CronJob(job_id=f"j{i}", schedule="* * * * *", prompt="p", channel="c")
            )
        self.assertEqual(len(self.svc.list_jobs()), 3)

    def test_set_job_handler(self):
        handler = MagicMock()
        self.svc.set_job_handler(handler)
        # handler stored internally
        self.assertIs(self.svc._job_handler, handler)


# ---------------------------------------------------------------------------
# CronService — disk persistence
# ---------------------------------------------------------------------------
class TestCronServicePersistence(unittest.TestCase):
    """Test CronService JSON disk persistence."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        )
        self.tmp.close()
        self.path = self.tmp.name

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_save_and_load(self):
        svc = CronService(persistence_path=self.path)
        svc.add_job(
            CronJob(
                job_id="j1",
                schedule="0 9 * * *",
                prompt="hi",
                channel="ch",
                metadata={"tag": "test"},
            )
        )
        svc.save()

        # Read raw file and check format
        with open(self.path) as f:
            data = json.load(f)
        self.assertEqual(data["version"], 1)
        self.assertIn("j1", data["jobs"])

        # Load into a new service
        svc2 = CronService(persistence_path=self.path)
        svc2.load()
        self.assertEqual(svc2.get_job("j1").prompt, "hi")
        self.assertEqual(svc2.get_job("j1").metadata, {"tag": "test"})

    def test_load_missing_file_no_error(self):
        """Loading from a non-existent file should not raise."""
        svc = CronService(persistence_path="/tmp/does_not_exist_morgan.json")
        svc.load()  # should silently do nothing
        self.assertEqual(svc.list_jobs(), [])

    def test_save_creates_file(self):
        os.unlink(self.path)  # remove it so save creates fresh
        svc = CronService(persistence_path=self.path)
        svc.add_job(CronJob(job_id="x", schedule="* * * * *", prompt="p", channel="c"))
        svc.save()
        self.assertTrue(os.path.exists(self.path))


# ---------------------------------------------------------------------------
# CronService — APScheduler integration
# ---------------------------------------------------------------------------
class TestCronServiceAPScheduler(unittest.TestCase):
    """Test that CronService handles APScheduler availability gracefully."""

    def test_start_without_apscheduler_logs_warning(self):
        """When APScheduler is not available, start() should log a warning, not crash."""
        svc = CronService(persistence_path=None)
        svc.add_job(CronJob(job_id="j1", schedule="0 9 * * *", prompt="hi", channel="ch"))

        with patch.object(svc, "_apscheduler_available", False):
            # Should not raise
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(svc.start())
            finally:
                loop.close()

    def test_stop_is_safe_when_not_started(self):
        svc = CronService(persistence_path=None)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(svc.stop())
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# HeartbeatManager
# ---------------------------------------------------------------------------
class TestHeartbeatManager(unittest.TestCase):
    """Test HeartbeatManager registration and beat execution."""

    def test_register_check(self):
        mgr = HeartbeatManager(interval_minutes=30, checks_per_beat=3)
        mgr.register_check("mem", fn=lambda: "ok", priority=5)
        self.assertEqual(len(mgr._checks), 1)
        self.assertEqual(mgr._checks["mem"].priority, 5)

    def test_register_duplicate_replaces(self):
        mgr = HeartbeatManager()
        mgr.register_check("c", fn=lambda: "a", priority=1)
        mgr.register_check("c", fn=lambda: "b", priority=2)
        self.assertEqual(mgr._checks["c"].priority, 2)

    def test_run_beat_picks_top_n(self):
        """_run_beat should pick the top checks_per_beat items."""
        mgr = HeartbeatManager(checks_per_beat=2)

        results = {}

        def make_fn(name):
            def fn():
                results[name] = True
                return f"{name}-done"
            return fn

        mgr.register_check("a", fn=make_fn("a"), priority=1)
        mgr.register_check("b", fn=make_fn("b"), priority=10)
        mgr.register_check("c", fn=make_fn("c"), priority=5)

        loop = asyncio.new_event_loop()
        try:
            beat_results = loop.run_until_complete(mgr._run_beat())
        finally:
            loop.close()

        # Should have picked 2 checks: b (prio 10) and c (prio 5) since all last_run=0
        self.assertEqual(len(beat_results), 2)
        self.assertIn("b", results)
        self.assertIn("c", results)

    def test_run_beat_async_check(self):
        """_run_beat should handle async check functions."""
        mgr = HeartbeatManager(checks_per_beat=1)

        async def async_fn():
            return "async-result"

        mgr.register_check("async-check", fn=async_fn, priority=1)

        loop = asyncio.new_event_loop()
        try:
            beat_results = loop.run_until_complete(mgr._run_beat())
        finally:
            loop.close()

        self.assertEqual(len(beat_results), 1)
        self.assertEqual(beat_results[0]["result"], "async-result")

    def test_run_beat_updates_last_run(self):
        mgr = HeartbeatManager(checks_per_beat=1)
        mgr.register_check("c", fn=lambda: "ok", priority=1)

        before = time.time()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(mgr._run_beat())
        finally:
            loop.close()
        after = time.time()

        self.assertGreaterEqual(mgr._checks["c"].last_run, before)
        self.assertLessEqual(mgr._checks["c"].last_run, after)

    def test_run_beat_result_handler_called(self):
        mgr = HeartbeatManager(checks_per_beat=1)
        mgr.register_check("c", fn=lambda: "val", priority=1)

        handler = MagicMock()
        mgr.set_result_handler(handler)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(mgr._run_beat())
        finally:
            loop.close()

        handler.assert_called_once()
        args = handler.call_args[0]
        self.assertIsInstance(args[0], list)
        self.assertEqual(args[0][0]["name"], "c")
        self.assertEqual(args[0][0]["result"], "val")

    def test_run_beat_async_result_handler(self):
        """Result handler can be async."""
        mgr = HeartbeatManager(checks_per_beat=1)
        mgr.register_check("c", fn=lambda: "val", priority=1)

        collected = []

        async def async_handler(results):
            collected.extend(results)

        mgr.set_result_handler(async_handler)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(mgr._run_beat())
        finally:
            loop.close()

        self.assertEqual(len(collected), 1)

    def test_run_beat_handles_exception_in_check(self):
        """A failing check should not break the whole beat."""
        mgr = HeartbeatManager(checks_per_beat=2)

        def bad_fn():
            raise RuntimeError("boom")

        mgr.register_check("good", fn=lambda: "ok", priority=1)
        mgr.register_check("bad", fn=bad_fn, priority=10)

        loop = asyncio.new_event_loop()
        try:
            beat_results = loop.run_until_complete(mgr._run_beat())
        finally:
            loop.close()

        # Both should appear in results; the bad one has an error
        self.assertEqual(len(beat_results), 2)
        names = {r["name"] for r in beat_results}
        self.assertIn("good", names)
        self.assertIn("bad", names)
        bad_result = next(r for r in beat_results if r["name"] == "bad")
        self.assertIn("error", bad_result)

    def test_run_beat_empty(self):
        """_run_beat with no checks registered returns empty list."""
        mgr = HeartbeatManager(checks_per_beat=3)
        loop = asyncio.new_event_loop()
        try:
            beat_results = loop.run_until_complete(mgr._run_beat())
        finally:
            loop.close()
        self.assertEqual(beat_results, [])

    def test_start_stop(self):
        """Start and stop should not raise even when called quickly."""
        mgr = HeartbeatManager(interval_minutes=60)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(mgr.start())
            self.assertTrue(mgr._running)
            loop.run_until_complete(mgr.stop())
            self.assertFalse(mgr._running)
        finally:
            loop.close()

    def test_heartbeat_jitter_range(self):
        """The jitter multiplier should be between 0.8 and 1.2."""
        mgr = HeartbeatManager(interval_minutes=10)
        for _ in range(100):
            jitter = mgr._jitter()
            self.assertGreaterEqual(jitter, 0.8)
            self.assertLessEqual(jitter, 1.2)


# ---------------------------------------------------------------------------
# Integration: CronService + HeartbeatManager together
# ---------------------------------------------------------------------------
class TestSchedulingIntegration(unittest.TestCase):
    """Lightweight integration tests."""

    def test_cron_service_save_load_multiple_jobs(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            svc = CronService(persistence_path=path)
            for i in range(5):
                svc.add_job(
                    CronJob(
                        job_id=f"job-{i}",
                        schedule="*/10 * * * *",
                        prompt=f"task {i}",
                        channel="ch",
                        isolated=(i % 2 == 0),
                    )
                )
            svc.save()

            svc2 = CronService(persistence_path=path)
            svc2.load()
            self.assertEqual(len(svc2.list_jobs()), 5)
            self.assertTrue(svc2.get_job("job-0").isolated)
            self.assertFalse(svc2.get_job("job-1").isolated)
        finally:
            os.unlink(path)

    def test_heartbeat_priority_rotation(self):
        """After a beat, previously-run checks should be deprioritised."""
        mgr = HeartbeatManager(checks_per_beat=1)
        mgr.register_check("a", fn=lambda: "a", priority=5)
        mgr.register_check("b", fn=lambda: "b", priority=5)

        loop = asyncio.new_event_loop()
        try:
            r1 = loop.run_until_complete(mgr._run_beat())
            # One of them ran; the other should run next time
            first_name = r1[0]["name"]

            r2 = loop.run_until_complete(mgr._run_beat())
            second_name = r2[0]["name"]

            self.assertNotEqual(first_name, second_name)
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
