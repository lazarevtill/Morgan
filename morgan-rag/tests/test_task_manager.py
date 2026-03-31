"""Tests for the task management system."""

from morgan.task_manager.types import TaskType, TaskStatus, TaskState
from morgan.task_manager.progress import ToolActivity, ProgressTracker
from morgan.task_manager.manager import TaskManager


# --- TaskState tests ---

class TestTaskState:
    def test_default_values(self):
        state = TaskState()
        assert len(state.task_id) == 12
        assert state.task_type == TaskType.AGENT
        assert state.status == TaskStatus.PENDING
        assert state.description == ""
        assert state.result is None
        assert state.error is None
        assert state.is_backgrounded is False
        assert state.metadata == {}

    def test_unique_ids(self):
        ids = {TaskState().task_id for _ in range(50)}
        assert len(ids) == 50

    def test_custom_values(self):
        state = TaskState(
            task_type=TaskType.CRON,
            description="nightly backup",
            is_backgrounded=True,
            metadata={"tag": "ops"},
        )
        assert state.task_type == TaskType.CRON
        assert state.description == "nightly backup"
        assert state.is_backgrounded is True
        assert state.metadata == {"tag": "ops"}


# --- ProgressTracker tests ---

class TestProgressTracker:
    def test_record_activity(self):
        tracker = ProgressTracker()
        tracker.record_activity(ToolActivity("grep", "pattern", "search"))
        assert tracker.tool_use_count == 1
        assert len(tracker.recent_activities) == 1
        assert tracker.recent_activities[0].tool_name == "grep"

    def test_cap_at_20(self):
        tracker = ProgressTracker()
        for i in range(30):
            tracker.record_activity(ToolActivity(f"tool_{i}"))
        assert len(tracker.recent_activities) == 20
        assert tracker.recent_activities[0].tool_name == "tool_10"
        assert tracker.recent_activities[-1].tool_name == "tool_29"
        assert tracker.tool_use_count == 30

    def test_add_tokens(self):
        tracker = ProgressTracker()
        tracker.add_tokens(100)
        tracker.add_tokens(250)
        assert tracker.cumulative_tokens == 350


# --- TaskManager tests ---

class TestTaskManager:
    def test_create_and_get(self):
        mgr = TaskManager()
        tid = mgr.create_task(TaskType.SHELL, "run ls")
        task = mgr.get_task(tid)
        assert task is not None
        assert task.task_type == TaskType.SHELL
        assert task.description == "run ls"
        assert task.status == TaskStatus.PENDING

    def test_get_missing(self):
        mgr = TaskManager()
        assert mgr.get_task("nonexistent") is None

    def test_update_status(self):
        mgr = TaskManager()
        tid = mgr.create_task(TaskType.AGENT, "think")
        assert mgr.update_status(tid, TaskStatus.IN_PROGRESS)
        assert mgr.get_task(tid).status == TaskStatus.IN_PROGRESS

    def test_update_with_result_and_error(self):
        mgr = TaskManager()
        tid = mgr.create_task(TaskType.DREAM, "dream")
        mgr.update_status(tid, TaskStatus.FAILED, error="timeout")
        task = mgr.get_task(tid)
        assert task.status == TaskStatus.FAILED
        assert task.error == "timeout"

    def test_update_missing_returns_false(self):
        mgr = TaskManager()
        assert mgr.update_status("nope", TaskStatus.COMPLETED) is False

    def test_list_tasks_all(self):
        mgr = TaskManager()
        mgr.create_task(TaskType.AGENT, "a")
        mgr.create_task(TaskType.SHELL, "b")
        assert len(mgr.list_tasks()) == 2

    def test_list_tasks_filtered(self):
        mgr = TaskManager()
        tid = mgr.create_task(TaskType.AGENT, "a")
        mgr.create_task(TaskType.SHELL, "b")
        mgr.update_status(tid, TaskStatus.COMPLETED)
        completed = mgr.list_tasks(status=TaskStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].task_id == tid

    def test_delete_task(self):
        mgr = TaskManager()
        tid = mgr.create_task(TaskType.CRON, "hourly")
        assert mgr.delete_task(tid) is True
        assert mgr.get_task(tid) is None

    def test_delete_missing_returns_false(self):
        mgr = TaskManager()
        assert mgr.delete_task("nope") is False

    def test_create_with_metadata(self):
        mgr = TaskManager()
        tid = mgr.create_task(TaskType.AGENT, "meta", metadata={"priority": "high"})
        assert mgr.get_task(tid).metadata == {"priority": "high"}
