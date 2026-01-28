"""
Task Planning System for Morgan.

Provides:
- Task decomposition from user requests
- Priority-based task ordering
- Progress tracking
- Dependency management
- Task execution orchestration
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from morgan.config import get_settings
from morgan.services.llm import LLMService, get_llm_service
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    """Status of a task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Waiting on dependencies
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Priority levels for tasks."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskType(str, Enum):
    """Types of tasks."""

    RESEARCH = "research"  # Find information
    ANALYSIS = "analysis"  # Analyze data/content
    GENERATION = "generation"  # Create content
    ACTION = "action"  # Perform an action
    VALIDATION = "validation"  # Validate results
    SYNTHESIS = "synthesis"  # Combine results


@dataclass
class Task:
    """A single task in a plan."""

    task_id: str
    name: str
    description: str
    task_type: TaskType
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "type": self.task_type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "progress": self.progress,
            "result": str(self.result)[:200] if self.result else None,
            "error": self.error,
        }


@dataclass
class TaskPlan:
    """A complete plan with multiple tasks."""

    plan_id: str
    goal: str
    tasks: List[Task]
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        """Calculate overall plan progress."""
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return completed / len(self.tasks)

    @property
    def is_complete(self) -> bool:
        """Check if all tasks are complete."""
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks)

    @property
    def has_failed(self) -> bool:
        """Check if any task has failed."""
        return any(t.status == TaskStatus.FAILED for t in self.tasks)

    def get_next_tasks(self) -> List[Task]:
        """Get tasks ready to execute (pending with satisfied dependencies)."""
        completed_ids = {
            t.task_id for t in self.tasks if t.status == TaskStatus.COMPLETED
        }

        ready = []
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            # Check if all dependencies are satisfied
            if all(dep_id in completed_ids for dep_id in task.dependencies):
                ready.append(task)

        # Sort by priority
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3,
        }
        ready.sort(key=lambda t: priority_order.get(t.priority, 2))

        return ready

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "status": self.status.value,
            "progress": self.progress,
            "tasks": [t.to_dict() for t in self.tasks],
            "is_complete": self.is_complete,
            "has_failed": self.has_failed,
        }


class TaskPlanner:
    """
    Task planning and execution orchestrator.

    Capabilities:
    - Decompose complex goals into tasks
    - Identify dependencies between tasks
    - Execute tasks in optimal order
    - Track progress and handle failures
    - Provide status updates

    Example:
        >>> planner = TaskPlanner()
        >>>
        >>> # Create a plan from a goal
        >>> plan = await planner.create_plan(
        ...     "Research and summarize the latest developments in AI safety"
        ... )
        >>>
        >>> # Execute the plan
        >>> result = await planner.execute_plan(plan)
        >>>
        >>> # Get progress
        >>> print(f"Progress: {plan.progress * 100:.0f}%")
    """

    PLANNING_PROMPT = """You are a task planner. Given a goal, break it down into specific, actionable tasks.

Goal: {goal}

Create a plan with 3-7 tasks. For each task, specify:
- name: Short task name
- description: What needs to be done
- type: research, analysis, generation, action, validation, or synthesis
- priority: critical, high, medium, or low
- dependencies: List of task numbers this depends on (0-indexed)

Respond in JSON format:
{{
    "tasks": [
        {{
            "name": "Task name",
            "description": "Detailed description",
            "type": "research|analysis|generation|action|validation|synthesis",
            "priority": "critical|high|medium|low",
            "depends_on": []
        }},
        ...
    ],
    "reasoning": "Brief explanation of your planning approach"
}}"""

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        max_concurrent_tasks: int = 3,
    ):
        """
        Initialize task planner.

        Args:
            llm_service: LLM service for planning
            max_concurrent_tasks: Maximum tasks to run concurrently
        """
        self.settings = get_settings()
        self.llm = llm_service or get_llm_service()
        self.max_concurrent = max_concurrent_tasks

        # Task executors by type
        self._executors: Dict[TaskType, Callable] = {}

        # Active plans
        self._plans: Dict[str, TaskPlan] = {}

        logger.info(f"TaskPlanner initialized: max_concurrent={max_concurrent_tasks}")

    def register_executor(
        self,
        task_type: TaskType,
        executor: Callable[[Task, Dict[str, Any]], Any],
    ):
        """
        Register an executor function for a task type.

        Args:
            task_type: Type of task this executor handles
            executor: Async function that executes the task
        """
        self._executors[task_type] = executor
        logger.info(f"Registered executor for task type: {task_type.value}")

    async def create_plan(
        self,
        goal: str,
        context: Optional[str] = None,
    ) -> TaskPlan:
        """
        Create a task plan from a goal.

        Args:
            goal: The goal to achieve
            context: Optional additional context

        Returns:
            TaskPlan with decomposed tasks
        """
        plan_id = str(uuid.uuid4())
        logger.info(f"Creating plan {plan_id} for goal: {goal[:100]}")

        try:
            # Use LLM to decompose goal into tasks
            prompt = self.PLANNING_PROMPT.format(goal=goal)
            if context:
                prompt += f"\n\nAdditional context: {context}"

            response = self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1500,
            )

            # Parse response
            import json
            import re

            content = response.content

            # Try to extract JSON
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                raise ValueError("Could not parse planning response")

            # Create tasks
            tasks = []
            task_ids = []

            for i, task_data in enumerate(data.get("tasks", [])):
                task_id = str(uuid.uuid4())
                task_ids.append(task_id)

                # Map dependencies to task IDs
                depends_on = task_data.get("depends_on", [])
                dependencies = [
                    task_ids[dep_idx]
                    for dep_idx in depends_on
                    if dep_idx < len(task_ids)
                ]

                task = Task(
                    task_id=task_id,
                    name=task_data.get("name", f"Task {i + 1}"),
                    description=task_data.get("description", ""),
                    task_type=TaskType(task_data.get("type", "analysis")),
                    priority=TaskPriority(task_data.get("priority", "medium")),
                    dependencies=dependencies,
                )
                tasks.append(task)

            plan = TaskPlan(
                plan_id=plan_id,
                goal=goal,
                tasks=tasks,
                metadata={"reasoning": data.get("reasoning", "")},
            )

            self._plans[plan_id] = plan

            logger.info(f"Created plan {plan_id} with {len(tasks)} tasks")

            return plan

        except Exception as e:
            logger.error(f"Failed to create plan: {e}")
            # Return a minimal plan with a single task
            task = Task(
                task_id=str(uuid.uuid4()),
                name="Complete Goal",
                description=goal,
                task_type=TaskType.GENERATION,
                priority=TaskPriority.HIGH,
            )
            plan = TaskPlan(
                plan_id=plan_id,
                goal=goal,
                tasks=[task],
                metadata={"error": str(e)},
            )
            self._plans[plan_id] = plan
            return plan

    async def execute_plan(
        self,
        plan: TaskPlan,
        on_progress: Optional[Callable[[TaskPlan], None]] = None,
    ) -> TaskPlan:
        """
        Execute all tasks in a plan.

        Args:
            plan: The plan to execute
            on_progress: Optional callback for progress updates

        Returns:
            Updated plan with results
        """
        logger.info(f"Executing plan {plan.plan_id}")
        plan.status = TaskStatus.IN_PROGRESS

        # Collect results from completed tasks for context
        results_context: Dict[str, Any] = {}

        while not plan.is_complete and not plan.has_failed:
            # Get ready tasks
            ready_tasks = plan.get_next_tasks()

            if not ready_tasks:
                # Check if blocked
                pending = [t for t in plan.tasks if t.status == TaskStatus.PENDING]
                if pending:
                    logger.warning(f"Plan blocked: {len(pending)} tasks waiting")
                    for task in pending:
                        task.status = TaskStatus.BLOCKED
                break

            # Execute tasks (respecting concurrency limit)
            batch = ready_tasks[: self.max_concurrent]

            await asyncio.gather(
                *[self._execute_task(task, results_context) for task in batch]
            )

            # Update context with new results
            for task in batch:
                if task.status == TaskStatus.COMPLETED and task.result:
                    results_context[task.task_id] = task.result

            # Progress callback
            if on_progress:
                on_progress(plan)

        # Update plan status
        if plan.is_complete:
            plan.status = TaskStatus.COMPLETED
            plan.completed_at = datetime.now(timezone.utc)
            logger.info(f"Plan {plan.plan_id} completed successfully")
        elif plan.has_failed:
            plan.status = TaskStatus.FAILED
            logger.warning(f"Plan {plan.plan_id} failed")
        else:
            plan.status = TaskStatus.BLOCKED
            logger.warning(f"Plan {plan.plan_id} blocked")

        return plan

    async def _execute_task(
        self,
        task: Task,
        context: Dict[str, Any],
    ):
        """Execute a single task."""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now(timezone.utc)

        logger.info(f"Executing task: {task.name}")

        try:
            # Check for registered executor
            if task.task_type in self._executors:
                executor = self._executors[task.task_type]
                result = await executor(task, context)
            else:
                # Default: use LLM to complete the task
                result = await self._default_executor(task, context)

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.completed_at = datetime.now(timezone.utc)

            logger.info(f"Task completed: {task.name}")

        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            logger.error(f"Task failed: {task.name} - {e}")

    async def _default_executor(
        self,
        task: Task,
        context: Dict[str, Any],
    ) -> str:
        """Default task executor using LLM."""
        # Build context from dependent task results
        context_str = ""
        for dep_id in task.dependencies:
            if dep_id in context:
                context_str += f"\nPrevious result: {context[dep_id]}"

        prompt = f"""Complete this task:

Task: {task.name}
Description: {task.description}
Type: {task.task_type.value}
{context_str}

Provide a thorough and helpful response."""

        response = self.llm.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=2000,
        )

        return response.content

    def get_plan(self, plan_id: str) -> Optional[TaskPlan]:
        """Get a plan by ID."""
        return self._plans.get(plan_id)

    def get_all_plans(self) -> List[TaskPlan]:
        """Get all plans."""
        return list(self._plans.values())


# Singleton instance
_planner: Optional[TaskPlanner] = None


def get_task_planner() -> TaskPlanner:
    """Get singleton task planner instance."""
    global _planner
    if _planner is None:
        _planner = TaskPlanner()
    return _planner
