"""
Task Anticipator for Proactive Assistance.

Anticipates user needs and prepares responses proactively:
- Predicts likely next questions
- Pre-computes common follow-ups
- Prepares context for anticipated tasks
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from morgan.config import get_settings
from morgan.services.llm_service import LLMService, get_llm_service
from morgan.proactive.monitor import ContextMonitor, UserContext, get_context_monitor
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class AnticipationType(str, Enum):
    """Types of anticipated tasks."""

    FOLLOW_UP_QUESTION = "follow_up_question"
    RELATED_TOPIC = "related_topic"
    CLARIFICATION = "clarification"
    DEEPER_DIVE = "deeper_dive"
    ALTERNATIVE_APPROACH = "alternative_approach"
    NEXT_STEP = "next_step"


@dataclass
class AnticipatedTask:
    """A task anticipated based on user context."""

    task_id: str
    anticipation_type: AnticipationType
    question: str  # Anticipated question
    probability: float  # 0.0 to 1.0
    prepared_response: Optional[str] = None  # Pre-computed response
    context_prepared: bool = False
    user_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "type": self.anticipation_type.value,
            "question": self.question,
            "probability": self.probability,
            "has_prepared_response": self.prepared_response is not None,
        }


class TaskAnticipator:
    """
    Anticipates user tasks and pre-computes responses.

    Capabilities:
    - Predict likely follow-up questions
    - Pre-fetch relevant context
    - Prepare responses for common follow-ups
    - Cache anticipated results

    Example:
        >>> anticipator = TaskAnticipator()
        >>>
        >>> # After a user query, anticipate next tasks
        >>> anticipated = await anticipator.anticipate(
        ...     user_id="user123",
        ...     current_query="How do I create a Docker container?",
        ...     response="To create a Docker container..."
        ... )
        >>>
        >>> for task in anticipated:
        ...     print(f"Likely question: {task.question} ({task.probability:.0%})")
        >>>
        >>> # Check if a query matches anticipated task
        >>> match = await anticipator.check_match(
        ...     user_id="user123",
        ...     query="How do I list running containers?"
        ... )
        >>> if match:
        ...     print(f"Using pre-computed response!")
    """

    ANTICIPATION_PROMPT = """Based on this conversation, predict what the user might ask next.

User's question: {query}
Your response: {response}
User's emotional state: {emotional_state}
Previous topics: {previous_topics}

Generate 3-5 likely follow-up questions with probability scores.

Respond in JSON format:
{{
    "anticipated": [
        {{
            "type": "follow_up_question|related_topic|clarification|deeper_dive|alternative_approach|next_step",
            "question": "The predicted question",
            "probability": 0.0-1.0,
            "reasoning": "Why this is likely"
        }},
        ...
    ]
}}"""

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        context_monitor: Optional[ContextMonitor] = None,
        max_anticipated: int = 5,
        min_probability: float = 0.3,
        auto_prepare_responses: bool = True,
    ):
        """
        Initialize task anticipator.

        Args:
            llm_service: LLM for generating anticipations
            context_monitor: Context monitor for user data
            max_anticipated: Maximum tasks to anticipate
            min_probability: Minimum probability threshold
            auto_prepare_responses: Auto-prepare responses for high-probability tasks
        """
        self.settings = get_settings()
        self.llm = llm_service or get_llm_service()
        self.monitor = context_monitor or get_context_monitor()
        self.max_anticipated = max_anticipated
        self.min_probability = min_probability
        self.auto_prepare = auto_prepare_responses

        # Anticipated tasks by user
        self._anticipated: Dict[str, List[AnticipatedTask]] = {}

        # Preparation queue
        self._preparation_queue: asyncio.Queue = asyncio.Queue()
        self._preparation_task: Optional[asyncio.Task] = None

        logger.info(
            f"TaskAnticipator initialized: max={max_anticipated}, "
            f"min_prob={min_probability}"
        )

    async def anticipate(
        self,
        user_id: str,
        current_query: str,
        response: str,
    ) -> List[AnticipatedTask]:
        """
        Anticipate likely next tasks based on current conversation.

        Args:
            user_id: User identifier
            current_query: User's current query
            response: Morgan's response

        Returns:
            List of anticipated tasks
        """
        logger.debug(f"Anticipating tasks for {user_id}")

        # Get context
        context = await self.monitor.get_context(user_id)

        try:
            # Generate anticipations
            prompt = self.ANTICIPATION_PROMPT.format(
                query=current_query,
                response=response[:500],  # Truncate long responses
                emotional_state=context.emotional_state if context else "Unknown",
                previous_topics=(
                    ", ".join(context.recent_queries[-3:]) if context else "None"
                ),
            )

            llm_response = self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=800,
            )

            # Parse response
            import json
            import re

            content = llm_response.content
            json_match = re.search(r"\{[\s\S]*\}", content)

            anticipated = []
            if json_match:
                data = json.loads(json_match.group(0))

                for item in data.get("anticipated", []):
                    probability = float(item.get("probability", 0.5))

                    if probability < self.min_probability:
                        continue

                    task = AnticipatedTask(
                        task_id=str(uuid.uuid4()),
                        anticipation_type=AnticipationType(
                            item.get("type", "follow_up_question")
                        ),
                        question=item.get("question", ""),
                        probability=probability,
                        user_id=user_id,
                        metadata={
                            "reasoning": item.get("reasoning", ""),
                            "source_query": current_query,
                        },
                    )
                    anticipated.append(task)

            # Sort by probability
            anticipated.sort(key=lambda t: t.probability, reverse=True)
            anticipated = anticipated[: self.max_anticipated]

            # Store
            self._anticipated[user_id] = anticipated

            # Queue high-probability tasks for response preparation
            if self.auto_prepare:
                for task in anticipated:
                    if task.probability >= 0.6:
                        await self._preparation_queue.put(task)

            logger.info(f"Anticipated {len(anticipated)} tasks for {user_id}")

            return anticipated

        except Exception as e:
            logger.warning(f"Anticipation failed: {e}")
            return []

    async def check_match(
        self,
        user_id: str,
        query: str,
        similarity_threshold: float = 0.7,
    ) -> Optional[AnticipatedTask]:
        """
        Check if a query matches an anticipated task.

        Args:
            user_id: User identifier
            query: User's query
            similarity_threshold: Minimum similarity for a match

        Returns:
            Matching anticipated task or None
        """
        anticipated = self._anticipated.get(user_id, [])

        if not anticipated:
            return None

        query_lower = query.lower()

        for task in anticipated:
            # Simple similarity check (could use embeddings for better matching)
            task_lower = task.question.lower()

            # Calculate Jaccard similarity
            query_words = set(query_lower.split())
            task_words = set(task_lower.split())

            if not query_words or not task_words:
                continue

            intersection = len(query_words & task_words)
            union = len(query_words | task_words)
            similarity = intersection / union

            if similarity >= similarity_threshold:
                logger.info(
                    f"Matched anticipated task: {task.task_id} "
                    f"(similarity: {similarity:.2f})"
                )
                return task

        return None

    async def get_prepared_response(
        self,
        task: AnticipatedTask,
    ) -> Optional[str]:
        """
        Get prepared response for an anticipated task.

        Args:
            task: The anticipated task

        Returns:
            Prepared response or None
        """
        if task.prepared_response:
            return task.prepared_response

        # Try to prepare now
        await self._prepare_response(task)
        return task.prepared_response

    async def _prepare_response(self, task: AnticipatedTask):
        """Prepare response for an anticipated task."""
        if task.prepared_response:
            return

        try:
            prompt = f"""Answer this anticipated question:

Question: {task.question}

Context: This is a follow-up to a previous conversation.

Provide a helpful, concise response."""

            response = self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1000,
            )

            task.prepared_response = response.content
            task.context_prepared = True

            logger.debug(f"Prepared response for task {task.task_id}")

        except Exception as e:
            logger.warning(f"Failed to prepare response: {e}")

    async def start_preparation_worker(self):
        """Start background worker for response preparation."""
        if self._preparation_task is None:
            self._preparation_task = asyncio.create_task(self._preparation_loop())
            logger.info("Started preparation worker")

    async def stop_preparation_worker(self):
        """Stop background preparation worker."""
        if self._preparation_task:
            self._preparation_task.cancel()
            self._preparation_task = None
            logger.info("Stopped preparation worker")

    async def _preparation_loop(self):
        """Background loop for preparing responses."""
        while True:
            try:
                task = await asyncio.wait_for(
                    self._preparation_queue.get(),
                    timeout=1.0,
                )
                await self._prepare_response(task)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in preparation loop: {e}")

    def get_anticipated(self, user_id: str) -> List[AnticipatedTask]:
        """Get anticipated tasks for a user."""
        return self._anticipated.get(user_id, [])

    def clear_anticipated(self, user_id: str):
        """Clear anticipated tasks for a user."""
        if user_id in self._anticipated:
            del self._anticipated[user_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get anticipator statistics."""
        total_tasks = sum(len(tasks) for tasks in self._anticipated.values())
        prepared_count = sum(
            1
            for tasks in self._anticipated.values()
            for task in tasks
            if task.prepared_response
        )

        return {
            "total_users": len(self._anticipated),
            "total_anticipated": total_tasks,
            "prepared_responses": prepared_count,
            "preparation_rate": prepared_count / total_tasks if total_tasks > 0 else 0,
        }


# Singleton instance
_anticipator: Optional[TaskAnticipator] = None


def get_task_anticipator() -> TaskAnticipator:
    """Get singleton task anticipator instance."""
    global _anticipator
    if _anticipator is None:
        _anticipator = TaskAnticipator()
    return _anticipator
