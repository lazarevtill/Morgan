"""
Multi-Step Reasoning Engine for Morgan.

Implements chain-of-thought reasoning with:
- Problem decomposition
- Step-by-step reasoning
- Self-reflection and correction
- Confidence tracking
- Reasoning explanation generation

Quality over speed - takes time to think through complex problems.
"""

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from morgan.config import get_settings
from morgan.services.llm import LLMService, get_llm_service
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class ReasoningType(str, Enum):
    """Types of reasoning approaches."""

    CHAIN_OF_THOUGHT = "chain_of_thought"  # Step-by-step logical reasoning
    DECOMPOSITION = "decomposition"  # Break down into sub-problems
    ANALOGY = "analogy"  # Reasoning by comparison
    HYPOTHESIS = "hypothesis"  # Form and test hypotheses
    REFLECTION = "reflection"  # Self-check and correct


class ReasoningStatus(str, Enum):
    """Status of reasoning process."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_CLARIFICATION = "needs_clarification"


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""

    step_id: str
    step_number: int
    reasoning_type: ReasoningType
    thought: str
    conclusion: Optional[str] = None
    confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    sub_steps: List["ReasoningStep"] = field(default_factory=list)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReasoningResult:
    """Complete result of a reasoning process."""

    reasoning_id: str
    query: str
    final_answer: str
    steps: List[ReasoningStep]
    overall_confidence: float
    reasoning_summary: str
    total_duration_ms: float
    status: ReasoningStatus
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reasoning_id": self.reasoning_id,
            "query": self.query,
            "final_answer": self.final_answer,
            "steps": [
                {
                    "step_number": s.step_number,
                    "type": s.reasoning_type.value,
                    "thought": s.thought,
                    "conclusion": s.conclusion,
                    "confidence": s.confidence,
                }
                for s in self.steps
            ],
            "overall_confidence": self.overall_confidence,
            "reasoning_summary": self.reasoning_summary,
            "total_duration_ms": self.total_duration_ms,
            "status": self.status.value,
        }


class ReasoningEngine:
    """
    Multi-step reasoning engine for complex problem solving.

    Implements chain-of-thought reasoning that:
    1. Analyzes the problem to understand its nature
    2. Breaks down complex problems into steps
    3. Reasons through each step carefully
    4. Synthesizes conclusions
    5. Self-reflects and validates

    Quality target: 5-10 seconds for complex reasoning (acceptable).

    Example:
        >>> engine = ReasoningEngine()
        >>> result = await engine.reason(
        ...     "How should I design a microservices architecture for an e-commerce platform?"
        ... )
        >>> print(result.reasoning_summary)
        >>> for step in result.steps:
        ...     print(f"Step {step.step_number}: {step.thought}")
    """

    # Prompt templates for different reasoning phases
    ANALYZE_PROMPT = """You are a reasoning assistant. Analyze this problem and determine:
1. What type of problem is this? (factual, analytical, creative, procedural, etc.)
2. How complex is it? (simple, moderate, complex)
3. What are the key components or aspects to consider?
4. What approach would work best?

Problem: {query}

Respond in JSON format:
{{
    "problem_type": "string",
    "complexity": "simple|moderate|complex",
    "key_aspects": ["aspect1", "aspect2", ...],
    "recommended_approach": "string",
    "needs_decomposition": true|false
}}"""

    DECOMPOSE_PROMPT = """Break down this complex problem into smaller, manageable sub-problems.

Problem: {query}

Key aspects to consider: {aspects}

List 3-5 sub-problems that, when solved together, will solve the main problem.

Respond in JSON format:
{{
    "sub_problems": [
        {{"id": 1, "description": "...", "priority": 1-5}},
        ...
    ]
}}"""

    REASON_STEP_PROMPT = """Think through this step carefully using chain-of-thought reasoning.

Main Problem: {query}
Current Step: {step_description}
Previous conclusions: {previous_conclusions}
Available context: {context}

Think step by step:
1. What do I know about this?
2. What logical connections can I make?
3. What's the conclusion for this step?

Respond in JSON format:
{{
    "thought_process": "Your detailed reasoning...",
    "conclusion": "Your conclusion for this step",
    "confidence": 0.0-1.0,
    "evidence": ["supporting point 1", "supporting point 2"]
}}"""

    SYNTHESIZE_PROMPT = """Synthesize all the reasoning steps into a final answer.

Original Question: {query}

Reasoning Steps:
{steps_summary}

Create a comprehensive, well-structured answer that:
1. Addresses the original question directly
2. Incorporates insights from all reasoning steps
3. Is clear and actionable

Respond in JSON format:
{{
    "final_answer": "Your comprehensive answer...",
    "key_insights": ["insight 1", "insight 2", ...],
    "confidence": 0.0-1.0
}}"""

    REFLECT_PROMPT = """Review and validate your reasoning and answer.

Original Question: {query}
Proposed Answer: {answer}
Reasoning Steps: {steps_summary}

Consider:
1. Is the answer complete and accurate?
2. Are there any logical gaps?
3. Could anything be improved?
4. What's your confidence level?

Respond in JSON format:
{{
    "is_valid": true|false,
    "issues": ["issue 1", ...] or [],
    "improvements": ["improvement 1", ...] or [],
    "final_confidence": 0.0-1.0,
    "revised_answer": "If changes needed, the revised answer, otherwise null"
}}"""

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        max_steps: int = 10,
        min_confidence: float = 0.7,
        enable_reflection: bool = True,
    ):
        """
        Initialize reasoning engine.

        Args:
            llm_service: LLM service for generating reasoning
            max_steps: Maximum reasoning steps
            min_confidence: Minimum confidence threshold
            enable_reflection: Enable self-reflection step
        """
        self.settings = get_settings()
        self.llm = llm_service or get_llm_service()
        self.max_steps = max_steps
        self.min_confidence = min_confidence
        self.enable_reflection = enable_reflection

        logger.info(
            f"ReasoningEngine initialized: max_steps={max_steps}, "
            f"min_confidence={min_confidence}"
        )

    async def reason(
        self,
        query: str,
        context: Optional[str] = None,
        reasoning_type: Optional[ReasoningType] = None,
    ) -> ReasoningResult:
        """
        Perform multi-step reasoning on a query.

        Args:
            query: The question or problem to reason about
            context: Optional additional context
            reasoning_type: Force a specific reasoning type

        Returns:
            Complete reasoning result with steps and conclusions
        """
        reasoning_id = str(uuid.uuid4())
        start_time = time.time()
        steps: List[ReasoningStep] = []

        logger.info(f"Starting reasoning process: {reasoning_id}")
        logger.debug(f"Query: {query[:100]}...")

        try:
            # Step 1: Analyze the problem
            analysis = await self._analyze_problem(query)
            logger.debug(f"Analysis: {analysis}")

            # Step 2: Decompose if complex
            sub_problems = []
            if analysis.get("needs_decomposition", False):
                sub_problems = await self._decompose_problem(
                    query, analysis.get("key_aspects", [])
                )
                logger.debug(f"Decomposed into {len(sub_problems)} sub-problems")

            # Step 3: Reason through each part
            if sub_problems:
                # Complex problem: reason through sub-problems
                for i, sub_problem in enumerate(sub_problems):
                    step = await self._reason_step(
                        query=query,
                        step_description=sub_problem["description"],
                        previous_conclusions=[
                            s.conclusion for s in steps if s.conclusion
                        ],
                        context=context,
                        step_number=i + 1,
                    )
                    steps.append(step)

                    # Check if we have enough steps
                    if len(steps) >= self.max_steps:
                        break
            else:
                # Simpler problem: direct reasoning
                step = await self._reason_step(
                    query=query,
                    step_description="Analyze and answer the question directly",
                    previous_conclusions=[],
                    context=context,
                    step_number=1,
                )
                steps.append(step)

            # Step 4: Synthesize final answer
            final_answer, confidence = await self._synthesize_answer(query, steps)

            # Step 5: Self-reflection (optional)
            if self.enable_reflection:
                final_answer, confidence = await self._reflect_and_validate(
                    query, final_answer, steps
                )

            # Calculate overall metrics
            total_duration = (time.time() - start_time) * 1000
            overall_confidence = self._calculate_overall_confidence(steps, confidence)

            # Generate reasoning summary
            reasoning_summary = self._generate_summary(steps)

            result = ReasoningResult(
                reasoning_id=reasoning_id,
                query=query,
                final_answer=final_answer,
                steps=steps,
                overall_confidence=overall_confidence,
                reasoning_summary=reasoning_summary,
                total_duration_ms=total_duration,
                status=ReasoningStatus.COMPLETED,
                metadata={
                    "analysis": analysis,
                    "sub_problems_count": len(sub_problems),
                    "reflection_enabled": self.enable_reflection,
                },
            )

            logger.info(
                f"Reasoning completed: {reasoning_id} in {total_duration:.0f}ms, "
                f"confidence: {overall_confidence:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return ReasoningResult(
                reasoning_id=reasoning_id,
                query=query,
                final_answer=f"I encountered an error while reasoning: {str(e)}",
                steps=steps,
                overall_confidence=0.0,
                reasoning_summary="Reasoning process failed",
                total_duration_ms=(time.time() - start_time) * 1000,
                status=ReasoningStatus.FAILED,
                metadata={"error": str(e)},
            )

    async def _analyze_problem(self, query: str) -> Dict[str, Any]:
        """Analyze the problem to determine approach."""
        try:
            prompt = self.ANALYZE_PROMPT.format(query=query)
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.3,  # Low temperature for analysis
                max_tokens=500,
            )

            return self._parse_json_response(response.content)
        except Exception as e:
            logger.warning(f"Problem analysis failed: {e}")
            return {
                "problem_type": "unknown",
                "complexity": "moderate",
                "key_aspects": [],
                "recommended_approach": "chain_of_thought",
                "needs_decomposition": False,
            }

    async def _decompose_problem(
        self, query: str, aspects: List[str]
    ) -> List[Dict[str, Any]]:
        """Decompose complex problem into sub-problems."""
        try:
            prompt = self.DECOMPOSE_PROMPT.format(
                query=query,
                aspects=", ".join(aspects) if aspects else "general analysis",
            )
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=800,
            )

            result = self._parse_json_response(response.content)
            return result.get("sub_problems", [])
        except Exception as e:
            logger.warning(f"Problem decomposition failed: {e}")
            return []

    async def _reason_step(
        self,
        query: str,
        step_description: str,
        previous_conclusions: List[str],
        context: Optional[str],
        step_number: int,
    ) -> ReasoningStep:
        """Reason through a single step."""
        step_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            prompt = self.REASON_STEP_PROMPT.format(
                query=query,
                step_description=step_description,
                previous_conclusions=(
                    "\n".join(f"- {c}" for c in previous_conclusions)
                    if previous_conclusions
                    else "None yet"
                ),
                context=context or "No additional context provided",
            )

            response = self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1000,
            )

            result = self._parse_json_response(response.content)

            return ReasoningStep(
                step_id=step_id,
                step_number=step_number,
                reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
                thought=result.get("thought_process", step_description),
                conclusion=result.get("conclusion"),
                confidence=float(result.get("confidence", 0.7)),
                supporting_evidence=result.get("evidence", []),
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.warning(f"Reasoning step failed: {e}")
            return ReasoningStep(
                step_id=step_id,
                step_number=step_number,
                reasoning_type=ReasoningType.CHAIN_OF_THOUGHT,
                thought=f"Error during reasoning: {e}",
                confidence=0.3,
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def _synthesize_answer(
        self, query: str, steps: List[ReasoningStep]
    ) -> Tuple[str, float]:
        """Synthesize final answer from reasoning steps."""
        try:
            steps_summary = "\n".join(
                f"Step {s.step_number}: {s.thought}\nConclusion: {s.conclusion}"
                for s in steps
            )

            prompt = self.SYNTHESIZE_PROMPT.format(
                query=query,
                steps_summary=steps_summary,
            )

            response = self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=2000,
            )

            result = self._parse_json_response(response.content)

            return (
                result.get("final_answer", "Unable to synthesize answer"),
                float(result.get("confidence", 0.7)),
            )

        except Exception as e:
            logger.warning(f"Answer synthesis failed: {e}")
            # Fallback: combine step conclusions
            conclusions = [s.conclusion for s in steps if s.conclusion]
            return " ".join(conclusions) if conclusions else str(e), 0.5

    async def _reflect_and_validate(
        self, query: str, answer: str, steps: List[ReasoningStep]
    ) -> Tuple[str, float]:
        """Self-reflect and validate the reasoning."""
        try:
            steps_summary = "\n".join(
                f"Step {s.step_number}: {s.conclusion}" for s in steps if s.conclusion
            )

            prompt = self.REFLECT_PROMPT.format(
                query=query,
                answer=answer,
                steps_summary=steps_summary,
            )

            response = self.llm.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=1000,
            )

            result = self._parse_json_response(response.content)

            # Use revised answer if provided
            revised = result.get("revised_answer")
            final_answer = revised if revised else answer
            confidence = float(result.get("final_confidence", 0.7))

            return final_answer, confidence

        except Exception as e:
            logger.warning(f"Reflection failed: {e}")
            return answer, 0.7

    def _calculate_overall_confidence(
        self, steps: List[ReasoningStep], synthesis_confidence: float
    ) -> float:
        """Calculate overall confidence from steps and synthesis."""
        if not steps:
            return synthesis_confidence

        step_confidences = [s.confidence for s in steps]
        avg_step_confidence = sum(step_confidences) / len(step_confidences)

        # Weight synthesis confidence higher
        overall = (synthesis_confidence * 0.6) + (avg_step_confidence * 0.4)
        return round(overall, 2)

    def _generate_summary(self, steps: List[ReasoningStep]) -> str:
        """Generate a human-readable reasoning summary."""
        if not steps:
            return "No reasoning steps were performed."

        summary_parts = [
            f"Reasoning process involved {len(steps)} step(s):",
        ]

        for step in steps:
            summary_parts.append(
                f"- Step {step.step_number}: {step.conclusion or step.thought[:100]}"
            )

        return "\n".join(summary_parts)

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling common issues."""
        try:
            # Try direct parsing
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in response
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Return empty dict if parsing fails
        logger.warning("Failed to parse JSON response")
        return {}


# Singleton instance
_engine: Optional[ReasoningEngine] = None


def get_reasoning_engine() -> ReasoningEngine:
    """Get singleton reasoning engine instance."""
    global _engine
    if _engine is None:
        _engine = ReasoningEngine()
    return _engine
