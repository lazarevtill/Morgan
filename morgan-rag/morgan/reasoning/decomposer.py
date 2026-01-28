"""
Problem Decomposer for Morgan's Reasoning Engine.

Breaks down complex problems into manageable sub-problems using:
- Hierarchical decomposition
- Dependency analysis
- Priority assessment
- Complexity estimation

Follows KISS principles with clear, focused decomposition strategies.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from morgan.services.llm import LLMService, get_llm_service
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class DecompositionStrategy(str, Enum):
    """Different strategies for problem decomposition."""
    
    HIERARCHICAL = "hierarchical"  # Break into levels/layers
    SEQUENTIAL = "sequential"      # Break into time-ordered steps
    FUNCTIONAL = "functional"      # Break by function/purpose
    COMPONENT = "component"        # Break by system components
    STAKEHOLDER = "stakeholder"    # Break by different perspectives


class ProblemType(str, Enum):
    """Types of problems that can be decomposed."""
    
    ANALYTICAL = "analytical"      # Analysis and evaluation
    DESIGN = "design"             # Creating something new
    PROCEDURAL = "procedural"     # Step-by-step processes
    DIAGNOSTIC = "diagnostic"     # Finding root causes
    OPTIMIZATION = "optimization" # Improving existing systems


@dataclass
class SubProblem:
    """A sub-problem within a larger problem decomposition."""
    
    id: str
    title: str
    description: str
    priority: int  # 1-5, where 5 is highest priority
    complexity: float  # 0.0-1.0, where 1.0 is most complex
    estimated_effort: str  # "low", "medium", "high"
    dependencies: List[str] = field(default_factory=list)  # IDs of prerequisite sub-problems
    tags: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    resources_needed: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "complexity": self.complexity,
            "estimated_effort": self.estimated_effort,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "success_criteria": self.success_criteria,
            "resources_needed": self.resources_needed
        }


@dataclass
class ProblemDecomposition:
    """Complete decomposition of a complex problem."""
    
    decomposition_id: str
    original_problem: str
    strategy: DecompositionStrategy
    problem_type: ProblemType
    sub_problems: List[SubProblem]
    execution_order: List[str]  # Ordered list of sub-problem IDs
    total_complexity: float
    estimated_duration: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decomposition_id": self.decomposition_id,
            "original_problem": self.original_problem,
            "strategy": self.strategy.value,
            "problem_type": self.problem_type.value,
            "sub_problems": [sp.to_dict() for sp in self.sub_problems],
            "execution_order": self.execution_order,
            "total_complexity": self.total_complexity,
            "estimated_duration": self.estimated_duration,
            "metadata": self.metadata
        }


class ProblemDecomposer:
    """
    Decomposes complex problems into manageable sub-problems.
    
    Uses AI-assisted analysis to:
    1. Identify the problem type and best decomposition strategy
    2. Break down the problem into logical sub-components
    3. Analyze dependencies between sub-problems
    4. Prioritize and order sub-problems for execution
    5. Estimate complexity and effort for each part
    
    Example:
        >>> decomposer = ProblemDecomposer()
        >>> result = await decomposer.decompose(
        ...     "Design a scalable microservices architecture for e-commerce"
        ... )
        >>> for sub_problem in result.sub_problems:
        ...     print(f"{sub_problem.title}: {sub_problem.description}")
    """
    
    # Prompt templates for different decomposition phases
    ANALYZE_PROBLEM_PROMPT = """Analyze this problem to determine the best decomposition approach.

Problem: {problem}

Consider:
1. What type of problem is this? (analytical, design, procedural, diagnostic, optimization)
2. What decomposition strategy would work best? (hierarchical, sequential, functional, component, stakeholder)
3. How complex is this problem overall? (0.0-1.0 scale)
4. What are the main aspects or dimensions to consider?

Respond in JSON format:
{{
    "problem_type": "analytical|design|procedural|diagnostic|optimization",
    "recommended_strategy": "hierarchical|sequential|functional|component|stakeholder",
    "overall_complexity": 0.0-1.0,
    "main_aspects": ["aspect1", "aspect2", ...],
    "reasoning": "Brief explanation of your analysis"
}}"""

    DECOMPOSE_PROBLEM_PROMPT = """Break down this problem into 3-7 manageable sub-problems.

Original Problem: {problem}
Problem Type: {problem_type}
Strategy: {strategy}
Main Aspects: {aspects}

For each sub-problem, provide:
- A clear, specific title
- Detailed description of what needs to be done
- Priority level (1-5, where 5 is highest)
- Complexity estimate (0.0-1.0)
- Effort estimate (low/medium/high)
- Success criteria
- Required resources or expertise

Respond in JSON format:
{{
    "sub_problems": [
        {{
            "title": "Clear, specific title",
            "description": "Detailed description of what needs to be done",
            "priority": 1-5,
            "complexity": 0.0-1.0,
            "estimated_effort": "low|medium|high",
            "success_criteria": ["criterion1", "criterion2"],
            "resources_needed": ["resource1", "resource2"],
            "tags": ["tag1", "tag2"]
        }},
        ...
    ]
}}"""

    ANALYZE_DEPENDENCIES_PROMPT = """Analyze the dependencies between these sub-problems.

Sub-problems:
{sub_problems_summary}

For each sub-problem, identify:
1. Which other sub-problems must be completed before it can start
2. Which sub-problems it blocks (that depend on it)
3. Which sub-problems can be done in parallel

Respond in JSON format:
{{
    "dependencies": {{
        "sub_problem_id_1": ["prerequisite_id_1", "prerequisite_id_2"],
        "sub_problem_id_2": [],
        ...
    }},
    "execution_order": ["id1", "id2", "id3", ...],
    "parallel_groups": [["id1", "id2"], ["id3", "id4"], ...]
}}"""

    def __init__(self, llm_service: Optional[LLMService] = None):
        """Initialize the problem decomposer."""
        self.llm = llm_service or get_llm_service()
        logger.info("ProblemDecomposer initialized")

    async def decompose(
        self,
        problem: str,
        strategy: Optional[DecompositionStrategy] = None,
        max_sub_problems: int = 7
    ) -> ProblemDecomposition:
        """
        Decompose a complex problem into manageable sub-problems.
        
        Args:
            problem: The complex problem to decompose
            strategy: Optional specific decomposition strategy to use
            max_sub_problems: Maximum number of sub-problems to create
            
        Returns:
            Complete problem decomposition with sub-problems and execution plan
        """
        decomposition_id = str(uuid.uuid4())
        logger.info(f"Starting problem decomposition: {decomposition_id}")
        logger.debug(f"Problem: {problem[:100]}...")
        
        try:
            # Step 1: Analyze the problem to determine approach
            analysis = await self._analyze_problem(problem)
            
            # Use provided strategy or the recommended one
            chosen_strategy = strategy or DecompositionStrategy(analysis.get("recommended_strategy", "hierarchical"))
            problem_type = ProblemType(analysis.get("problem_type", "analytical"))
            
            logger.debug(f"Analysis: strategy={chosen_strategy}, type={problem_type}")
            
            # Step 2: Decompose into sub-problems
            sub_problems = await self._decompose_into_sub_problems(
                problem=problem,
                strategy=chosen_strategy,
                problem_type=problem_type,
                aspects=analysis.get("main_aspects", []),
                max_count=max_sub_problems
            )
            
            logger.debug(f"Created {len(sub_problems)} sub-problems")
            
            # Step 3: Analyze dependencies and create execution order
            dependencies, execution_order = await self._analyze_dependencies(sub_problems)
            
            # Update sub-problems with dependencies
            for sub_problem in sub_problems:
                sub_problem.dependencies = dependencies.get(sub_problem.id, [])
            
            # Step 4: Calculate overall metrics
            total_complexity = self._calculate_total_complexity(sub_problems)
            estimated_duration = self._estimate_duration(sub_problems)
            
            decomposition = ProblemDecomposition(
                decomposition_id=decomposition_id,
                original_problem=problem,
                strategy=chosen_strategy,
                problem_type=problem_type,
                sub_problems=sub_problems,
                execution_order=execution_order,
                total_complexity=total_complexity,
                estimated_duration=estimated_duration,
                metadata={
                    "analysis": analysis,
                    "max_sub_problems": max_sub_problems
                }
            )
            
            logger.info(
                f"Decomposition completed: {len(sub_problems)} sub-problems, "
                f"complexity: {total_complexity:.2f}"
            )
            
            return decomposition
            
        except Exception as e:
            logger.error(f"Problem decomposition failed: {e}")
            # Return a simple fallback decomposition
            return self._create_fallback_decomposition(problem, str(e))

    async def _analyze_problem(self, problem: str) -> Dict[str, Any]:
        """Analyze the problem to determine the best decomposition approach."""
        try:
            prompt = self.ANALYZE_PROBLEM_PROMPT.format(problem=problem)
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=600
            )
            
            return self._parse_json_response(response.content)
            
        except Exception as e:
            logger.warning(f"Problem analysis failed: {e}")
            return {
                "problem_type": "analytical",
                "recommended_strategy": "hierarchical",
                "overall_complexity": 0.7,
                "main_aspects": ["analysis", "implementation", "validation"],
                "reasoning": "Fallback analysis due to error"
            }

    async def _decompose_into_sub_problems(
        self,
        problem: str,
        strategy: DecompositionStrategy,
        problem_type: ProblemType,
        aspects: List[str],
        max_count: int
    ) -> List[SubProblem]:
        """Decompose the problem into specific sub-problems."""
        try:
            prompt = self.DECOMPOSE_PROBLEM_PROMPT.format(
                problem=problem,
                problem_type=problem_type.value,
                strategy=strategy.value,
                aspects=", ".join(aspects) if aspects else "general analysis"
            )
            
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            result = self._parse_json_response(response.content)
            sub_problems_data = result.get("sub_problems", [])
            
            # Convert to SubProblem objects
            sub_problems = []
            for i, sp_data in enumerate(sub_problems_data[:max_count]):
                sub_problem = SubProblem(
                    id=f"sp_{i+1}",
                    title=sp_data.get("title", f"Sub-problem {i+1}"),
                    description=sp_data.get("description", ""),
                    priority=int(sp_data.get("priority", 3)),
                    complexity=float(sp_data.get("complexity", 0.5)),
                    estimated_effort=sp_data.get("estimated_effort", "medium"),
                    success_criteria=sp_data.get("success_criteria", []),
                    resources_needed=sp_data.get("resources_needed", []),
                    tags=sp_data.get("tags", [])
                )
                sub_problems.append(sub_problem)
            
            return sub_problems
            
        except Exception as e:
            logger.warning(f"Sub-problem creation failed: {e}")
            # Return basic fallback sub-problems
            return [
                SubProblem(
                    id="sp_1",
                    title="Analyze Requirements",
                    description="Understand and analyze the problem requirements",
                    priority=5,
                    complexity=0.3,
                    estimated_effort="low"
                ),
                SubProblem(
                    id="sp_2", 
                    title="Design Solution",
                    description="Create a solution design based on requirements",
                    priority=4,
                    complexity=0.7,
                    estimated_effort="high"
                ),
                SubProblem(
                    id="sp_3",
                    title="Implement Solution",
                    description="Execute the designed solution",
                    priority=3,
                    complexity=0.8,
                    estimated_effort="high"
                )
            ]

    async def _analyze_dependencies(
        self, sub_problems: List[SubProblem]
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        """Analyze dependencies between sub-problems."""
        try:
            # Create summary of sub-problems for analysis
            summary = "\n".join([
                f"ID: {sp.id}, Title: {sp.title}, Description: {sp.description[:100]}"
                for sp in sub_problems
            ])
            
            prompt = self.ANALYZE_DEPENDENCIES_PROMPT.format(
                sub_problems_summary=summary
            )
            
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=1000
            )
            
            result = self._parse_json_response(response.content)
            dependencies = result.get("dependencies", {})
            execution_order = result.get("execution_order", [sp.id for sp in sub_problems])
            
            return dependencies, execution_order
            
        except Exception as e:
            logger.warning(f"Dependency analysis failed: {e}")
            # Return simple sequential dependencies
            dependencies = {}
            execution_order = []
            
            for i, sp in enumerate(sub_problems):
                if i > 0:
                    dependencies[sp.id] = [sub_problems[i-1].id]
                else:
                    dependencies[sp.id] = []
                execution_order.append(sp.id)
            
            return dependencies, execution_order

    def _calculate_total_complexity(self, sub_problems: List[SubProblem]) -> float:
        """Calculate total complexity score for the decomposition."""
        if not sub_problems:
            return 0.0
        
        # Weight by priority and sum complexities
        total_weighted = sum(sp.complexity * (sp.priority / 5.0) for sp in sub_problems)
        return round(total_weighted / len(sub_problems), 2)

    def _estimate_duration(self, sub_problems: List[SubProblem]) -> str:
        """Estimate overall duration based on sub-problems."""
        if not sub_problems:
            return "unknown"
        
        effort_counts = {"low": 0, "medium": 0, "high": 0}
        for sp in sub_problems:
            effort_counts[sp.estimated_effort] += 1
        
        # Simple heuristic based on effort distribution
        if effort_counts["high"] >= 3:
            return "weeks to months"
        elif effort_counts["high"] >= 1 or effort_counts["medium"] >= 4:
            return "days to weeks"
        else:
            return "hours to days"

    def _create_fallback_decomposition(self, problem: str, error: str) -> ProblemDecomposition:
        """Create a basic fallback decomposition when AI analysis fails."""
        sub_problems = [
            SubProblem(
                id="fallback_1",
                title="Problem Analysis",
                description=f"Analyze the problem: {problem[:100]}...",
                priority=5,
                complexity=0.5,
                estimated_effort="medium"
            ),
            SubProblem(
                id="fallback_2",
                title="Solution Development", 
                description="Develop a solution approach",
                priority=4,
                complexity=0.7,
                estimated_effort="high",
                dependencies=["fallback_1"]
            ),
            SubProblem(
                id="fallback_3",
                title="Implementation",
                description="Implement the solution",
                priority=3,
                complexity=0.6,
                estimated_effort="high",
                dependencies=["fallback_2"]
            )
        ]
        
        return ProblemDecomposition(
            decomposition_id=str(uuid.uuid4()),
            original_problem=problem,
            strategy=DecompositionStrategy.SEQUENTIAL,
            problem_type=ProblemType.ANALYTICAL,
            sub_problems=sub_problems,
            execution_order=["fallback_1", "fallback_2", "fallback_3"],
            total_complexity=0.6,
            estimated_duration="days to weeks",
            metadata={"error": error, "fallback": True}
        )

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling common issues."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        import re
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
        
        logger.warning("Failed to parse JSON response")
        return {}

    def get_execution_plan(self, decomposition: ProblemDecomposition) -> List[List[str]]:
        """
        Get an execution plan showing which sub-problems can be done in parallel.
        
        Returns:
            List of lists, where each inner list contains sub-problem IDs that can be executed in parallel
        """
        # Build dependency graph
        dependencies = {sp.id: sp.dependencies for sp in decomposition.sub_problems}
        completed = set()
        execution_plan = []
        
        while len(completed) < len(decomposition.sub_problems):
            # Find sub-problems that can be started (all dependencies completed)
            ready = []
            for sp in decomposition.sub_problems:
                if sp.id not in completed and all(dep in completed for dep in sp.dependencies):
                    ready.append(sp.id)
            
            if not ready:
                # Circular dependency or other issue - add remaining items
                remaining = [sp.id for sp in decomposition.sub_problems if sp.id not in completed]
                execution_plan.append(remaining)
                break
            
            execution_plan.append(ready)
            completed.update(ready)
        
        return execution_plan


# Singleton instance
_decomposer: Optional[ProblemDecomposer] = None


def get_problem_decomposer() -> ProblemDecomposer:
    """Get singleton problem decomposer instance."""
    global _decomposer
    if _decomposer is None:
        _decomposer = ProblemDecomposer()
    return _decomposer