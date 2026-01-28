"""
Reasoning Explainer for Morgan's Reasoning Engine.

Generates clear, human-readable explanations of reasoning processes:
- Step-by-step reasoning breakdowns
- Visual reasoning flow diagrams
- Confidence and uncertainty explanations
- Alternative reasoning paths
- Assumption impact explanations

Follows KISS principles with clear, accessible explanations.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from morgan.reasoning.chains import ReasoningChain, ChainStep
from morgan.reasoning.decomposer import ProblemDecomposition, SubProblem
from morgan.reasoning.assumptions import AssumptionSet, Assumption
from morgan.reasoning.engine import ReasoningResult, ReasoningStep
from morgan.services.llm import LLMService, get_llm_service
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class ExplanationStyle(str, Enum):
    """Different styles of explanation."""
    
    SIMPLE = "simple"           # Basic, non-technical explanation
    DETAILED = "detailed"       # Comprehensive technical explanation
    VISUAL = "visual"          # Diagram-focused explanation
    CONVERSATIONAL = "conversational"  # Natural, dialogue-style
    ACADEMIC = "academic"      # Formal, scholarly style


class ExplanationAudience(str, Enum):
    """Target audience for explanations."""
    
    GENERAL = "general"        # General public
    TECHNICAL = "technical"    # Technical professionals
    EXPERT = "expert"         # Domain experts
    STUDENT = "student"       # Learning-focused
    EXECUTIVE = "executive"   # High-level decision makers


@dataclass
class ExplanationSection:
    """A section of a reasoning explanation."""
    
    title: str
    content: str
    section_type: str  # overview, steps, assumptions, conclusions, etc.
    confidence_notes: Optional[str] = None
    visual_elements: List[str] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "content": self.content,
            "section_type": self.section_type,
            "confidence_notes": self.confidence_notes,
            "visual_elements": self.visual_elements,
            "key_points": self.key_points
        }


@dataclass
class ReasoningExplanation:
    """Complete explanation of a reasoning process."""
    
    explanation_id: str
    title: str
    summary: str
    sections: List[ExplanationSection]
    style: ExplanationStyle
    audience: ExplanationAudience
    confidence_level: float
    key_insights: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "explanation_id": self.explanation_id,
            "title": self.title,
            "summary": self.summary,
            "sections": [section.to_dict() for section in self.sections],
            "style": self.style.value,
            "audience": self.audience.value,
            "confidence_level": self.confidence_level,
            "key_insights": self.key_insights,
            "limitations": self.limitations,
            "next_steps": self.next_steps,
            "metadata": self.metadata
        }
    
    def to_markdown(self) -> str:
        """Convert explanation to markdown format."""
        md_parts = [
            f"# {self.title}",
            "",
            f"**Summary:** {self.summary}",
            "",
            f"**Confidence Level:** {self.confidence_level:.1%}",
            ""
        ]
        
        for section in self.sections:
            md_parts.extend([
                f"## {section.title}",
                "",
                section.content,
                ""
            ])
            
            if section.key_points:
                md_parts.extend([
                    "**Key Points:**",
                    ""
                ])
                for point in section.key_points:
                    md_parts.append(f"- {point}")
                md_parts.append("")
            
            if section.confidence_notes:
                md_parts.extend([
                    f"*Confidence Note: {section.confidence_notes}*",
                    ""
                ])
        
        if self.key_insights:
            md_parts.extend([
                "## Key Insights",
                ""
            ])
            for insight in self.key_insights:
                md_parts.append(f"- {insight}")
            md_parts.append("")
        
        if self.limitations:
            md_parts.extend([
                "## Limitations",
                ""
            ])
            for limitation in self.limitations:
                md_parts.append(f"- {limitation}")
            md_parts.append("")
        
        if self.next_steps:
            md_parts.extend([
                "## Recommended Next Steps",
                ""
            ])
            for step in self.next_steps:
                md_parts.append(f"- {step}")
        
        return "\n".join(md_parts)


class ReasoningExplainer:
    """
    Generates clear explanations of reasoning processes.
    
    Can explain different types of reasoning artifacts:
    - Complete reasoning results from the reasoning engine
    - Problem decompositions
    - Reasoning chains (deductive, inductive, etc.)
    - Assumption sets and their implications
    
    Adapts explanations based on:
    - Target audience (general, technical, expert, etc.)
    - Explanation style (simple, detailed, visual, etc.)
    - Specific focus areas (assumptions, confidence, alternatives)
    
    Example:
        >>> explainer = ReasoningExplainer()
        >>> explanation = await explainer.explain_reasoning_result(
        ...     reasoning_result,
        ...     style=ExplanationStyle.CONVERSATIONAL,
        ...     audience=ExplanationAudience.GENERAL
        ... )
        >>> print(explanation.to_markdown())
    """
    
    # Prompt templates for different explanation types
    EXPLAIN_REASONING_PROMPT = """Create a clear explanation of this reasoning process.

Reasoning Process:
Query: {query}
Final Answer: {final_answer}
Confidence: {confidence}

Reasoning Steps:
{steps_summary}

Style: {style}
Audience: {audience}

Create an explanation that:
1. Summarizes what was being reasoned about
2. Explains the reasoning approach taken
3. Walks through the key steps clearly
4. Discusses confidence levels and uncertainties
5. Identifies key insights and limitations

Adapt the language and detail level for the {audience} audience using a {style} style.

Respond in JSON format:
{{
    "title": "Clear title for the explanation",
    "summary": "Brief overview of the reasoning process",
    "sections": [
        {{
            "title": "Section title",
            "content": "Detailed explanation content",
            "section_type": "overview|steps|assumptions|conclusions|confidence",
            "key_points": ["point1", "point2"],
            "confidence_notes": "Notes about confidence/uncertainty"
        }},
        ...
    ],
    "key_insights": ["insight1", "insight2"],
    "limitations": ["limitation1", "limitation2"],
    "next_steps": ["step1", "step2"]
}}"""

    EXPLAIN_DECOMPOSITION_PROMPT = """Explain this problem decomposition clearly.

Original Problem: {problem}
Strategy: {strategy}
Sub-problems: {sub_problems_summary}

Style: {style}
Audience: {audience}

Create an explanation that:
1. Explains why the problem needed to be broken down
2. Describes the decomposition strategy used
3. Walks through each sub-problem and its role
4. Explains the execution order and dependencies
5. Discusses complexity and effort estimates

Respond in JSON format with the same structure as the reasoning explanation."""

    EXPLAIN_ASSUMPTIONS_PROMPT = """Explain this set of assumptions and their implications.

Assumptions: {assumptions_summary}
Overall Risk: {overall_risk}
Critical Assumptions: {critical_assumptions}

Style: {style}
Audience: {audience}

Create an explanation that:
1. Explains what assumptions are and why they matter
2. Describes each key assumption clearly
3. Discusses the risks if assumptions are wrong
4. Explains which assumptions are most critical
5. Suggests ways to validate or test assumptions

Respond in JSON format with the same structure as other explanations."""

    def __init__(self, llm_service: Optional[LLMService] = None):
        """Initialize the reasoning explainer."""
        self.llm = llm_service or get_llm_service()
        logger.info("ReasoningExplainer initialized")

    async def explain_reasoning_result(
        self,
        reasoning_result: ReasoningResult,
        style: ExplanationStyle = ExplanationStyle.CONVERSATIONAL,
        audience: ExplanationAudience = ExplanationAudience.GENERAL,
        focus_areas: Optional[List[str]] = None
    ) -> ReasoningExplanation:
        """
        Explain a complete reasoning result.
        
        Args:
            reasoning_result: The reasoning result to explain
            style: Explanation style to use
            audience: Target audience for the explanation
            focus_areas: Specific areas to focus on (optional)
            
        Returns:
            Complete explanation of the reasoning process
        """
        logger.info(f"Explaining reasoning result: {reasoning_result.reasoning_id}")
        
        try:
            # Prepare steps summary
            steps_summary = self._format_reasoning_steps(reasoning_result.steps)
            
            prompt = self.EXPLAIN_REASONING_PROMPT.format(
                query=reasoning_result.query,
                final_answer=reasoning_result.final_answer,
                confidence=f"{reasoning_result.overall_confidence:.1%}",
                steps_summary=steps_summary,
                style=style.value,
                audience=audience.value
            )
            
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=2500
            )
            
            result = self._parse_json_response(response.content)
            
            # Create explanation sections
            sections = []
            for section_data in result.get("sections", []):
                section = ExplanationSection(
                    title=section_data.get("title", ""),
                    content=section_data.get("content", ""),
                    section_type=section_data.get("section_type", "general"),
                    confidence_notes=section_data.get("confidence_notes"),
                    key_points=section_data.get("key_points", [])
                )
                sections.append(section)
            
            explanation = ReasoningExplanation(
                explanation_id=f"exp_{reasoning_result.reasoning_id}",
                title=result.get("title", "Reasoning Explanation"),
                summary=result.get("summary", ""),
                sections=sections,
                style=style,
                audience=audience,
                confidence_level=reasoning_result.overall_confidence,
                key_insights=result.get("key_insights", []),
                limitations=result.get("limitations", []),
                next_steps=result.get("next_steps", []),
                metadata={
                    "original_reasoning_id": reasoning_result.reasoning_id,
                    "focus_areas": focus_areas or []
                }
            )
            
            logger.info("Reasoning explanation generated successfully")
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain reasoning result: {e}")
            return self._create_fallback_explanation(
                "Reasoning Process Explanation",
                f"Unable to generate detailed explanation: {e}",
                style,
                audience
            )

    async def explain_problem_decomposition(
        self,
        decomposition: ProblemDecomposition,
        style: ExplanationStyle = ExplanationStyle.DETAILED,
        audience: ExplanationAudience = ExplanationAudience.TECHNICAL
    ) -> ReasoningExplanation:
        """
        Explain a problem decomposition.
        
        Args:
            decomposition: The problem decomposition to explain
            style: Explanation style to use
            audience: Target audience for the explanation
            
        Returns:
            Complete explanation of the decomposition process
        """
        logger.info(f"Explaining problem decomposition: {decomposition.decomposition_id}")
        
        try:
            # Prepare sub-problems summary
            sub_problems_summary = self._format_sub_problems(decomposition.sub_problems)
            
            prompt = self.EXPLAIN_DECOMPOSITION_PROMPT.format(
                problem=decomposition.original_problem,
                strategy=decomposition.strategy.value,
                sub_problems_summary=sub_problems_summary,
                style=style.value,
                audience=audience.value
            )
            
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            result = self._parse_json_response(response.content)
            
            # Create explanation sections
            sections = []
            for section_data in result.get("sections", []):
                section = ExplanationSection(
                    title=section_data.get("title", ""),
                    content=section_data.get("content", ""),
                    section_type=section_data.get("section_type", "general"),
                    key_points=section_data.get("key_points", [])
                )
                sections.append(section)
            
            explanation = ReasoningExplanation(
                explanation_id=f"exp_{decomposition.decomposition_id}",
                title=result.get("title", "Problem Decomposition Explanation"),
                summary=result.get("summary", ""),
                sections=sections,
                style=style,
                audience=audience,
                confidence_level=1.0 - decomposition.total_complexity,  # Higher complexity = lower confidence
                key_insights=result.get("key_insights", []),
                limitations=result.get("limitations", []),
                next_steps=result.get("next_steps", []),
                metadata={
                    "original_decomposition_id": decomposition.decomposition_id,
                    "strategy": decomposition.strategy.value,
                    "sub_problem_count": len(decomposition.sub_problems)
                }
            )
            
            logger.info("Decomposition explanation generated successfully")
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain decomposition: {e}")
            return self._create_fallback_explanation(
                "Problem Decomposition Explanation",
                f"Unable to generate detailed explanation: {e}",
                style,
                audience
            )

    async def explain_assumption_set(
        self,
        assumption_set: AssumptionSet,
        style: ExplanationStyle = ExplanationStyle.DETAILED,
        audience: ExplanationAudience = ExplanationAudience.TECHNICAL
    ) -> ReasoningExplanation:
        """
        Explain a set of assumptions and their implications.
        
        Args:
            assumption_set: The assumption set to explain
            style: Explanation style to use
            audience: Target audience for the explanation
            
        Returns:
            Complete explanation of the assumptions
        """
        logger.info(f"Explaining assumption set: {assumption_set.set_id}")
        
        try:
            # Prepare assumptions summary
            assumptions_summary = self._format_assumptions(assumption_set.assumptions)
            critical_assumptions = ", ".join(assumption_set.critical_assumptions)
            
            prompt = self.EXPLAIN_ASSUMPTIONS_PROMPT.format(
                assumptions_summary=assumptions_summary,
                overall_risk=assumption_set.overall_risk.value,
                critical_assumptions=critical_assumptions,
                style=style.value,
                audience=audience.value
            )
            
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            result = self._parse_json_response(response.content)
            
            # Create explanation sections
            sections = []
            for section_data in result.get("sections", []):
                section = ExplanationSection(
                    title=section_data.get("title", ""),
                    content=section_data.get("content", ""),
                    section_type=section_data.get("section_type", "general"),
                    key_points=section_data.get("key_points", [])
                )
                sections.append(section)
            
            # Calculate confidence based on assumption risk
            confidence = 1.0 if assumption_set.overall_risk.value == "low" else \
                        0.8 if assumption_set.overall_risk.value == "medium" else \
                        0.6 if assumption_set.overall_risk.value == "high" else 0.4
            
            explanation = ReasoningExplanation(
                explanation_id=f"exp_{assumption_set.set_id}",
                title=result.get("title", "Assumption Analysis"),
                summary=result.get("summary", ""),
                sections=sections,
                style=style,
                audience=audience,
                confidence_level=confidence,
                key_insights=result.get("key_insights", []),
                limitations=result.get("limitations", []),
                next_steps=result.get("next_steps", []),
                metadata={
                    "original_assumption_set_id": assumption_set.set_id,
                    "assumption_count": len(assumption_set.assumptions),
                    "overall_risk": assumption_set.overall_risk.value
                }
            )
            
            logger.info("Assumption explanation generated successfully")
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain assumptions: {e}")
            return self._create_fallback_explanation(
                "Assumption Analysis",
                f"Unable to generate detailed explanation: {e}",
                style,
                audience
            )

    def create_visual_summary(self, explanation: ReasoningExplanation) -> str:
        """
        Create a visual text-based summary of the explanation.
        
        Args:
            explanation: The explanation to summarize visually
            
        Returns:
            ASCII art style visual summary
        """
        lines = [
            "┌─" + "─" * (len(explanation.title) + 2) + "─┐",
            f"│ {explanation.title} │",
            "└─" + "─" * (len(explanation.title) + 2) + "─┘",
            "",
            f"Confidence: {'█' * int(explanation.confidence_level * 10)}{'░' * (10 - int(explanation.confidence_level * 10))} {explanation.confidence_level:.1%}",
            "",
            "Summary:",
            f"  {explanation.summary}",
            ""
        ]
        
        if explanation.key_insights:
            lines.extend([
                "Key Insights:",
                *[f"  • {insight}" for insight in explanation.key_insights],
                ""
            ])
        
        if explanation.limitations:
            lines.extend([
                "Limitations:",
                *[f"  ⚠ {limitation}" for limitation in explanation.limitations],
                ""
            ])
        
        return "\n".join(lines)

    def _format_reasoning_steps(self, steps: List[ReasoningStep]) -> str:
        """Format reasoning steps for inclusion in prompts."""
        formatted_steps = []
        for i, step in enumerate(steps, 1):
            formatted_steps.append(
                f"Step {i}: {step.thought}\n"
                f"Conclusion: {step.conclusion or 'No specific conclusion'}\n"
                f"Confidence: {step.confidence:.1%}"
            )
        return "\n\n".join(formatted_steps)

    def _format_sub_problems(self, sub_problems: List[SubProblem]) -> str:
        """Format sub-problems for inclusion in prompts."""
        formatted = []
        for sp in sub_problems:
            formatted.append(
                f"• {sp.title} (Priority: {sp.priority}, Complexity: {sp.complexity:.1f})\n"
                f"  {sp.description}"
            )
        return "\n\n".join(formatted)

    def _format_assumptions(self, assumptions: List[Assumption]) -> str:
        """Format assumptions for inclusion in prompts."""
        formatted = []
        for assumption in assumptions:
            formatted.append(
                f"• {assumption.content}\n"
                f"  Type: {assumption.assumption_type.value}, "
                f"Risk: {assumption.risk_level.value}, "
                f"Confidence: {assumption.confidence:.1%}"
            )
        return "\n\n".join(formatted)

    def _create_fallback_explanation(
        self,
        title: str,
        error_message: str,
        style: ExplanationStyle,
        audience: ExplanationAudience
    ) -> ReasoningExplanation:
        """Create a basic fallback explanation when generation fails."""
        fallback_section = ExplanationSection(
            title="Error",
            content=error_message,
            section_type="error"
        )
        
        return ReasoningExplanation(
            explanation_id=f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=title,
            summary="Explanation generation encountered an error",
            sections=[fallback_section],
            style=style,
            audience=audience,
            confidence_level=0.0,
            limitations=["Explanation generation failed"],
            metadata={"fallback": True, "error": error_message}
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


# Singleton instance
_explainer: Optional[ReasoningExplainer] = None


def get_reasoning_explainer() -> ReasoningExplainer:
    """Get singleton reasoning explainer instance."""
    global _explainer
    if _explainer is None:
        _explainer = ReasoningExplainer()
    return _explainer