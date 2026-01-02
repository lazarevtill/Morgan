"""
Logical Reasoning Chains for Morgan's Reasoning Engine.

Implements different types of logical reasoning patterns:
- Deductive reasoning (general to specific)
- Inductive reasoning (specific to general)
- Abductive reasoning (best explanation)
- Causal reasoning (cause and effect)
- Analogical reasoning (similarity-based)

Follows KISS principles with clear, focused reasoning chain implementations.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from morgan.services.llm import LLMService, get_llm_service
from morgan.utils.logger import get_logger

logger = get_logger(__name__)


class ReasoningChainType(str, Enum):
    """Types of logical reasoning chains."""
    
    DEDUCTIVE = "deductive"      # General principles → specific conclusions
    INDUCTIVE = "inductive"      # Specific observations → general patterns
    ABDUCTIVE = "abductive"      # Best explanation for observations
    CAUSAL = "causal"           # Cause and effect relationships
    ANALOGICAL = "analogical"    # Reasoning by similarity/comparison
    HYPOTHETICAL = "hypothetical" # If-then conditional reasoning


class ChainStepType(str, Enum):
    """Types of steps in a reasoning chain."""
    
    PREMISE = "premise"          # Starting assumption or fact
    INFERENCE = "inference"      # Logical step from premises
    CONCLUSION = "conclusion"    # Final result of reasoning
    EVIDENCE = "evidence"        # Supporting evidence
    ASSUMPTION = "assumption"    # Explicit assumption made
    COUNTERARGUMENT = "counterargument"  # Potential objection


@dataclass
class ChainStep:
    """A single step in a logical reasoning chain."""
    
    step_id: str
    step_number: int
    step_type: ChainStepType
    content: str
    confidence: float  # 0.0-1.0
    supporting_evidence: List[str] = field(default_factory=list)
    logical_connection: Optional[str] = None  # How this connects to previous steps
    assumptions: List[str] = field(default_factory=list)
    potential_flaws: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "logical_connection": self.logical_connection,
            "assumptions": self.assumptions,
            "potential_flaws": self.potential_flaws
        }


@dataclass
class ReasoningChain:
    """A complete logical reasoning chain."""
    
    chain_id: str
    chain_type: ReasoningChainType
    premise: str
    conclusion: str
    steps: List[ChainStep]
    overall_confidence: float
    logical_validity: float  # How logically sound the chain is
    strength_of_evidence: float  # How well-supported the conclusion is
    potential_weaknesses: List[str] = field(default_factory=list)
    alternative_conclusions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chain_id": self.chain_id,
            "chain_type": self.chain_type.value,
            "premise": self.premise,
            "conclusion": self.conclusion,
            "steps": [step.to_dict() for step in self.steps],
            "overall_confidence": self.overall_confidence,
            "logical_validity": self.logical_validity,
            "strength_of_evidence": self.strength_of_evidence,
            "potential_weaknesses": self.potential_weaknesses,
            "alternative_conclusions": self.alternative_conclusions,
            "metadata": self.metadata
        }


class ReasoningChainBuilder:
    """
    Builds logical reasoning chains for different types of reasoning.
    
    Supports multiple reasoning patterns:
    - Deductive: Start with general principles, derive specific conclusions
    - Inductive: Start with specific observations, infer general patterns
    - Abductive: Start with observations, find best explanation
    - Causal: Trace cause-and-effect relationships
    - Analogical: Reason by comparing to similar situations
    
    Example:
        >>> builder = ReasoningChainBuilder()
        >>> chain = await builder.build_deductive_chain(
        ...     premise="All software has bugs",
        ...     specific_case="This is a software system",
        ...     context="We're evaluating system reliability"
        ... )
        >>> print(chain.conclusion)
    """
    
    # Prompt templates for different reasoning types
    DEDUCTIVE_CHAIN_PROMPT = """Build a deductive reasoning chain from general principle to specific conclusion.

General Principle/Premise: {premise}
Specific Case: {specific_case}
Context: {context}

Create a logical chain that:
1. States the general principle clearly
2. Identifies the specific case
3. Makes the logical connection
4. Draws the specific conclusion

For each step, provide:
- The logical content
- Confidence level (0.0-1.0)
- How it connects to previous steps
- Any assumptions made

Respond in JSON format:
{{
    "steps": [
        {{
            "step_type": "premise|inference|conclusion",
            "content": "The logical statement",
            "confidence": 0.0-1.0,
            "logical_connection": "How this follows from previous steps",
            "assumptions": ["assumption1", "assumption2"]
        }},
        ...
    ],
    "final_conclusion": "The specific conclusion reached",
    "logical_validity": 0.0-1.0,
    "potential_flaws": ["flaw1", "flaw2"]
}}"""

    INDUCTIVE_CHAIN_PROMPT = """Build an inductive reasoning chain from specific observations to general pattern.

Observations: {observations}
Context: {context}

Create a logical chain that:
1. Lists the specific observations
2. Identifies patterns or commonalities
3. Considers the sample size and representativeness
4. Draws a general conclusion with appropriate confidence

For each step, assess:
- How strong the pattern is
- How representative the sample is
- What alternative explanations exist

Respond in JSON format:
{{
    "steps": [
        {{
            "step_type": "evidence|inference|conclusion",
            "content": "The logical statement",
            "confidence": 0.0-1.0,
            "supporting_evidence": ["evidence1", "evidence2"],
            "assumptions": ["assumption1", "assumption2"]
        }},
        ...
    ],
    "final_conclusion": "The general pattern or rule inferred",
    "strength_of_evidence": 0.0-1.0,
    "alternative_conclusions": ["alternative1", "alternative2"]
}}"""

    ABDUCTIVE_CHAIN_PROMPT = """Build an abductive reasoning chain to find the best explanation.

Observations/Facts: {observations}
Context: {context}

Create a logical chain that:
1. Lists the key observations that need explaining
2. Generates multiple possible explanations
3. Evaluates each explanation's plausibility
4. Selects the best explanation

Consider:
- Which explanation accounts for the most observations
- Which requires the fewest assumptions
- Which is most consistent with known facts

Respond in JSON format:
{{
    "steps": [
        {{
            "step_type": "evidence|inference|conclusion",
            "content": "The logical statement",
            "confidence": 0.0-1.0,
            "supporting_evidence": ["evidence1", "evidence2"],
            "assumptions": ["assumption1", "assumption2"]
        }},
        ...
    ],
    "final_conclusion": "The best explanation for the observations",
    "alternative_explanations": ["explanation1", "explanation2"],
    "explanation_quality": 0.0-1.0
}}"""

    CAUSAL_CHAIN_PROMPT = """Build a causal reasoning chain to trace cause-and-effect relationships.

Situation: {situation}
Context: {context}

Create a logical chain that:
1. Identifies potential causes
2. Traces the causal mechanism
3. Considers intervening factors
4. Evaluates the strength of causal links

Consider:
- Temporal sequence (cause before effect)
- Mechanism (how cause leads to effect)
- Alternative causes
- Confounding factors

Respond in JSON format:
{{
    "steps": [
        {{
            "step_type": "premise|inference|conclusion",
            "content": "The causal statement",
            "confidence": 0.0-1.0,
            "logical_connection": "The causal mechanism",
            "assumptions": ["assumption1", "assumption2"]
        }},
        ...
    ],
    "final_conclusion": "The causal relationship identified",
    "causal_strength": 0.0-1.0,
    "alternative_causes": ["cause1", "cause2"]
}}"""

    ANALOGICAL_CHAIN_PROMPT = """Build an analogical reasoning chain using similarity-based reasoning.

Source Situation: {source_situation}
Target Situation: {target_situation}
Context: {context}

Create a logical chain that:
1. Identifies key similarities between situations
2. Maps relevant features from source to target
3. Considers important differences
4. Draws conclusions based on the analogy

Evaluate:
- How similar are the core structures?
- Which similarities are most relevant?
- What differences might limit the analogy?

Respond in JSON format:
{{
    "steps": [
        {{
            "step_type": "premise|inference|conclusion",
            "content": "The analogical statement",
            "confidence": 0.0-1.0,
            "supporting_evidence": ["similarity1", "similarity2"],
            "assumptions": ["assumption1", "assumption2"]
        }},
        ...
    ],
    "final_conclusion": "The conclusion drawn from the analogy",
    "analogy_strength": 0.0-1.0,
    "key_differences": ["difference1", "difference2"]
}}"""

    def __init__(self, llm_service: Optional[LLMService] = None):
        """Initialize the reasoning chain builder."""
        self.llm = llm_service or get_llm_service()
        logger.info("ReasoningChainBuilder initialized")

    async def build_deductive_chain(
        self,
        premise: str,
        specific_case: str,
        context: Optional[str] = None
    ) -> ReasoningChain:
        """
        Build a deductive reasoning chain from general to specific.
        
        Args:
            premise: General principle or rule
            specific_case: Specific instance to apply the rule to
            context: Additional context for the reasoning
            
        Returns:
            Complete deductive reasoning chain
        """
        return await self._build_chain(
            chain_type=ReasoningChainType.DEDUCTIVE,
            prompt_template=self.DEDUCTIVE_CHAIN_PROMPT,
            premise=premise,
            specific_case=specific_case,
            context=context or "General deductive reasoning"
        )

    async def build_inductive_chain(
        self,
        observations: List[str],
        context: Optional[str] = None
    ) -> ReasoningChain:
        """
        Build an inductive reasoning chain from specific to general.
        
        Args:
            observations: List of specific observations or examples
            context: Additional context for the reasoning
            
        Returns:
            Complete inductive reasoning chain
        """
        observations_text = "\n".join(f"- {obs}" for obs in observations)
        return await self._build_chain(
            chain_type=ReasoningChainType.INDUCTIVE,
            prompt_template=self.INDUCTIVE_CHAIN_PROMPT,
            observations=observations_text,
            context=context or "General inductive reasoning"
        )

    async def build_abductive_chain(
        self,
        observations: List[str],
        context: Optional[str] = None
    ) -> ReasoningChain:
        """
        Build an abductive reasoning chain to find best explanation.
        
        Args:
            observations: List of facts or observations to explain
            context: Additional context for the reasoning
            
        Returns:
            Complete abductive reasoning chain
        """
        observations_text = "\n".join(f"- {obs}" for obs in observations)
        return await self._build_chain(
            chain_type=ReasoningChainType.ABDUCTIVE,
            prompt_template=self.ABDUCTIVE_CHAIN_PROMPT,
            observations=observations_text,
            context=context or "Finding best explanation"
        )

    async def build_causal_chain(
        self,
        situation: str,
        context: Optional[str] = None
    ) -> ReasoningChain:
        """
        Build a causal reasoning chain to trace cause-and-effect.
        
        Args:
            situation: The situation to analyze for causal relationships
            context: Additional context for the reasoning
            
        Returns:
            Complete causal reasoning chain
        """
        return await self._build_chain(
            chain_type=ReasoningChainType.CAUSAL,
            prompt_template=self.CAUSAL_CHAIN_PROMPT,
            situation=situation,
            context=context or "Causal analysis"
        )

    async def build_analogical_chain(
        self,
        source_situation: str,
        target_situation: str,
        context: Optional[str] = None
    ) -> ReasoningChain:
        """
        Build an analogical reasoning chain using similarity.
        
        Args:
            source_situation: The known situation to reason from
            target_situation: The new situation to reason about
            context: Additional context for the reasoning
            
        Returns:
            Complete analogical reasoning chain
        """
        return await self._build_chain(
            chain_type=ReasoningChainType.ANALOGICAL,
            prompt_template=self.ANALOGICAL_CHAIN_PROMPT,
            source_situation=source_situation,
            target_situation=target_situation,
            context=context or "Analogical reasoning"
        )

    async def _build_chain(
        self,
        chain_type: ReasoningChainType,
        prompt_template: str,
        **kwargs
    ) -> ReasoningChain:
        """Build a reasoning chain using the specified template and parameters."""
        chain_id = str(uuid.uuid4())
        logger.info(f"Building {chain_type.value} reasoning chain: {chain_id}")
        
        try:
            # Generate the reasoning chain using LLM
            prompt = prompt_template.format(**kwargs)
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            result = self._parse_json_response(response.content)
            
            # Convert steps to ChainStep objects
            steps = []
            for i, step_data in enumerate(result.get("steps", [])):
                step = ChainStep(
                    step_id=f"{chain_id}_step_{i+1}",
                    step_number=i + 1,
                    step_type=ChainStepType(step_data.get("step_type", "inference")),
                    content=step_data.get("content", ""),
                    confidence=float(step_data.get("confidence", 0.7)),
                    supporting_evidence=step_data.get("supporting_evidence", []),
                    logical_connection=step_data.get("logical_connection"),
                    assumptions=step_data.get("assumptions", []),
                    potential_flaws=step_data.get("potential_flaws", [])
                )
                steps.append(step)
            
            # Extract chain-level metrics
            overall_confidence = self._calculate_overall_confidence(steps)
            logical_validity = float(result.get("logical_validity", 0.8))
            strength_of_evidence = float(result.get("strength_of_evidence", 0.7))
            
            # Determine premise and conclusion
            premise = self._extract_premise(kwargs, steps)
            conclusion = result.get("final_conclusion", "No conclusion reached")
            
            # Extract weaknesses and alternatives
            potential_weaknesses = result.get("potential_flaws", [])
            alternative_conclusions = result.get("alternative_conclusions", [])
            
            chain = ReasoningChain(
                chain_id=chain_id,
                chain_type=chain_type,
                premise=premise,
                conclusion=conclusion,
                steps=steps,
                overall_confidence=overall_confidence,
                logical_validity=logical_validity,
                strength_of_evidence=strength_of_evidence,
                potential_weaknesses=potential_weaknesses,
                alternative_conclusions=alternative_conclusions,
                metadata={
                    "input_parameters": kwargs,
                    "step_count": len(steps)
                }
            )
            
            logger.info(
                f"Reasoning chain completed: {len(steps)} steps, "
                f"confidence: {overall_confidence:.2f}"
            )
            
            return chain
            
        except Exception as e:
            logger.error(f"Failed to build {chain_type.value} reasoning chain: {e}")
            return self._create_fallback_chain(chain_type, str(e), kwargs)

    def _calculate_overall_confidence(self, steps: List[ChainStep]) -> float:
        """Calculate overall confidence from individual step confidences."""
        if not steps:
            return 0.0
        
        # Use geometric mean for confidence (more conservative)
        product = 1.0
        for step in steps:
            product *= step.confidence
        
        return round(product ** (1.0 / len(steps)), 2)

    def _extract_premise(self, kwargs: Dict[str, Any], steps: List[ChainStep]) -> str:
        """Extract the premise from input parameters or first step."""
        # Try to get premise from input parameters
        if "premise" in kwargs:
            return kwargs["premise"]
        elif "observations" in kwargs:
            return f"Observations: {kwargs['observations'][:200]}..."
        elif "situation" in kwargs:
            return f"Situation: {kwargs['situation'][:200]}..."
        elif steps and steps[0].step_type == ChainStepType.PREMISE:
            return steps[0].content
        else:
            return "Starting premise not clearly identified"

    def _create_fallback_chain(
        self,
        chain_type: ReasoningChainType,
        error: str,
        kwargs: Dict[str, Any]
    ) -> ReasoningChain:
        """Create a basic fallback chain when AI generation fails."""
        fallback_step = ChainStep(
            step_id="fallback_1",
            step_number=1,
            step_type=ChainStepType.INFERENCE,
            content=f"Unable to complete {chain_type.value} reasoning due to error: {error}",
            confidence=0.3,
            assumptions=["Fallback reasoning due to system error"]
        )
        
        return ReasoningChain(
            chain_id=str(uuid.uuid4()),
            chain_type=chain_type,
            premise=str(kwargs),
            conclusion="Reasoning could not be completed",
            steps=[fallback_step],
            overall_confidence=0.3,
            logical_validity=0.0,
            strength_of_evidence=0.0,
            potential_weaknesses=["System error prevented proper reasoning"],
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

    def validate_chain(self, chain: ReasoningChain) -> Dict[str, Any]:
        """
        Validate the logical structure and quality of a reasoning chain.
        
        Returns:
            Dictionary with validation results and suggestions
        """
        validation = {
            "is_valid": True,
            "issues": [],
            "suggestions": [],
            "quality_score": 0.0
        }
        
        # Check for basic structure
        if not chain.steps:
            validation["is_valid"] = False
            validation["issues"].append("No reasoning steps found")
            return validation
        
        # Check for logical flow
        premise_found = any(step.step_type == ChainStepType.PREMISE for step in chain.steps)
        conclusion_found = any(step.step_type == ChainStepType.CONCLUSION for step in chain.steps)
        
        if not premise_found:
            validation["issues"].append("No clear premise identified")
        
        if not conclusion_found:
            validation["issues"].append("No clear conclusion reached")
        
        # Check confidence levels
        low_confidence_steps = [step for step in chain.steps if step.confidence < 0.5]
        if low_confidence_steps:
            validation["suggestions"].append(
                f"{len(low_confidence_steps)} steps have low confidence - consider strengthening evidence"
            )
        
        # Check for assumptions
        total_assumptions = sum(len(step.assumptions) for step in chain.steps)
        if total_assumptions > len(chain.steps) * 2:
            validation["suggestions"].append("Many assumptions made - consider validating key assumptions")
        
        # Calculate quality score
        structure_score = 0.4 if premise_found and conclusion_found else 0.2
        confidence_score = chain.overall_confidence * 0.3
        validity_score = chain.logical_validity * 0.3
        
        validation["quality_score"] = structure_score + confidence_score + validity_score
        
        if validation["quality_score"] < 0.6:
            validation["is_valid"] = False
            validation["issues"].append("Overall quality score too low")
        
        return validation


# Singleton instance
_chain_builder: Optional[ReasoningChainBuilder] = None


def get_reasoning_chain_builder() -> ReasoningChainBuilder:
    """Get singleton reasoning chain builder instance."""
    global _chain_builder
    if _chain_builder is None:
        _chain_builder = ReasoningChainBuilder()
    return _chain_builder