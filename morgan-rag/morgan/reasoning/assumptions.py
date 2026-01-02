"""
Assumption Tracker for Morgan's Reasoning Engine.

Tracks, validates, and manages assumptions made during reasoning:
- Explicit assumption identification
- Assumption validation and testing
- Impact assessment of assumptions
- Alternative scenario analysis
- Assumption dependency tracking

Follows KISS principles with clear assumption management.
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


class AssumptionType(str, Enum):
    """Types of assumptions that can be made."""
    
    FACTUAL = "factual"          # Assumptions about facts or data
    CAUSAL = "causal"           # Assumptions about cause-effect relationships
    BEHAVIORAL = "behavioral"    # Assumptions about human/system behavior
    CONTEXTUAL = "contextual"   # Assumptions about context or environment
    METHODOLOGICAL = "methodological"  # Assumptions about methods or approaches
    TEMPORAL = "temporal"       # Assumptions about timing or sequence


class AssumptionStatus(str, Enum):
    """Status of assumption validation."""
    
    UNVALIDATED = "unvalidated"  # Not yet checked
    VALIDATED = "validated"      # Confirmed to be reasonable
    QUESTIONABLE = "questionable"  # Potentially problematic
    INVALID = "invalid"          # Clearly wrong or unreasonable
    CRITICAL = "critical"        # High impact on conclusions


class AssumptionRisk(str, Enum):
    """Risk level if assumption is wrong."""
    
    LOW = "low"                 # Minor impact on conclusions
    MEDIUM = "medium"           # Moderate impact on conclusions
    HIGH = "high"               # Major impact on conclusions
    CRITICAL = "critical"       # Invalidates conclusions


@dataclass
class Assumption:
    """A single assumption made during reasoning."""
    
    assumption_id: str
    content: str
    assumption_type: AssumptionType
    context: str  # Where/why this assumption was made
    confidence: float  # 0.0-1.0, how confident we are this is true
    risk_level: AssumptionRisk  # Impact if assumption is wrong
    status: AssumptionStatus = AssumptionStatus.UNVALIDATED
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Other assumption IDs this depends on
    implications: List[str] = field(default_factory=list)  # What follows if this is true
    alternatives: List[str] = field(default_factory=list)  # Alternative assumptions
    validation_notes: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "assumption_id": self.assumption_id,
            "content": self.content,
            "assumption_type": self.assumption_type.value,
            "context": self.context,
            "confidence": self.confidence,
            "risk_level": self.risk_level.value,
            "status": self.status.value,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "dependencies": self.dependencies,
            "implications": self.implications,
            "alternatives": self.alternatives,
            "validation_notes": self.validation_notes
        }


@dataclass
class AssumptionSet:
    """A collection of related assumptions."""
    
    set_id: str
    name: str
    description: str
    assumptions: List[Assumption]
    overall_risk: AssumptionRisk
    critical_assumptions: List[str]  # IDs of most important assumptions
    assumption_conflicts: List[Tuple[str, str]] = field(default_factory=list)  # Conflicting assumption pairs
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "set_id": self.set_id,
            "name": self.name,
            "description": self.description,
            "assumptions": [a.to_dict() for a in self.assumptions],
            "overall_risk": self.overall_risk.value,
            "critical_assumptions": self.critical_assumptions,
            "assumption_conflicts": self.assumption_conflicts,
            "metadata": self.metadata
        }


class AssumptionTracker:
    """
    Tracks and manages assumptions made during reasoning processes.
    
    Key capabilities:
    1. Automatically identify assumptions in reasoning text
    2. Categorize and assess risk levels of assumptions
    3. Validate assumptions against available evidence
    4. Track dependencies between assumptions
    5. Analyze impact of assumption changes on conclusions
    
    Example:
        >>> tracker = AssumptionTracker()
        >>> assumptions = await tracker.extract_assumptions(
        ...     "If we assume users will adopt the new interface quickly, "
        ...     "then we can expect 50% adoption within 3 months."
        ... )
        >>> for assumption in assumptions:
        ...     print(f"{assumption.content} (Risk: {assumption.risk_level})")
    """
    
    # Prompt templates for assumption analysis
    EXTRACT_ASSUMPTIONS_PROMPT = """Identify all explicit and implicit assumptions in this reasoning text.

Text: {text}
Context: {context}

For each assumption, determine:
1. What exactly is being assumed
2. What type of assumption it is (factual, causal, behavioral, contextual, methodological, temporal)
3. How confident we should be in this assumption (0.0-1.0)
4. What the risk level is if this assumption is wrong (low, medium, high, critical)
5. What evidence might support or contradict it

Respond in JSON format:
{{
    "assumptions": [
        {{
            "content": "Clear statement of what is assumed",
            "assumption_type": "factual|causal|behavioral|contextual|methodological|temporal",
            "confidence": 0.0-1.0,
            "risk_level": "low|medium|high|critical",
            "supporting_evidence": ["evidence1", "evidence2"],
            "implications": ["what follows if true"],
            "alternatives": ["alternative assumption"]
        }},
        ...
    ]
}}"""

    VALIDATE_ASSUMPTION_PROMPT = """Validate this assumption against available evidence and reasoning.

Assumption: {assumption}
Context: {context}
Available Evidence: {evidence}

Evaluate:
1. Is this assumption reasonable given the evidence?
2. What evidence supports or contradicts it?
3. How critical is this assumption to the overall reasoning?
4. What are the main risks if this assumption is wrong?
5. What alternative assumptions might be more valid?

Respond in JSON format:
{{
    "validation_status": "validated|questionable|invalid|critical",
    "confidence_adjustment": 0.0-1.0,
    "supporting_points": ["point1", "point2"],
    "contradicting_points": ["point1", "point2"],
    "risk_assessment": "Explanation of risks if assumption is wrong",
    "alternatives": ["alternative1", "alternative2"],
    "recommendation": "Keep, modify, or replace this assumption"
}}"""

    ANALYZE_DEPENDENCIES_PROMPT = """Analyze the dependencies and relationships between these assumptions.

Assumptions:
{assumptions_summary}

Identify:
1. Which assumptions depend on others
2. Which assumptions conflict with each other
3. Which assumptions are most critical to the overall reasoning
4. What happens if key assumptions change

Respond in JSON format:
{{
    "dependencies": {{
        "assumption_id_1": ["depends_on_id_1", "depends_on_id_2"],
        "assumption_id_2": [],
        ...
    }},
    "conflicts": [
        ["conflicting_id_1", "conflicting_id_2", "explanation"]
    ],
    "critical_assumptions": ["id1", "id2"],
    "impact_analysis": {{
        "assumption_id": "What happens if this changes"
    }}
}}"""

    def __init__(self, llm_service: Optional[LLMService] = None):
        """Initialize the assumption tracker."""
        self.llm = llm_service or get_llm_service()
        self.assumption_sets: Dict[str, AssumptionSet] = {}
        logger.info("AssumptionTracker initialized")

    async def extract_assumptions(
        self,
        text: str,
        context: Optional[str] = None
    ) -> List[Assumption]:
        """
        Extract assumptions from reasoning text.
        
        Args:
            text: The reasoning text to analyze
            context: Additional context about the reasoning
            
        Returns:
            List of identified assumptions
        """
        logger.info("Extracting assumptions from text")
        
        try:
            prompt = self.EXTRACT_ASSUMPTIONS_PROMPT.format(
                text=text,
                context=context or "General reasoning analysis"
            )
            
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=1500
            )
            
            result = self._parse_json_response(response.content)
            assumptions_data = result.get("assumptions", [])
            
            assumptions = []
            for i, assumption_data in enumerate(assumptions_data):
                assumption = Assumption(
                    assumption_id=f"assumption_{uuid.uuid4().hex[:8]}",
                    content=assumption_data.get("content", ""),
                    assumption_type=AssumptionType(assumption_data.get("assumption_type", "factual")),
                    context=context or "Extracted from reasoning text",
                    confidence=float(assumption_data.get("confidence", 0.7)),
                    risk_level=AssumptionRisk(assumption_data.get("risk_level", "medium")),
                    supporting_evidence=assumption_data.get("supporting_evidence", []),
                    implications=assumption_data.get("implications", []),
                    alternatives=assumption_data.get("alternatives", [])
                )
                assumptions.append(assumption)
            
            logger.info(f"Extracted {len(assumptions)} assumptions")
            return assumptions
            
        except Exception as e:
            logger.error(f"Failed to extract assumptions: {e}")
            return []

    async def validate_assumption(
        self,
        assumption: Assumption,
        evidence: Optional[List[str]] = None
    ) -> Assumption:
        """
        Validate an assumption against available evidence.
        
        Args:
            assumption: The assumption to validate
            evidence: Available evidence to check against
            
        Returns:
            Updated assumption with validation results
        """
        logger.info(f"Validating assumption: {assumption.assumption_id}")
        
        try:
            evidence_text = "\n".join(evidence) if evidence else "No specific evidence provided"
            
            prompt = self.VALIDATE_ASSUMPTION_PROMPT.format(
                assumption=assumption.content,
                context=assumption.context,
                evidence=evidence_text
            )
            
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000
            )
            
            result = self._parse_json_response(response.content)
            
            # Update assumption with validation results
            assumption.status = AssumptionStatus(result.get("validation_status", "unvalidated"))
            assumption.confidence = float(result.get("confidence_adjustment", assumption.confidence))
            assumption.supporting_evidence.extend(result.get("supporting_points", []))
            assumption.contradicting_evidence = result.get("contradicting_points", [])
            assumption.alternatives.extend(result.get("alternatives", []))
            assumption.validation_notes = result.get("recommendation", "")
            
            logger.info(f"Assumption validated: {assumption.status.value}")
            return assumption
            
        except Exception as e:
            logger.error(f"Failed to validate assumption: {e}")
            assumption.status = AssumptionStatus.QUESTIONABLE
            assumption.validation_notes = f"Validation failed: {e}"
            return assumption

    async def create_assumption_set(
        self,
        name: str,
        description: str,
        assumptions: List[Assumption]
    ) -> AssumptionSet:
        """
        Create a set of related assumptions and analyze their relationships.
        
        Args:
            name: Name for the assumption set
            description: Description of what these assumptions relate to
            assumptions: List of assumptions to include
            
        Returns:
            AssumptionSet with dependency and conflict analysis
        """
        set_id = str(uuid.uuid4())
        logger.info(f"Creating assumption set: {name}")
        
        try:
            # Analyze dependencies and relationships
            dependencies, conflicts, critical_assumptions = await self._analyze_assumption_relationships(assumptions)
            
            # Calculate overall risk
            overall_risk = self._calculate_overall_risk(assumptions)
            
            assumption_set = AssumptionSet(
                set_id=set_id,
                name=name,
                description=description,
                assumptions=assumptions,
                overall_risk=overall_risk,
                critical_assumptions=critical_assumptions,
                assumption_conflicts=conflicts,
                metadata={
                    "total_assumptions": len(assumptions),
                    "high_risk_count": len([a for a in assumptions if a.risk_level in [AssumptionRisk.HIGH, AssumptionRisk.CRITICAL]])
                }
            )
            
            # Store the assumption set
            self.assumption_sets[set_id] = assumption_set
            
            logger.info(f"Created assumption set with {len(assumptions)} assumptions")
            return assumption_set
            
        except Exception as e:
            logger.error(f"Failed to create assumption set: {e}")
            # Return basic assumption set without analysis
            return AssumptionSet(
                set_id=set_id,
                name=name,
                description=description,
                assumptions=assumptions,
                overall_risk=AssumptionRisk.MEDIUM,
                critical_assumptions=[],
                metadata={"error": str(e)}
            )

    async def _analyze_assumption_relationships(
        self, assumptions: List[Assumption]
    ) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]], List[str]]:
        """Analyze dependencies and conflicts between assumptions."""
        try:
            # Create summary of assumptions for analysis
            assumptions_summary = "\n".join([
                f"ID: {a.assumption_id}, Content: {a.content}, Type: {a.assumption_type.value}"
                for a in assumptions
            ])
            
            prompt = self.ANALYZE_DEPENDENCIES_PROMPT.format(
                assumptions_summary=assumptions_summary
            )
            
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=1200
            )
            
            result = self._parse_json_response(response.content)
            
            dependencies = result.get("dependencies", {})
            conflicts = [(c[0], c[1]) for c in result.get("conflicts", []) if len(c) >= 2]
            critical_assumptions = result.get("critical_assumptions", [])
            
            return dependencies, conflicts, critical_assumptions
            
        except Exception as e:
            logger.warning(f"Failed to analyze assumption relationships: {e}")
            return {}, [], []

    def _calculate_overall_risk(self, assumptions: List[Assumption]) -> AssumptionRisk:
        """Calculate overall risk level for a set of assumptions."""
        if not assumptions:
            return AssumptionRisk.LOW
        
        risk_counts = {
            AssumptionRisk.CRITICAL: 0,
            AssumptionRisk.HIGH: 0,
            AssumptionRisk.MEDIUM: 0,
            AssumptionRisk.LOW: 0
        }
        
        for assumption in assumptions:
            risk_counts[assumption.risk_level] += 1
        
        # Determine overall risk based on highest risks present
        if risk_counts[AssumptionRisk.CRITICAL] > 0:
            return AssumptionRisk.CRITICAL
        elif risk_counts[AssumptionRisk.HIGH] >= 2:
            return AssumptionRisk.HIGH
        elif risk_counts[AssumptionRisk.HIGH] >= 1 or risk_counts[AssumptionRisk.MEDIUM] >= 3:
            return AssumptionRisk.MEDIUM
        else:
            return AssumptionRisk.LOW

    def get_assumption_summary(self, assumption_set_id: str) -> Dict[str, Any]:
        """Get a summary of assumptions in a set."""
        if assumption_set_id not in self.assumption_sets:
            return {"error": "Assumption set not found"}
        
        assumption_set = self.assumption_sets[assumption_set_id]
        
        return {
            "set_name": assumption_set.name,
            "total_assumptions": len(assumption_set.assumptions),
            "overall_risk": assumption_set.overall_risk.value,
            "critical_count": len(assumption_set.critical_assumptions),
            "conflicts_count": len(assumption_set.assumption_conflicts),
            "status_breakdown": self._get_status_breakdown(assumption_set.assumptions),
            "risk_breakdown": self._get_risk_breakdown(assumption_set.assumptions),
            "type_breakdown": self._get_type_breakdown(assumption_set.assumptions)
        }

    def _get_status_breakdown(self, assumptions: List[Assumption]) -> Dict[str, int]:
        """Get breakdown of assumptions by validation status."""
        breakdown = {status.value: 0 for status in AssumptionStatus}
        for assumption in assumptions:
            breakdown[assumption.status.value] += 1
        return breakdown

    def _get_risk_breakdown(self, assumptions: List[Assumption]) -> Dict[str, int]:
        """Get breakdown of assumptions by risk level."""
        breakdown = {risk.value: 0 for risk in AssumptionRisk}
        for assumption in assumptions:
            breakdown[assumption.risk_level.value] += 1
        return breakdown

    def _get_type_breakdown(self, assumptions: List[Assumption]) -> Dict[str, int]:
        """Get breakdown of assumptions by type."""
        breakdown = {atype.value: 0 for atype in AssumptionType}
        for assumption in assumptions:
            breakdown[assumption.assumption_type.value] += 1
        return breakdown

    def test_assumption_impact(
        self, assumption_set_id: str, assumption_id: str, new_value: str
    ) -> Dict[str, Any]:
        """
        Test the impact of changing an assumption.
        
        Args:
            assumption_set_id: ID of the assumption set
            assumption_id: ID of the assumption to change
            new_value: New assumption content
            
        Returns:
            Analysis of the impact of the change
        """
        if assumption_set_id not in self.assumption_sets:
            return {"error": "Assumption set not found"}
        
        assumption_set = self.assumption_sets[assumption_set_id]
        target_assumption = None
        
        for assumption in assumption_set.assumptions:
            if assumption.assumption_id == assumption_id:
                target_assumption = assumption
                break
        
        if not target_assumption:
            return {"error": "Assumption not found"}
        
        # Find dependent assumptions
        dependent_assumptions = []
        for assumption in assumption_set.assumptions:
            if assumption_id in assumption.dependencies:
                dependent_assumptions.append(assumption.assumption_id)
        
        return {
            "original_assumption": target_assumption.content,
            "new_assumption": new_value,
            "risk_level": target_assumption.risk_level.value,
            "dependent_assumptions": dependent_assumptions,
            "implications": target_assumption.implications,
            "impact_assessment": f"Changing this {target_assumption.risk_level.value}-risk assumption would affect {len(dependent_assumptions)} other assumptions"
        }

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
_assumption_tracker: Optional[AssumptionTracker] = None


def get_assumption_tracker() -> AssumptionTracker:
    """Get singleton assumption tracker instance."""
    global _assumption_tracker
    if _assumption_tracker is None:
        _assumption_tracker = AssumptionTracker()
    return _assumption_tracker