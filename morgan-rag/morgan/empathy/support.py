"""
Crisis detection and support module.

Provides crisis detection capabilities and appropriate support responses for users
experiencing emotional distress, mental health challenges, or crisis situations.
"""

import threading
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import json

from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.services.llm_service import get_llm_service
from morgan.emotional.models import (
    EmotionalState, EmotionType, ConversationContext
)

logger = get_logger(__name__)


class CrisisLevel(Enum):
    """Crisis severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SupportType(Enum):
    """Types of support responses."""
    EMOTIONAL_SUPPORT = "emotional_support"
    CRISIS_INTERVENTION = "crisis_intervention"
    RESOURCE_REFERRAL = "resource_referral"
    SAFETY_PLANNING = "safety_planning"
    PROFESSIONAL_HELP = "professional_help"


class CrisisSupport:
    """
    Crisis detection and support system.
    
    Provides capabilities to:
    - Detect crisis situations and emotional distress
    - Assess crisis severity levels
    - Provide appropriate support responses
    - Offer resources and professional help referrals
    - Create safety plans when needed
    """
    
    # Crisis detection patterns (high-risk indicators)
    CRISIS_PATTERNS = {
        "suicide_risk": [
            r'\b(kill myself|end it all|not worth living|want to die|suicide|take my life)\b',
            r'\b(better off dead|can\'t go on|no point|give up|end the pain)\b',
            r'\b(plan to|thinking about|considering|want to hurt myself)\b'
        ],
        "self_harm": [
            r'\b(hurt myself|cut myself|harm myself|self harm|self-harm)\b',
            r'\b(cutting|burning|hitting myself|punishing myself)\b',
            r'\b(deserve pain|need to hurt|make it stop)\b'
        ],
        "severe_depression": [
            r'\b(hopeless|worthless|useless|failure|burden|hate myself)\b',
            r'\b(nothing matters|empty inside|numb|can\'t feel|dead inside)\b',
            r'\b(no future|no hope|pointless|meaningless)\b'
        ],
        "panic_crisis": [
            r'\b(can\'t breathe|heart racing|panic attack|losing control)\b',
            r'\b(going crazy|losing my mind|can\'t handle|overwhelming)\b',
            r'\b(emergency|help me|crisis|desperate)\b'
        ],
        "substance_crisis": [
            r'\b(overdose|too much|can\'t stop|addiction|relapse)\b',
            r'\b(drinking too much|using again|out of control)\b'
        ],
        "abuse_situation": [
            r'\b(being hurt|someone hurting me|abuse|violence|unsafe)\b',
            r'\b(scared of|threatening me|hitting me|controlling)\b'
        ]
    }
    
    # Support resources by crisis type
    CRISIS_RESOURCES = {
        "suicide_risk": {
            "immediate": [
                "National Suicide Prevention Lifeline: 988 (US)",
                "Crisis Text Line: Text HOME to 741741",
                "International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/"
            ],
            "ongoing": [
                "National Alliance on Mental Illness (NAMI): https://www.nami.org/",
                "Mental Health America: https://www.mhanational.org/",
                "Local mental health services and counselors"
            ]
        },
        "self_harm": {
            "immediate": [
                "Crisis Text Line: Text HOME to 741741",
                "Self-Injury Outreach & Support: http://sioutreach.org/",
                "To Write Love on Her Arms: https://twloha.com/"
            ],
            "ongoing": [
                "Dialectical Behavior Therapy (DBT) resources",
                "Self-harm recovery support groups",
                "Mental health counseling services"
            ]
        },
        "severe_depression": {
            "immediate": [
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
                "SAMHSA National Helpline: 1-800-662-4357"
            ],
            "ongoing": [
                "Depression and Bipolar Support Alliance: https://www.dbsalliance.org/",
                "Therapy and counseling services",
                "Support groups for depression"
            ]
        },
        "panic_crisis": {
            "immediate": [
                "Crisis Text Line: Text HOME to 741741",
                "Anxiety and Depression Association of America: https://adaa.org/",
                "Local emergency services if experiencing severe panic"
            ],
            "ongoing": [
                "Anxiety treatment and therapy",
                "Panic disorder support groups",
                "Mindfulness and relaxation resources"
            ]
        },
        "substance_crisis": {
            "immediate": [
                "SAMHSA National Helpline: 1-800-662-4357",
                "Crisis Text Line: Text HOME to 741741",
                "Local addiction crisis services"
            ],
            "ongoing": [
                "Alcoholics Anonymous: https://www.aa.org/",
                "Narcotics Anonymous: https://www.na.org/",
                "SMART Recovery: https://www.smartrecovery.org/"
            ]
        },
        "abuse_situation": {
            "immediate": [
                "National Domestic Violence Hotline: 1-800-799-7233",
                "Crisis Text Line: Text HOME to 741741",
                "Local law enforcement: 911 (if in immediate danger)"
            ],
            "ongoing": [
                "National Coalition Against Domestic Violence: https://ncadv.org/",
                "Local domestic violence shelters and services",
                "Legal aid and advocacy services"
            ]
        }
    }
    
    def __init__(self):
        """Initialize crisis support system."""
        self.settings = get_settings()
        self.llm_service = get_llm_service()
        
        # Track crisis interactions for follow-up
        self.crisis_history: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("Crisis Support system initialized")
    
    def detect_crisis(
        self,
        text: str,
        emotional_state: EmotionalState,
        context: ConversationContext
    ) -> Tuple[CrisisLevel, List[str]]:
        """
        Detect crisis situation from text and emotional state.
        
        Args:
            text: User's message text
            emotional_state: User's emotional state
            context: Conversation context
            
        Returns:
            Tuple of (crisis_level, detected_crisis_types)
        """
        detected_crises = []
        crisis_scores = {}
        
        text_lower = text.lower()
        
        # Check for crisis patterns
        for crisis_type, patterns in self.CRISIS_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    score += matches * 0.4
            
            if score > 0.3:  # Threshold for crisis detection
                detected_crises.append(crisis_type)
                crisis_scores[crisis_type] = score
        
        # Factor in emotional state
        crisis_level = self._assess_crisis_level(
            detected_crises, crisis_scores, emotional_state, text
        )
        
        # Log crisis detection for monitoring
        if crisis_level != CrisisLevel.NONE:
            self._log_crisis_detection(context.user_id, crisis_level, detected_crises, text)
        
        return crisis_level, detected_crises
    
    def generate_crisis_response(
        self,
        crisis_level: CrisisLevel,
        crisis_types: List[str],
        emotional_state: EmotionalState,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """
        Generate appropriate crisis response.
        
        Args:
            crisis_level: Detected crisis severity level
            crisis_types: Types of crisis detected
            emotional_state: User's emotional state
            context: Conversation context
            
        Returns:
            Crisis response with support message, resources, and actions
        """
        if crisis_level == CrisisLevel.NONE:
            return self._generate_emotional_support(emotional_state, context)
        
        response = {
            "crisis_level": crisis_level.value,
            "crisis_types": crisis_types,
            "support_message": self._generate_crisis_support_message(
                crisis_level, crisis_types, emotional_state
            ),
            "immediate_resources": self._get_immediate_resources(crisis_types),
            "ongoing_resources": self._get_ongoing_resources(crisis_types),
            "safety_suggestions": self._generate_safety_suggestions(crisis_level, crisis_types),
            "follow_up_needed": crisis_level in [CrisisLevel.HIGH, CrisisLevel.CRITICAL],
            "professional_help_recommended": crisis_level in [CrisisLevel.MEDIUM, CrisisLevel.HIGH, CrisisLevel.CRITICAL]
        }
        
        # Add emergency contact info for high-risk situations
        if crisis_level in [CrisisLevel.HIGH, CrisisLevel.CRITICAL]:
            response["emergency_contacts"] = self._get_emergency_contacts()
        
        # Store crisis interaction for follow-up
        self._store_crisis_interaction(context.user_id, response)
        
        return response
    
    def create_safety_plan(
        self,
        crisis_types: List[str],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a personalized safety plan.
        
        Args:
            crisis_types: Types of crisis to address
            user_context: User's personal context and preferences
            
        Returns:
            Personalized safety plan
        """
        safety_plan = {
            "warning_signs": self._identify_warning_signs(crisis_types),
            "coping_strategies": self._suggest_coping_strategies(crisis_types),
            "support_contacts": self._suggest_support_contacts(),
            "professional_contacts": self._get_professional_contacts(crisis_types),
            "safe_environment": self._suggest_environment_safety(crisis_types),
            "emergency_plan": self._create_emergency_plan(crisis_types),
            "daily_wellness": self._suggest_daily_wellness_activities()
        }
        
        return safety_plan
    
    def assess_ongoing_risk(
        self,
        user_id: str,
        timeframe_days: int = 7
    ) -> Dict[str, Any]:
        """
        Assess ongoing crisis risk for a user.
        
        Args:
            user_id: User identifier
            timeframe_days: Days to look back for assessment
            
        Returns:
            Risk assessment with recommendations
        """
        if user_id not in self.crisis_history:
            return {"risk_level": "low", "recommendations": []}
        
        recent_crises = [
            crisis for crisis in self.crisis_history[user_id]
            if (datetime.utcnow() - crisis["timestamp"]).days <= timeframe_days
        ]
        
        if not recent_crises:
            return {"risk_level": "low", "recommendations": ["Continue regular check-ins"]}
        
        # Analyze patterns
        crisis_levels = [crisis["crisis_level"] for crisis in recent_crises]
        crisis_types = []
        for crisis in recent_crises:
            crisis_types.extend(crisis["crisis_types"])
        
        # Assess risk
        risk_assessment = {
            "risk_level": self._calculate_ongoing_risk_level(crisis_levels),
            "frequent_crisis_types": self._get_frequent_crisis_types(crisis_types),
            "pattern_analysis": self._analyze_crisis_patterns(recent_crises),
            "recommendations": self._generate_risk_recommendations(crisis_levels, crisis_types),
            "follow_up_frequency": self._suggest_follow_up_frequency(crisis_levels)
        }
        
        return risk_assessment
    
    def _assess_crisis_level(
        self,
        detected_crises: List[str],
        crisis_scores: Dict[str, float],
        emotional_state: EmotionalState,
        text: str
    ) -> CrisisLevel:
        """Assess overall crisis level."""
        if not detected_crises:
            # Check for high emotional distress without explicit crisis language
            if (emotional_state.primary_emotion in [EmotionType.SADNESS, EmotionType.FEAR, EmotionType.ANGER] 
                and emotional_state.intensity > 0.8):
                return CrisisLevel.LOW
            return CrisisLevel.NONE
        
        # High-risk crisis types
        high_risk_types = ["suicide_risk", "self_harm", "abuse_situation"]
        if any(crisis_type in high_risk_types for crisis_type in detected_crises):
            max_score = max(crisis_scores.values())
            if max_score > 0.8:
                return CrisisLevel.CRITICAL
            elif max_score > 0.6:
                return CrisisLevel.HIGH
            else:
                return CrisisLevel.MEDIUM
        
        # Medium-risk crisis types
        medium_risk_types = ["severe_depression", "panic_crisis", "substance_crisis"]
        if any(crisis_type in medium_risk_types for crisis_type in detected_crises):
            max_score = max(crisis_scores.values())
            if max_score > 0.7:
                return CrisisLevel.HIGH
            elif max_score > 0.5:
                return CrisisLevel.MEDIUM
            else:
                return CrisisLevel.LOW
        
        return CrisisLevel.LOW
    
    def _generate_crisis_support_message(
        self,
        crisis_level: CrisisLevel,
        crisis_types: List[str],
        emotional_state: EmotionalState
    ) -> str:
        """Generate appropriate crisis support message."""
        if crisis_level == CrisisLevel.CRITICAL:
            return (
                "I'm very concerned about what you're going through right now. "
                "Your safety and wellbeing are the most important things. "
                "Please reach out to a crisis helpline immediately - they have trained professionals "
                "who can provide the support you need right now. You don't have to face this alone."
            )
        
        elif crisis_level == CrisisLevel.HIGH:
            return (
                "I can see you're going through an incredibly difficult time, and I'm worried about you. "
                "What you're experiencing sounds overwhelming, and it's important that you get proper support. "
                "Please consider reaching out to a mental health professional or crisis service. "
                "You deserve help and support through this."
            )
        
        elif crisis_level == CrisisLevel.MEDIUM:
            return (
                "I can sense you're struggling with some really difficult feelings and situations. "
                "It takes courage to share what you're going through. "
                "While I'm here to listen and support you, I think it would be helpful for you "
                "to connect with a mental health professional who can provide more comprehensive support."
            )
        
        else:  # LOW
            return (
                "I can see you're going through a tough time, and I want you to know that your feelings are valid. "
                "It's okay to not be okay sometimes. I'm here to support you, and there are also "
                "resources available if you need additional help."
            )
    
    def _get_immediate_resources(self, crisis_types: List[str]) -> List[str]:
        """Get immediate crisis resources."""
        resources = set()
        
        for crisis_type in crisis_types:
            if crisis_type in self.CRISIS_RESOURCES:
                resources.update(self.CRISIS_RESOURCES[crisis_type]["immediate"])
        
        # Add general crisis resources if none specific
        if not resources:
            resources.update([
                "Crisis Text Line: Text HOME to 741741",
                "National Suicide Prevention Lifeline: 988",
                "SAMHSA National Helpline: 1-800-662-4357"
            ])
        
        return list(resources)
    
    def _get_ongoing_resources(self, crisis_types: List[str]) -> List[str]:
        """Get ongoing support resources."""
        resources = set()
        
        for crisis_type in crisis_types:
            if crisis_type in self.CRISIS_RESOURCES:
                resources.update(self.CRISIS_RESOURCES[crisis_type]["ongoing"])
        
        # Add general ongoing resources
        resources.update([
            "Psychology Today therapist finder: https://www.psychologytoday.com/",
            "Local community mental health centers",
            "Employee Assistance Programs (if available through work)"
        ])
        
        return list(resources)
    
    def _generate_safety_suggestions(
        self,
        crisis_level: CrisisLevel,
        crisis_types: List[str]
    ) -> List[str]:
        """Generate safety suggestions based on crisis type and level."""
        suggestions = []
        
        if crisis_level in [CrisisLevel.HIGH, CrisisLevel.CRITICAL]:
            suggestions.extend([
                "Stay with someone you trust or call someone to be with you",
                "Remove any means of self-harm from your immediate environment",
                "Go to a safe place where you feel secure",
                "Call a crisis helpline or emergency services if you feel unsafe"
            ])
        
        # Crisis-specific suggestions
        if "suicide_risk" in crisis_types or "self_harm" in crisis_types:
            suggestions.extend([
                "Reach out to a trusted friend, family member, or mental health professional",
                "Use coping strategies that have helped you before",
                "Remember that these intense feelings are temporary"
            ])
        
        if "panic_crisis" in crisis_types:
            suggestions.extend([
                "Practice deep breathing: breathe in for 4, hold for 4, out for 4",
                "Ground yourself: name 5 things you can see, 4 you can touch, 3 you can hear",
                "Remind yourself that panic attacks are temporary and will pass"
            ])
        
        if "abuse_situation" in crisis_types:
            suggestions.extend([
                "If you're in immediate danger, call 911",
                "Reach out to domestic violence resources for safety planning",
                "Trust your instincts about your safety"
            ])
        
        return suggestions
    
    def _get_emergency_contacts(self) -> Dict[str, str]:
        """Get emergency contact information."""
        return {
            "National Suicide Prevention Lifeline": "988",
            "Crisis Text Line": "Text HOME to 741741",
            "Emergency Services": "911",
            "SAMHSA National Helpline": "1-800-662-4357",
            "National Domestic Violence Hotline": "1-800-799-7233"
        }
    
    def _log_crisis_detection(
        self,
        user_id: str,
        crisis_level: CrisisLevel,
        crisis_types: List[str],
        text: str
    ):
        """Log crisis detection for monitoring and follow-up."""
        logger.warning(
            f"Crisis detected for user {user_id}: "
            f"Level={crisis_level.value}, Types={crisis_types}"
        )
        
        # In a production system, this would also:
        # - Alert human moderators for high-risk cases
        # - Store in secure database for follow-up
        # - Trigger automated safety protocols
    
    def _store_crisis_interaction(self, user_id: str, response: Dict[str, Any]):
        """Store crisis interaction for follow-up tracking."""
        if user_id not in self.crisis_history:
            self.crisis_history[user_id] = []
        
        interaction = {
            "timestamp": datetime.utcnow(),
            "crisis_level": response["crisis_level"],
            "crisis_types": response["crisis_types"],
            "resources_provided": len(response["immediate_resources"]) + len(response["ongoing_resources"]),
            "follow_up_needed": response["follow_up_needed"]
        }
        
        self.crisis_history[user_id].append(interaction)
        
        # Keep only recent history (last 30 days)
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        self.crisis_history[user_id] = [
            interaction for interaction in self.crisis_history[user_id]
            if interaction["timestamp"] >= cutoff_date
        ]
    
    def _generate_emotional_support(
        self,
        emotional_state: EmotionalState,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Generate emotional support for non-crisis situations."""
        return {
            "crisis_level": "none",
            "crisis_types": [],
            "support_message": self._generate_general_support_message(emotional_state),
            "coping_suggestions": self._get_general_coping_suggestions(emotional_state),
            "resources": self._get_general_mental_health_resources(),
            "follow_up_needed": False,
            "professional_help_recommended": emotional_state.intensity > 0.7
        }
    
    def _generate_general_support_message(self, emotional_state: EmotionalState) -> str:
        """Generate general emotional support message."""
        support_messages = {
            EmotionType.SADNESS: (
                "I can see you're going through a difficult time. "
                "It's okay to feel sad, and you don't have to go through this alone. "
                "I'm here to listen and support you."
            ),
            EmotionType.ANGER: (
                "I can sense your frustration and anger. "
                "These feelings are valid, and it's important to find healthy ways to process them. "
                "I'm here to help you work through this."
            ),
            EmotionType.FEAR: (
                "I can feel your worry and concern. "
                "It's natural to feel anxious when facing uncertainty. "
                "You're stronger than you know, and I'm here to support you."
            )
        }
        
        return support_messages.get(
            emotional_state.primary_emotion,
            "I can see you're experiencing some difficult emotions. "
            "Your feelings are valid, and I'm here to support you through this."
        )
    
    def _get_general_coping_suggestions(self, emotional_state: EmotionalState) -> List[str]:
        """Get general coping suggestions."""
        general_suggestions = [
            "Take some deep breaths and try to ground yourself in the present moment",
            "Reach out to someone you trust to talk about what you're experiencing",
            "Practice self-care activities that usually help you feel better",
            "Remember that difficult emotions are temporary and will pass"
        ]
        
        emotion_specific = {
            EmotionType.SADNESS: [
                "Allow yourself to feel the sadness without judgment",
                "Consider gentle activities like taking a walk or listening to music"
            ],
            EmotionType.ANGER: [
                "Try physical exercise or other healthy outlets for your energy",
                "Consider what boundary or value might have been crossed"
            ],
            EmotionType.FEAR: [
                "Focus on what you can control in the situation",
                "Break down overwhelming problems into smaller, manageable steps"
            ]
        }
        
        suggestions = general_suggestions.copy()
        suggestions.extend(emotion_specific.get(emotional_state.primary_emotion, []))
        
        return suggestions
    
    def _get_general_mental_health_resources(self) -> List[str]:
        """Get general mental health resources."""
        return [
            "National Alliance on Mental Illness (NAMI): https://www.nami.org/",
            "Mental Health America: https://www.mhanational.org/",
            "Psychology Today therapist finder: https://www.psychologytoday.com/",
            "Crisis Text Line: Text HOME to 741741 (for crisis support)",
            "SAMHSA National Helpline: 1-800-662-4357"
        ]
    
    def _identify_warning_signs(self, crisis_types: List[str]) -> List[str]:
        """Identify warning signs for safety planning."""
        warning_signs = {
            "suicide_risk": [
                "Thoughts of death or dying",
                "Feeling hopeless or worthless",
                "Withdrawing from friends and family",
                "Giving away possessions"
            ],
            "self_harm": [
                "Urges to hurt yourself",
                "Feeling overwhelmed by emotions",
                "Isolation from others",
                "Increased stress or anxiety"
            ],
            "severe_depression": [
                "Persistent sadness or emptiness",
                "Loss of interest in activities",
                "Changes in sleep or appetite",
                "Difficulty concentrating"
            ]
        }
        
        signs = set()
        for crisis_type in crisis_types:
            if crisis_type in warning_signs:
                signs.update(warning_signs[crisis_type])
        
        return list(signs)
    
    def _suggest_coping_strategies(self, crisis_types: List[str]) -> List[str]:
        """Suggest coping strategies for safety planning."""
        strategies = {
            "suicide_risk": [
                "Call a trusted friend or family member",
                "Use grounding techniques (5-4-3-2-1 method)",
                "Engage in physical activity",
                "Practice mindfulness or meditation"
            ],
            "self_harm": [
                "Hold ice cubes in your hands",
                "Draw on your skin with a red marker",
                "Do intense exercise",
                "Call a crisis helpline"
            ],
            "panic_crisis": [
                "Practice deep breathing exercises",
                "Use progressive muscle relaxation",
                "Ground yourself with sensory techniques",
                "Remind yourself that panic attacks are temporary"
            ]
        }
        
        coping_strategies = set()
        for crisis_type in crisis_types:
            if crisis_type in strategies:
                coping_strategies.update(strategies[crisis_type])
        
        # Add general strategies
        coping_strategies.update([
            "Listen to calming music",
            "Take a warm bath or shower",
            "Write in a journal",
            "Practice gratitude exercises"
        ])
        
        return list(coping_strategies)
    
    def _suggest_support_contacts(self) -> List[str]:
        """Suggest support contacts for safety planning."""
        return [
            "Trusted friend or family member",
            "Mental health counselor or therapist",
            "Primary care doctor",
            "Spiritual or religious leader",
            "Crisis helpline counselor"
        ]
    
    def _get_professional_contacts(self, crisis_types: List[str]) -> List[str]:
        """Get professional contacts for safety planning."""
        contacts = [
            "Local mental health crisis team",
            "Emergency room or urgent care",
            "Primary care physician",
            "Mental health counselor or therapist"
        ]
        
        # Add specific contacts based on crisis type
        if "substance_crisis" in crisis_types:
            contacts.append("Addiction counselor or treatment center")
        
        if "abuse_situation" in crisis_types:
            contacts.extend([
                "Domestic violence advocate",
                "Legal aid attorney",
                "Law enforcement (if safe to contact)"
            ])
        
        return contacts
    
    def _suggest_environment_safety(self, crisis_types: List[str]) -> List[str]:
        """Suggest environment safety measures."""
        safety_measures = [
            "Remove or secure any means of self-harm",
            "Stay in areas where you feel safe and supported",
            "Avoid isolation - stay around trusted people when possible"
        ]
        
        if "substance_crisis" in crisis_types:
            safety_measures.extend([
                "Remove alcohol and substances from your environment",
                "Avoid places and people associated with substance use"
            ])
        
        if "abuse_situation" in crisis_types:
            safety_measures.extend([
                "Identify safe places you can go quickly",
                "Keep important documents and emergency money accessible",
                "Have a safety plan for leaving if necessary"
            ])
        
        return safety_measures
    
    def _create_emergency_plan(self, crisis_types: List[str]) -> List[str]:
        """Create emergency plan steps."""
        plan = [
            "Call crisis helpline: 988 or text HOME to 741741",
            "Contact trusted person from your support list",
            "Go to emergency room if in immediate danger",
            "Use coping strategies from your safety plan"
        ]
        
        if "abuse_situation" in crisis_types:
            plan.insert(0, "Call 911 if in immediate physical danger")
        
        return plan
    
    def _suggest_daily_wellness_activities(self) -> List[str]:
        """Suggest daily wellness activities."""
        return [
            "Maintain regular sleep schedule",
            "Eat nutritious meals regularly",
            "Engage in physical activity",
            "Practice mindfulness or meditation",
            "Connect with supportive people",
            "Engage in meaningful activities",
            "Limit alcohol and substance use",
            "Take prescribed medications as directed"
        ]
    
    def _calculate_ongoing_risk_level(self, crisis_levels: List[str]) -> str:
        """Calculate ongoing risk level from recent crises."""
        if "critical" in crisis_levels:
            return "high"
        elif "high" in crisis_levels:
            return "medium-high"
        elif "medium" in crisis_levels:
            return "medium"
        elif "low" in crisis_levels:
            return "low-medium"
        else:
            return "low"
    
    def _get_frequent_crisis_types(self, crisis_types: List[str]) -> List[str]:
        """Get most frequent crisis types."""
        from collections import Counter
        crisis_counts = Counter(crisis_types)
        return [crisis_type for crisis_type, count in crisis_counts.most_common(3)]
    
    def _analyze_crisis_patterns(self, recent_crises: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in recent crises."""
        if len(recent_crises) < 2:
            return {"pattern": "insufficient_data"}
        
        # Simple pattern analysis
        crisis_frequency = len(recent_crises)
        time_span = (recent_crises[-1]["timestamp"] - recent_crises[0]["timestamp"]).days
        
        if time_span > 0:
            frequency_per_week = (crisis_frequency / time_span) * 7
        else:
            frequency_per_week = crisis_frequency
        
        if frequency_per_week > 2:
            pattern = "high_frequency"
        elif frequency_per_week > 1:
            pattern = "moderate_frequency"
        else:
            pattern = "low_frequency"
        
        return {
            "pattern": pattern,
            "frequency_per_week": frequency_per_week,
            "total_incidents": crisis_frequency,
            "time_span_days": time_span
        }
    
    def _generate_risk_recommendations(
        self,
        crisis_levels: List[str],
        crisis_types: List[str]
    ) -> List[str]:
        """Generate recommendations based on risk assessment."""
        recommendations = []
        
        if "critical" in crisis_levels or "high" in crisis_levels:
            recommendations.extend([
                "Immediate professional mental health evaluation recommended",
                "Consider intensive outpatient or inpatient treatment",
                "Daily check-ins with mental health professional",
                "Safety planning with crisis team"
            ])
        elif "medium" in crisis_levels:
            recommendations.extend([
                "Regular therapy sessions recommended",
                "Weekly check-ins with mental health professional",
                "Medication evaluation if not already done",
                "Support group participation"
            ])
        else:
            recommendations.extend([
                "Regular mental health maintenance",
                "Bi-weekly or monthly therapy sessions",
                "Stress management and coping skills development"
            ])
        
        return recommendations
    
    def _suggest_follow_up_frequency(self, crisis_levels: List[str]) -> str:
        """Suggest follow-up frequency based on crisis levels."""
        if "critical" in crisis_levels:
            return "daily"
        elif "high" in crisis_levels:
            return "every_2_days"
        elif "medium" in crisis_levels:
            return "weekly"
        else:
            return "bi_weekly"


# Singleton instance
_crisis_support_instance = None
_crisis_support_lock = threading.Lock()


def get_crisis_support() -> CrisisSupport:
    """
    Get singleton crisis support instance.
    
    Returns:
        Shared CrisisSupport instance
    """
    global _crisis_support_instance
    
    if _crisis_support_instance is None:
        with _crisis_support_lock:
            if _crisis_support_instance is None:
                _crisis_support_instance = CrisisSupport()
    
    return _crisis_support_instance