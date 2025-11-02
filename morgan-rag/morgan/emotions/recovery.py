"""
Emotional recovery tracking module.

Provides focused emotional recovery monitoring, resilience assessment,
and recovery pattern analysis for enhanced emotional support and user well-being.
"""

import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import statistics

from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.emotions.memory import get_emotional_memory_storage, EmotionalMemory
from morgan.emotional.models import EmotionalState, EmotionType

logger = get_logger(__name__)


class EmotionalRecovery:
    """
    Represents an emotional recovery instance.
    
    Features:
    - Recovery timeline tracking
    - Resilience measurement
    - Recovery pattern identification
    - Support effectiveness assessment
    """
    
    def __init__(
        self,
        recovery_id: str,
        user_id: str,
        initial_state: EmotionalState,
        recovery_state: EmotionalState,
        recovery_duration: timedelta,
        recovery_type: str,
        support_factors: List[str],
        recovery_quality: float
    ):
        """Initialize emotional recovery instance."""
        self.recovery_id = recovery_id
        self.user_id = user_id
        self.initial_state = initial_state
        self.recovery_state = recovery_state
        self.recovery_duration = recovery_duration
        self.recovery_type = recovery_type
        self.support_factors = support_factors
        self.recovery_quality = recovery_quality
        self.detected_at = datetime.utcnow()
        self.intermediate_states = []
        self.recovery_milestones = []
    
    def add_intermediate_state(self, emotional_state: EmotionalState):
        """Add intermediate emotional state during recovery."""
        self.intermediate_states.append(emotional_state)
    
    def add_recovery_milestone(self, milestone: str, timestamp: datetime):
        """Add recovery milestone."""
        self.recovery_milestones.append({
            'milestone': milestone,
            'timestamp': timestamp,
            'time_from_start': timestamp - self.initial_state.timestamp
        })
    
    def calculate_recovery_effectiveness(self) -> float:
        """
        Calculate recovery effectiveness score.
        
        Returns:
            Recovery effectiveness (0.0 to 1.0)
        """
        # Base effectiveness on emotional improvement
        initial_valence = self._get_emotional_valence(self.initial_state.primary_emotion)
        recovery_valence = self._get_emotional_valence(self.recovery_state.primary_emotion)
        
        valence_improvement = recovery_valence - initial_valence
        intensity_improvement = self.initial_state.intensity - self.recovery_state.intensity
        
        # Normalize improvements
        valence_score = max(0.0, min(1.0, (valence_improvement + 2.0) / 4.0))
        intensity_score = max(0.0, min(1.0, intensity_improvement))
        
        # Consider recovery speed (faster recovery is better, up to a point)
        hours = self.recovery_duration.total_seconds() / 3600
        speed_score = max(0.1, min(1.0, 24 / max(1, hours)))  # Optimal around 24 hours
        
        # Combine scores
        effectiveness = (valence_score * 0.4) + (intensity_score * 0.4) + (speed_score * 0.2)
        return min(1.0, effectiveness)
    
    def _get_emotional_valence(self, emotion: EmotionType) -> float:
        """Get emotional valence score (-2.0 to 2.0)."""
        valence_map = {
            EmotionType.JOY: 2.0,
            EmotionType.SURPRISE: 0.5,
            EmotionType.NEUTRAL: 0.0,
            EmotionType.FEAR: -1.0,
            EmotionType.DISGUST: -1.2,
            EmotionType.SADNESS: -1.5,
            EmotionType.ANGER: -1.8
        }
        return valence_map.get(emotion, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert recovery to dictionary for storage."""
        return {
            'recovery_id': self.recovery_id,
            'user_id': self.user_id,
            'initial_state': {
                'primary_emotion': self.initial_state.primary_emotion.value,
                'intensity': self.initial_state.intensity,
                'timestamp': self.initial_state.timestamp.isoformat()
            },
            'recovery_state': {
                'primary_emotion': self.recovery_state.primary_emotion.value,
                'intensity': self.recovery_state.intensity,
                'timestamp': self.recovery_state.timestamp.isoformat()
            },
            'recovery_duration_seconds': self.recovery_duration.total_seconds(),
            'recovery_type': self.recovery_type,
            'support_factors': self.support_factors,
            'recovery_quality': self.recovery_quality,
            'detected_at': self.detected_at.isoformat(),
            'intermediate_states': [
                {
                    'primary_emotion': state.primary_emotion.value,
                    'intensity': state.intensity,
                    'timestamp': state.timestamp.isoformat()
                }
                for state in self.intermediate_states
            ],
            'recovery_milestones': self.recovery_milestones
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalRecovery':
        """Create recovery from dictionary."""
        initial_state = EmotionalState(
            primary_emotion=EmotionType(data['initial_state']['primary_emotion']),
            intensity=data['initial_state']['intensity'],
            confidence=0.8,  # Default confidence
            timestamp=datetime.fromisoformat(data['initial_state']['timestamp'])
        )
        
        recovery_state = EmotionalState(
            primary_emotion=EmotionType(data['recovery_state']['primary_emotion']),
            intensity=data['recovery_state']['intensity'],
            confidence=0.8,  # Default confidence
            timestamp=datetime.fromisoformat(data['recovery_state']['timestamp'])
        )
        
        recovery = cls(
            recovery_id=data['recovery_id'],
            user_id=data['user_id'],
            initial_state=initial_state,
            recovery_state=recovery_state,
            recovery_duration=timedelta(seconds=data['recovery_duration_seconds']),
            recovery_type=data['recovery_type'],
            support_factors=data['support_factors'],
            recovery_quality=data['recovery_quality']
        )
        
        # Restore metadata
        recovery.detected_at = datetime.fromisoformat(data['detected_at'])
        recovery.recovery_milestones = data['recovery_milestones']
        
        # Restore intermediate states
        for state_data in data['intermediate_states']:
            state = EmotionalState(
                primary_emotion=EmotionType(state_data['primary_emotion']),
                intensity=state_data['intensity'],
                confidence=0.8,
                timestamp=datetime.fromisoformat(state_data['timestamp'])
            )
            recovery.intermediate_states.append(state)
        
        return recovery


class EmotionalRecoveryTracker:
    """
    Tracks and analyzes emotional recovery patterns.
    
    Features:
    - Recovery detection and monitoring
    - Resilience assessment
    - Recovery pattern analysis
    - Support strategy effectiveness
    """
    
    # Recovery type definitions
    RECOVERY_TYPES = {
        'natural': 'Natural recovery without intervention',
        'supported': 'Recovery with emotional support',
        'distraction': 'Recovery through distraction/activity',
        'cognitive': 'Recovery through cognitive reframing',
        'social': 'Recovery through social interaction',
        'time_based': 'Recovery through passage of time'
    }
    
    def __init__(self):
        """Initialize emotional recovery tracker."""
        self.settings = get_settings()
        self.memory_storage = get_emotional_memory_storage()
        
        # Recovery storage
        self._detected_recoveries = {}  # user_id -> List[EmotionalRecovery]
        self._recovery_cache = {}
        
        logger.info("Emotional Recovery Tracker initialized")
    
    def detect_recoveries(
        self,
        user_id: str,
        analysis_days: int = 30,
        min_recovery_improvement: float = 0.3
    ) -> List[EmotionalRecovery]:
        """
        Detect emotional recovery instances for a user.
        
        Args:
            user_id: User identifier
            analysis_days: Days of history to analyze
            min_recovery_improvement: Minimum improvement to consider recovery
            
        Returns:
            List of detected emotional recoveries
        """
        # Retrieve emotional memories for analysis
        memories = self.memory_storage.retrieve_memories(
            user_id=user_id,
            max_age_days=analysis_days,
            min_importance=0.2,
            limit=200
        )
        
        if len(memories) < 10:
            logger.debug(f"Insufficient data for recovery detection: {len(memories)} memories")
            return []
        
        # Sort memories by timestamp
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        
        detected_recoveries = []
        
        # Look for recovery patterns
        negative_emotions = {EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR}
        positive_emotions = {EmotionType.JOY, EmotionType.SURPRISE}
        neutral_emotions = {EmotionType.NEUTRAL}
        
        i = 0
        while i < len(sorted_memories) - 1:
            current_memory = sorted_memories[i]
            current_emotion = current_memory.emotional_state.primary_emotion
            
            # Look for negative emotional states
            if current_emotion in negative_emotions:
                recovery = self._analyze_recovery_sequence(
                    sorted_memories, i, min_recovery_improvement
                )
                if recovery:
                    detected_recoveries.append(recovery)
                    # Skip ahead to avoid overlapping recoveries
                    i += max(1, len(recovery.intermediate_states))
                else:
                    i += 1
            else:
                i += 1
        
        # Store recoveries for user
        self._detected_recoveries[user_id] = detected_recoveries
        
        logger.info(f"Detected {len(detected_recoveries)} emotional recoveries for user {user_id}")
        return detected_recoveries
    
    def get_user_recoveries(
        self,
        user_id: str,
        recovery_types: Optional[List[str]] = None,
        min_effectiveness: float = 0.0
    ) -> List[EmotionalRecovery]:
        """
        Get detected recoveries for a user.
        
        Args:
            user_id: User identifier
            recovery_types: Optional filter by recovery types
            min_effectiveness: Minimum recovery effectiveness
            
        Returns:
            List of user's emotional recoveries
        """
        user_recoveries = self._detected_recoveries.get(user_id, [])
        
        # Filter by recovery types
        if recovery_types:
            user_recoveries = [
                r for r in user_recoveries
                if r.recovery_type in recovery_types
            ]
        
        # Filter by effectiveness
        if min_effectiveness > 0.0:
            user_recoveries = [
                r for r in user_recoveries
                if r.calculate_recovery_effectiveness() >= min_effectiveness
            ]
        
        return user_recoveries
    
    def analyze_recovery_patterns(
        self,
        user_id: str,
        timeframe_days: int = 60
    ) -> Dict[str, Any]:
        """
        Analyze recovery patterns for a user.
        
        Args:
            user_id: User identifier
            timeframe_days: Analysis timeframe in days
            
        Returns:
            Recovery pattern analysis
        """
        recoveries = self.get_user_recoveries(user_id)
        
        if not recoveries:
            return {'pattern': 'insufficient_recovery_data'}
        
        # Calculate recovery metrics
        recovery_times = [r.recovery_duration.total_seconds() / 3600 for r in recoveries]  # in hours
        effectiveness_scores = [r.calculate_recovery_effectiveness() for r in recoveries]
        recovery_types = [r.recovery_type for r in recoveries]
        
        # Recovery time statistics
        avg_recovery_time = statistics.mean(recovery_times)
        median_recovery_time = statistics.median(recovery_times)
        recovery_time_variance = statistics.variance(recovery_times) if len(recovery_times) > 1 else 0
        
        # Effectiveness statistics
        avg_effectiveness = statistics.mean(effectiveness_scores)
        effectiveness_trend = self._calculate_effectiveness_trend(recoveries)
        
        # Recovery type analysis
        type_counts = {}
        for recovery_type in recovery_types:
            type_counts[recovery_type] = type_counts.get(recovery_type, 0) + 1
        
        most_effective_type = self._find_most_effective_recovery_type(recoveries)
        
        # Resilience assessment
        resilience_score = self._calculate_resilience_score(recoveries)
        
        # Support factor analysis
        support_effectiveness = self._analyze_support_effectiveness(recoveries)
        
        return {
            'total_recoveries': len(recoveries),
            'average_recovery_time_hours': avg_recovery_time,
            'median_recovery_time_hours': median_recovery_time,
            'recovery_time_consistency': max(0.0, 1.0 - (recovery_time_variance / max(1, avg_recovery_time))),
            'average_effectiveness': avg_effectiveness,
            'effectiveness_trend': effectiveness_trend,
            'recovery_type_distribution': type_counts,
            'most_effective_recovery_type': most_effective_type,
            'resilience_score': resilience_score,
            'support_effectiveness': support_effectiveness,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    def assess_current_recovery_potential(
        self,
        user_id: str,
        current_emotional_state: EmotionalState
    ) -> Dict[str, Any]:
        """
        Assess recovery potential for current emotional state.
        
        Args:
            user_id: User identifier
            current_emotional_state: Current emotional state
            
        Returns:
            Recovery potential assessment
        """
        user_recoveries = self.get_user_recoveries(user_id)
        
        if not user_recoveries:
            return {
                'recovery_potential': 'unknown',
                'confidence': 0.0,
                'recommendations': ['Insufficient recovery history for assessment']
            }
        
        # Find similar past recovery situations
        similar_recoveries = []
        for recovery in user_recoveries:
            if recovery.initial_state.primary_emotion == current_emotional_state.primary_emotion:
                intensity_diff = abs(recovery.initial_state.intensity - current_emotional_state.intensity)
                if intensity_diff <= 0.3:  # Similar intensity
                    similar_recoveries.append(recovery)
        
        if not similar_recoveries:
            return {
                'recovery_potential': 'moderate',
                'confidence': 0.3,
                'recommendations': ['No similar recovery patterns found']
            }
        
        # Analyze similar recoveries
        avg_effectiveness = statistics.mean([
            r.calculate_recovery_effectiveness() for r in similar_recoveries
        ])
        avg_recovery_time = statistics.mean([
            r.recovery_duration.total_seconds() / 3600 for r in similar_recoveries
        ])
        
        # Determine recovery potential
        if avg_effectiveness > 0.7:
            potential = 'high'
        elif avg_effectiveness > 0.4:
            potential = 'moderate'
        else:
            potential = 'low'
        
        # Generate recommendations
        recommendations = self._generate_recovery_recommendations(similar_recoveries)
        
        return {
            'recovery_potential': potential,
            'confidence': min(1.0, len(similar_recoveries) / 5.0),
            'predicted_recovery_time_hours': avg_recovery_time,
            'predicted_effectiveness': avg_effectiveness,
            'similar_recoveries_count': len(similar_recoveries),
            'recommendations': recommendations
        }
    
    def track_ongoing_recovery(
        self,
        user_id: str,
        initial_state: EmotionalState,
        current_state: EmotionalState,
        support_factors: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Track an ongoing recovery process.
        
        Args:
            user_id: User identifier
            initial_state: Initial emotional state
            current_state: Current emotional state
            support_factors: Support factors being applied
            
        Returns:
            Recovery tracking information
        """
        # Check if this represents recovery progress
        initial_valence = self._get_emotional_valence(initial_state.primary_emotion)
        current_valence = self._get_emotional_valence(current_state.primary_emotion)
        
        valence_improvement = current_valence - initial_valence
        intensity_improvement = initial_state.intensity - current_state.intensity
        
        if valence_improvement > 0.2 or intensity_improvement > 0.2:
            # Recovery in progress
            recovery_duration = current_state.timestamp - initial_state.timestamp
            
            # Estimate recovery quality
            recovery_quality = self._estimate_recovery_quality(
                initial_state, current_state, recovery_duration
            )
            
            return {
                'recovery_in_progress': True,
                'recovery_duration_hours': recovery_duration.total_seconds() / 3600,
                'valence_improvement': valence_improvement,
                'intensity_improvement': intensity_improvement,
                'recovery_quality': recovery_quality,
                'support_factors': support_factors,
                'next_check_recommended_hours': 2.0
            }
        
        return None
    
    def _analyze_recovery_sequence(
        self,
        memories: List[EmotionalMemory],
        start_index: int,
        min_improvement: float
    ) -> Optional[EmotionalRecovery]:
        """Analyze a potential recovery sequence starting from a negative state."""
        initial_memory = memories[start_index]
        initial_state = initial_memory.emotional_state
        
        # Look ahead for recovery
        max_look_ahead = min(10, len(memories) - start_index - 1)
        
        for i in range(1, max_look_ahead + 1):
            if start_index + i >= len(memories):
                break
            
            current_memory = memories[start_index + i]
            current_state = current_memory.emotional_state
            
            # Check for recovery
            initial_valence = self._get_emotional_valence(initial_state.primary_emotion)
            current_valence = self._get_emotional_valence(current_state.primary_emotion)
            
            valence_improvement = current_valence - initial_valence
            intensity_improvement = initial_state.intensity - current_state.intensity
            
            # Check if this represents sufficient recovery
            if valence_improvement >= min_improvement or intensity_improvement >= min_improvement:
                recovery_duration = current_state.timestamp - initial_state.timestamp
                
                # Determine recovery type
                recovery_type = self._classify_recovery_type(
                    memories[start_index:start_index + i + 1]
                )
                
                # Extract support factors
                support_factors = self._extract_support_factors(
                    memories[start_index:start_index + i + 1]
                )
                
                # Calculate recovery quality
                recovery_quality = self._estimate_recovery_quality(
                    initial_state, current_state, recovery_duration
                )
                
                # Create recovery instance
                recovery_id = f"{initial_memory.user_id}_{initial_state.timestamp.isoformat()}"
                
                recovery = EmotionalRecovery(
                    recovery_id=recovery_id,
                    user_id=initial_memory.user_id,
                    initial_state=initial_state,
                    recovery_state=current_state,
                    recovery_duration=recovery_duration,
                    recovery_type=recovery_type,
                    support_factors=support_factors,
                    recovery_quality=recovery_quality
                )
                
                # Add intermediate states
                for j in range(1, i):
                    if start_index + j < len(memories):
                        intermediate_state = memories[start_index + j].emotional_state
                        recovery.add_intermediate_state(intermediate_state)
                
                return recovery
        
        return None
    
    def _classify_recovery_type(self, memory_sequence: List[EmotionalMemory]) -> str:
        """Classify the type of recovery based on memory sequence."""
        # Simple heuristic-based classification
        
        # Check for support-related keywords
        all_content = ' '.join([
            m.conversation_context.message_text.lower()
            for m in memory_sequence
        ])
        
        if any(word in all_content for word in ['help', 'support', 'talk', 'listen', 'understand']):
            return 'supported'
        elif any(word in all_content for word in ['distract', 'activity', 'do something', 'busy']):
            return 'distraction'
        elif any(word in all_content for word in ['think', 'realize', 'understand', 'perspective']):
            return 'cognitive'
        elif any(word in all_content for word in ['friend', 'family', 'social', 'together']):
            return 'social'
        elif len(memory_sequence) > 5:  # Long recovery suggests time-based
            return 'time_based'
        else:
            return 'natural'
    
    def _extract_support_factors(self, memory_sequence: List[EmotionalMemory]) -> List[str]:
        """Extract support factors from memory sequence."""
        support_factors = []
        
        all_content = ' '.join([
            m.conversation_context.message_text.lower()
            for m in memory_sequence
        ])
        
        # Check for various support factors
        if 'empathy' in all_content or 'understand' in all_content:
            support_factors.append('empathetic_response')
        if 'advice' in all_content or 'suggest' in all_content:
            support_factors.append('practical_advice')
        if 'validate' in all_content or 'normal' in all_content:
            support_factors.append('emotional_validation')
        if 'perspective' in all_content or 'view' in all_content:
            support_factors.append('perspective_shift')
        if 'calm' in all_content or 'relax' in all_content:
            support_factors.append('calming_presence')
        
        # Check for positive feedback
        positive_feedback_count = sum(
            1 for m in memory_sequence
            if m.conversation_context.user_feedback and m.conversation_context.user_feedback >= 4
        )
        
        if positive_feedback_count > 0:
            support_factors.append('positive_interaction')
        
        return support_factors if support_factors else ['unknown_factors']
    
    def _estimate_recovery_quality(
        self,
        initial_state: EmotionalState,
        recovery_state: EmotionalState,
        recovery_duration: timedelta
    ) -> float:
        """Estimate the quality of recovery."""
        # Base quality on emotional improvement
        initial_valence = self._get_emotional_valence(initial_state.primary_emotion)
        recovery_valence = self._get_emotional_valence(recovery_state.primary_emotion)
        
        valence_improvement = recovery_valence - initial_valence
        intensity_improvement = initial_state.intensity - recovery_state.intensity
        
        # Normalize improvements
        valence_score = max(0.0, min(1.0, (valence_improvement + 2.0) / 4.0))
        intensity_score = max(0.0, min(1.0, intensity_improvement))
        
        # Consider recovery speed (moderate speed is optimal)
        hours = recovery_duration.total_seconds() / 3600
        if hours < 1:
            speed_score = 0.5  # Too fast might not be sustainable
        elif hours <= 24:
            speed_score = 1.0  # Optimal range
        elif hours <= 72:
            speed_score = 0.8  # Still good
        else:
            speed_score = 0.6  # Slower recovery
        
        # Combine scores
        quality = (valence_score * 0.4) + (intensity_score * 0.4) + (speed_score * 0.2)
        return min(1.0, quality)
    
    def _get_emotional_valence(self, emotion: EmotionType) -> float:
        """Get emotional valence score (-2.0 to 2.0)."""
        valence_map = {
            EmotionType.JOY: 2.0,
            EmotionType.SURPRISE: 0.5,
            EmotionType.NEUTRAL: 0.0,
            EmotionType.FEAR: -1.0,
            EmotionType.DISGUST: -1.2,
            EmotionType.SADNESS: -1.5,
            EmotionType.ANGER: -1.8
        }
        return valence_map.get(emotion, 0.0)
    
    def _calculate_effectiveness_trend(self, recoveries: List[EmotionalRecovery]) -> str:
        """Calculate trend in recovery effectiveness over time."""
        if len(recoveries) < 3:
            return 'insufficient_data'
        
        # Sort by detection time
        sorted_recoveries = sorted(recoveries, key=lambda r: r.detected_at)
        
        # Split into early and recent recoveries
        split_point = len(sorted_recoveries) // 2
        early_recoveries = sorted_recoveries[:split_point]
        recent_recoveries = sorted_recoveries[split_point:]
        
        early_avg = statistics.mean([
            r.calculate_recovery_effectiveness() for r in early_recoveries
        ])
        recent_avg = statistics.mean([
            r.calculate_recovery_effectiveness() for r in recent_recoveries
        ])
        
        improvement = recent_avg - early_avg
        
        if improvement > 0.1:
            return 'improving'
        elif improvement < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _find_most_effective_recovery_type(
        self,
        recoveries: List[EmotionalRecovery]
    ) -> Dict[str, Any]:
        """Find the most effective recovery type."""
        type_effectiveness = defaultdict(list)
        
        for recovery in recoveries:
            effectiveness = recovery.calculate_recovery_effectiveness()
            type_effectiveness[recovery.recovery_type].append(effectiveness)
        
        # Calculate average effectiveness for each type
        type_averages = {}
        for recovery_type, effectiveness_scores in type_effectiveness.items():
            type_averages[recovery_type] = statistics.mean(effectiveness_scores)
        
        if not type_averages:
            return {'type': 'unknown', 'effectiveness': 0.0}
        
        best_type = max(type_averages, key=type_averages.get)
        
        return {
            'type': best_type,
            'effectiveness': type_averages[best_type],
            'sample_size': len(type_effectiveness[best_type])
        }
    
    def _calculate_resilience_score(self, recoveries: List[EmotionalRecovery]) -> float:
        """Calculate overall resilience score."""
        if not recoveries:
            return 0.0
        
        # Factors for resilience:
        # 1. Recovery frequency (more recoveries = more resilient)
        # 2. Average recovery effectiveness
        # 3. Recovery speed consistency
        # 4. Ability to recover from severe states
        
        effectiveness_scores = [r.calculate_recovery_effectiveness() for r in recoveries]
        avg_effectiveness = statistics.mean(effectiveness_scores)
        
        # Recovery frequency factor (normalized)
        frequency_factor = min(1.0, len(recoveries) / 10.0)
        
        # Consistency factor
        if len(effectiveness_scores) > 1:
            consistency = 1.0 - (statistics.stdev(effectiveness_scores) / max(0.1, avg_effectiveness))
            consistency = max(0.0, min(1.0, consistency))
        else:
            consistency = 0.5
        
        # Severe state recovery factor
        severe_recoveries = [
            r for r in recoveries
            if r.initial_state.intensity > 0.7
        ]
        severe_factor = len(severe_recoveries) / max(1, len(recoveries))
        
        # Combine factors
        resilience = (
            (avg_effectiveness * 0.4) +
            (frequency_factor * 0.3) +
            (consistency * 0.2) +
            (severe_factor * 0.1)
        )
        
        return min(1.0, resilience)
    
    def _analyze_support_effectiveness(
        self,
        recoveries: List[EmotionalRecovery]
    ) -> Dict[str, float]:
        """Analyze effectiveness of different support factors."""
        support_effectiveness = defaultdict(list)
        
        for recovery in recoveries:
            effectiveness = recovery.calculate_recovery_effectiveness()
            for support_factor in recovery.support_factors:
                support_effectiveness[support_factor].append(effectiveness)
        
        # Calculate average effectiveness for each support factor
        result = {}
        for support_factor, effectiveness_scores in support_effectiveness.items():
            result[support_factor] = statistics.mean(effectiveness_scores)
        
        return result
    
    def _generate_recovery_recommendations(
        self,
        similar_recoveries: List[EmotionalRecovery]
    ) -> List[str]:
        """Generate recovery recommendations based on similar past recoveries."""
        recommendations = []
        
        # Find most effective support factors
        support_effectiveness = self._analyze_support_effectiveness(similar_recoveries)
        
        if support_effectiveness:
            best_support = max(support_effectiveness, key=support_effectiveness.get)
            recommendations.append(f"Consider using {best_support.replace('_', ' ')} - it has been effective before")
        
        # Recovery type recommendations
        type_effectiveness = {}
        for recovery in similar_recoveries:
            recovery_type = recovery.recovery_type
            effectiveness = recovery.calculate_recovery_effectiveness()
            if recovery_type not in type_effectiveness:
                type_effectiveness[recovery_type] = []
            type_effectiveness[recovery_type].append(effectiveness)
        
        if type_effectiveness:
            best_type = max(type_effectiveness, key=lambda t: statistics.mean(type_effectiveness[t]))
            recommendations.append(f"Try {best_type.replace('_', ' ')} recovery approach")
        
        # Time-based recommendations
        recovery_times = [r.recovery_duration.total_seconds() / 3600 for r in similar_recoveries]
        avg_time = statistics.mean(recovery_times)
        
        if avg_time < 6:
            recommendations.append("Recovery typically happens quickly - be patient")
        elif avg_time < 24:
            recommendations.append("Recovery usually takes several hours - give it time")
        else:
            recommendations.append("Recovery may take a day or more - focus on self-care")
        
        return recommendations if recommendations else ["Continue with current support approach"]


# Singleton instance
_recovery_tracker_instance = None
_recovery_tracker_lock = threading.Lock()


def get_emotional_recovery_tracker() -> EmotionalRecoveryTracker:
    """
    Get singleton emotional recovery tracker instance.
    
    Returns:
        Shared EmotionalRecoveryTracker instance
    """
    global _recovery_tracker_instance
    
    if _recovery_tracker_instance is None:
        with _recovery_tracker_lock:
            if _recovery_tracker_instance is None:
                _recovery_tracker_instance = EmotionalRecoveryTracker()
    
    return _recovery_tracker_instance