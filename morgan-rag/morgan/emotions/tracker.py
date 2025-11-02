"""
Emotional state history tracking module.

Provides persistent storage and retrieval of emotional states with temporal
organization, search capabilities, and historical analysis.
"""

import threading
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import sqlite3
from contextlib import contextmanager

from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.emotional.models import EmotionalState, EmotionType, ConversationContext

logger = get_logger(__name__)


class EmotionalStateTracker:
    """
    Persistent tracking and storage of emotional states over time.
    
    Features:
    - SQLite-based persistent storage
    - Temporal organization and indexing
    - Efficient querying by user, timeframe, and emotion type
    - Historical trend analysis
    - Data export and backup capabilities
    """
    
    def __init__(self):
        """Initialize emotional state tracker with database setup."""
        self.settings = get_settings()
        
        # Setup database path
        from pathlib import Path
        self.db_path = Path(self.settings.morgan_data_dir) / "emotional_states.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Emotional State Tracker initialized with database: {self.db_path}")
    
    def track_emotional_state(
        self,
        user_id: str,
        emotional_state: EmotionalState,
        context: Optional[ConversationContext] = None
    ) -> str:
        """
        Store an emotional state in the tracking system.
        
        Args:
            user_id: User identifier
            emotional_state: Emotional state to track
            context: Optional conversation context
            
        Returns:
            Unique tracking ID for the stored state
        """
        tracking_id = self._generate_tracking_id(user_id, emotional_state.timestamp)
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare context data
            context_data = {}
            if context:
                context_data = {
                    "conversation_id": context.conversation_id,
                    "message_text": context.message_text[:500],  # Truncate for storage
                    "user_feedback": context.user_feedback,
                    "session_duration": context.session_duration.total_seconds() if context.session_duration else None
                }
            
            # Insert emotional state
            cursor.execute("""
                INSERT INTO emotional_states (
                    tracking_id, user_id, primary_emotion, intensity, confidence,
                    secondary_emotions, emotional_indicators, context_data,
                    timestamp, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tracking_id,
                user_id,
                emotional_state.primary_emotion.value,
                emotional_state.intensity,
                emotional_state.confidence,
                json.dumps([e.value for e in emotional_state.secondary_emotions]),
                json.dumps(emotional_state.emotional_indicators),
                json.dumps(context_data),
                emotional_state.timestamp.isoformat(),
                datetime.utcnow().isoformat()
            ))
            
            conn.commit()
        
        logger.debug(f"Tracked emotional state {tracking_id} for user {user_id}")
        return tracking_id
    
    def get_emotional_history(
        self,
        user_id: str,
        timeframe: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        emotion_types: Optional[List[EmotionType]] = None,
        limit: Optional[int] = None
    ) -> List[EmotionalState]:
        """
        Retrieve emotional history for a user with filtering options.
        
        Args:
            user_id: User identifier
            timeframe: Relative timeframe (e.g., "7d", "30d")
            start_date: Absolute start date
            end_date: Absolute end date
            emotion_types: Filter by specific emotion types
            limit: Maximum number of results
            
        Returns:
            List of emotional states matching criteria
        """
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM emotional_states WHERE user_id = ?"
            params = [user_id]
            
            # Add time filtering
            if timeframe:
                days = int(timeframe.rstrip('d'))
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                query += " AND timestamp >= ?"
                params.append(cutoff_date.isoformat())
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            # Add emotion type filtering
            if emotion_types:
                emotion_values = [e.value for e in emotion_types]
                placeholders = ",".join("?" * len(emotion_values))
                query += f" AND primary_emotion IN ({placeholders})"
                params.extend(emotion_values)
            
            # Add ordering and limit
            query += " ORDER BY timestamp DESC"
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        # Convert to EmotionalState objects
        emotional_states = []
        for row in rows:
            emotional_state = self._row_to_emotional_state(row)
            emotional_states.append(emotional_state)
        
        logger.debug(f"Retrieved {len(emotional_states)} emotional states for user {user_id}")
        return emotional_states
    
    def get_emotion_statistics(
        self,
        user_id: str,
        timeframe: str = "30d"
    ) -> Dict[str, Any]:
        """
        Get statistical summary of emotions for a user.
        
        Args:
            user_id: User identifier
            timeframe: Analysis timeframe
            
        Returns:
            Statistical summary of emotional data
        """
        emotions = self.get_emotional_history(user_id, timeframe=timeframe)
        
        if not emotions:
            return {
                "total_records": 0,
                "timeframe": timeframe,
                "emotion_distribution": {},
                "average_intensity": 0.0,
                "average_confidence": 0.0
            }
        
        # Calculate statistics
        emotion_counts = {}
        total_intensity = 0.0
        total_confidence = 0.0
        
        for emotion in emotions:
            emotion_type = emotion.primary_emotion.value
            emotion_counts[emotion_type] = emotion_counts.get(emotion_type, 0) + 1
            total_intensity += emotion.intensity
            total_confidence += emotion.confidence
        
        # Calculate percentages
        total_records = len(emotions)
        emotion_distribution = {
            emotion: (count / total_records) * 100
            for emotion, count in emotion_counts.items()
        }
        
        return {
            "total_records": total_records,
            "timeframe": timeframe,
            "emotion_distribution": emotion_distribution,
            "average_intensity": total_intensity / total_records,
            "average_confidence": total_confidence / total_records,
            "most_common_emotion": max(emotion_counts, key=emotion_counts.get),
            "date_range": {
                "earliest": emotions[-1].timestamp.isoformat(),
                "latest": emotions[0].timestamp.isoformat()
            }
        }
    
    def get_emotional_timeline(
        self,
        user_id: str,
        timeframe: str = "7d",
        granularity: str = "daily"
    ) -> List[Dict[str, Any]]:
        """
        Get emotional timeline with aggregated data points.
        
        Args:
            user_id: User identifier
            timeframe: Analysis timeframe
            granularity: Timeline granularity ("hourly", "daily", "weekly")
            
        Returns:
            List of timeline data points
        """
        emotions = self.get_emotional_history(user_id, timeframe=timeframe)
        
        if not emotions:
            return []
        
        # Group emotions by time period
        timeline_data = {}
        
        for emotion in emotions:
            # Determine time bucket based on granularity
            if granularity == "hourly":
                time_key = emotion.timestamp.strftime("%Y-%m-%d %H:00")
            elif granularity == "daily":
                time_key = emotion.timestamp.strftime("%Y-%m-%d")
            elif granularity == "weekly":
                # Get Monday of the week
                monday = emotion.timestamp - timedelta(days=emotion.timestamp.weekday())
                time_key = monday.strftime("%Y-%m-%d")
            else:
                time_key = emotion.timestamp.strftime("%Y-%m-%d")
            
            if time_key not in timeline_data:
                timeline_data[time_key] = {
                    "timestamp": time_key,
                    "emotions": [],
                    "emotion_counts": {},
                    "total_intensity": 0.0,
                    "count": 0
                }
            
            bucket = timeline_data[time_key]
            bucket["emotions"].append(emotion)
            bucket["count"] += 1
            bucket["total_intensity"] += emotion.intensity
            
            emotion_type = emotion.primary_emotion.value
            bucket["emotion_counts"][emotion_type] = bucket["emotion_counts"].get(emotion_type, 0) + 1
        
        # Calculate aggregated metrics for each time bucket
        timeline = []
        for time_key in sorted(timeline_data.keys()):
            bucket = timeline_data[time_key]
            
            # Find dominant emotion
            dominant_emotion = max(bucket["emotion_counts"], key=bucket["emotion_counts"].get)
            
            timeline.append({
                "timestamp": time_key,
                "dominant_emotion": dominant_emotion,
                "emotion_distribution": bucket["emotion_counts"],
                "average_intensity": bucket["total_intensity"] / bucket["count"],
                "total_records": bucket["count"]
            })
        
        return timeline
    
    def search_emotional_states(
        self,
        user_id: str,
        search_criteria: Dict[str, Any]
    ) -> List[EmotionalState]:
        """
        Search emotional states using flexible criteria.
        
        Args:
            user_id: User identifier
            search_criteria: Search parameters
            
        Returns:
            List of matching emotional states
        """
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM emotional_states WHERE user_id = ?"
            params = [user_id]
            
            # Add search criteria
            if "min_intensity" in search_criteria:
                query += " AND intensity >= ?"
                params.append(search_criteria["min_intensity"])
            
            if "max_intensity" in search_criteria:
                query += " AND intensity <= ?"
                params.append(search_criteria["max_intensity"])
            
            if "min_confidence" in search_criteria:
                query += " AND confidence >= ?"
                params.append(search_criteria["min_confidence"])
            
            if "indicator_contains" in search_criteria:
                query += " AND emotional_indicators LIKE ?"
                params.append(f"%{search_criteria['indicator_contains']}%")
            
            if "has_context" in search_criteria and search_criteria["has_context"]:
                query += " AND context_data != '{}'"
            
            query += " ORDER BY timestamp DESC"
            
            if "limit" in search_criteria:
                query += " LIMIT ?"
                params.append(search_criteria["limit"])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        return [self._row_to_emotional_state(row) for row in rows]
    
    def delete_emotional_history(
        self,
        user_id: str,
        before_date: Optional[datetime] = None
    ) -> int:
        """
        Delete emotional history for a user.
        
        Args:
            user_id: User identifier
            before_date: Optional cutoff date (delete records before this date)
            
        Returns:
            Number of records deleted
        """
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            if before_date:
                cursor.execute(
                    "DELETE FROM emotional_states WHERE user_id = ? AND timestamp < ?",
                    (user_id, before_date.isoformat())
                )
            else:
                cursor.execute(
                    "DELETE FROM emotional_states WHERE user_id = ?",
                    (user_id,)
                )
            
            deleted_count = cursor.rowcount
            conn.commit()
        
        logger.info(f"Deleted {deleted_count} emotional state records for user {user_id}")
        return deleted_count
    
    def export_emotional_data(
        self,
        user_id: str,
        format: str = "json",
        timeframe: Optional[str] = None
    ) -> str:
        """
        Export emotional data for a user.
        
        Args:
            user_id: User identifier
            format: Export format ("json", "csv")
            timeframe: Optional timeframe filter
            
        Returns:
            Exported data as string
        """
        emotions = self.get_emotional_history(user_id, timeframe=timeframe)
        
        if format == "json":
            export_data = []
            for emotion in emotions:
                export_data.append({
                    "timestamp": emotion.timestamp.isoformat(),
                    "primary_emotion": emotion.primary_emotion.value,
                    "intensity": emotion.intensity,
                    "confidence": emotion.confidence,
                    "secondary_emotions": [e.value for e in emotion.secondary_emotions],
                    "emotional_indicators": emotion.emotional_indicators
                })
            return json.dumps(export_data, indent=2)
        
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "timestamp", "primary_emotion", "intensity", "confidence",
                "secondary_emotions", "emotional_indicators"
            ])
            
            # Write data
            for emotion in emotions:
                writer.writerow([
                    emotion.timestamp.isoformat(),
                    emotion.primary_emotion.value,
                    emotion.intensity,
                    emotion.confidence,
                    "|".join(e.value for e in emotion.secondary_emotions),
                    "|".join(emotion.emotional_indicators)
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create emotional_states table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emotional_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tracking_id TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    primary_emotion TEXT NOT NULL,
                    intensity REAL NOT NULL,
                    confidence REAL NOT NULL,
                    secondary_emotions TEXT,
                    emotional_indicators TEXT,
                    context_data TEXT,
                    timestamp TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for efficient querying
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_timestamp 
                ON emotional_states(user_id, timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_emotion_type 
                ON emotional_states(primary_emotion)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_intensity 
                ON emotional_states(intensity)
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def _generate_tracking_id(self, user_id: str, timestamp: datetime) -> str:
        """Generate unique tracking ID."""
        import hashlib
        
        id_input = f"{user_id}:{timestamp.isoformat()}:{datetime.utcnow().microsecond}"
        return hashlib.sha256(id_input.encode()).hexdigest()[:16]
    
    def _row_to_emotional_state(self, row: sqlite3.Row) -> EmotionalState:
        """Convert database row to EmotionalState object."""
        secondary_emotions = []
        if row["secondary_emotions"]:
            secondary_emotion_values = json.loads(row["secondary_emotions"])
            secondary_emotions = [EmotionType(e) for e in secondary_emotion_values]
        
        emotional_indicators = []
        if row["emotional_indicators"]:
            emotional_indicators = json.loads(row["emotional_indicators"])
        
        return EmotionalState(
            primary_emotion=EmotionType(row["primary_emotion"]),
            intensity=row["intensity"],
            confidence=row["confidence"],
            secondary_emotions=secondary_emotions,
            emotional_indicators=emotional_indicators,
            timestamp=datetime.fromisoformat(row["timestamp"])
        )


# Singleton instance
_tracker_instance = None
_tracker_lock = threading.Lock()


def get_emotional_state_tracker() -> EmotionalStateTracker:
    """
    Get singleton emotional state tracker instance.
    
    Returns:
        Shared EmotionalStateTracker instance
    """
    global _tracker_instance
    
    if _tracker_instance is None:
        with _tracker_lock:
            if _tracker_instance is None:
                _tracker_instance = EmotionalStateTracker()
    
    return _tracker_instance