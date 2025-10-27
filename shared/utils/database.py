"""
PostgreSQL database connection and utilities for Morgan AI Assistant
"""
import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager
import os

import asyncpg
from asyncpg import Pool, Connection

from shared.models.database import (
    ConversationModel,
    MessageModel,
    StreamingSessionModel,
    AudioTranscriptionModel,
    TTSGenerationModel,
    UserPreferencesModel,
    SystemMetricModel
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """PostgreSQL database manager with connection pooling"""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None,
        min_size: int = 5,
        max_size: int = 20
    ):
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = database or os.getenv("POSTGRES_DB", "morgan")
        self.user = user or os.getenv("POSTGRES_USER", "morgan")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "morgan_secure_password")
        self.min_size = min_size
        self.max_size = max_size
        
        self.pool: Optional[Pool] = None
        self.logger = logger

    async def connect(self):
        """Create database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=60
            )
            self.logger.info(f"Connected to PostgreSQL database: {self.database}@{self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self.logger.info("Disconnected from PostgreSQL database")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool"""
        async with self.pool.acquire() as connection:
            yield connection

    async def execute(self, query: str, *args) -> str:
        """Execute a query"""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetch multiple rows"""
        async with self.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch a single row"""
        async with self.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value"""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    # Conversation operations
    async def create_conversation(self, conversation: ConversationModel) -> ConversationModel:
        """Create a new conversation"""
        query = """
            INSERT INTO conversations (conversation_id, user_id, title, metadata)
            VALUES ($1, $2, $3, $4)
            RETURNING id, conversation_id, user_id, title, created_at, updated_at, 
                      last_message_at, message_count, metadata, is_active
        """
        row = await self.fetchrow(
            query,
            conversation.conversation_id,
            conversation.user_id,
            conversation.title,
            conversation.metadata
        )
        return ConversationModel(**row)

    async def get_conversation(self, conversation_id: str) -> Optional[ConversationModel]:
        """Get conversation by ID"""
        query = """
            SELECT * FROM conversations WHERE conversation_id = $1
        """
        row = await self.fetchrow(query, conversation_id)
        return ConversationModel(**row) if row else None

    async def get_user_conversations(self, user_id: str, limit: int = 50) -> List[ConversationModel]:
        """Get conversations for a user"""
        query = """
            SELECT * FROM conversations 
            WHERE user_id = $1 AND is_active = TRUE
            ORDER BY updated_at DESC
            LIMIT $2
        """
        rows = await self.fetch(query, user_id, limit)
        return [ConversationModel(**row) for row in rows]

    async def update_conversation(self, conversation_id: str, **kwargs) -> Optional[ConversationModel]:
        """Update conversation"""
        set_clauses = []
        values = []
        idx = 1
        
        for key, value in kwargs.items():
            if key not in ['id', 'conversation_id', 'created_at']:
                set_clauses.append(f"{key} = ${idx}")
                values.append(value)
                idx += 1
        
        if not set_clauses:
            return await self.get_conversation(conversation_id)
        
        values.append(conversation_id)
        query = f"""
            UPDATE conversations 
            SET {', '.join(set_clauses)}
            WHERE conversation_id = ${idx}
            RETURNING *
        """
        row = await self.fetchrow(query, *values)
        return ConversationModel(**row) if row else None

    # Message operations
    async def add_message(self, message: MessageModel) -> MessageModel:
        """Add a message to conversation"""
        query = """
            INSERT INTO messages (conversation_id, role, content, sequence_number, metadata, 
                                 embedding, tokens_used, processing_time_ms)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *
        """
        row = await self.fetchrow(
            query,
            message.conversation_id,
            message.role,
            message.content,
            message.sequence_number,
            message.metadata,
            message.embedding,
            message.tokens_used,
            message.processing_time_ms
        )
        return MessageModel(**row)

    async def get_conversation_messages(
        self, 
        conversation_id: str, 
        limit: int = 100,
        offset: int = 0
    ) -> List[MessageModel]:
        """Get messages for a conversation"""
        query = """
            SELECT m.* FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.conversation_id = $1
            ORDER BY m.sequence_number DESC
            LIMIT $2 OFFSET $3
        """
        rows = await self.fetch(query, conversation_id, limit, offset)
        return [MessageModel(**row) for row in rows]

    async def get_recent_messages(
        self, 
        conversation_id: str, 
        count: int = 10
    ) -> List[MessageModel]:
        """Get recent messages for context"""
        query = """
            SELECT m.* FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.conversation_id = $1
            ORDER BY m.sequence_number DESC
            LIMIT $2
        """
        rows = await self.fetch(query, conversation_id, count)
        messages = [MessageModel(**row) for row in rows]
        return list(reversed(messages))  # Return in chronological order

    # Streaming session operations
    async def create_streaming_session(self, session: StreamingSessionModel) -> StreamingSessionModel:
        """Create a new streaming session"""
        query = """
            INSERT INTO streaming_sessions (session_id, user_id, conversation_id, status, 
                                           session_type, metadata)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *
        """
        row = await self.fetchrow(
            query,
            session.session_id,
            session.user_id,
            session.conversation_id,
            session.status,
            session.session_type,
            session.metadata
        )
        return StreamingSessionModel(**row)

    async def get_streaming_session(self, session_id: str) -> Optional[StreamingSessionModel]:
        """Get streaming session by ID"""
        query = """
            SELECT * FROM streaming_sessions WHERE session_id = $1
        """
        row = await self.fetchrow(query, session_id)
        return StreamingSessionModel(**row) if row else None

    async def update_streaming_session(self, session_id: str, **kwargs) -> Optional[StreamingSessionModel]:
        """Update streaming session"""
        set_clauses = []
        values = []
        idx = 1
        
        for key, value in kwargs.items():
            if key not in ['id', 'session_id', 'created_at']:
                set_clauses.append(f"{key} = ${idx}")
                values.append(value)
                idx += 1
        
        if not set_clauses:
            return await self.get_streaming_session(session_id)
        
        values.append(session_id)
        query = f"""
            UPDATE streaming_sessions 
            SET {', '.join(set_clauses)}
            WHERE session_id = ${idx}
            RETURNING *
        """
        row = await self.fetchrow(query, *values)
        return StreamingSessionModel(**row) if row else None

    async def end_streaming_session(self, session_id: str) -> Optional[StreamingSessionModel]:
        """End a streaming session"""
        return await self.update_streaming_session(
            session_id,
            status="ended",
            ended_at="NOW()"
        )

    # User preferences operations
    async def get_user_preferences(self, user_id: str) -> Optional[UserPreferencesModel]:
        """Get user preferences"""
        query = """
            SELECT * FROM user_preferences WHERE user_id = $1
        """
        row = await self.fetchrow(query, user_id)
        return UserPreferencesModel(**row) if row else None

    async def upsert_user_preferences(self, preferences: UserPreferencesModel) -> UserPreferencesModel:
        """Create or update user preferences"""
        query = """
            INSERT INTO user_preferences (user_id, preferred_language, preferred_voice, 
                                         tts_speed, llm_temperature, llm_max_tokens, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (user_id) DO UPDATE SET
                preferred_language = EXCLUDED.preferred_language,
                preferred_voice = EXCLUDED.preferred_voice,
                tts_speed = EXCLUDED.tts_speed,
                llm_temperature = EXCLUDED.llm_temperature,
                llm_max_tokens = EXCLUDED.llm_max_tokens,
                metadata = EXCLUDED.metadata,
                updated_at = CURRENT_TIMESTAMP
            RETURNING *
        """
        row = await self.fetchrow(
            query,
            preferences.user_id,
            preferences.preferred_language,
            preferences.preferred_voice,
            preferences.tts_speed,
            preferences.llm_temperature,
            preferences.llm_max_tokens,
            preferences.metadata
        )
        return UserPreferencesModel(**row)

    # Audio transcription operations
    async def add_audio_transcription(self, transcription: AudioTranscriptionModel) -> AudioTranscriptionModel:
        """Add audio transcription record"""
        query = """
            INSERT INTO audio_transcriptions (session_id, message_id, transcription, language, 
                                             confidence, duration_ms, audio_format, sample_rate, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING *
        """
        row = await self.fetchrow(
            query,
            transcription.session_id,
            transcription.message_id,
            transcription.transcription,
            transcription.language,
            transcription.confidence,
            transcription.duration_ms,
            transcription.audio_format,
            transcription.sample_rate,
            transcription.metadata
        )
        return AudioTranscriptionModel(**row)

    # System metrics operations
    async def add_system_metric(self, metric: SystemMetricModel) -> SystemMetricModel:
        """Add system metric"""
        query = """
            INSERT INTO system_metrics (metric_type, metric_value, service_name, metadata)
            VALUES ($1, $2, $3, $4)
            RETURNING *
        """
        row = await self.fetchrow(
            query,
            metric.metric_type,
            metric.metric_value,
            metric.service_name,
            metric.metadata
        )
        return SystemMetricModel(**row)

    async def get_recent_metrics(
        self, 
        service_name: str = None, 
        metric_type: str = None,
        limit: int = 100
    ) -> List[SystemMetricModel]:
        """Get recent system metrics"""
        where_clauses = []
        values = []
        idx = 1
        
        if service_name:
            where_clauses.append(f"service_name = ${idx}")
            values.append(service_name)
            idx += 1
        
        if metric_type:
            where_clauses.append(f"metric_type = ${idx}")
            values.append(metric_type)
            idx += 1
        
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        values.append(limit)
        
        query = f"""
            SELECT * FROM system_metrics
            {where_sql}
            ORDER BY created_at DESC
            LIMIT ${idx}
        """
        rows = await self.fetch(query, *values)
        return [SystemMetricModel(**row) for row in rows]

    async def health_check(self) -> bool:
        """Check database health"""
        try:
            await self.fetchval("SELECT 1")
            return True
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


async def get_db_manager() -> DatabaseManager:
    """Get or create global database manager"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
        await db_manager.connect()
    return db_manager


async def close_db_manager():
    """Close global database manager"""
    global db_manager
    if db_manager:
        await db_manager.disconnect()
        db_manager = None

