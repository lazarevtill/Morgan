"""
PostgreSQL database utilities for Morgan AI Assistant
"""

import asyncio
import json
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg
from asyncpg import Pool, Connection

from shared.models.database import (
    ConversationModel,
    MessageModel,
    StreamingSessionModel,
    AudioTranscriptionModel,
    TTSGenerationModel,
    UserPreferencesModel,
    SystemMetricModel,
)

logger = logging.getLogger(__name__)


class DatabaseClient:
    """PostgreSQL database client with connection pooling"""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_size: int = 10,
        max_size: int = 20,
    ):
        import os

        # Load from environment variables with fallback to parameters
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = database or os.getenv("POSTGRES_DB", "morgan")
        self.user = user or os.getenv("POSTGRES_USER", "morgan")
        self.password = password or os.getenv("POSTGRES_PASSWORD")

        if not self.password:
            raise ValueError(
                "PostgreSQL password must be provided via POSTGRES_PASSWORD environment variable or password parameter"
            )

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
                command_timeout=60,
            )
            self.logger.info(
                f"Connected to PostgreSQL at {self.host}:{self.port}/{self.database}"
            )
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self.logger.info("Disconnected from PostgreSQL")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool"""
        if not self.pool:
            raise RuntimeError("Database pool not initialized. Call connect() first.")

        async with self.pool.acquire() as conn:
            yield conn

    async def health_check(self) -> bool:
        """Check database connection health"""
        try:
            async with self.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False

    # Conversation operations
    async def create_conversation(self, conversation: ConversationModel) -> UUID:
        """Create a new conversation"""
        async with self.acquire() as conn:
            # Convert metadata dict to JSON string for PostgreSQL
            metadata_json = (
                json.dumps(conversation.metadata) if conversation.metadata else "{}"
            )

            row = await conn.fetchrow(
                """
                INSERT INTO conversations (conversation_id, user_id, title, metadata)
                VALUES ($1, $2, $3, $4::jsonb)
                RETURNING id
                """,
                conversation.conversation_id,
                conversation.user_id,
                conversation.title,
                metadata_json,
            )
            return row["id"]

    async def get_conversation(
        self, conversation_id: str
    ) -> Optional[ConversationModel]:
        """Get conversation by conversation_id"""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM conversations WHERE conversation_id = $1",
                conversation_id,
            )
            if not row:
                return None

            # Convert row to dict and parse JSON metadata
            row_dict = dict(row)
            if isinstance(row_dict.get("metadata"), str):
                try:
                    row_dict["metadata"] = json.loads(row_dict["metadata"])
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(
                        f"Failed to parse metadata JSON for conversation {conversation_id}: {e}"
                    )
                    row_dict["metadata"] = {}

            return ConversationModel(**row_dict)

    async def update_conversation(self, conversation_id: str, **kwargs):
        """Update conversation fields"""
        if not kwargs:
            return

        set_clause = ", ".join(
            [f"{key} = ${i+2}" for i, key in enumerate(kwargs.keys())]
        )
        query = f"UPDATE conversations SET {set_clause} WHERE conversation_id = $1"

        async with self.acquire() as conn:
            await conn.execute(query, conversation_id, *kwargs.values())

    async def get_user_conversations(
        self, user_id: str, limit: int = 50, offset: int = 0, active_only: bool = True
    ) -> List[ConversationModel]:
        """Get user's conversations"""
        query = """
            SELECT * FROM conversations 
            WHERE user_id = $1
        """
        if active_only:
            query += " AND is_active = TRUE"
        query += " ORDER BY updated_at DESC LIMIT $2 OFFSET $3"

        async with self.acquire() as conn:
            rows = await conn.fetch(query, user_id, limit, offset)
            conversations = []
            for row in rows:
                row_dict = dict(row)
                # Parse JSON metadata string to dict
                if isinstance(row_dict.get("metadata"), str):
                    try:
                        row_dict["metadata"] = json.loads(row_dict["metadata"])
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(
                            f"Failed to parse metadata JSON for conversation: {e}"
                        )
                        row_dict["metadata"] = {}
                conversations.append(ConversationModel(**row_dict))
            return conversations

    # Message operations
    async def add_message(self, message: MessageModel) -> UUID:
        """Add a message to conversation"""
        async with self.acquire() as conn:
            # Convert metadata dict to JSON string for PostgreSQL
            metadata_json = json.dumps(message.metadata) if message.metadata else "{}"

            row = await conn.fetchrow(
                """
                INSERT INTO messages (
                    conversation_id, role, content, sequence_number,
                    metadata, tokens_used, processing_time_ms
                )
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
                RETURNING id
                """,
                message.conversation_id,
                message.role,
                message.content,
                message.sequence_number,
                metadata_json,
                message.tokens_used,
                message.processing_time_ms,
            )
            return row["id"]

    async def get_conversation_messages(
        self, conversation_id: UUID, limit: int = 100, offset: int = 0
    ) -> List[MessageModel]:
        """Get messages for a conversation"""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM messages
                WHERE conversation_id = $1
                ORDER BY sequence_number ASC
                LIMIT $2 OFFSET $3
                """,
                conversation_id,
                limit,
                offset,
            )
            return [MessageModel(**dict(row)) for row in rows]

    async def get_recent_messages(
        self, conversation_id: UUID, count: int = 10
    ) -> List[MessageModel]:
        """Get recent messages from conversation"""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM messages
                WHERE conversation_id = $1
                ORDER BY sequence_number DESC
                LIMIT $2
                """,
                conversation_id,
                count,
            )
            return [MessageModel(**dict(row)) for row in reversed(rows)]

    # Streaming session operations
    async def create_streaming_session(self, session: StreamingSessionModel) -> UUID:
        """Create a streaming session"""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO streaming_sessions (
                    session_id, user_id, conversation_id, status,
                    session_type, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                session.session_id,
                session.user_id,
                session.conversation_id,
                session.status,
                session.session_type,
                session.metadata,
            )
            return row["id"]

    async def update_streaming_session(self, session_id: str, **kwargs):
        """Update streaming session"""
        if not kwargs:
            return

        set_clause = ", ".join(
            [f"{key} = ${i+2}" for i, key in enumerate(kwargs.keys())]
        )
        query = f"UPDATE streaming_sessions SET {set_clause} WHERE session_id = $1"

        async with self.acquire() as conn:
            await conn.execute(query, session_id, *kwargs.values())

    async def get_active_sessions(
        self, user_id: Optional[str] = None
    ) -> List[StreamingSessionModel]:
        """Get active streaming sessions"""
        query = "SELECT * FROM streaming_sessions WHERE status = 'active'"
        params = []

        if user_id:
            query += " AND user_id = $1"
            params.append(user_id)

        async with self.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [StreamingSessionModel(**dict(row)) for row in rows]

    # Audio transcription operations
    async def add_transcription(self, transcription: AudioTranscriptionModel) -> UUID:
        """Add audio transcription"""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO audio_transcriptions (
                    session_id, message_id, transcription, language,
                    confidence, duration_ms, audio_format, sample_rate, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
                """,
                transcription.session_id,
                transcription.message_id,
                transcription.transcription,
                transcription.language,
                transcription.confidence,
                transcription.duration_ms,
                transcription.audio_format,
                transcription.sample_rate,
                transcription.metadata,
            )
            return row["id"]

    # TTS cache operations
    async def get_tts_cache(self, text_hash: str) -> Optional[TTSGenerationModel]:
        """Get cached TTS generation"""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM tts_generations
                WHERE text_hash = $1
                AND created_at > NOW() - INTERVAL '7 days'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                text_hash,
            )
            return TTSGenerationModel(**dict(row)) if row else None

    async def cache_tts_generation(self, generation: TTSGenerationModel) -> UUID:
        """Cache TTS generation"""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO tts_generations (
                    message_id, text_hash, voice, speed,
                    audio_format, sample_rate, duration_ms, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
                """,
                generation.message_id,
                generation.text_hash,
                generation.voice,
                generation.speed,
                generation.audio_format,
                generation.sample_rate,
                generation.duration_ms,
                generation.metadata,
            )
            return row["id"]

    # User preferences operations
    async def get_user_preferences(
        self, user_id: str
    ) -> Optional[UserPreferencesModel]:
        """Get user preferences"""
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM user_preferences WHERE user_id = $1", user_id
            )
            return UserPreferencesModel(**dict(row)) if row else None

    async def update_user_preferences(self, user_id: str, **kwargs):
        """Update user preferences"""
        if not kwargs:
            return

        set_clause = ", ".join(
            [f"{key} = ${i+2}" for i, key in enumerate(kwargs.keys())]
        )
        query = f"UPDATE user_preferences SET {set_clause} WHERE user_id = $1"

        async with self.acquire() as conn:
            await conn.execute(query, user_id, *kwargs.values())

    # System metrics operations
    async def log_metric(self, metric: SystemMetricModel):
        """Log system metric"""
        async with self.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO system_metrics (metric_type, metric_value, service_name, metadata)
                VALUES ($1, $2, $3, $4)
                """,
                metric.metric_type,
                metric.metric_value,
                metric.service_name,
                metric.metadata,
            )


# Global database client instance
db_client: Optional[DatabaseClient] = None


def get_db_client() -> DatabaseClient:
    """Get global database client instance"""
    global db_client
    if db_client is None:
        raise RuntimeError(
            "Database client not initialized. Call init_db_client() first."
        )
    return db_client


def init_db_client(**kwargs) -> DatabaseClient:
    """Initialize global database client"""
    global db_client
    db_client = DatabaseClient(**kwargs)
    return db_client
