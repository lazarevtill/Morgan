"""
Redis client utilities for Morgan AI Assistant
"""
import json
import logging
from typing import Optional, Any, Dict, List
from datetime import timedelta

import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis client for caching and session state"""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        decode_responses: bool = True,
        max_connections: int = 50
    ):
        import os
        # Load from environment variables with fallback to parameters
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.db = db or int(os.getenv("REDIS_DB", "0"))
        self.password = password or os.getenv("REDIS_PASSWORD")  # Optional for Redis
        self.decode_responses = decode_responses
        self.max_connections = max_connections
        self.client: Optional[Redis] = None
        self.logger = logger

    async def connect(self):
        """Connect to Redis"""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=self.decode_responses,
                max_connections=self.max_connections
            )
            # Test connection
            await self.client.ping()
            self.logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            await self.client.close()
            self.logger.info("Disconnected from Redis")

    async def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            if self.client:
                await self.client.ping()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Redis health check failed: {e}")
            return False

    # Key-value operations
    async def get(self, key: str) -> Optional[str]:
        """Get value by key"""
        try:
            return await self.client.get(key)
        except Exception as e:
            self.logger.error(f"Redis GET failed for key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: str,
        expire: Optional[int] = None
    ) -> bool:
        """Set key-value pair with optional expiration (seconds)"""
        try:
            if expire:
                return await self.client.setex(key, expire, value)
            else:
                return await self.client.set(key, value)
        except Exception as e:
            self.logger.error(f"Redis SET failed for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key"""
        try:
            return await self.client.delete(key) > 0
        except Exception as e:
            self.logger.error(f"Redis DELETE failed for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            self.logger.error(f"Redis EXISTS failed for key {key}: {e}")
            return False

    async def expire(self, key: str, seconds: int) -> bool:
        """Set key expiration"""
        try:
            return await self.client.expire(key, seconds)
        except Exception as e:
            self.logger.error(f"Redis EXPIRE failed for key {key}: {e}")
            return False

    async def ttl(self, key: str) -> int:
        """Get key time-to-live"""
        try:
            return await self.client.ttl(key)
        except Exception as e:
            self.logger.error(f"Redis TTL failed for key {key}: {e}")
            return -1

    # JSON operations
    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON value by key"""
        try:
            value = await self.get(key)
            return json.loads(value) if value else None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON for key {key}: {e}")
            return None

    async def set_json(
        self,
        key: str,
        value: Dict[str, Any],
        expire: Optional[int] = None
    ) -> bool:
        """Set JSON value with optional expiration"""
        try:
            json_str = json.dumps(value)
            return await self.set(key, json_str, expire)
        except Exception as e:
            self.logger.error(f"Failed to set JSON for key {key}: {e}")
            return False

    # Hash operations
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field value"""
        try:
            return await self.client.hget(name, key)
        except Exception as e:
            self.logger.error(f"Redis HGET failed for {name}.{key}: {e}")
            return None

    async def hset(self, name: str, key: str, value: str) -> bool:
        """Set hash field value"""
        try:
            return await self.client.hset(name, key, value) >= 0
        except Exception as e:
            self.logger.error(f"Redis HSET failed for {name}.{key}: {e}")
            return False

    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields"""
        try:
            return await self.client.hgetall(name)
        except Exception as e:
            self.logger.error(f"Redis HGETALL failed for {name}: {e}")
            return {}

    async def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields"""
        try:
            return await self.client.hdel(name, *keys)
        except Exception as e:
            self.logger.error(f"Redis HDEL failed for {name}: {e}")
            return 0

    # List operations
    async def lpush(self, key: str, *values: str) -> int:
        """Push values to list head"""
        try:
            return await self.client.lpush(key, *values)
        except Exception as e:
            self.logger.error(f"Redis LPUSH failed for {key}: {e}")
            return 0

    async def rpush(self, key: str, *values: str) -> int:
        """Push values to list tail"""
        try:
            return await self.client.rpush(key, *values)
        except Exception as e:
            self.logger.error(f"Redis RPUSH failed for {key}: {e}")
            return 0

    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        """Get list range"""
        try:
            return await self.client.lrange(key, start, end)
        except Exception as e:
            self.logger.error(f"Redis LRANGE failed for {key}: {e}")
            return []

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        """Trim list"""
        try:
            return await self.client.ltrim(key, start, end)
        except Exception as e:
            self.logger.error(f"Redis LTRIM failed for {key}: {e}")
            return False

    # Session state operations
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get streaming session state"""
        return await self.get_json(f"session:{session_id}")

    async def set_session(
        self,
        session_id: str,
        data: Dict[str, Any],
        expire: int = 3600
    ) -> bool:
        """Set streaming session state with expiration"""
        return await self.set_json(f"session:{session_id}", data, expire)

    async def delete_session(self, session_id: str) -> bool:
        """Delete streaming session state"""
        return await self.delete(f"session:{session_id}")

    async def update_session(
        self,
        session_id: str,
        **kwargs
    ) -> bool:
        """Update session state fields"""
        session = await self.get_session(session_id)
        if session:
            session.update(kwargs)
            return await self.set_session(session_id, session)
        return False

    # Conversation cache operations
    async def cache_conversation_messages(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]],
        expire: int = 1800
    ) -> bool:
        """Cache conversation messages"""
        return await self.set_json(
            f"conv:messages:{conversation_id}",
            {"messages": messages},
            expire
        )

    async def get_cached_messages(
        self,
        conversation_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached conversation messages"""
        data = await self.get_json(f"conv:messages:{conversation_id}")
        return data.get("messages") if data else None

    # Audio chunk buffering
    async def buffer_audio_chunk(
        self,
        session_id: str,
        chunk_data: str,
        max_chunks: int = 100
    ) -> bool:
        """Buffer audio chunk for session"""
        try:
            await self.rpush(f"audio:buffer:{session_id}", chunk_data)
            # Keep only last N chunks
            await self.ltrim(f"audio:buffer:{session_id}", -max_chunks, -1)
            # Set expiration
            await self.expire(f"audio:buffer:{session_id}", 300)  # 5 minutes
            return True
        except Exception as e:
            self.logger.error(f"Failed to buffer audio chunk: {e}")
            return False

    async def get_audio_buffer(
        self,
        session_id: str
    ) -> List[str]:
        """Get all buffered audio chunks for session"""
        return await self.lrange(f"audio:buffer:{session_id}", 0, -1)

    async def clear_audio_buffer(self, session_id: str) -> bool:
        """Clear audio buffer for session"""
        return await self.delete(f"audio:buffer:{session_id}")

    # Rate limiting
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: int
    ) -> bool:
        """Check if rate limit is exceeded"""
        try:
            count = await self.client.incr(key)
            if count == 1:
                await self.expire(key, window)
            return count <= limit
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            return True  # Allow on error

    # Metrics
    async def increment_metric(
        self,
        metric_name: str,
        value: int = 1
    ) -> int:
        """Increment metric counter"""
        try:
            return await self.client.incrby(f"metric:{metric_name}", value)
        except Exception as e:
            self.logger.error(f"Failed to increment metric {metric_name}: {e}")
            return 0

    async def get_metric(self, metric_name: str) -> int:
        """Get metric value"""
        try:
            value = await self.get(f"metric:{metric_name}")
            return int(value) if value else 0
        except Exception as e:
            self.logger.error(f"Failed to get metric {metric_name}: {e}")
            return 0


# Global Redis client instance
redis_client: Optional[RedisClient] = None


def get_redis_client() -> RedisClient:
    """Get global Redis client instance"""
    global redis_client
    if redis_client is None:
        raise RuntimeError("Redis client not initialized. Call init_redis_client() first.")
    return redis_client


def init_redis_client(**kwargs) -> RedisClient:
    """Initialize global Redis client"""
    global redis_client
    redis_client = RedisClient(**kwargs)
    return redis_client
