"""
Redis client utilities for Morgan AI Assistant
"""
import asyncio
import logging
import json
import os
from typing import Optional, Any, Dict, List
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio import Redis, ConnectionPool

logger = logging.getLogger(__name__)


class RedisManager:
    """Redis connection manager for caching and session state"""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = 0,
        password: str = None,
        max_connections: int = 50,
        decode_responses: bool = True
    ):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.db = db
        self.password = password or os.getenv("REDIS_PASSWORD", None)
        self.max_connections = max_connections
        self.decode_responses = decode_responses
        
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[Redis] = None
        self.logger = logger

    async def connect(self):
        """Create Redis connection pool"""
        try:
            self.pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=self.decode_responses
            )
            self.client = Redis(connection_pool=self.pool)
            
            # Test connection
            await self.client.ping()
            self.logger.info(f"Connected to Redis: {self.host}:{self.port}/{self.db}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            self.logger.info("Disconnected from Redis")

    async def get(self, key: str) -> Optional[str]:
        """Get value by key"""
        try:
            return await self.client.get(key)
        except Exception as e:
            self.logger.error(f"Redis GET error for key {key}: {e}")
            return None

    async def set(
        self, 
        key: str, 
        value: str, 
        ex: int = None, 
        px: int = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """Set key-value pair with optional expiration"""
        try:
            return await self.client.set(key, value, ex=ex, px=px, nx=nx, xx=xx)
        except Exception as e:
            self.logger.error(f"Redis SET error for key {key}: {e}")
            return False

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys"""
        try:
            return await self.client.delete(*keys)
        except Exception as e:
            self.logger.error(f"Redis DELETE error: {e}")
            return 0

    async def exists(self, *keys: str) -> int:
        """Check if keys exist"""
        try:
            return await self.client.exists(*keys)
        except Exception as e:
            self.logger.error(f"Redis EXISTS error: {e}")
            return 0

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on a key"""
        try:
            return await self.client.expire(key, seconds)
        except Exception as e:
            self.logger.error(f"Redis EXPIRE error for key {key}: {e}")
            return False

    async def ttl(self, key: str) -> int:
        """Get time to live for a key"""
        try:
            return await self.client.ttl(key)
        except Exception as e:
            self.logger.error(f"Redis TTL error for key {key}: {e}")
            return -2

    # JSON operations
    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value by key"""
        value = await self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error for key {key}: {e}")
        return None

    async def set_json(self, key: str, value: Any, ex: int = None) -> bool:
        """Set JSON value"""
        try:
            json_str = json.dumps(value)
            return await self.set(key, json_str, ex=ex)
        except Exception as e:
            self.logger.error(f"JSON encode/set error for key {key}: {e}")
            return False

    # Hash operations
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field value"""
        try:
            return await self.client.hget(name, key)
        except Exception as e:
            self.logger.error(f"Redis HGET error for {name}:{key}: {e}")
            return None

    async def hset(self, name: str, key: str, value: str) -> int:
        """Set hash field value"""
        try:
            return await self.client.hset(name, key, value)
        except Exception as e:
            self.logger.error(f"Redis HSET error for {name}:{key}: {e}")
            return 0

    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields"""
        try:
            return await self.client.hgetall(name)
        except Exception as e:
            self.logger.error(f"Redis HGETALL error for {name}: {e}")
            return {}

    async def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields"""
        try:
            return await self.client.hdel(name, *keys)
        except Exception as e:
            self.logger.error(f"Redis HDEL error for {name}: {e}")
            return 0

    # List operations
    async def lpush(self, key: str, *values: str) -> int:
        """Push values to list head"""
        try:
            return await self.client.lpush(key, *values)
        except Exception as e:
            self.logger.error(f"Redis LPUSH error for {key}: {e}")
            return 0

    async def rpush(self, key: str, *values: str) -> int:
        """Push values to list tail"""
        try:
            return await self.client.rpush(key, *values)
        except Exception as e:
            self.logger.error(f"Redis RPUSH error for {key}: {e}")
            return 0

    async def lpop(self, key: str) -> Optional[str]:
        """Pop value from list head"""
        try:
            return await self.client.lpop(key)
        except Exception as e:
            self.logger.error(f"Redis LPOP error for {key}: {e}")
            return None

    async def rpop(self, key: str) -> Optional[str]:
        """Pop value from list tail"""
        try:
            return await self.client.rpop(key)
        except Exception as e:
            self.logger.error(f"Redis RPOP error for {key}: {e}")
            return None

    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        """Get list range"""
        try:
            return await self.client.lrange(key, start, end)
        except Exception as e:
            self.logger.error(f"Redis LRANGE error for {key}: {e}")
            return []

    async def llen(self, key: str) -> int:
        """Get list length"""
        try:
            return await self.client.llen(key)
        except Exception as e:
            self.logger.error(f"Redis LLEN error for {key}: {e}")
            return 0

    # Pub/Sub operations
    async def publish(self, channel: str, message: str) -> int:
        """Publish message to channel"""
        try:
            return await self.client.publish(channel, message)
        except Exception as e:
            self.logger.error(f"Redis PUBLISH error for channel {channel}: {e}")
            return 0

    async def subscribe(self, *channels: str):
        """Subscribe to channels"""
        pubsub = self.client.pubsub()
        await pubsub.subscribe(*channels)
        return pubsub

    # Cache helpers for Morgan-specific operations
    async def cache_conversation_context(
        self, 
        user_id: str, 
        context: Dict[str, Any], 
        ttl: int = 3600
    ) -> bool:
        """Cache conversation context"""
        key = f"conversation:{user_id}"
        return await self.set_json(key, context, ex=ttl)

    async def get_conversation_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached conversation context"""
        key = f"conversation:{user_id}"
        return await self.get_json(key)

    async def cache_streaming_session(
        self, 
        session_id: str, 
        data: Dict[str, Any], 
        ttl: int = 1800
    ) -> bool:
        """Cache streaming session data"""
        key = f"stream_session:{session_id}"
        return await self.set_json(key, data, ex=ttl)

    async def get_streaming_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached streaming session"""
        key = f"stream_session:{session_id}"
        return await self.get_json(key)

    async def delete_streaming_session(self, session_id: str) -> int:
        """Delete streaming session from cache"""
        key = f"stream_session:{session_id}"
        return await self.delete(key)

    async def cache_user_preferences(
        self, 
        user_id: str, 
        preferences: Dict[str, Any], 
        ttl: int = 86400
    ) -> bool:
        """Cache user preferences (24h default)"""
        key = f"user_prefs:{user_id}"
        return await self.set_json(key, preferences, ex=ttl)

    async def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user preferences"""
        key = f"user_prefs:{user_id}"
        return await self.get_json(key)

    async def add_audio_chunk(self, session_id: str, chunk_data: str) -> int:
        """Add audio chunk to session buffer"""
        key = f"audio_buffer:{session_id}"
        result = await self.rpush(key, chunk_data)
        # Set expiration on first chunk
        if result == 1:
            await self.expire(key, 1800)  # 30 minutes
        return result

    async def get_audio_chunks(self, session_id: str) -> List[str]:
        """Get all audio chunks for session"""
        key = f"audio_buffer:{session_id}"
        return await self.lrange(key, 0, -1)

    async def clear_audio_buffer(self, session_id: str) -> int:
        """Clear audio buffer for session"""
        key = f"audio_buffer:{session_id}"
        return await self.delete(key)

    async def increment_metric(self, metric_key: str, amount: int = 1) -> int:
        """Increment a metric counter"""
        try:
            return await self.client.incrby(metric_key, amount)
        except Exception as e:
            self.logger.error(f"Redis INCRBY error for {metric_key}: {e}")
            return 0

    async def set_active_user(self, user_id: str, ttl: int = 300) -> bool:
        """Mark user as active (5 min default)"""
        key = f"active_user:{user_id}"
        return await self.set(key, "1", ex=ttl)

    async def is_user_active(self, user_id: str) -> bool:
        """Check if user is active"""
        key = f"active_user:{user_id}"
        return await self.exists(key) > 0

    async def health_check(self) -> bool:
        """Check Redis health"""
        try:
            await self.client.ping()
            return True
        except Exception as e:
            self.logger.error(f"Redis health check failed: {e}")
            return False


# Global Redis manager instance
redis_manager: Optional[RedisManager] = None


async def get_redis_manager() -> RedisManager:
    """Get or create global Redis manager"""
    global redis_manager
    if redis_manager is None:
        redis_manager = RedisManager()
        await redis_manager.connect()
    return redis_manager


async def close_redis_manager():
    """Close global Redis manager"""
    global redis_manager
    if redis_manager:
        await redis_manager.disconnect()
        redis_manager = None

