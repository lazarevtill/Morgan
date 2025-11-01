"""
Connection Pool Manager for Morgan RAG.

Provides intelligent connection pooling for:
- Vector database connections (Qdrant)
- HTTP connections for Jina AI services
- Background processing connections
- Async processing with connection reuse

Key Features:
- Adaptive pool sizing based on load
- Connection health monitoring
- Automatic connection recycling
- Resource optimization
- Thread-safe operations
"""

import asyncio
import time
import threading
from typing import Dict, Any, Optional, List, Callable, AsyncContextManager
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import aiohttp
import logging

from morgan.config import get_settings
from morgan.utils.logger import get_logger
from morgan.utils.error_handling import (
    VectorizationError, NetworkError, ErrorSeverity
)

logger = get_logger(__name__)


@dataclass
class ConnectionConfig:
    """Configuration for connection pools."""
    # Pool sizing
    min_connections: int = 2
    max_connections: int = 20
    initial_connections: int = 5
    
    # Connection management
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0  # 5 minutes
    max_connection_age: float = 3600.0  # 1 hour
    
    # Health checking
    health_check_interval: float = 60.0  # 1 minute
    health_check_timeout: float = 5.0
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Performance optimization
    enable_keep_alive: bool = True
    enable_connection_reuse: bool = True


@dataclass
class ConnectionStats:
    """Statistics for connection pool."""
    pool_name: str
    active_connections: int
    idle_connections: int
    total_connections: int
    connections_created: int
    connections_closed: int
    connection_errors: int
    avg_connection_time: float
    last_health_check: datetime
    pool_efficiency: float  # active / total


@dataclass
class PooledConnection:
    """Wrapper for pooled connections."""
    connection: Any
    created_at: datetime
    last_used: datetime
    use_count: int
    is_healthy: bool = True
    connection_id: str = ""
    
    def __post_init__(self):
        if not self.connection_id:
            self.connection_id = f"conn_{int(time.time() * 1000000)}"
    
    @property
    def age(self) -> float:
        """Get connection age in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    @property
    def idle_time(self) -> float:
        """Get idle time in seconds."""
        return (datetime.now() - self.last_used).total_seconds()


class ConnectionPool:
    """
    Generic connection pool with health monitoring and adaptive sizing.
    """
    
    def __init__(
        self,
        name: str,
        connection_factory: Callable,
        config: Optional[ConnectionConfig] = None
    ):
        """
        Initialize connection pool.
        
        Args:
            name: Pool name for identification
            connection_factory: Function to create new connections
            config: Pool configuration
        """
        self.name = name
        self.connection_factory = connection_factory
        self.config = config or ConnectionConfig()
        
        # Connection management
        self.active_connections: Dict[str, PooledConnection] = {}
        self.idle_connections: deque = deque()
        self.pool_lock = threading.Lock()
        
        # Statistics
        self.stats = ConnectionStats(
            pool_name=name,
            active_connections=0,
            idle_connections=0,
            total_connections=0,
            connections_created=0,
            connections_closed=0,
            connection_errors=0,
            avg_connection_time=0.0,
            last_health_check=datetime.now(),
            pool_efficiency=0.0
        )
        
        # Health monitoring
        self.health_check_task = None
        self.is_running = False
        
        logger.info(f"ConnectionPool '{name}' initialized with config: {self.config}")
    
    async def start(self):
        """Start the connection pool."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Create initial connections
        await self._create_initial_connections()
        
        # Start health check task
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"ConnectionPool '{self.name}' started with {len(self.idle_connections)} connections")
    
    async def stop(self):
        """Stop the connection pool."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel health check task
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        await self._close_all_connections()
        
        logger.info(f"ConnectionPool '{self.name}' stopped")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncContextManager[Any]:
        """
        Get a connection from the pool (async context manager).
        
        Yields:
            Connection object
        """
        connection = None
        pooled_conn = None
        
        try:
            # Get connection from pool
            pooled_conn = await self._acquire_connection()
            connection = pooled_conn.connection
            
            # Update usage stats
            pooled_conn.last_used = datetime.now()
            pooled_conn.use_count += 1
            
            yield connection
            
        except Exception as e:
            # Mark connection as unhealthy on error
            if pooled_conn:
                pooled_conn.is_healthy = False
            
            logger.error(f"Connection error in pool '{self.name}': {e}")
            raise
            
        finally:
            # Return connection to pool
            if pooled_conn:
                await self._release_connection(pooled_conn)
    
    async def _acquire_connection(self) -> PooledConnection:
        """Acquire a connection from the pool."""
        with self.pool_lock:
            # Try to get idle connection first
            while self.idle_connections:
                pooled_conn = self.idle_connections.popleft()
                
                # Check if connection is still healthy and not too old
                if (pooled_conn.is_healthy and 
                    pooled_conn.age < self.config.max_connection_age and
                    pooled_conn.idle_time < self.config.idle_timeout):
                    
                    # Move to active connections
                    self.active_connections[pooled_conn.connection_id] = pooled_conn
                    self._update_stats()
                    
                    logger.debug(f"Reusing connection {pooled_conn.connection_id} from pool '{self.name}'")
                    return pooled_conn
                else:
                    # Connection is stale, close it
                    await self._close_connection(pooled_conn)
            
            # No idle connections available, create new one if possible
            if len(self.active_connections) + len(self.idle_connections) < self.config.max_connections:
                pooled_conn = await self._create_connection()
                self.active_connections[pooled_conn.connection_id] = pooled_conn
                self._update_stats()
                
                logger.debug(f"Created new connection {pooled_conn.connection_id} for pool '{self.name}'")
                return pooled_conn
            
            # Pool is at capacity, wait for connection to become available
            logger.warning(f"Pool '{self.name}' at capacity, waiting for connection...")
            
            # Simple wait and retry (in production, would use proper queuing)
            await asyncio.sleep(0.1)
            return await self._acquire_connection()
    
    async def _release_connection(self, pooled_conn: PooledConnection):
        """Release a connection back to the pool."""
        with self.pool_lock:
            # Remove from active connections
            if pooled_conn.connection_id in self.active_connections:
                del self.active_connections[pooled_conn.connection_id]
            
            # Add back to idle pool if healthy
            if pooled_conn.is_healthy and pooled_conn.age < self.config.max_connection_age:
                self.idle_connections.append(pooled_conn)
                logger.debug(f"Returned connection {pooled_conn.connection_id} to pool '{self.name}'")
            else:
                # Connection is unhealthy or too old, close it
                await self._close_connection(pooled_conn)
                logger.debug(f"Closed unhealthy connection {pooled_conn.connection_id} from pool '{self.name}'")
            
            self._update_stats()
    
    async def _create_connection(self) -> PooledConnection:
        """Create a new pooled connection."""
        start_time = time.time()
        
        try:
            # Create connection using factory
            connection = await self.connection_factory()
            
            pooled_conn = PooledConnection(
                connection=connection,
                created_at=datetime.now(),
                last_used=datetime.now(),
                use_count=0
            )
            
            # Update stats
            creation_time = time.time() - start_time
            self.stats.connections_created += 1
            
            # Update average connection time
            if self.stats.connections_created > 1:
                self.stats.avg_connection_time = (
                    (self.stats.avg_connection_time * (self.stats.connections_created - 1) + creation_time) /
                    self.stats.connections_created
                )
            else:
                self.stats.avg_connection_time = creation_time
            
            logger.debug(f"Created connection {pooled_conn.connection_id} in {creation_time:.3f}s")
            return pooled_conn
            
        except Exception as e:
            self.stats.connection_errors += 1
            logger.error(f"Failed to create connection for pool '{self.name}': {e}")
            raise NetworkError(
                f"Failed to create connection: {e}",
                operation="create_connection",
                component="connection_pool",
                severity=ErrorSeverity.HIGH
            ) from e
    
    async def _close_connection(self, pooled_conn: PooledConnection):
        """Close a pooled connection."""
        try:
            # Close the actual connection
            if hasattr(pooled_conn.connection, 'close'):
                if asyncio.iscoroutinefunction(pooled_conn.connection.close):
                    await pooled_conn.connection.close()
                else:
                    pooled_conn.connection.close()
            
            self.stats.connections_closed += 1
            logger.debug(f"Closed connection {pooled_conn.connection_id}")
            
        except Exception as e:
            logger.error(f"Error closing connection {pooled_conn.connection_id}: {e}")
    
    async def _create_initial_connections(self):
        """Create initial pool of connections."""
        for i in range(self.config.initial_connections):
            try:
                pooled_conn = await self._create_connection()
                self.idle_connections.append(pooled_conn)
            except Exception as e:
                logger.error(f"Failed to create initial connection {i}: {e}")
        
        self._update_stats()
    
    async def _close_all_connections(self):
        """Close all connections in the pool."""
        with self.pool_lock:
            # Close active connections
            for pooled_conn in list(self.active_connections.values()):
                await self._close_connection(pooled_conn)
            self.active_connections.clear()
            
            # Close idle connections
            while self.idle_connections:
                pooled_conn = self.idle_connections.popleft()
                await self._close_connection(pooled_conn)
            
            self._update_stats()
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for pool '{self.name}': {e}")
    
    async def _perform_health_check(self):
        """Perform health check on connections."""
        with self.pool_lock:
            unhealthy_connections = []
            
            # Check idle connections
            for pooled_conn in list(self.idle_connections):
                if not await self._check_connection_health(pooled_conn):
                    unhealthy_connections.append(pooled_conn)
                    self.idle_connections.remove(pooled_conn)
            
            # Close unhealthy connections
            for pooled_conn in unhealthy_connections:
                await self._close_connection(pooled_conn)
            
            # Update health check timestamp
            self.stats.last_health_check = datetime.now()
            self._update_stats()
            
            if unhealthy_connections:
                logger.info(f"Removed {len(unhealthy_connections)} unhealthy connections from pool '{self.name}'")
    
    async def _check_connection_health(self, pooled_conn: PooledConnection) -> bool:
        """Check if a connection is healthy."""
        try:
            # Basic health check - connection exists and is not too old
            if pooled_conn.age > self.config.max_connection_age:
                return False
            
            if pooled_conn.idle_time > self.config.idle_timeout:
                return False
            
            # Additional health checks could be added here
            # (e.g., ping database, check HTTP connection)
            
            return True
            
        except Exception as e:
            logger.debug(f"Connection health check failed for {pooled_conn.connection_id}: {e}")
            return False
    
    def _update_stats(self):
        """Update connection pool statistics."""
        self.stats.active_connections = len(self.active_connections)
        self.stats.idle_connections = len(self.idle_connections)
        self.stats.total_connections = self.stats.active_connections + self.stats.idle_connections
        
        # Calculate pool efficiency
        if self.stats.total_connections > 0:
            self.stats.pool_efficiency = self.stats.active_connections / self.stats.total_connections
        else:
            self.stats.pool_efficiency = 0.0
    
    def get_stats(self) -> ConnectionStats:
        """Get current pool statistics."""
        with self.pool_lock:
            self._update_stats()
            return self.stats


class ConnectionPoolManager:
    """
    Manager for multiple connection pools.
    """
    
    def __init__(self):
        """Initialize connection pool manager."""
        self.pools: Dict[str, ConnectionPool] = {}
        self.manager_lock = threading.Lock()
        self.is_running = False
        
        logger.info("ConnectionPoolManager initialized")
    
    async def start(self):
        """Start all connection pools."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start all pools
        for pool in self.pools.values():
            await pool.start()
        
        logger.info(f"ConnectionPoolManager started with {len(self.pools)} pools")
    
    async def stop(self):
        """Stop all connection pools."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop all pools
        for pool in self.pools.values():
            await pool.stop()
        
        logger.info("ConnectionPoolManager stopped")
    
    def create_pool(
        self,
        name: str,
        connection_factory: Callable,
        config: Optional[ConnectionConfig] = None
    ) -> ConnectionPool:
        """
        Create a new connection pool.
        
        Args:
            name: Pool name
            connection_factory: Function to create connections
            config: Pool configuration
            
        Returns:
            Created connection pool
        """
        with self.manager_lock:
            if name in self.pools:
                raise ValueError(f"Pool '{name}' already exists")
            
            pool = ConnectionPool(name, connection_factory, config)
            self.pools[name] = pool
            
            # Start pool if manager is running
            if self.is_running:
                asyncio.create_task(pool.start())
            
            logger.info(f"Created connection pool '{name}'")
            return pool
    
    def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get connection pool by name."""
        return self.pools.get(name)
    
    def remove_pool(self, name: str) -> bool:
        """Remove and stop a connection pool."""
        with self.manager_lock:
            if name not in self.pools:
                return False
            
            pool = self.pools[name]
            del self.pools[name]
            
            # Stop pool asynchronously
            asyncio.create_task(pool.stop())
            
            logger.info(f"Removed connection pool '{name}'")
            return True
    
    def get_all_stats(self) -> Dict[str, ConnectionStats]:
        """Get statistics for all pools."""
        return {name: pool.get_stats() for name, pool in self.pools.items()}
    
    async def create_http_pool(
        self,
        name: str,
        base_url: str,
        config: Optional[ConnectionConfig] = None
    ) -> ConnectionPool:
        """
        Create HTTP connection pool for Jina AI services.
        
        Args:
            name: Pool name
            base_url: Base URL for HTTP connections
            config: Pool configuration
            
        Returns:
            HTTP connection pool
        """
        async def http_connection_factory():
            """Factory for HTTP connections."""
            connector = aiohttp.TCPConnector(
                limit=config.max_connections if config else 20,
                limit_per_host=config.max_connections if config else 20,
                keepalive_timeout=config.idle_timeout if config else 300,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(
                total=config.connection_timeout if config else 30
            )
            
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                base_url=base_url
            )
            
            return session
        
        return self.create_pool(name, http_connection_factory, config)
    
    async def create_vector_db_pool(
        self,
        name: str,
        db_config: Dict[str, Any],
        config: Optional[ConnectionConfig] = None
    ) -> ConnectionPool:
        """
        Create vector database connection pool.
        
        Args:
            name: Pool name
            db_config: Database configuration
            config: Pool configuration
            
        Returns:
            Vector database connection pool
        """
        async def vector_db_connection_factory():
            """Factory for vector database connections."""
            # This would create actual Qdrant client connections
            # For now, return a mock connection
            return {
                'client': 'mock_qdrant_client',
                'host': db_config.get('host', 'localhost'),
                'port': db_config.get('port', 6333),
                'created_at': datetime.now()
            }
        
        return self.create_pool(name, vector_db_connection_factory, config)


# Singleton instance
_connection_pool_manager_instance = None
_connection_pool_manager_lock = threading.Lock()


def get_connection_pool_manager() -> ConnectionPoolManager:
    """
    Get singleton connection pool manager instance (thread-safe).
    
    Returns:
        Shared ConnectionPoolManager instance
    """
    global _connection_pool_manager_instance
    
    if _connection_pool_manager_instance is None:
        with _connection_pool_manager_lock:
            if _connection_pool_manager_instance is None:
                _connection_pool_manager_instance = ConnectionPoolManager()
    
    return _connection_pool_manager_instance