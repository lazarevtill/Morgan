"""
HTTP/WebSocket Client for Morgan Server

This module provides the client implementation for communicating with the Morgan server.
It handles HTTP REST API calls and WebSocket connections for real-time chat.
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Callable
from urllib.parse import urljoin

import aiohttp
import websockets
from pydantic import BaseModel, Field, ValidationError


# ============================================================================
# Configuration
# ============================================================================

class ClientConfig(BaseModel):
    """Client configuration."""
    
    model_config = {"frozen": True}
    
    server_url: str = Field(default="http://localhost:8080", description="Server URL")
    api_key: Optional[str] = Field(default=None, description="API key (if required)")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    timeout_seconds: int = Field(default=60, ge=1, description="Request timeout")
    retry_attempts: int = Field(default=3, ge=0, description="Number of retry attempts")
    retry_delay_seconds: int = Field(default=2, ge=0, description="Delay between retries")


# ============================================================================
# Connection Status
# ============================================================================

class ConnectionStatus(str, Enum):
    """Connection status enumeration."""
    
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


# ============================================================================
# Exceptions
# ============================================================================

class MorganClientError(Exception):
    """Base exception for Morgan client errors."""
    pass


class ConnectionError(MorganClientError):
    """Raised when connection to server fails."""
    pass


class RequestError(MorganClientError):
    """Raised when a request fails."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


class TimeoutError(MorganClientError):
    """Raised when a request times out."""
    pass


class ValidationError(MorganClientError):
    """Raised when response validation fails."""
    pass


# ============================================================================
# HTTP Client
# ============================================================================

class HTTPClient:
    """HTTP client for REST API calls."""
    
    def __init__(self, config: ClientConfig):
        """
        Initialize HTTP client.
        
        Args:
            config: Client configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._session: Optional[aiohttp.ClientSession] = None
        self._status = ConnectionStatus.DISCONNECTED
        self._status_callbacks: List[Callable[[ConnectionStatus], None]] = []
    
    @property
    def status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._status
    
    def add_status_callback(self, callback: Callable[[ConnectionStatus], None]) -> None:
        """
        Add a callback for status changes.
        
        Args:
            callback: Function to call when status changes
        """
        self._status_callbacks.append(callback)
    
    def _set_status(self, status: ConnectionStatus) -> None:
        """
        Set connection status and notify callbacks.
        
        Args:
            status: New connection status
        """
        if status != self._status:
            self._status = status
            for callback in self._status_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    self.logger.error(f"Error in status callback: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def connect(self) -> None:
        """Establish connection (create session)."""
        if self._session is None or self._session.closed:
            self._set_status(ConnectionStatus.CONNECTING)
            
            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout,
                raise_for_status=False
            )
            
            self._set_status(ConnectionStatus.CONNECTED)
            self.logger.info(f"Connected to {self.config.server_url}")
    
    async def close(self) -> None:
        """Close connection (close session)."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            self._set_status(ConnectionStatus.DISCONNECTED)
            self.logger.info("Disconnected from server")
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., "/api/chat")
            json_data: JSON request body
            params: Query parameters
            retry_count: Current retry attempt
            
        Returns:
            Response data as dictionary
            
        Raises:
            ConnectionError: If connection fails
            RequestError: If request fails
            TimeoutError: If request times out
        """
        if not self._session or self._session.closed:
            await self.connect()
        
        url = urljoin(self.config.server_url, endpoint)
        
        try:
            async with self._session.request(
                method,
                url,
                json=json_data,
                params=params
            ) as response:
                # Read response body
                try:
                    response_data = await response.json()
                except aiohttp.ContentTypeError:
                    response_text = await response.text()
                    response_data = {"error": "INVALID_RESPONSE", "message": response_text}
                
                # Handle error responses
                if response.status >= 400:
                    error_msg = response_data.get("message", f"Request failed with status {response.status}")
                    error_code = response_data.get("error", "UNKNOWN_ERROR")
                    
                    # Retry on 5xx errors
                    if response.status >= 500 and retry_count < self.config.retry_attempts:
                        self.logger.warning(
                            f"Request failed with {response.status}, retrying "
                            f"({retry_count + 1}/{self.config.retry_attempts})..."
                        )
                        await asyncio.sleep(self.config.retry_delay_seconds)
                        return await self._request(method, endpoint, json_data, params, retry_count + 1)
                    
                    raise RequestError(
                        error_msg,
                        status_code=response.status,
                        details=response_data.get("details", {})
                    )
                
                return response_data
        
        except asyncio.TimeoutError:
            if retry_count < self.config.retry_attempts:
                self.logger.warning(
                    f"Request timed out, retrying ({retry_count + 1}/{self.config.retry_attempts})..."
                )
                await asyncio.sleep(self.config.retry_delay_seconds)
                return await self._request(method, endpoint, json_data, params, retry_count + 1)
            
            raise TimeoutError(f"Request to {endpoint} timed out after {self.config.timeout_seconds}s")
        
        except aiohttp.ClientError as e:
            if retry_count < self.config.retry_attempts:
                self.logger.warning(
                    f"Connection error: {e}, retrying ({retry_count + 1}/{self.config.retry_attempts})..."
                )
                self._set_status(ConnectionStatus.RECONNECTING)
                await asyncio.sleep(self.config.retry_delay_seconds)
                await self.connect()
                return await self._request(method, endpoint, json_data, params, retry_count + 1)
            
            self._set_status(ConnectionStatus.ERROR)
            raise ConnectionError(f"Failed to connect to {url}: {e}")
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GET request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data
        """
        return await self._request("GET", endpoint, params=params)
    
    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a POST request.
        
        Args:
            endpoint: API endpoint
            data: Request body
            
        Returns:
            Response data
        """
        return await self._request("POST", endpoint, json_data=data)
    
    async def put(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a PUT request.
        
        Args:
            endpoint: API endpoint
            data: Request body
            
        Returns:
            Response data
        """
        return await self._request("PUT", endpoint, json_data=data)
    
    async def delete(self, endpoint: str) -> Dict[str, Any]:
        """
        Make a DELETE request.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            Response data
        """
        return await self._request("DELETE", endpoint)
    
    # ========================================================================
    # API Methods
    # ========================================================================
    
    async def chat(
        self,
        message: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a chat message.
        
        Args:
            message: User message
            user_id: User identifier
            conversation_id: Conversation identifier
            metadata: Additional metadata
            
        Returns:
            Chat response
        """
        data = {
            "message": message,
            "user_id": user_id or self.config.user_id,
            "conversation_id": conversation_id,
            "metadata": metadata or {}
        }
        return await self.post("/api/chat", data)
    
    async def get_memory_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Args:
            user_id: User identifier
            
        Returns:
            Memory statistics
        """
        params = {"user_id": user_id or self.config.user_id} if user_id or self.config.user_id else None
        return await self.get("/api/memory/stats", params=params)
    
    async def search_memory(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search conversation memory.
        
        Args:
            query: Search query
            user_id: User identifier
            limit: Maximum results
            
        Returns:
            List of search results
        """
        data = {
            "query": query,
            "user_id": user_id or self.config.user_id,
            "limit": limit
        }
        response = await self.post("/api/memory/search", data)
        return response.get("results", [])
    
    async def cleanup_memory(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Clean up old conversations.
        
        Args:
            user_id: User identifier
            
        Returns:
            Cleanup result
        """
        params = {"user_id": user_id or self.config.user_id} if user_id or self.config.user_id else None
        return await self.delete("/api/memory/cleanup")
    
    async def learn(
        self,
        source: Optional[str] = None,
        url: Optional[str] = None,
        content: Optional[str] = None,
        doc_type: str = "auto",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add documents to knowledge base.
        
        Args:
            source: File path
            url: URL to fetch
            content: Direct content
            doc_type: Document type
            metadata: Document metadata
            
        Returns:
            Learn response
        """
        data = {
            "source": source,
            "url": url,
            "content": content,
            "doc_type": doc_type,
            "metadata": metadata or {}
        }
        return await self.post("/api/knowledge/learn", data)
    
    async def search_knowledge(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of search results
        """
        params = {"query": query, "limit": limit}
        response = await self.get("/api/knowledge/search", params=params)
        return response.get("results", [])
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Returns:
            Knowledge statistics
        """
        return await self.get("/api/knowledge/stats")
    
    async def get_profile(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get user profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile
        """
        uid = user_id or self.config.user_id or "default"
        return await self.get(f"/api/profile/{uid}")
    
    async def update_profile(
        self,
        user_id: Optional[str] = None,
        communication_style: Optional[str] = None,
        response_length: Optional[str] = None,
        topics_of_interest: Optional[List[str]] = None,
        preferred_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update user preferences.
        
        Args:
            user_id: User identifier
            communication_style: Communication style preference
            response_length: Response length preference
            topics_of_interest: Topics of interest
            preferred_name: Preferred name
            
        Returns:
            Updated profile
        """
        uid = user_id or self.config.user_id or "default"
        data = {
            "communication_style": communication_style,
            "response_length": response_length,
            "topics_of_interest": topics_of_interest,
            "preferred_name": preferred_name
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return await self.put(f"/api/profile/{uid}", data)
    
    async def get_timeline(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get user timeline.
        
        Args:
            user_id: User identifier
            
        Returns:
            User timeline
        """
        uid = user_id or self.config.user_id or "default"
        return await self.get(f"/api/timeline/{uid}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check server health.
        
        Returns:
            Health status
        """
        return await self.get("/health")
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get detailed server status.
        
        Returns:
            Server status
        """
        return await self.get("/api/status")


# ============================================================================
# WebSocket Client
# ============================================================================

class WebSocketClient:
    """WebSocket client for real-time chat."""
    
    def __init__(self, config: ClientConfig):
        """
        Initialize WebSocket client.
        
        Args:
            config: Client configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._status = ConnectionStatus.DISCONNECTED
        self._status_callbacks: List[Callable[[ConnectionStatus], None]] = []
        self._reconnect_task: Optional[asyncio.Task] = None
        self._should_reconnect = True
    
    @property
    def status(self) -> ConnectionStatus:
        """Get current connection status."""
        return self._status
    
    def add_status_callback(self, callback: Callable[[ConnectionStatus], None]) -> None:
        """
        Add a callback for status changes.
        
        Args:
            callback: Function to call when status changes
        """
        self._status_callbacks.append(callback)
    
    def _set_status(self, status: ConnectionStatus) -> None:
        """
        Set connection status and notify callbacks.
        
        Args:
            status: New connection status
        """
        if status != self._status:
            self._status = status
            for callback in self._status_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    self.logger.error(f"Error in status callback: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def _get_ws_url(self) -> str:
        """Get WebSocket URL from server URL."""
        server_url = self.config.server_url
        # Convert http:// to ws:// and https:// to wss://
        if server_url.startswith("http://"):
            ws_url = server_url.replace("http://", "ws://", 1)
        elif server_url.startswith("https://"):
            ws_url = server_url.replace("https://", "wss://", 1)
        else:
            ws_url = f"ws://{server_url}"
        
        user_id = self.config.user_id or "default"
        return f"{ws_url}/ws/{user_id}"
    
    async def connect(self) -> None:
        """Establish WebSocket connection."""
        if self._ws and not self._ws.closed:
            return
        
        self._set_status(ConnectionStatus.CONNECTING)
        
        ws_url = self._get_ws_url()
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        try:
            self._ws = await websockets.connect(
                ws_url,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            self._set_status(ConnectionStatus.CONNECTED)
            self.logger.info(f"WebSocket connected to {ws_url}")
        except Exception as e:
            self._set_status(ConnectionStatus.ERROR)
            raise ConnectionError(f"Failed to connect to WebSocket: {e}")
    
    async def close(self) -> None:
        """Close WebSocket connection."""
        self._should_reconnect = False
        
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        
        if self._ws and not self._ws.closed:
            await self._ws.close()
            self._ws = None
            self._set_status(ConnectionStatus.DISCONNECTED)
            self.logger.info("WebSocket disconnected")
    
    async def send_message(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send a message over WebSocket.
        
        Args:
            message: User message
            conversation_id: Conversation identifier
            metadata: Additional metadata
            
        Raises:
            ConnectionError: If not connected
        """
        if not self._ws or self._ws.closed:
            raise ConnectionError("WebSocket not connected")
        
        data = {
            "message": message,
            "conversation_id": conversation_id,
            "metadata": metadata or {}
        }
        
        await self._ws.send(json.dumps(data))
    
    async def receive_messages(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Receive messages from WebSocket.
        
        Yields:
            Message data
            
        Raises:
            ConnectionError: If connection fails
        """
        if not self._ws or self._ws.closed:
            raise ConnectionError("WebSocket not connected")
        
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    yield data
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse message: {e}")
                    continue
        except websockets.exceptions.ConnectionClosed:
            self._set_status(ConnectionStatus.DISCONNECTED)
            if self._should_reconnect:
                self.logger.info("WebSocket connection closed, attempting to reconnect...")
                self._set_status(ConnectionStatus.RECONNECTING)
                await self._reconnect()
        except Exception as e:
            self._set_status(ConnectionStatus.ERROR)
            raise ConnectionError(f"WebSocket error: {e}")
    
    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        retry_count = 0
        while self._should_reconnect and retry_count < self.config.retry_attempts:
            try:
                delay = self.config.retry_delay_seconds * (2 ** retry_count)
                self.logger.info(f"Reconnecting in {delay}s...")
                await asyncio.sleep(delay)
                
                await self.connect()
                self.logger.info("Reconnected successfully")
                return
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Reconnection attempt {retry_count} failed: {e}")
        
        if retry_count >= self.config.retry_attempts:
            self._set_status(ConnectionStatus.ERROR)
            self.logger.error("Max reconnection attempts reached")


# ============================================================================
# Unified Client
# ============================================================================

class MorganClient:
    """
    Unified client for Morgan server.
    
    Provides both HTTP and WebSocket interfaces.
    """
    
    def __init__(self, config: ClientConfig):
        """
        Initialize Morgan client.
        
        Args:
            config: Client configuration
        """
        self.config = config
        self.http = HTTPClient(config)
        self.ws = WebSocketClient(config)
        self.logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.http.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self) -> None:
        """Close all connections."""
        await self.http.close()
        await self.ws.close()
    
    @property
    def status(self) -> ConnectionStatus:
        """Get HTTP connection status."""
        return self.http.status
    
    @property
    def ws_status(self) -> ConnectionStatus:
        """Get WebSocket connection status."""
        return self.ws.status
