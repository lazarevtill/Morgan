"""
HTTP client utilities for service-to-service communication
"""
import asyncio
import json
from typing import Dict, Any, Optional, Union
import logging
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientTimeout, ClientError

from .errors import ServiceError, ErrorCode, ErrorHandler
from ..models.base import ProcessingResult

logger = logging.getLogger(__name__)


class MorganHTTPClient:
    """Enhanced HTTP client for service communication"""

    def __init__(self, service_name: str, base_url: str, timeout: float = 30.0,
                 max_retries: int = 3, retry_delay: float = 1.0,
                 logger: Optional[logging.Logger] = None):
        self.service_name = service_name
        self.base_url = base_url.rstrip('/')
        self.timeout = ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session: Optional[aiohttp.ClientSession] = None
        self.error_handler = ErrorHandler(logger or logging.getLogger(__name__))

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    async def connect(self):
        """Establish HTTP session"""
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

    async def disconnect(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> ProcessingResult:
        """Make HTTP request with retries and error handling"""
        url = urljoin(self.base_url, endpoint)

        for attempt in range(self.max_retries):
            try:
                if not self.session:
                    await self.connect()

                async with self.session.request(method, url, **kwargs) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        raise ServiceError(
                            f"HTTP {response.status}: {error_text}",
                            ErrorCode.SERVICE_ERROR,
                            {"status": response.status, "url": url}
                        )

                    content = await response.json()
                    return ProcessingResult(
                        success=True,
                        data=content,
                        metadata={"status": response.status, "url": url}
                    )

            except ClientError as e:
                if attempt == self.max_retries - 1:
                    raise ServiceError(
                        f"Failed to connect to {self.service_name}: {e}",
                        ErrorCode.CONNECTION_ERROR,
                        {"url": url, "attempts": attempt + 1}
                    )

                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

            except asyncio.TimeoutError:
                if attempt == self.max_retries - 1:
                    raise ServiceError(
                        f"Timeout connecting to {self.service_name}",
                        ErrorCode.SERVICE_TIMEOUT,
                        {"url": url, "timeout": self.timeout.total}
                    )

                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                await asyncio.sleep(self.retry_delay * (2 ** attempt))

            except Exception as e:
                raise ServiceError(
                    f"Unexpected error in {self.service_name}: {e}",
                    ErrorCode.SERVICE_ERROR,
                    {"url": url, "error_type": type(e).__name__}
                )

        # This should never be reached, but just in case
        raise ServiceError(
            f"Max retries exceeded for {self.service_name}",
            ErrorCode.SERVICE_ERROR,
            {"url": url, "max_retries": self.max_retries}
        )

    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Make GET request"""
        return await self._make_request("GET", endpoint, params=params)

    async def post(self, endpoint: str, data: Optional[Union[Dict[str, Any], str]] = None,
                   json_data: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Make POST request"""
        if json_data:
            return await self._make_request("POST", endpoint, json=json_data)
        else:
            return await self._make_request("POST", endpoint, data=data)

    async def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Make PUT request"""
        return await self._make_request("PUT", endpoint, json=data)

    async def delete(self, endpoint: str) -> ProcessingResult:
        """Make DELETE request"""
        return await self._make_request("DELETE", endpoint)

    async def health_check(self) -> bool:
        """Check if service is healthy"""
        try:
            result = await self.get("/health")
            return result.success
        except Exception:
            return False


class ServiceRegistry:
    """Registry for managing service clients"""

    def __init__(self):
        self.clients: Dict[str, MorganHTTPClient] = {}
        self.logger = logging.getLogger(__name__)

    def register_service(self, name: str, base_url: str, **kwargs) -> MorganHTTPClient:
        """Register a service client"""
        if name in self.clients:
            self.logger.warning(f"Service {name} already registered, replacing")

        client = MorganHTTPClient(name, base_url, **kwargs)
        self.clients[name] = client
        return client

    async def get_service(self, name: str) -> MorganHTTPClient:
        """Get a registered service client"""
        if name not in self.clients:
            raise ValueError(f"Service {name} not registered")

        client = self.clients[name]
        if not client.session:
            await client.connect()

        return client

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all registered services"""
        results = {}
        for name, client in self.clients.items():
            results[name] = await client.health_check()
        return results

    async def disconnect_all(self):
        """Disconnect all service clients"""
        for client in self.clients.values():
            await client.disconnect()


# Global service registry instance
service_registry = ServiceRegistry()
