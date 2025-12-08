"""
Configuration management for Morgan CLI.

This module handles loading configuration from environment variables,
command-line arguments, and default values.
"""

import os
from typing import Optional
from pathlib import Path

from pydantic import BaseModel, Field
from dotenv import load_dotenv


# Load environment variables from .env file if present
load_dotenv()


class Config(BaseModel):
    """Client configuration."""

    model_config = {"frozen": True}

    server_url: str = Field(
        default_factory=lambda: os.getenv(
            "MORGAN_SERVER_URL", "http://localhost:8080"
        ),
        description="Morgan server URL"
    )
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("MORGAN_API_KEY"),
        description="API key for authentication"
    )
    user_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("MORGAN_USER_ID"),
        description="User identifier"
    )
    timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("MORGAN_TIMEOUT", "60")),
        ge=1,
        description="Request timeout in seconds"
    )
    retry_attempts: int = Field(
        default_factory=lambda: int(os.getenv("MORGAN_RETRY_ATTEMPTS", "3")),
        ge=0,
        description="Number of retry attempts"
    )
    retry_delay_seconds: int = Field(
        default_factory=lambda: int(os.getenv("MORGAN_RETRY_DELAY", "2")),
        ge=0,
        description="Delay between retries in seconds"
    )

    @classmethod
    def from_env(cls) -> "Config":
        """
        Create configuration from environment variables.

        Returns:
            Config instance
        """
        return cls()

    def with_overrides(
        self,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        retry_attempts: Optional[int] = None,
        retry_delay_seconds: Optional[int] = None
    ) -> "Config":
        """
        Create a new config with overridden values.

        Args:
            server_url: Override server URL
            api_key: Override API key
            user_id: Override user ID
            timeout_seconds: Override timeout
            retry_attempts: Override retry attempts
            retry_delay_seconds: Override retry delay

        Returns:
            New Config instance with overrides
        """
        data = self.model_dump()

        if server_url is not None:
            data["server_url"] = server_url
        if api_key is not None:
            data["api_key"] = api_key
        if user_id is not None:
            data["user_id"] = user_id
        if timeout_seconds is not None:
            data["timeout_seconds"] = timeout_seconds
        if retry_attempts is not None:
            data["retry_attempts"] = retry_attempts
        if retry_delay_seconds is not None:
            data["retry_delay_seconds"] = retry_delay_seconds

        return Config(**data)


def get_config(
    server_url: Optional[str] = None,
    api_key: Optional[str] = None,
    user_id: Optional[str] = None
) -> Config:
    """
    Get configuration with optional overrides.

    Args:
        server_url: Override server URL
        api_key: Override API key
        user_id: Override user ID

    Returns:
        Config instance
    """
    config = Config.from_env()

    if server_url or api_key or user_id:
        config = config.with_overrides(
            server_url=server_url,
            api_key=api_key,
            user_id=user_id
        )

    return config
