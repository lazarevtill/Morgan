# Copyright 2025 Morgan AI Assistant Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Singleton Factory Utility for Morgan AI Assistant.

Provides a thread-safe singleton pattern that can be used across all services.
Eliminates the 50+ repeated singleton patterns throughout the codebase.

Usage:
    from morgan.utils.singleton import SingletonFactory

    # Define a service class
    class MyService:
        def __init__(self, config=None):
            self.config = config

    # Create a singleton factory
    my_service_factory = SingletonFactory(MyService)

    # Get singleton instance
    service = my_service_factory.get_instance(config="my_config")

    # Force new instance (useful for testing)
    new_service = my_service_factory.get_instance(force_new=True)

    # Reset instance
    my_service_factory.reset()
"""

import threading
from typing import Any, Callable, Dict, Optional, TypeVar, Generic

T = TypeVar("T")


class SingletonFactory(Generic[T]):
    """
    Thread-safe singleton factory.

    Creates and manages a single instance of a class with support for:
    - Thread-safe initialization
    - Force new instance (for testing)
    - Instance reset
    - Cleanup callbacks

    Example:
        >>> factory = SingletonFactory(MyService)
        >>> service1 = factory.get_instance()
        >>> service2 = factory.get_instance()
        >>> assert service1 is service2  # Same instance

        >>> # Force new instance
        >>> service3 = factory.get_instance(force_new=True)
        >>> assert service1 is not service3  # Different instance
    """

    def __init__(
        self,
        cls: Callable[..., T],
        cleanup_method: Optional[str] = None,
    ):
        """
        Initialize singleton factory.

        Args:
            cls: Class to create singleton of
            cleanup_method: Optional method name to call on reset (e.g., "shutdown")
        """
        self._cls = cls
        self._instance: Optional[T] = None
        self._lock = threading.Lock()
        self._cleanup_method = cleanup_method

    def get_instance(
        self,
        force_new: bool = False,
        **kwargs: Any,
    ) -> T:
        """
        Get singleton instance.

        Args:
            force_new: Force create new instance (resets existing)
            **kwargs: Arguments to pass to class constructor

        Returns:
            Singleton instance of the class
        """
        if self._instance is None or force_new:
            with self._lock:
                if self._instance is None or force_new:
                    # Cleanup existing instance if forcing new
                    if force_new and self._instance is not None:
                        self._cleanup()

                    self._instance = self._cls(**kwargs)

        return self._instance

    def reset(self) -> None:
        """
        Reset singleton instance.

        Calls cleanup method if defined, then clears the instance.
        """
        with self._lock:
            self._cleanup()
            self._instance = None

    def _cleanup(self) -> None:
        """Call cleanup method on instance if defined."""
        if self._instance is not None and self._cleanup_method:
            cleanup_fn = getattr(self._instance, self._cleanup_method, None)
            if callable(cleanup_fn):
                try:
                    cleanup_fn()
                except Exception:
                    pass  # Ignore cleanup errors

    @property
    def has_instance(self) -> bool:
        """Check if instance exists."""
        return self._instance is not None

    @property
    def instance(self) -> Optional[T]:
        """Get current instance without creating new one."""
        return self._instance


class SingletonMeta(type):
    """
    Metaclass for creating singleton classes.

    Alternative to SingletonFactory for classes that should always be singletons.

    Usage:
        class MyService(metaclass=SingletonMeta):
            def __init__(self, config=None):
                self.config = config

        # All instances are the same
        service1 = MyService()
        service2 = MyService()
        assert service1 is service2
    """

    _instances: Dict[type, Any] = {}
    _locks: Dict[type, threading.Lock] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._locks:
            cls._locks[cls] = threading.Lock()

        with cls._locks[cls]:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]

    @classmethod
    def reset(mcs, cls: type) -> None:
        """Reset singleton instance for a class."""
        if cls in mcs._instances:
            del mcs._instances[cls]


def singleton(cleanup_method: Optional[str] = None):
    """
    Decorator to make a class a singleton.

    Usage:
        @singleton()
        class MyService:
            def __init__(self, config=None):
                self.config = config

        # Get instance
        service = MyService.get_instance()

        # Reset
        MyService.reset_instance()
    """

    def decorator(cls):
        factory = SingletonFactory(cls, cleanup_method=cleanup_method)

        # Add class methods for singleton access
        cls.get_instance = staticmethod(factory.get_instance)
        cls.reset_instance = staticmethod(factory.reset)
        cls._singleton_factory = factory

        return cls

    return decorator
