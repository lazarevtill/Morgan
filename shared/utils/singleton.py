"""
Thread-safe singleton factory for services.

Provides a unified singleton pattern used across all Morgan services.
"""
import threading
from typing import TypeVar, Type, Optional, Dict, Any, Callable
import logging

T = TypeVar('T')

logger = logging.getLogger(__name__)


class SingletonFactory:
    """
    Thread-safe singleton factory with cleanup support.

    Usage:
        # Get or create singleton
        service = SingletonFactory.get_or_create(
            MyService,
            factory=lambda: MyService(config)
        )

        # Reset singleton (calls cleanup methods)
        SingletonFactory.reset(MyService)

        # Reset all singletons
        SingletonFactory.reset_all()
    """

    _instances: Dict[Type, Any] = {}
    _locks: Dict[Type, threading.Lock] = {}
    _global_lock = threading.Lock()

    @classmethod
    def get_or_create(
        cls,
        service_class: Type[T],
        factory: Optional[Callable[[], T]] = None,
        force_new: bool = False,
        **kwargs
    ) -> T:
        """
        Get existing instance or create new one.

        Args:
            service_class: The class to instantiate
            factory: Optional factory function to create instance
            force_new: Force creation of new instance
            **kwargs: Arguments passed to constructor if no factory

        Returns:
            Singleton instance of the class
        """
        # Ensure lock exists for this class
        with cls._global_lock:
            if service_class not in cls._locks:
                cls._locks[service_class] = threading.Lock()

        # Double-checked locking pattern
        if service_class not in cls._instances or force_new:
            with cls._locks[service_class]:
                if service_class not in cls._instances or force_new:
                    if force_new and service_class in cls._instances:
                        # Clean up old instance first
                        cls._cleanup_instance(cls._instances[service_class])

                    if factory:
                        cls._instances[service_class] = factory()
                    else:
                        cls._instances[service_class] = service_class(**kwargs)

                    logger.debug(f"Created singleton instance: {service_class.__name__}")

        return cls._instances[service_class]

    @classmethod
    def _cleanup_instance(cls, instance: Any) -> None:
        """Clean up an instance by calling cleanup methods."""
        cleanup_methods = ['shutdown', 'close', 'cleanup', 'clear_cache']

        for method_name in cleanup_methods:
            if hasattr(instance, method_name):
                try:
                    method = getattr(instance, method_name)
                    if callable(method):
                        method()
                        logger.debug(f"Called {method_name}() on {type(instance).__name__}")
                        break
                except Exception as e:
                    logger.warning(f"Error calling {method_name}() on {type(instance).__name__}: {e}")

    @classmethod
    def reset(cls, service_class: Type[T]) -> None:
        """
        Reset singleton instance with cleanup.

        Args:
            service_class: The class to reset
        """
        with cls._global_lock:
            if service_class in cls._instances:
                instance = cls._instances[service_class]
                cls._cleanup_instance(instance)
                del cls._instances[service_class]
                logger.debug(f"Reset singleton instance: {service_class.__name__}")

    @classmethod
    def reset_all(cls) -> None:
        """Reset all singleton instances."""
        with cls._global_lock:
            for service_class in list(cls._instances.keys()):
                cls.reset(service_class)
            logger.debug("Reset all singleton instances")

    @classmethod
    def has_instance(cls, service_class: Type[T]) -> bool:
        """Check if a singleton instance exists."""
        return service_class in cls._instances

    @classmethod
    def get_instance(cls, service_class: Type[T]) -> Optional[T]:
        """Get existing instance without creating new one."""
        return cls._instances.get(service_class)


def singleton(cls: Type[T]) -> Type[T]:
    """
    Decorator to make a class a singleton.

    Usage:
        @singleton
        class MyService:
            def __init__(self):
                pass

        # Always returns same instance
        service1 = MyService()
        service2 = MyService()
        assert service1 is service2
    """
    original_new = cls.__new__

    def new_new(cls_arg, *args, **kwargs):
        return SingletonFactory.get_or_create(
            cls_arg,
            lambda: object.__new__(cls_arg)
        )

    cls.__new__ = new_new
    return cls
