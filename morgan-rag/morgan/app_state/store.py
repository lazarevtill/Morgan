"""Thread-safe application state store with subscriber notifications."""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class AppState:
    """Snapshot of the application's runtime state."""

    verbose: bool = False
    main_model: str = "qwen2.5:7b"
    permission_mode: str = "interactive"
    tasks: dict[str, Any] = field(default_factory=dict)
    channels: dict[str, Any] = field(default_factory=dict)
    plugins: dict[str, Any] = field(default_factory=dict)
    skills: list[str] = field(default_factory=list)
    status_text: Optional[str] = None


class AppStateStore:
    """Thread-safe observable state container.

    ``get_state`` returns a *copy* so callers cannot mutate the store
    directly. Use ``set_state(updater)`` to apply changes atomically.
    """

    def __init__(self, initial: Optional[AppState] = None) -> None:
        self._lock = threading.RLock()
        self._state: AppState = initial if initial is not None else AppState()
        self._listeners: list[Callable[[AppState], Any]] = []

    def get_state(self) -> AppState:
        """Return a deep copy of the current state."""
        with self._lock:
            return copy.deepcopy(self._state)

    def set_state(self, updater: Callable[[AppState], None]) -> None:
        """Apply *updater* to the state and notify subscribers.

        The updater receives the live state object under the lock -- it
        should mutate it in-place.
        """
        with self._lock:
            updater(self._state)
            snapshot = copy.deepcopy(self._state)
        # Notify outside the lock to avoid deadlocks in listeners.
        for listener in list(self._listeners):
            listener(snapshot)

    def subscribe(self, listener: Callable[[AppState], Any]) -> Callable[[], None]:
        """Register a listener called on every state change.

        Returns an ``unsubscribe`` callable.
        """
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                try:
                    self._listeners.remove(listener)
                except ValueError:
                    pass

        return unsubscribe
