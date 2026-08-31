"""Role router — maps a *role* (e.g. "strong", "fast") to a capable ``ChatClient``.

Usage::

    reg = CapabilityRegistry.from_packaged()
    router = RoleRouter(reg=reg, bindings={
        "strong": [Binding("llamacpp", "qwen2.5:7b", client)],
    })
    client, model = router.chat_for("strong", needs_tools=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from morgan_brain.providers.capability import CapabilityRegistry, JsonMode

if TYPE_CHECKING:
    from morgan_brain.interfaces.llm import ChatClient


@dataclass
class Binding:
    """Associates a provider/model pair with a concrete ``ChatClient`` instance."""

    provider: str
    model: str
    client: ChatClient


class RoleRouter:
    """Selects the first binding for a *role* that satisfies all requested capabilities.

    Args:
        reg:      ``CapabilityRegistry`` used to look up descriptor for each binding.
        bindings: Map of role name → ordered list of ``Binding`` objects (priority order,
                  first match wins).

    Raises:
        LookupError: If the role is not registered, or no binding satisfies the request.
    """

    def __init__(
        self,
        reg: CapabilityRegistry,
        bindings: dict[str, list[Binding]],
    ) -> None:
        self._reg = reg
        self._bindings = bindings

    def chat_for(
        self,
        role: str,
        *,
        needs_tools: bool = False,
        needs_json_schema: bool = False,
        needs_vision: bool = False,
        min_context: int = 0,
    ) -> tuple[ChatClient, str]:
        """Return ``(client, model)`` for the first binding that satisfies all caps.

        Args:
            role:             Role name to look up (must be registered in *bindings*).
            needs_tools:      Binding must have ``supports_tools=True``.
            needs_json_schema: Binding must have ``json_mode == JsonMode.JSON_SCHEMA``.
            needs_vision:     Binding must have ``supports_vision=True``.
            min_context:      Binding must have ``context_window >= min_context``.

        Raises:
            LookupError: No registered binding satisfies the request.
        """
        candidates = self._bindings.get(role)
        if not candidates:
            raise LookupError(
                f"No bindings registered for role {role!r}. Known roles: {list(self._bindings)}"
            )

        for binding in candidates:
            desc = self._reg.get(binding.provider, binding.model)
            if needs_tools and not desc.supports_tools:
                continue
            if needs_json_schema and desc.json_mode != JsonMode.JSON_SCHEMA:
                continue
            if needs_vision and not desc.supports_vision:
                continue
            if min_context and desc.context_window < min_context:
                continue
            return binding.client, binding.model

        raise LookupError(
            f"No binding for role {role!r} satisfies the requested capabilities "
            f"(needs_tools={needs_tools}, needs_json_schema={needs_json_schema}, "
            f"needs_vision={needs_vision}, min_context={min_context})."
        )

    def bindings_for(self, role: str) -> list[Binding]:
        """Return the full binding list for a role (used by RoleFallback)."""
        return list(self._bindings.get(role, []))
