"""Capability descriptors and registry.

A CapabilityDescriptor records what a provider/model can do (context window, tool support,
JSON mode, vision, embedding dim, cost). The CapabilityRegistry stores these descriptors by
"provider/model" key and returns conservative defaults on miss.

Capability is explicit (vendored seed + runtime override), never inferred from the response.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class JsonMode(str, Enum):
    """Levels of native JSON/structured-output support a model advertises."""

    NONE = "none"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class CapabilityDescriptor(BaseModel):
    """All capability facts for one provider/model pair.

    Defaults are deliberately conservative so that code that forgets to seed a model still
    works — it just won't use advanced features.
    """

    provider: str
    model: str

    # Context / output
    context_window: int = 4096
    max_output: int = 2048

    # Tool / function calling
    supports_tools: bool = False
    supports_parallel_tools: bool = False

    # Structured output
    json_mode: JsonMode = JsonMode.NONE
    supports_grammar: bool = False  # GBNF or equivalent grammar-constrained decoding

    # Multimodal
    supports_vision: bool = False

    # Embeddings (0 means this is a chat model, not an embedding model)
    embedding_dim: int = 0

    # Cost per 1 M tokens (USD); 0.0 = local/unknown
    cost_in: float = 0.0
    cost_out: float = 0.0


class CapabilityRegistry:
    """Maps "provider/model" → CapabilityDescriptor.

    Conservative defaults are returned on miss so callers don't need to guard None.
    """

    def __init__(self, store: dict[str, CapabilityDescriptor]) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_seed(cls, data: dict[str, Any]) -> "CapabilityRegistry":
        """Build a registry from a plain dict keyed by "provider/model".

        Each value is a dict of field overrides; missing fields use conservative defaults.
        The provider and model fields are inferred from the key if not present.
        """
        store: dict[str, CapabilityDescriptor] = {}
        for key, overrides in data.items():
            # Skip metadata/comment keys that are not "provider/model" entries.
            if not isinstance(overrides, dict):
                continue
            provider, sep, model = key.partition("/")
            if not sep:
                continue  # malformed key — skip silently
            fields: dict[str, Any] = {"provider": provider, "model": model}
            fields.update(overrides)
            store[key] = CapabilityDescriptor(**fields)
        return cls(store)

    @classmethod
    def from_packaged(cls) -> "CapabilityRegistry":
        """Load the vendored seed from ``providers/data/model_capabilities.json``."""
        data_path = Path(__file__).parent / "data" / "model_capabilities.json"
        with data_path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = json.load(fh)
        return cls.from_seed(raw)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, provider: str, model: str) -> CapabilityDescriptor:
        """Return the descriptor for *provider/model*, or a conservative default on miss."""
        key = f"{provider}/{model}"
        if key in self._store:
            return self._store[key]
        # Conservative default — unknown model gets the safest assumptions.
        return CapabilityDescriptor(provider=provider, model=model)

    def override(self, provider: str, model: str, partial: dict[str, Any]) -> None:
        """Merge *partial* field updates into an existing or default descriptor."""
        existing = self.get(provider, model)
        updated_fields = existing.model_dump()
        updated_fields.update(partial)
        key = f"{provider}/{model}"
        self._store[key] = CapabilityDescriptor(**updated_fields)
