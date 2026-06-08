"""Factory functions — build a ``RoleRouter`` or ``Embedder`` from ``Settings``.

This is the only place that translates Settings values into concrete adapters.
Everything above this layer depends on the seam types (``ChatClient``, ``Embedder``,
``RoleRouter``) rather than concrete adapters or SDKs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from morgan_brain.config import Settings
from morgan_brain.interfaces.embedding import Embedder
from morgan_brain.providers.adapters.ollama import OllamaAdapter
from morgan_brain.providers.adapters.openai_compat import (
    OpenAICompatAdapter,
    OpenAICompatEmbedder,
)
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter


def _make_chat_adapter(provider: str, provider_cfg: dict[str, Any]) -> OpenAICompatAdapter:
    """Instantiate the right chat adapter for a provider name."""
    base_url: str = provider_cfg.get("base_url", "http://localhost:11434/v1")
    api_key: str = provider_cfg.get("api_key", "")

    if provider == "ollama":
        return OllamaAdapter(base_url=base_url, api_key=api_key or "ollama")
    # For all other OpenAI-compatible providers (openai, vllm, llamacpp, openrouter…)
    return OpenAICompatAdapter(base_url=base_url, api_key=api_key, provider=provider)


def build_router(settings: Settings) -> RoleRouter:
    """Build a ``RoleRouter`` from ``Settings``.

    Steps:
    1. Load the packaged capability registry.
    2. Apply any YAML override file if ``settings.models_yaml`` is set.
    3. Instantiate one adapter per unique provider referenced in the bindings.
    4. Build ``Binding`` objects per role.
    """
    reg = CapabilityRegistry.from_packaged()

    # Optional YAML capability overrides
    if settings.models_yaml:
        _apply_yaml_overrides(reg, settings.models_yaml)

    # Build one adapter per provider (de-duplicated)
    provider_adapters: dict[str, OpenAICompatAdapter] = {}
    for role_specs in settings.role_bindings.values():
        for spec in role_specs:
            provider, _, _model = spec.partition(":")
            if provider not in provider_adapters:
                provider_cfg = settings.providers.get(provider, {})
                provider_adapters[provider] = _make_chat_adapter(provider, provider_cfg)

    # Build bindings per role
    bindings: dict[str, list[Binding]] = {}
    for role, role_specs in settings.role_bindings.items():
        role_bindings: list[Binding] = []
        for spec in role_specs:
            provider, _, model = spec.partition(":")
            if not model:
                continue  # malformed spec — skip
            adapter = provider_adapters.get(provider)
            if adapter is None:
                continue
            role_bindings.append(Binding(provider=provider, model=model, client=adapter))
        if role_bindings:
            bindings[role] = role_bindings

    return RoleRouter(reg=reg, bindings=bindings)


def build_embedder(settings: Settings) -> Embedder:
    """Build an ``Embedder`` from ``Settings``.

    Uses the Ollama embedding endpoint by default.  If the ``ollama`` provider
    config overrides the base_url, that URL is used.
    """
    ollama_cfg = settings.providers.get("ollama", {})
    base_url: str = ollama_cfg.get("base_url", settings.llm_endpoint)
    api_key: str = ollama_cfg.get("api_key", "ollama")
    model: str = settings.embedding_model

    return OpenAICompatEmbedder(base_url=base_url, api_key=api_key, model=model)


def _apply_yaml_overrides(reg: CapabilityRegistry, yaml_path: str) -> None:
    """Load a YAML file of capability overrides and apply them to the registry.

    The YAML must have the same structure as ``model_capabilities.json``:
    top-level keys are ``"provider/model"`` strings; values are partial
    ``CapabilityDescriptor`` field dicts.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        # PyYAML not installed — skip silently (factory still works without it)
        return

    path = Path(yaml_path)
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    for key, partial in data.items():
        if not isinstance(partial, dict):
            continue
        provider, sep, model = key.partition("/")
        if not sep:
            continue
        reg.override(provider, model, partial)
