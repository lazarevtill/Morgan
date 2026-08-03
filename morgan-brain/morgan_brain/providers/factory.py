"""Factory functions — build a ``RoleRouter`` from ``Settings``.

This is the only place that translates Settings values into concrete adapters.
Everything above this layer depends on the seam types (``ChatClient``, ``RoleRouter``)
rather than concrete adapters or SDKs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from morgan_brain.config import Settings
from morgan_brain.modules.memory.indexing.embedder import Embedder, FakeEmbedder
from morgan_brain.providers.adapters.embeddings import OpenAICompatEmbedder
from morgan_brain.providers.adapters.ollama import OllamaAdapter
from morgan_brain.providers.adapters.openai_compat import OpenAICompatAdapter
from morgan_brain.providers.capability import CapabilityRegistry
from morgan_brain.providers.router import Binding, RoleRouter


def _make_chat_adapter(provider: str, provider_cfg: dict[str, Any]) -> OpenAICompatAdapter:
    """Instantiate the right chat adapter for a provider name."""
    base_url: str = provider_cfg.get("base_url", "http://localhost:8081/v1")
    api_key: str = provider_cfg.get("api_key", "")
    timeout: float = provider_cfg.get("timeout", 120.0)

    if provider == "ollama":
        # Still supported as a non-default key for owners who haven't moved to llama.cpp.
        return OllamaAdapter(base_url=base_url, api_key=api_key or "ollama", timeout=timeout)
    # For all other OpenAI-compatible providers (llamacpp — the default, openai, vllm,
    # openrouter…)
    return OpenAICompatAdapter(
        base_url=base_url, api_key=api_key, provider=provider, timeout=timeout
    )


def build_hash_embedder(dim: int = 1024) -> Embedder:
    """Build the deterministic sha256-hash embedder stub (``embedding_backend="hash"``).

    Reuses ``FakeEmbedder`` rather than a second implementation. Unlike a builtin-``hash()``
    stub, sha256 is stable across processes regardless of ``PYTHONHASHSEED`` — required for the
    CLI (a subprocess) and the store to agree on vectors for the same text.
    """
    return FakeEmbedder(dim=dim)


def build_embedder(settings: Settings) -> Embedder:
    """Build the configured ``Embedder`` (provider or hash) from Settings.

    The single place that decides live-provider vs. deterministic stub — composition.py must
    not construct an embedder directly, so switching backends stays a config change.
    """
    if settings.embedding_backend == "hash":
        return build_hash_embedder(dim=settings.embedding_dim)
    return OpenAICompatEmbedder(
        settings.llm_endpoint,
        settings.embedding_model,
        timeout=settings.llm_timeout_seconds,
        api_key=settings.llm_api_key or None,
    )


async def check_llm_reachable(settings: Settings, *, timeout: float = 5.0) -> bool:
    """Best-effort reachability check for the configured LLM endpoint.

    GETs the OpenAI-compatible ``/models`` listing, which llama-server, vLLM, and Ollama's
    ``/v1`` shim all serve. Deliberately short-timeout and non-raising: this exists for the
    CLI's ``doctor`` command (Task 17) to answer "which llama-server am I talking to, and can
    I reach it right now" — the first question anyone asks when a remote endpoint is down.
    """
    import httpx

    url = settings.llm_endpoint.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
        return resp.status_code < 500
    except Exception:  # noqa: BLE001 — unreachable is a normal answer, not an error to surface
        return False


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
