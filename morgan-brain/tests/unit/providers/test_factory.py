"""Unit tests for providers/factory.py.

All tests use in-process adapters or Ollama adapters that construct fine offline.
No network calls are made.
"""
from __future__ import annotations

import pytest

from morgan_brain.config import Settings
from morgan_brain.interfaces.embedding import Embedder
from morgan_brain.interfaces.llm import ChatClient
from morgan_brain.providers.factory import build_embedder, build_router
from morgan_brain.providers.router import RoleRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_settings(**overrides: object) -> Settings:
    """Return a Settings that avoids reading .env / env vars."""
    defaults: dict[str, object] = {
        "llm_model": "qwen2.5:7b",
        "llm_fast_model": "qwen2.5:7b",
        "llm_endpoint": "http://localhost:11434/v1",
        "embedding_model": "qwen3-embedding:4b",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_router
# ---------------------------------------------------------------------------


def test_build_router_returns_role_router():
    s = _minimal_settings()
    router = build_router(s)
    assert isinstance(router, RoleRouter)


def test_build_router_strong_role_exists():
    s = _minimal_settings()
    router = build_router(s)
    # chat_for must not raise for "strong"
    client, model = router.chat_for("strong")
    assert isinstance(client, ChatClient)
    assert isinstance(model, str) and model


def test_build_router_fast_role_exists():
    s = _minimal_settings()
    router = build_router(s)
    client, model = router.chat_for("fast")
    assert isinstance(client, ChatClient)


def test_build_router_custom_role_bindings():
    """A custom role_bindings dict in Settings is honoured."""
    s = _minimal_settings(
        role_bindings={"strong": ["ollama:qwen2.5:7b"], "fast": ["ollama:qwen2.5:7b"]},
    )
    router = build_router(s)
    client, model = router.chat_for("strong")
    assert model == "qwen2.5:7b"


def test_build_router_unknown_role_raises_lookup_error():
    s = _minimal_settings()
    router = build_router(s)
    with pytest.raises(LookupError):
        router.chat_for("nonexistent_role")


# ---------------------------------------------------------------------------
# build_embedder
# ---------------------------------------------------------------------------


def test_build_embedder_returns_embedder():
    s = _minimal_settings()
    emb = build_embedder(s)
    assert isinstance(emb, Embedder)


def test_build_embedder_custom_endpoint():
    s = _minimal_settings(llm_endpoint="http://custom:11434/v1")
    emb = build_embedder(s)
    assert isinstance(emb, Embedder)


# ---------------------------------------------------------------------------
# YAML override (models_yaml=None path)
# ---------------------------------------------------------------------------


def test_build_router_no_yaml_override_does_not_raise():
    s = _minimal_settings(models_yaml=None)
    router = build_router(s)
    assert isinstance(router, RoleRouter)
