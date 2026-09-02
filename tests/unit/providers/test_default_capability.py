"""Unit tests: CapabilityRegistry.from_packaged() returns usable descriptors.

Asserts that the default production models in model_capabilities.json have
descriptors that are complete and capable enough for the default config to
work — so the default config can never silently break tool routing.

The key invariant: ``CapabilityRegistry.from_packaged().get('ollama', 'qwen2.5:7b')``
must return a descriptor that:
- has supports_tools=True  (required for the tool-call loop)
- has json_mode != 'none'  (required for structured output / consolidation)
- has a realistic context_window (not 4096 conservative default)
- is not the default miss-fallback (meaning the key actually exists in the seed)
"""

from __future__ import annotations

from morgan_brain.providers.capability import CapabilityDescriptor, CapabilityRegistry, JsonMode


def test_packaged_qwen25_7b_is_tool_capable() -> None:
    """ollama/qwen2.5:7b from packaged seed must have supports_tools=True.

    This is the DEFAULT_LLM_MODEL in settings.  If this breaks, the tool-call
    loop silently falls back to no-tools mode in production.
    """
    reg = CapabilityRegistry.from_packaged()
    d = reg.get("ollama", "qwen2.5:7b")

    assert d.supports_tools is True, (
        "ollama/qwen2.5:7b must support tools — tool-call loop depends on this"
    )
    assert d.json_mode in (JsonMode.JSON_OBJECT, JsonMode.JSON_SCHEMA), (
        "ollama/qwen2.5:7b must support JSON mode — consolidation depends on this"
    )
    assert d.context_window >= 8192, (
        f"ollama/qwen2.5:7b context_window should be realistic, got {d.context_window}"
    )


def test_packaged_qwen25_7b_is_not_conservative_fallback() -> None:
    """The qwen2.5:7b entry must exist in the seed (not a miss returning defaults)."""
    reg = CapabilityRegistry.from_packaged()
    d = reg.get("ollama", "qwen2.5:7b")

    # Conservative default has context_window=4096; the real entry has 32768.
    # A miss would return the conservative default.
    assert d.context_window != 4096, (
        "ollama/qwen2.5:7b appears to be missing from model_capabilities.json "
        "(got conservative default context_window=4096)"
    )


def test_packaged_embedding_model_has_nonzero_dim() -> None:
    """The default embedding model must have embedding_dim > 0."""
    reg = CapabilityRegistry.from_packaged()
    d = reg.get("ollama", "qwen3-embedding:4b")

    assert d.embedding_dim > 0, (
        "ollama/qwen3-embedding:4b must have embedding_dim > 0 in the packaged seed"
    )
    assert d.supports_tools is False, "Embedding models must not claim tool support"


def test_packaged_descriptor_fields_are_complete() -> None:
    """All required fields are present and well-typed in the packaged descriptor."""
    reg = CapabilityRegistry.from_packaged()
    d = reg.get("ollama", "qwen2.5:7b")

    assert isinstance(d, CapabilityDescriptor)
    assert isinstance(d.context_window, int) and d.context_window > 0
    assert isinstance(d.max_output, int) and d.max_output >= 0
    assert isinstance(d.supports_tools, bool)
    assert isinstance(d.supports_parallel_tools, bool)
    assert isinstance(d.json_mode, JsonMode)
    assert isinstance(d.embedding_dim, int) and d.embedding_dim >= 0
    assert isinstance(d.cost_in, float) and d.cost_in >= 0.0
    assert isinstance(d.cost_out, float) and d.cost_out >= 0.0


def test_packaged_registry_conservative_miss_still_usable() -> None:
    """Unknown models return a conservative default — not None, not an exception."""
    reg = CapabilityRegistry.from_packaged()
    d = reg.get("unknown-provider", "unknown-model-xyz")

    # Must be a valid CapabilityDescriptor, not None.
    assert isinstance(d, CapabilityDescriptor)
    # Conservative defaults: no tools, no JSON, small context.
    assert d.supports_tools is False
    assert d.json_mode == JsonMode.NONE
