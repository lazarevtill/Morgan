from morgan_brain.providers.capability import CapabilityDescriptor, CapabilityRegistry, JsonMode


def test_descriptor_defaults_conservative():
    d = CapabilityDescriptor(provider="ollama", model="qwen2.5:7b")
    assert d.json_mode in (JsonMode.NONE, JsonMode.JSON_OBJECT, JsonMode.JSON_SCHEMA)
    assert d.context_window >= 2048


def test_registry_lookup_and_override():
    reg = CapabilityRegistry.from_seed(
        {
            "ollama/qwen2.5:7b": {
                "context_window": 32768,
                "supports_tools": True,
                "json_mode": "json_schema",
                "embedding_dim": 0,
            }
        }
    )
    d = reg.get("ollama", "qwen2.5:7b")
    assert d.context_window == 32768 and d.supports_tools is True
    reg.override("ollama", "qwen2.5:7b", {"context_window": 8192})
    assert reg.get("ollama", "qwen2.5:7b").context_window == 8192


def test_registry_unknown_returns_conservative_default():
    reg = CapabilityRegistry.from_seed({})
    d = reg.get("x", "y")
    assert d.supports_tools is False and d.json_mode == JsonMode.NONE


def test_registry_from_packaged_loads_known_models():
    reg = CapabilityRegistry.from_packaged()
    d = reg.get("ollama", "qwen2.5:7b")
    assert d.context_window >= 32768 and d.supports_tools is True


def test_registry_from_packaged_embedding_model():
    reg = CapabilityRegistry.from_packaged()
    d = reg.get("ollama", "qwen3-embedding:4b")
    assert d.embedding_dim > 0 and d.supports_tools is False


def test_descriptor_conservative_defaults():
    d = CapabilityDescriptor(provider="p", model="m")
    assert d.context_window == 4096
    assert d.max_output == 2048
    assert d.supports_tools is False
    assert d.supports_parallel_tools is False
    assert d.json_mode == JsonMode.NONE
    assert d.supports_grammar is False
    assert d.supports_vision is False
    assert d.embedding_dim == 0
    assert d.cost_in == 0.0
    assert d.cost_out == 0.0
