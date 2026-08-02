"""Wave: llama.cpp defaults, judge/reflection role bindings, promotion flag.

Covers the composition-root hardcode removal described in Task 16: the default
provider key is no longer the literal "ollama", the judge/reflection roles are
bound (making the eval-gated optimize path reachable for the first time), champion
promotion ships disarmed by default, and the hash embedder stub is process-stable.

Also covers the follow-up correction: the owner's real deployment reaches a REMOTE
llama-server over an overlay network, not a loopback socket — so the provider surface
needs an outbound API key distinct from the inbound one, a network-sane request
timeout, and a non-fatal reachability probe.
"""

from __future__ import annotations

import pytest

from morgan_brain.config import Settings
from morgan_brain.modules.memory.indexing.embedder import OllamaEmbedder
from morgan_brain.providers.factory import check_llm_reachable


def test_default_provider_is_not_ollama() -> None:
    s = Settings()
    assert "ollama" not in s.providers
    assert all("ollama:" not in b for bs in s.role_bindings.values() for b in bs)


def test_judge_and_reflection_roles_are_bound() -> None:
    s = Settings()
    assert set(s.role_bindings) >= {"strong", "fast", "judge", "reflection"}


def test_promotion_is_disarmed_by_default() -> None:
    assert Settings().enable_champion_promotion is False


def test_hash_backend_is_stable_across_processes() -> None:
    """PYTHONHASHSEED randomises builtin hash(); the stub must not use it."""
    import json
    import subprocess
    import sys

    code = (
        "import asyncio,json;"
        "from morgan_brain.providers.factory import build_hash_embedder;"
        "print(json.dumps(asyncio.run(build_hash_embedder().embed('harbor'))))"
    )
    runs = [
        json.loads(
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, check=True
            ).stdout
        )
        for _ in range(2)
    ]
    assert runs[0] == runs[1]


def test_outbound_llm_api_key_is_distinct_from_inbound_api_key() -> None:
    """MORGAN_LLM_API_KEY (outbound, to the model server) must never be conflated with
    MORGAN_API_KEY (inbound, from Morgan's own clients) — they point in opposite directions."""
    s = Settings(api_key="inbound-secret", llm_api_key="outbound-secret")
    assert s.api_key == "inbound-secret"
    assert s.llm_api_key == "outbound-secret"
    assert s.providers["llamacpp"]["api_key"] == "outbound-secret"


def test_llamacpp_provider_falls_back_to_placeholder_key_when_unset() -> None:
    """No llm_api_key configured (the common homelab case) still yields a non-empty api_key —
    the openai SDK client requires SOME string even when the server enforces no auth."""
    s = Settings()
    assert s.llm_api_key == ""
    assert s.providers["llamacpp"]["api_key"] == "llamacpp"


def test_llm_timeout_is_network_sized_and_configurable() -> None:
    """Default must assume a network hop (remote llama-server), not a loopback socket."""
    assert Settings().llm_timeout_seconds == 120.0
    s = Settings(llm_timeout_seconds=45.0)
    assert s.providers["llamacpp"]["timeout"] == 45.0


def test_ollama_embedder_sends_bearer_header_only_when_api_key_set() -> None:
    """Remote llama-server needs auth on embeddings too — the embedder previously sent no
    Authorization header at all, which silently 401s against an --api-key-protected server."""
    with_key = OllamaEmbedder("http://example:8081/v1", "m", api_key="secret")
    assert with_key._headers == {"Authorization": "Bearer secret"}

    without_key = OllamaEmbedder("http://example:8081/v1", "m")
    assert without_key._headers == {}


@pytest.mark.asyncio
async def test_check_llm_reachable_is_non_fatal_when_endpoint_is_down() -> None:
    """An unreachable model server (laptop off the network, homelab rebooting) is a normal
    transient in the remote-first topology — the probe must return False, never raise."""
    s = Settings(llm_endpoint="http://127.0.0.1:1/v1")
    assert await check_llm_reachable(s, timeout=1.0) is False
