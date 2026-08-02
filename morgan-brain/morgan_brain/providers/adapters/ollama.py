"""Ollama adapter — ``OpenAICompatAdapter`` preconfigured for a local Ollama server.

Ollama exposes an OpenAI-compatible API at ``/v1``.  The only differences from a
stock OpenAI endpoint are:

* The default base URL is ``http://localhost:11434/v1``.
* Any non-empty api_key works (we use ``"ollama"``).
* Context-window sizing: Ollama honours the ``num_ctx`` option; for structured
  requests that need a large context, pass ``num_ctx`` via ``extra_body`` (not
  yet wired here — left for a future increment).

Usage::

    from morgan_brain.providers.adapters.ollama import OllamaAdapter

    adapter = OllamaAdapter()                           # local defaults
    adapter = OllamaAdapter(base_url="http://host:11434/v1")  # remote
"""

from __future__ import annotations

from morgan_brain.providers.adapters.openai_compat import OpenAICompatAdapter

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_API_KEY = "ollama"
_PROVIDER = "ollama"


class OllamaAdapter(OpenAICompatAdapter):
    """``OpenAICompatAdapter`` pre-configured for Ollama.

    Args:
        base_url: Override the Ollama base URL (default: ``http://localhost:11434/v1``).
        api_key:  Override the API key (default: ``"ollama"``).
        timeout:  Request timeout in seconds (default: 120.0 — see OpenAICompatAdapter).
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        api_key: str = _DEFAULT_API_KEY,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key, provider=_PROVIDER, timeout=timeout)
