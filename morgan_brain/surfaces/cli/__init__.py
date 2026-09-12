"""The ``morgan`` CLI -- the first surface an owner touches directly.

``remember``/``recall``/``facts``/``forget``/``doctor`` are direct memory operations against
``composition.build_memory_context`` (no LLM router required). ``ask`` is a full chat turn
against ``composition.build_app_context`` (requires a reachable LLM). See ``__main__.py``.
"""

from __future__ import annotations
