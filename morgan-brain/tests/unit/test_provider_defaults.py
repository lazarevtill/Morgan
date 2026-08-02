"""Wave: llama.cpp defaults, judge/reflection role bindings, promotion flag.

Covers the composition-root hardcode removal described in Task 16: the default
provider key is no longer the literal "ollama", the judge/reflection roles are
bound (making the eval-gated optimize path reachable for the first time), champion
promotion ships disarmed by default, and the hash embedder stub is process-stable.
"""

from __future__ import annotations

from morgan_brain.config import Settings


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
