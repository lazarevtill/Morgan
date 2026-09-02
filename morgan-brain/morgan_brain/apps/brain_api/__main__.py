"""Run brain-api: ``python -m morgan_brain.apps.brain_api``.

Bind host and port come from ``MORGAN_API_HOST`` / ``MORGAN_API_PORT`` and default to
loopback. Binding beyond loopback requires ``MORGAN_API_KEY`` -- see
``security/network.py::assert_safe_bind``, which refuses to start otherwise.
"""

from __future__ import annotations

import uvicorn

from morgan_brain.config import get_settings
from morgan_brain.logging_setup import configure_logging
from morgan_brain.security.network import assert_safe_bind


def main() -> None:
    configure_logging()
    settings = get_settings()
    assert_safe_bind(host=settings.api_host, api_key=settings.api_key, surface="brain-api")
    uvicorn.run(
        "morgan_brain.apps.brain_api.app:app", host=settings.api_host, port=settings.api_port
    )


if __name__ == "__main__":
    main()
