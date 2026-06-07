"""morgan CLI entrypoint. Phase 0: a health check against brain-api.

Phase 1+ adds: interactive chat, `morgan memory`, `morgan skills`, `morgan workspace`.
"""
from __future__ import annotations

import os
import sys

import httpx

BASE_URL = os.environ.get("MORGAN_API_URL", "http://localhost:8080")


def main() -> None:
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        print(resp.json())
    except httpx.HTTPError as exc:
        print(f"morgan: cannot reach brain-api at {BASE_URL}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
